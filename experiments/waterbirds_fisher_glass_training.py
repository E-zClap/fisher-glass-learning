"""Waterbirds Fisher-Glass training on frozen features.

This is a deliberately lightweight training experiment: it uses the local
Waterbirds-WILDS split, frozen ImageNet MobileNetV3-small features, and a
trainable linear classifier. The Fisher-Glass objective is optimized in the
same low-dimensional operational geometry used by the diagnostic scripts:
the current class-margin logit is the task coordinate and a fixed background
probe score is the nuisance coordinate.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from wilds import get_dataset

from ridge_exact_audit import audit_scalar_environments
from waterbirds_fisher_glass import as_numpy, cvar, extract_features, fisher_schur


GROUP_NAMES = {
    0: "landbird/land",
    1: "landbird/water",
    2: "waterbird/land",
    3: "waterbird/water",
}


@dataclass
class SplitData:
    x: np.ndarray
    y: np.ndarray
    bg: np.ndarray
    group: np.ndarray
    nuisance: np.ndarray


def standardize(train: np.ndarray, values: np.ndarray) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    mean = train.mean(axis=0, keepdims=True)
    std = train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    return ((values - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def normalize_scores(train_scores: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, dict[str, float]]:
    mean = float(train_scores.mean())
    std = float(train_scores.std())
    if std < 1e-6:
        std = 1.0
    return ((scores - mean) / std).astype(np.float32), {"mean": mean, "std": std}


def group_ids(y: np.ndarray, bg: np.ndarray) -> np.ndarray:
    return (2 * y.astype(int) + bg.astype(int)).astype(int)


def torch_cvar(values: torch.Tensor, alpha: float) -> torch.Tensor:
    k = max(1, int(np.ceil((1.0 - alpha) * values.numel())))
    return torch.topk(values, k=k, largest=True).values.mean()


def fisher_glass_penalty(
    logits: torch.Tensor,
    nuisance: torch.Tensor,
    groups: torch.Tensor,
    ridge: float,
    eps: float,
    alpha: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = (logits - logits.mean()) / (logits.std(unbiased=False) + 1e-6)
    losses = []
    fishers = []
    for gid in range(4):
        mask = groups == gid
        zg = z[mask]
        bg = nuisance[mask]
        j_tt = torch.mean(zg * zg)
        j_tn = torch.mean(zg * bg)
        j_nn = torch.mean(bg * bg)
        projected = j_tt - (j_tn * j_tn) / (j_nn + ridge)
        projected = torch.clamp(projected, min=eps)
        fishers.append(projected)
        losses.append(1.0 / (projected + eps))
    inv = torch.stack(losses)
    return torch_cvar(inv, alpha), torch.stack(fishers)


def initialize_linear(dim: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    w = torch.randn(dim, generator=generator, dtype=torch.float32) * 0.01
    b = torch.zeros((), dtype=torch.float32)
    w.requires_grad_(True)
    b.requires_grad_(True)
    return w, b


def train_linear(
    train: SplitData,
    method: str,
    seed: int,
    epochs: int,
    lr: float,
    weight_decay: float,
    ridge: float,
    eps: float,
    fg_alpha: float,
    fg_lambda: float,
    jtt_weights: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    torch.manual_seed(seed)
    x = torch.tensor(train.x, dtype=torch.float32)
    y = torch.tensor(train.y.astype(np.float32), dtype=torch.float32)
    groups = torch.tensor(train.group, dtype=torch.long)
    nuisance = torch.tensor(train.nuisance.astype(np.float32), dtype=torch.float32)
    sample_weights = None if jtt_weights is None else torch.tensor(jtt_weights.astype(np.float32), dtype=torch.float32)

    w, bias = initialize_linear(x.shape[1], seed)
    opt = torch.optim.AdamW([w, bias], lr=lr, weight_decay=weight_decay)
    history = {}
    for _ in range(epochs):
        opt.zero_grad()
        logits = x @ w + bias
        losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        if method == "erm" or method == "jtt":
            if sample_weights is None:
                objective = losses.mean()
            else:
                objective = (losses * sample_weights).sum() / sample_weights.sum()
        elif method == "group_dro":
            group_losses = torch.stack([losses[groups == gid].mean() for gid in range(4)])
            objective = torch.max(group_losses)
        elif method == "loss_cvar":
            objective = torch_cvar(losses, 0.9)
        elif method == "fisher_glass":
            fg_loss, fishers = fisher_glass_penalty(logits, nuisance, groups, ridge, eps, fg_alpha)
            objective = losses.mean() + fg_lambda * fg_loss
            history = {"train_fg_penalty": float(fg_loss.detach()), "train_min_projected_fisher": float(fishers.min().detach())}
        elif method == "group_dro_fisher_glass":
            group_losses = torch.stack([losses[groups == gid].mean() for gid in range(4)])
            fg_loss, fishers = fisher_glass_penalty(logits, nuisance, groups, ridge, eps, fg_alpha)
            objective = torch.max(group_losses) + fg_lambda * fg_loss
            history = {
                "train_fg_penalty": float(fg_loss.detach()),
                "train_min_projected_fisher": float(fishers.min().detach()),
            }
        else:
            raise ValueError(f"Unknown method: {method}")
        objective.backward()
        opt.step()
    params = np.concatenate([w.detach().numpy(), np.array([float(bias.detach())], dtype=np.float32)])
    return params, history


def predict_logits(params: np.ndarray, x: np.ndarray) -> np.ndarray:
    return x @ params[:-1] + params[-1]


def evaluate(
    params: np.ndarray,
    train: SplitData,
    split: SplitData,
    alpha: float,
    ridge: float,
    eps: float,
    audit_bootstrap_replicates: int = 0,
    audit_confidence: float = 0.95,
    audit_seed: int = 20260729,
) -> dict:
    train_logits = predict_logits(params, train.x)
    logits = predict_logits(params, split.x)
    probs = 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))
    preds = (probs >= 0.5).astype(int)
    losses = -(split.y * np.log(np.clip(probs, 1e-12, 1.0)) + (1 - split.y) * np.log(np.clip(1.0 - probs, 1e-12, 1.0)))
    z, _ = normalize_scores(train_logits, logits)

    rows = []
    for gid in range(4):
        mask = split.group == gid
        f_feature = fisher_schur(z[mask], split.nuisance[mask], ridge)
        rows.append(
            {
                "group": GROUP_NAMES[gid],
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(split.y[mask], preds[mask])),
                "loss": float(losses[mask].mean()),
                "projected_feature_fisher": f_feature["projected"],
                "exact_projected_feature_fisher": f_feature["exact_projected"],
                "feature_ridge_to_exact_gap": f_feature["ridge_to_exact_gap"],
                "feature_j_tn": f_feature["j_tn"],
                "feature_j_nn": f_feature["j_nn"],
                "feature_audit_status": f_feature["audit_status"],
                "inverse_feature_loss": float(1.0 / (f_feature["projected"] + eps)),
            }
        )
    ridge_exact_audit = audit_scalar_environments(
        [
            {
                "name": GROUP_NAMES[gid],
                "z": z[split.group == gid],
                "b": split.nuisance[split.group == gid],
            }
            for gid in range(4)
        ],
        ridge=ridge,
        eps=eps,
        alpha=alpha,
        bootstrap_replicates=audit_bootstrap_replicates,
        confidence=audit_confidence,
        seed=audit_seed,
    )
    inverse = np.array([row["inverse_feature_loss"] for row in rows])
    group_losses = np.array([row["loss"] for row in rows])
    group_acc = np.array([row["accuracy"] for row in rows])
    return {
        "overall_accuracy": float(accuracy_score(split.y, preds)),
        "overall_loss": float(log_loss(split.y, probs, labels=[0, 1])),
        "worst_group_accuracy": float(group_acc.min()),
        "group_loss_cvar": cvar(group_losses, alpha),
        "fisher_glass_certificate": float(1.0 / cvar(inverse, alpha)),
        "exact_fisher_glass_certificate": ridge_exact_audit["aggregate"]["exact_certificate"],
        "conservative_lower_certificate": ridge_exact_audit["aggregate"]["conservative_lower_certificate"],
        "inverse_loss_p95": float(np.quantile(inverse, 0.95)),
        "weakest_slice": max(rows, key=lambda row: row["inverse_feature_loss"]),
        "worst_accuracy_slice": min(rows, key=lambda row: row["accuracy"]),
        "groups": rows,
        "ridge_exact_audit": ridge_exact_audit,
    }


def load_waterbirds(args) -> tuple[SplitData, SplitData, SplitData]:
    dataset = get_dataset(dataset="waterbirds", root_dir=args.root_dir, download=args.download)
    y = as_numpy(dataset._y_array).astype(int)
    metadata = as_numpy(dataset._metadata_array).astype(int)
    split = as_numpy(dataset._split_array).astype(int)
    bg = metadata[:, 0].astype(int)

    train_idx = np.flatnonzero(split == dataset.split_dict["train"])
    val_idx = np.flatnonzero(split == dataset.split_dict["val"])
    test_idx = np.flatnonzero(split == dataset.split_dict["test"])
    all_idx = np.concatenate([train_idx, val_idx, test_idx])
    feat_idx, features = extract_features(dataset, all_idx, args.batch_size, args.device, Path(args.cache))
    order = {int(idx): i for i, idx in enumerate(feat_idx)}
    train_rows = np.array([order[int(i)] for i in train_idx])
    val_rows = np.array([order[int(i)] for i in val_idx])
    test_rows = np.array([order[int(i)] for i in test_idx])

    x_train_raw = features[train_rows].astype(np.float32)
    x_train, norm = standardize(x_train_raw, x_train_raw)
    x_val = ((features[val_rows] - norm["mean"]) / norm["std"]).astype(np.float32)
    x_test = ((features[test_rows] - norm["mean"]) / norm["std"]).astype(np.float32)

    bg_probe = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=args.seeds[0] + 1)
    bg_probe.fit(x_train, bg[train_idx])
    train_bg_score = bg_probe.decision_function(x_train)
    val_bg_score = bg_probe.decision_function(x_val)
    test_bg_score = bg_probe.decision_function(x_test)
    nuisance_train, n_norm = normalize_scores(train_bg_score, train_bg_score)
    nuisance_val = ((val_bg_score - n_norm["mean"]) / n_norm["std"]).astype(np.float32)
    nuisance_test = ((test_bg_score - n_norm["mean"]) / n_norm["std"]).astype(np.float32)

    train = SplitData(x_train, y[train_idx], bg[train_idx], group_ids(y[train_idx], bg[train_idx]), nuisance_train)
    val = SplitData(x_val, y[val_idx], bg[val_idx], group_ids(y[val_idx], bg[val_idx]), nuisance_val)
    test = SplitData(x_test, y[test_idx], bg[test_idx], group_ids(y[test_idx], bg[test_idx]), nuisance_test)
    return train, val, test


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0
    return mean, float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def optional_mean_ci(values: list[float | None]) -> tuple[float, float] | None:
    available = [float(value) for value in values if value is not None]
    return mean_ci(available) if available else None


def optional_min(values: list[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return min(available) if available else None


def optional_max(values: list[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return max(available) if available else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/wilds")
    parser.add_argument("--cache", default="build/features/waterbirds_mobilenetv3_small.npz")
    parser.add_argument("--output", default="build/results/waterbirds_fisher_glass_training_results.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19])
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--fg-alpha", type=float, default=0.75)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--audit-bootstrap-replicates", type=int, default=100000)
    parser.add_argument("--audit-confidence", type=float, default=0.95)
    parser.add_argument("--audit-seed", type=int, default=20260729)
    parser.add_argument("--fg-lambdas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    parser.add_argument("--gdro-fg-lambdas", type=float, nargs="+", default=[0.01, 0.03, 0.1, 0.3])
    parser.add_argument("--selection-worst-tolerance", type=float, default=0.02)
    parser.add_argument("--jtt-upweight", type=float, default=20.0)
    args = parser.parse_args()

    train, val, test = load_waterbirds(args)
    methods = ["erm", "group_dro", "loss_cvar", "jtt"]
    rows = []
    for seed in args.seeds:
        seed_rows = []
        erm_params, _ = train_linear(train, "erm", seed, args.epochs, args.lr, args.weight_decay, args.ridge, args.eps, args.fg_alpha, 0.0)
        train_preds = (predict_logits(erm_params, train.x) >= 0).astype(int)
        jtt_weights = np.ones_like(train.y, dtype=np.float32)
        jtt_weights[train_preds != train.y] = args.jtt_upweight
        for method in methods:
            if method == "erm":
                params = erm_params
                history = {}
            elif method == "jtt":
                params, history = train_linear(train, "jtt", seed + 1000, args.epochs, args.lr, args.weight_decay, args.ridge, args.eps, args.fg_alpha, 0.0, jtt_weights)
                history["upweighted_fraction"] = float(np.mean(train_preds != train.y))
            else:
                params, history = train_linear(train, method, seed, args.epochs, args.lr, args.weight_decay, args.ridge, args.eps, args.fg_alpha, 0.0)
            val_metrics = evaluate(
                params, train, val, args.alpha, args.ridge, args.eps,
                args.audit_bootstrap_replicates, args.audit_confidence, args.audit_seed,
            )
            test_metrics = evaluate(
                params, train, test, args.alpha, args.ridge, args.eps,
                args.audit_bootstrap_replicates, args.audit_confidence, args.audit_seed + 1,
            )
            seed_rows.append({"method": method, "seed": seed, "lambda_fg": 0.0, "selection": "fixed", "history": history, "val": val_metrics, "test": test_metrics})

        fg_candidates = []
        for lambda_fg in args.fg_lambdas:
            params, history = train_linear(train, "fisher_glass", seed, args.epochs, args.lr, args.weight_decay, args.ridge, args.eps, args.fg_alpha, lambda_fg)
            val_metrics = evaluate(
                params, train, val, args.alpha, args.ridge, args.eps,
                args.audit_bootstrap_replicates, args.audit_confidence, args.audit_seed,
            )
            test_metrics = evaluate(
                params, train, test, args.alpha, args.ridge, args.eps,
                args.audit_bootstrap_replicates, args.audit_confidence, args.audit_seed + 1,
            )
            fg_candidates.append({"method": "fisher_glass", "seed": seed, "lambda_fg": lambda_fg, "selection": "candidate", "history": history, "val": val_metrics, "test": test_metrics})
        best_val_worst = max(row["val"]["worst_group_accuracy"] for row in fg_candidates)
        eligible = [
            row
            for row in fg_candidates
            if row["val"]["worst_group_accuracy"] >= best_val_worst - args.selection_worst_tolerance
        ]
        selected = max(eligible, key=lambda row: (row["val"]["fisher_glass_certificate"], row["val"]["worst_group_accuracy"]))
        selected = dict(selected)
        selected["selection"] = "selected_by_val_certificate_with_worst_group_tolerance"
        seed_rows.extend(fg_candidates)
        seed_rows.append(selected)

        gdro_fg_candidates = []
        for lambda_fg in args.gdro_fg_lambdas:
            params, history = train_linear(
                train,
                "group_dro_fisher_glass",
                seed,
                args.epochs,
                args.lr,
                args.weight_decay,
                args.ridge,
                args.eps,
                args.fg_alpha,
                lambda_fg,
            )
            val_metrics = evaluate(
                params, train, val, args.alpha, args.ridge, args.eps,
                args.audit_bootstrap_replicates, args.audit_confidence, args.audit_seed,
            )
            test_metrics = evaluate(
                params, train, test, args.alpha, args.ridge, args.eps,
                args.audit_bootstrap_replicates, args.audit_confidence, args.audit_seed + 1,
            )
            gdro_fg_candidates.append(
                {
                    "method": "group_dro_fisher_glass",
                    "seed": seed,
                    "lambda_fg": lambda_fg,
                    "selection": "candidate",
                    "history": history,
                    "val": val_metrics,
                    "test": test_metrics,
                }
            )
        best_val_worst = max(row["val"]["worst_group_accuracy"] for row in gdro_fg_candidates)
        eligible = [
            row
            for row in gdro_fg_candidates
            if row["val"]["worst_group_accuracy"] >= best_val_worst - args.selection_worst_tolerance
        ]
        selected = max(eligible, key=lambda row: (row["val"]["fisher_glass_certificate"], row["val"]["worst_group_accuracy"]))
        selected = dict(selected)
        selected["selection"] = "selected_by_val_certificate_with_worst_group_tolerance"
        seed_rows.extend(gdro_fg_candidates)
        seed_rows.append(selected)
        rows.extend(seed_rows)

    selected_rows = [row for row in rows if row["selection"] != "candidate"]
    summary = []
    for method in sorted({row["method"] for row in selected_rows}):
        method_rows = [row for row in selected_rows if row["method"] == method]
        summary.append(
            {
                "method": method,
                "n": len(method_rows),
                "lambda_fg_values": sorted({row["lambda_fg"] for row in method_rows}),
                "avg_accuracy_mean_ci": mean_ci([row["test"]["overall_accuracy"] for row in method_rows]),
                "worst_group_accuracy_mean_ci": mean_ci([row["test"]["worst_group_accuracy"] for row in method_rows]),
                "loss_cvar_mean_ci": mean_ci([row["test"]["group_loss_cvar"] for row in method_rows]),
                "fisher_glass_certificate_mean_ci": mean_ci([row["test"]["fisher_glass_certificate"] for row in method_rows]),
                "exact_fisher_glass_certificate_mean_ci": optional_mean_ci(
                    [row["test"]["exact_fisher_glass_certificate"] for row in method_rows]
                ),
                "exact_fisher_glass_certificate_min": optional_min(
                    [row["test"]["exact_fisher_glass_certificate"] for row in method_rows]
                ),
                "conservative_lower_certificate_mean_ci": optional_mean_ci(
                    [row["test"]["conservative_lower_certificate"] for row in method_rows]
                ),
                "conservative_lower_certificate_min": optional_min(
                    [row["test"]["conservative_lower_certificate"] for row in method_rows]
                ),
                "minimum_nuisance_eigenvalue": optional_min(
                    [
                        row["test"]["ridge_exact_audit"]["aggregate"]["min_nuisance_eigenvalue"]
                        for row in method_rows
                    ]
                ),
                "maximum_ridge_to_exact_gap": optional_max(
                    [
                        row["test"]["ridge_exact_audit"]["aggregate"]["max_operator_gap"]
                        for row in method_rows
                    ]
                ),
                "audit_statuses": sorted(
                    {
                        row["test"]["ridge_exact_audit"]["aggregate"]["status"]
                        for row in method_rows
                    }
                ),
                "inverse_loss_p95_mean_ci": mean_ci([row["test"]["inverse_loss_p95"] for row in method_rows]),
            }
        )

    result = {
        "schema_version": 2,
        "dataset": "Waterbirds via WILDS loader",
        "backbone": "Frozen ImageNet MobileNetV3-small features",
        "classifier": "Linear logistic classifier trained in PyTorch",
        "seeds": args.seeds,
        "epochs": args.epochs,
        "ridge": args.ridge,
        "eps": args.eps,
        "alpha": args.alpha,
        "audit_bootstrap_replicates": args.audit_bootstrap_replicates,
        "audit_confidence": args.audit_confidence,
        "selection_rule": (
            "Fisher-Glass lambda selected by validation certificate among candidates whose validation "
            f"worst-group accuracy is within {args.selection_worst_tolerance:.3f} of the best Fisher-Glass candidate; "
            "baselines use fixed hyperparameters."
        ),
        "summary": summary,
        "runs": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"summary": summary}, indent=2))


if __name__ == "__main__":
    main()
