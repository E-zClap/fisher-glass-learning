"""Waterbirds diagnostic via the WILDS loader.

This script runs a lightweight image benchmark on the official WILDS
Waterbirds split. It extracts frozen ImageNet MobileNetV3-small features,
trains a linear bird-label classifier and a linear background probe, then
computes a 2D feature-Schur Fisher-glass diagnostic over the four
label/background groups.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from tqdm import tqdm
from wilds import get_dataset

from ridge_exact_audit import audit_scalar_environments, schur_from_moments


def as_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def cvar(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    threshold = np.quantile(values, alpha)
    return float(values[values >= threshold].mean())


def normalize(train_scores: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, dict]:
    mean = float(np.mean(train_scores))
    std = float(np.std(train_scores))
    if std < 1e-12:
        std = 1.0
    return (scores - mean) / std, {"mean": mean, "std": std}


def fisher_schur(z: np.ndarray, b: np.ndarray, ridge: float, weight: np.ndarray | None = None) -> dict:
    if weight is None:
        weight = np.ones_like(z, dtype=float)
    weight = np.asarray(weight, dtype=float)
    weight = weight / max(float(weight.sum()), 1e-12)
    j_tt = float(np.sum(weight * z * z))
    j_tn = float(np.sum(weight * z * b))
    j_nn = float(np.sum(weight * b * b))
    audit = schur_from_moments(j_tt, j_tn, j_nn, ridge)
    return {
        "raw": j_tt,
        "projected": audit["ridge_information"],
        "exact_projected": audit["exact_information"],
        "ridge_to_exact_gap": audit["operator_gap"],
        "j_tn": j_tn,
        "j_nn": j_nn,
        "audit_status": audit["status"],
        "audit_reasons": audit["reasons"],
    }


def summarize_basis(
    name: str,
    group_defs: list[dict],
    y_test: np.ndarray,
    pred_test: np.ndarray,
    losses: np.ndarray,
    z: np.ndarray,
    b: np.ndarray,
    ridge: float,
    eps: float,
    alpha: float,
) -> dict:
    rows = []
    for group in group_defs:
        mask = group["mask"]
        f_feature = fisher_schur(z[mask], b[mask], ridge)
        rows.append(
            {
                "group": group["name"],
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(y_test[mask], pred_test[mask])),
                "loss": float(np.mean(losses[mask])),
                "projected_feature_fisher": f_feature["projected"],
                "inverse_feature_loss": float(1.0 / (f_feature["projected"] + eps)),
            }
        )
    inverse = np.array([row["inverse_feature_loss"] for row in rows], dtype=float)
    exact_values = [
        fisher_schur(z[group["mask"]], b[group["mask"]], ridge)["exact_projected"]
        for group in group_defs
    ]
    exact_certificate = None
    if all(value is not None for value in exact_values):
        exact_inverse = 1.0 / (np.asarray(exact_values, dtype=float) + eps)
        exact_certificate = float(1.0 / cvar(exact_inverse, alpha))
    acc = np.array([row["accuracy"] for row in rows], dtype=float)
    weakest = max(rows, key=lambda row: row["inverse_feature_loss"])
    return {
        "basis": name,
        "fisher_glass_certificate": float(1.0 / cvar(inverse, alpha)),
        "exact_fisher_glass_certificate": exact_certificate,
        "inverse_loss_p95": float(np.quantile(inverse, 0.95)),
        "worst_group_accuracy": float(acc.min()),
        "weakest_slice": weakest["group"],
        "weakest_slice_accuracy": weakest["accuracy"],
        "weakest_slice_projected_fisher": weakest["projected_feature_fisher"],
    }


class WildsImageDataset(Dataset):
    def __init__(self, dataset, indices: np.ndarray, transform):
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.transform = transform

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        image, _, _ = self.dataset[idx]
        return self.transform(image), idx


def extract_features(dataset, indices: np.ndarray, batch_size: int, device: str, cache_path: Path) -> tuple[np.ndarray, np.ndarray]:
    if cache_path.exists():
        cached = np.load(cache_path)
        return cached["indices"], cached["features"]

    weights = MobileNet_V3_Small_Weights.DEFAULT
    transform = weights.transforms()
    model = mobilenet_v3_small(weights=weights)
    model.eval().to(device)

    loader = DataLoader(WildsImageDataset(dataset, indices, transform), batch_size=batch_size, shuffle=False, num_workers=0)
    all_indices: list[np.ndarray] = []
    all_features: list[np.ndarray] = []
    with torch.inference_mode():
        for images, batch_indices in tqdm(loader, desc="extract Waterbirds features"):
            images = images.to(device)
            feats = model.features(images)
            feats = model.avgpool(feats)
            feats = torch.flatten(feats, 1)
            all_indices.append(batch_indices.numpy())
            all_features.append(feats.cpu().numpy())

    out_indices = np.concatenate(all_indices)
    out_features = np.concatenate(all_features)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, indices=out_indices, features=out_features)
    return out_indices, out_features


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/wilds")
    parser.add_argument("--output", default="build/results/waterbirds_fisher_glass_results.json")
    parser.add_argument("--cache", default="build/features/waterbirds_mobilenetv3_small.npz")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--audit-bootstrap-replicates", type=int, default=100000)
    parser.add_argument("--audit-confidence", type=float, default=0.95)
    parser.add_argument("--audit-seed", type=int, default=20260729)
    parser.add_argument("--reserve-r", type=float, default=0.05)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = get_dataset(dataset="waterbirds", root_dir=args.root_dir, download=args.download)
    y = as_numpy(dataset._y_array).astype(int)
    metadata = as_numpy(dataset._metadata_array).astype(int)
    split = as_numpy(dataset._split_array).astype(int)
    background = metadata[:, 0].astype(int)

    train_idx = np.flatnonzero(split == dataset.split_dict["train"])
    val_idx = np.flatnonzero(split == dataset.split_dict["val"])
    test_idx = np.flatnonzero(split == dataset.split_dict["test"])
    all_idx = np.concatenate([train_idx, val_idx, test_idx])

    feat_idx, features = extract_features(dataset, all_idx, args.batch_size, device, Path(args.cache))
    order = {int(idx): i for i, idx in enumerate(feat_idx)}
    train_rows = np.array([order[int(i)] for i in train_idx])
    test_rows = np.array([order[int(i)] for i in test_idx])

    x_train = features[train_rows]
    x_test = features[test_rows]
    y_train = y[train_idx]
    y_test = y[test_idx]
    bg_train = background[train_idx]
    bg_test = background[test_idx]

    clf = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=args.seed)
    clf.fit(x_train, y_train)
    bg_probe = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=args.seed + 1)
    bg_probe.fit(x_train, bg_train)

    train_task_score = clf.decision_function(x_train)
    test_task_score = clf.decision_function(x_test)
    train_bg_score = bg_probe.decision_function(x_train)
    test_bg_score = bg_probe.decision_function(x_test)
    z_test, z_norm = normalize(train_task_score, test_task_score)
    b_test, b_norm = normalize(train_bg_score, test_bg_score)

    prob_test = clf.predict_proba(x_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)
    losses = -(y_test * np.log(np.clip(prob_test, 1e-12, 1.0)) + (1 - y_test) * np.log(np.clip(1.0 - prob_test, 1e-12, 1.0)))
    predictive_weight = prob_test * (1.0 - prob_test)

    group_defs = [
        {"name": f"y={label}/background={bg}", "mask": (y_test == label) & (bg_test == bg)}
        for label in (0, 1)
        for bg in (0, 1)
    ]
    b_metadata, _ = normalize(bg_train.astype(float), bg_test.astype(float))
    random_scores = rng.normal(size=y_test.size)
    b_random, _ = normalize(random_scores, random_scores)
    b_under = np.zeros_like(z_test)
    over_scores = z_test + 0.01 * rng.normal(size=z_test.size)
    b_over, _ = normalize(over_scores, over_scores)
    b_hidden_loss, _ = normalize(losses, losses)
    nuisance_ablation = [
        summarize_basis("metadata nuisance", group_defs, y_test, pred_test, losses, z_test, b_metadata, args.ridge, args.eps, args.alpha),
        summarize_basis("learned probe nuisance", group_defs, y_test, pred_test, losses, z_test, b_test, args.ridge, args.eps, args.alpha),
        summarize_basis("random nuisance", group_defs, y_test, pred_test, losses, z_test, b_random, args.ridge, args.eps, args.alpha),
        summarize_basis("under-projected nuisance", group_defs, y_test, pred_test, losses, z_test, b_under, args.ridge, args.eps, args.alpha),
        summarize_basis("over-projected nuisance", group_defs, y_test, pred_test, losses, z_test, b_over, args.ridge, args.eps, args.alpha),
        summarize_basis("hidden-loss nuisance", group_defs, y_test, pred_test, losses, z_test, b_hidden_loss, args.ridge, args.eps, args.alpha),
    ]

    group_rows = []
    for group in group_defs:
        mask = group["mask"]
        f_feature = fisher_schur(z_test[mask], b_test[mask], args.ridge)
        f_predictive = fisher_schur(z_test[mask], b_test[mask], args.ridge, predictive_weight[mask])
        group_rows.append(
            {
                "group": group["name"],
                "n": int(mask.sum()),
                "accuracy": float(accuracy_score(y_test[mask], pred_test[mask])),
                "loss": float(np.mean(losses[mask])),
                "raw_feature_fisher": f_feature["raw"],
                "projected_feature_fisher": f_feature["projected"],
                "exact_projected_feature_fisher": f_feature["exact_projected"],
                "feature_ridge_to_exact_gap": f_feature["ridge_to_exact_gap"],
                "feature_j_tn": f_feature["j_tn"],
                "feature_j_nn": f_feature["j_nn"],
                "feature_audit_status": f_feature["audit_status"],
                "inverse_feature_loss": float(1.0 / (f_feature["projected"] + args.eps)),
                "raw_predictive_fisher": f_predictive["raw"],
                "projected_predictive_fisher": f_predictive["projected"],
                "exact_projected_predictive_fisher": f_predictive["exact_projected"],
                "predictive_ridge_to_exact_gap": f_predictive["ridge_to_exact_gap"],
                "predictive_j_tn": f_predictive["j_tn"],
                "predictive_j_nn": f_predictive["j_nn"],
                "inverse_predictive_loss": float(1.0 / (f_predictive["projected"] + args.eps)),
            }
        )

    audit_environments = [
        {
            "name": group["name"],
            "z": z_test[group["mask"]],
            "b": b_test[group["mask"]],
        }
        for group in group_defs
    ]
    ridge_exact_audit = audit_scalar_environments(
        audit_environments,
        ridge=args.ridge,
        eps=args.eps,
        alpha=args.alpha,
        bootstrap_replicates=args.audit_bootstrap_replicates,
        confidence=args.audit_confidence,
        seed=args.audit_seed,
    )
    predictive_ridge_exact_audit = audit_scalar_environments(
        [
            {
                **environment,
                "weight": predictive_weight[group["mask"]],
            }
            for environment, group in zip(audit_environments, group_defs)
        ],
        ridge=args.ridge,
        eps=args.eps,
        alpha=args.alpha,
        bootstrap_replicates=args.audit_bootstrap_replicates,
        confidence=args.audit_confidence,
        seed=args.audit_seed + 1,
    )

    # Balanced four-group anchor from training data.
    anchor_parts = []
    min_group = min(int(((y_train == label) & (bg_train == bg)).sum()) for label in (0, 1) for bg in (0, 1))
    for label in (0, 1):
        for bg in (0, 1):
            candidates = train_idx[(y_train == label) & (bg_train == bg)]
            anchor_parts.append(rng.choice(candidates, size=min_group, replace=False))
    anchor_idx = np.concatenate(anchor_parts)
    anchor_rows = np.array([order[int(i)] for i in anchor_idx])
    anchor_z_raw = clf.decision_function(features[anchor_rows])
    anchor_b_raw = bg_probe.decision_function(features[anchor_rows])
    anchor_z = (anchor_z_raw - z_norm["mean"]) / z_norm["std"]
    anchor_b = (anchor_b_raw - b_norm["mean"]) / b_norm["std"]
    anchor_fisher = fisher_schur(anchor_z, anchor_b, args.ridge)["projected"]

    projected = np.array([row["projected_feature_fisher"] for row in group_rows])
    inverse = np.array([row["inverse_feature_loss"] for row in group_rows])
    predictive_inverse = np.array([row["inverse_predictive_loss"] for row in group_rows])
    group_losses = np.array([row["loss"] for row in group_rows])
    group_acc = np.array([row["accuracy"] for row in group_rows])
    mixed = (1.0 - args.reserve_r) * projected + args.reserve_r * anchor_fisher
    mixed_inverse = 1.0 / (mixed + args.eps)

    result = {
        "schema_version": 2,
        "dataset": "Waterbirds via WILDS loader",
        "model": "Frozen ImageNet MobileNetV3-small features + logistic regression",
        "seed": args.seed,
        "device": device,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "n_groups": len(group_rows),
        "ridge": args.ridge,
        "eps": args.eps,
        "alpha": args.alpha,
        "overall_accuracy": float(accuracy_score(y_test, pred_test)),
        "overall_loss": float(log_loss(y_test, prob_test, labels=[0, 1])),
        "group_loss_cvar": cvar(group_losses, args.alpha),
        "worst_group_accuracy": float(group_acc.min()),
        "feature_raw_fisher_mean": float(np.mean([row["raw_feature_fisher"] for row in group_rows])),
        "feature_projected_fisher_min": float(projected.min()),
        "feature_inverse_loss_p95": float(np.quantile(inverse, 0.95)),
        "feature_inverse_loss_cvar": cvar(inverse, args.alpha),
        "fisher_glass_certificate": float(1.0 / cvar(inverse, args.alpha)),
        "predictive_fisher_glass_certificate": float(1.0 / cvar(predictive_inverse, args.alpha)),
        "ridge_exact_audit": ridge_exact_audit,
        "predictive_ridge_exact_audit": predictive_ridge_exact_audit,
        "reserve_r": args.reserve_r,
        "reserve_anchor_projected_fisher": float(anchor_fisher),
        "reserve_fisher_glass_certificate": float(1.0 / cvar(mixed_inverse, args.alpha)),
        "reserve_theory_floor": float(args.reserve_r * anchor_fisher),
        "worst_group_by_accuracy": min(group_rows, key=lambda row: row["accuracy"]),
        "worst_group_by_inverse_fisher": max(group_rows, key=lambda row: row["inverse_feature_loss"]),
        "nuisance_ablation": nuisance_ablation,
        "groups": sorted(group_rows, key=lambda row: row["inverse_feature_loss"], reverse=True),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in [
        "dataset",
        "model",
        "device",
        "n_train",
        "n_test",
        "overall_accuracy",
        "worst_group_accuracy",
        "overall_loss",
        "group_loss_cvar",
        "fisher_glass_certificate",
        "feature_inverse_loss_p95",
        "reserve_fisher_glass_certificate",
        "reserve_theory_floor",
        "worst_group_by_accuracy",
        "worst_group_by_inverse_fisher",
    ]}, indent=2))


if __name__ == "__main__":
    main()
