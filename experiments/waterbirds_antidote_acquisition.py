"""Waterbirds Fisher-antidote and tail-transversality acquisition experiment.

This experiment turns the Fisher reserve idea into an acquisition rule.  We
train a base classifier, find the weakest nuisance-projected Fisher tail on a
held-out split, compute efficient-score residuals

    phi = z - beta b,  beta = J_TN / (J_NN + ridge),

for candidate examples, and add a tiny budget of examples that maximize either
the weakest-mode residual phi^2 or the tail-weighted transversality gain.  The
comparison is not "more data helps"; it is whether targeted efficient-residual
examples lift the Fisher-Glass certificate more efficiently than random,
group-balanced, high-loss, raw-Fisher, or uncertainty-selected examples.

The candidate pool is the Waterbirds validation split, so selected examples are
real images with physical label/background variation rather than synthetic
feature perturbations.  The final evaluation is on the official test split.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss

from waterbirds_fisher_glass import cvar, fisher_schur
from waterbirds_fisher_glass_training import (
    GROUP_NAMES,
    SplitData,
    evaluate,
    load_waterbirds,
    predict_logits,
    train_linear,
)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -60.0, 60.0)))


def normalize_scores(train_scores: np.ndarray, scores: np.ndarray) -> np.ndarray:
    mean = float(train_scores.mean())
    std = float(train_scores.std())
    if std < 1e-6:
        std = 1.0
    return ((scores - mean) / std).astype(np.float32)


def group_counts(groups: np.ndarray) -> dict[str, int]:
    return {GROUP_NAMES[int(gid)]: int(np.sum(groups == gid)) for gid in range(4)}


def mean_ci(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean()) if arr.size else float("nan")
    if arr.size <= 1:
        return mean, 0.0
    return mean, float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))


def make_augmented_train(train: SplitData, pool: SplitData, selected: np.ndarray) -> SplitData:
    return SplitData(
        x=np.concatenate([train.x, pool.x[selected]], axis=0),
        y=np.concatenate([train.y, pool.y[selected]], axis=0),
        bg=np.concatenate([train.bg, pool.bg[selected]], axis=0),
        group=np.concatenate([train.group, pool.group[selected]], axis=0),
        nuisance=np.concatenate([train.nuisance, pool.nuisance[selected]], axis=0),
    )


def fit_classifier(args: argparse.Namespace, data: SplitData, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    if args.trainer == "sklearn":
        if args.objective != "erm":
            raise ValueError("--trainer sklearn currently supports --objective erm only")
        clf = LogisticRegression(
            max_iter=args.max_iter,
            class_weight="balanced",
            solver="lbfgs",
            random_state=seed,
        )
        clf.fit(data.x, data.y)
        params = np.concatenate([clf.coef_.reshape(-1), clf.intercept_.astype(np.float32)])
        return params.astype(np.float32), {"trainer": "sklearn_logistic", "max_iter": args.max_iter}
    params, history = train_linear(
        data,
        args.objective,
        seed,
        args.epochs,
        args.lr,
        args.weight_decay,
        args.ridge,
        args.eps,
        args.fg_alpha,
        0.0,
    )
    history = dict(history)
    history["trainer"] = "torch_linear"
    return params, history


def select_group_balanced(rng: np.random.Generator, groups: np.ndarray, k: int) -> np.ndarray:
    selected: list[int] = []
    per_group = max(1, int(np.ceil(k / 4)))
    for gid in range(4):
        candidates = np.flatnonzero(groups == gid)
        if candidates.size == 0:
            continue
        take = min(per_group, candidates.size)
        selected.extend(rng.choice(candidates, size=take, replace=False).tolist())
    if len(selected) < k:
        remaining = np.setdiff1d(np.arange(groups.size), np.asarray(selected, dtype=int), assume_unique=False)
        if remaining.size > 0:
            take = min(k - len(selected), remaining.size)
            selected.extend(rng.choice(remaining, size=take, replace=False).tolist())
    return np.asarray(selected[:k], dtype=int)


def top_k(scores: np.ndarray, k: int, allowed: np.ndarray | None = None) -> np.ndarray:
    if allowed is None:
        allowed = np.arange(scores.size)
    if allowed.size == 0:
        return np.asarray([], dtype=int)
    k = min(k, allowed.size)
    order = allowed[np.argsort(scores[allowed])[::-1]]
    return order[:k].astype(int)


def stable_selector_offset(selector: str) -> int:
    digest = hashlib.blake2s(selector.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % 10000


def score_candidate_pool(
    params: np.ndarray,
    train: SplitData,
    pool: SplitData,
    alpha: float,
    ridge: float,
    eps: float,
) -> dict[str, Any]:
    train_logits = predict_logits(params, train.x)
    pool_logits = predict_logits(params, pool.x)
    z_pool = normalize_scores(train_logits, pool_logits)
    probs = sigmoid(pool_logits)
    losses = -(
        pool.y * np.log(np.clip(probs, 1e-12, 1.0))
        + (1 - pool.y) * np.log(np.clip(1.0 - probs, 1e-12, 1.0))
    )

    rows = []
    for gid in range(4):
        mask = pool.group == gid
        f = fisher_schur(z_pool[mask], pool.nuisance[mask], ridge)
        beta_g = float(f["j_tn"] / (f["j_nn"] + ridge))
        rows.append(
            {
                "gid": gid,
                "group": GROUP_NAMES[gid],
                "n": int(mask.sum()),
                "projected_feature_fisher": f["projected"],
                "inverse_feature_loss": float(1.0 / (f["projected"] + eps)),
                "j_tn": f["j_tn"],
                "j_nn": f["j_nn"],
                "beta": beta_g,
            }
        )
    weakest = max(rows, key=lambda row: row["inverse_feature_loss"])
    beta = float(weakest["j_tn"] / (weakest["j_nn"] + ridge))
    phi = z_pool - beta * pool.nuisance
    phi2 = phi * phi
    inverse = np.asarray([row["inverse_feature_loss"] for row in rows], dtype=float)
    tail_threshold = float(np.quantile(inverse, alpha))
    tail_rows = [row for row in rows if row["inverse_feature_loss"] >= tail_threshold]
    if not tail_rows:
        tail_rows = [weakest]
    tail_gain = np.zeros_like(z_pool, dtype=np.float64)
    tail_weight_sum = 0.0
    for row in tail_rows:
        lam = float(row["projected_feature_fisher"])
        weight = 1.0 / ((lam + eps) ** 2)
        phi_g = z_pool - float(row["beta"]) * pool.nuisance
        tail_gain += weight * phi_g * phi_g
        tail_weight_sum += weight
    tail_gain = tail_gain / max(tail_weight_sum, 1e-12)
    return {
        "weakest": weakest,
        "tail_rows": tail_rows,
        "pool_rows": rows,
        "pool_certificate": float(1.0 / cvar(inverse, alpha)),
        "beta": beta,
        "z": z_pool,
        "phi2": phi2,
        "raw_fisher": z_pool * z_pool,
        "tail_transverse_gain": tail_gain.astype(np.float32),
        "loss": losses,
        "uncertainty": probs * (1.0 - probs),
        "low_loss_mask": losses <= float(np.median(losses)),
    }


def run_one(
    args: argparse.Namespace,
    train: SplitData,
    val: SplitData,
    test: SplitData,
    seed: int,
    budget_fraction: float,
    selector: str,
    base_params: np.ndarray,
    base_history: dict[str, Any],
    base_val: dict[str, Any],
    base_test: dict[str, Any],
    scored: dict[str, Any],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + int(100000 * budget_fraction) + stable_selector_offset(selector))
    k = max(1, int(round(budget_fraction * len(train.y))))
    k = min(k, len(val.y))
    weak_gid = int(scored["weakest"]["gid"])
    weak_y = weak_gid // 2
    weak_bg = weak_gid % 2
    weak_allowed = np.flatnonzero(val.group == weak_gid)
    paired_allowed = np.flatnonzero((val.y == weak_y) & (val.bg != weak_bg))
    low_loss_allowed = np.flatnonzero(scored["low_loss_mask"])

    if selector == "fisher_antidote":
        selected = top_k(scored["phi2"], k)
    elif selector == "fg_tail_transverse":
        selected = top_k(scored["tail_transverse_gain"], k)
    elif selector == "low_loss_tail_transverse":
        selected = top_k(scored["tail_transverse_gain"], k, low_loss_allowed)
    elif selector == "raw_fisher":
        selected = top_k(scored["raw_fisher"], k)
    elif selector == "low_loss_antidote":
        selected = top_k(scored["phi2"], k, low_loss_allowed)
    elif selector == "weak_slice_antidote":
        selected = top_k(scored["phi2"], k, weak_allowed)
    elif selector == "paired_antidote":
        selected = top_k(scored["phi2"], k, paired_allowed)
    elif selector == "paired_random":
        take = min(k, paired_allowed.size)
        selected = rng.choice(paired_allowed, size=take, replace=False).astype(int)
    elif selector == "weak_slice_random":
        take = min(k, weak_allowed.size)
        selected = rng.choice(weak_allowed, size=take, replace=False).astype(int)
    elif selector == "high_loss":
        selected = top_k(scored["loss"], k)
    elif selector == "uncertainty":
        selected = top_k(scored["uncertainty"], k)
    elif selector == "group_balanced":
        selected = select_group_balanced(rng, val.group, k)
    elif selector == "random":
        selected = rng.choice(np.arange(len(val.y)), size=k, replace=False).astype(int)
    else:
        raise ValueError(f"Unknown selector: {selector}")

    augmented = make_augmented_train(train, val, selected)
    params, history = fit_classifier(args, augmented, seed + 1000)
    test_metrics = evaluate(params, augmented, test, args.alpha, args.ridge, args.eps)
    val_metrics = evaluate(params, augmented, val, args.alpha, args.ridge, args.eps)

    selected_phi2 = scored["phi2"][selected]
    selected_gain = scored["tail_transverse_gain"][selected]
    selected_raw = scored["raw_fisher"][selected]
    selected_loss = scored["loss"][selected]
    base_phi2_weak = float(np.mean(scored["phi2"][weak_allowed])) if weak_allowed.size else float("nan")
    selected_phi2_mean = float(np.mean(selected_phi2)) if selected.size else float("nan")
    first_order_lift = selected_phi2_mean - base_phi2_weak
    return {
        "seed": seed,
        "objective": args.objective,
        "budget_fraction": float(budget_fraction),
        "budget_n": int(selected.size),
        "selector": selector,
        "weakest_validation_slice": scored["weakest"],
        "tail_validation_slices": scored["tail_rows"],
        "pool_certificate": scored["pool_certificate"],
        "base_history": base_history,
        "history": history,
        "base_val": base_val,
        "base_test": base_test,
        "val": val_metrics,
        "test": test_metrics,
        "acquisition": {
            "selected_group_counts": group_counts(val.group[selected]),
            "selected_phi2_mean": selected_phi2_mean,
            "selected_phi2_p95": float(np.quantile(selected_phi2, 0.95)) if selected.size else float("nan"),
            "selected_tail_transverse_gain_mean": float(np.mean(selected_gain)) if selected.size else float("nan"),
            "selected_raw_fisher_mean": float(np.mean(selected_raw)) if selected.size else float("nan"),
            "selected_loss_mean": float(np.mean(selected_loss)) if selected.size else float("nan"),
            "selected_loss_p95": float(np.quantile(selected_loss, 0.95)) if selected.size else float("nan"),
            "selected_uncertainty_mean": float(np.mean(scored["uncertainty"][selected])) if selected.size else float("nan"),
            "selected_low_loss_fraction": float(np.mean(scored["low_loss_mask"][selected])) if selected.size else float("nan"),
            "weak_slice_phi2_mean": base_phi2_weak,
            "old_projector_first_order_lift": first_order_lift,
        },
        "deltas_vs_base": {
            "overall_accuracy": test_metrics["overall_accuracy"] - base_test["overall_accuracy"],
            "worst_group_accuracy": test_metrics["worst_group_accuracy"] - base_test["worst_group_accuracy"],
            "fisher_glass_certificate": test_metrics["fisher_glass_certificate"] - base_test["fisher_glass_certificate"],
            "inverse_loss_p95": test_metrics["inverse_loss_p95"] - base_test["inverse_loss_p95"],
            "group_loss_cvar": test_metrics["group_loss_cvar"] - base_test["group_loss_cvar"],
        },
    }


def aggregate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary = []
    keys = sorted({(row["objective"], row["budget_fraction"], row["selector"]) for row in rows})
    for objective, budget, selector in keys:
        items = [row for row in rows if row["objective"] == objective and row["budget_fraction"] == budget and row["selector"] == selector]
        summary.append(
            {
                "objective": objective,
                "budget_fraction": budget,
                "selector": selector,
                "n": len(items),
                "test_avg_accuracy_mean_ci": mean_ci([row["test"]["overall_accuracy"] for row in items]),
                "test_worst_group_accuracy_mean_ci": mean_ci([row["test"]["worst_group_accuracy"] for row in items]),
                "test_fisher_glass_certificate_mean_ci": mean_ci([row["test"]["fisher_glass_certificate"] for row in items]),
                "test_inverse_loss_p95_mean_ci": mean_ci([row["test"]["inverse_loss_p95"] for row in items]),
                "delta_worst_group_accuracy_mean_ci": mean_ci([row["deltas_vs_base"]["worst_group_accuracy"] for row in items]),
                "delta_fisher_glass_certificate_mean_ci": mean_ci([row["deltas_vs_base"]["fisher_glass_certificate"] for row in items]),
                "delta_inverse_loss_p95_mean_ci": mean_ci([row["deltas_vs_base"]["inverse_loss_p95"] for row in items]),
                "selected_phi2_mean_ci": mean_ci([row["acquisition"]["selected_phi2_mean"] for row in items]),
                "selected_tail_transverse_gain_mean_ci": mean_ci([row["acquisition"]["selected_tail_transverse_gain_mean"] for row in items]),
                "selected_raw_fisher_mean_ci": mean_ci([row["acquisition"]["selected_raw_fisher_mean"] for row in items]),
                "selected_loss_mean_ci": mean_ci([row["acquisition"]["selected_loss_mean"] for row in items]),
                "selected_low_loss_fraction_mean_ci": mean_ci([row["acquisition"]["selected_low_loss_fraction"] for row in items]),
                "first_order_lift_mean_ci": mean_ci([row["acquisition"]["old_projector_first_order_lift"] for row in items]),
            }
        )
    return summary


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "objective",
        "budget_fraction",
        "selector",
        "n",
        "test_avg_accuracy",
        "test_worst_group_accuracy",
        "test_fisher_glass_certificate",
        "test_inverse_loss_p95",
        "delta_worst_group_accuracy",
        "delta_fisher_glass_certificate",
        "delta_inverse_loss_p95",
        "selected_phi2",
        "selected_tail_transverse_gain",
        "selected_raw_fisher",
        "selected_loss",
        "selected_low_loss_fraction",
        "first_order_lift",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            flat = {
                "objective": row["objective"],
                "budget_fraction": row["budget_fraction"],
                "selector": row["selector"],
                "n": row["n"],
            }
            for out_name, key in [
                ("test_avg_accuracy", "test_avg_accuracy_mean_ci"),
                ("test_worst_group_accuracy", "test_worst_group_accuracy_mean_ci"),
                ("test_fisher_glass_certificate", "test_fisher_glass_certificate_mean_ci"),
                ("test_inverse_loss_p95", "test_inverse_loss_p95_mean_ci"),
                ("delta_worst_group_accuracy", "delta_worst_group_accuracy_mean_ci"),
                ("delta_fisher_glass_certificate", "delta_fisher_glass_certificate_mean_ci"),
                ("delta_inverse_loss_p95", "delta_inverse_loss_p95_mean_ci"),
                ("selected_phi2", "selected_phi2_mean_ci"),
                ("selected_tail_transverse_gain", "selected_tail_transverse_gain_mean_ci"),
                ("selected_raw_fisher", "selected_raw_fisher_mean_ci"),
                ("selected_loss", "selected_loss_mean_ci"),
                ("selected_low_loss_fraction", "selected_low_loss_fraction_mean_ci"),
                ("first_order_lift", "first_order_lift_mean_ci"),
            ]:
                flat[out_name] = row[key][0]
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/wilds")
    parser.add_argument("--cache", default="build/features/waterbirds_mobilenetv3_small.npz")
    parser.add_argument("--output", default="build/results/waterbirds_antidote_acquisition.json")
    parser.add_argument("--summary-csv", default="build/results/waterbirds_antidote_acquisition_summary.csv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--objective", choices=["erm", "group_dro"], default="erm")
    parser.add_argument("--trainer", choices=["sklearn", "torch"], default="sklearn")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--fg-alpha", type=float, default=0.75)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=[
            "fisher_antidote",
            "fg_tail_transverse",
            "low_loss_tail_transverse",
            "raw_fisher",
            "paired_antidote",
            "paired_random",
            "low_loss_antidote",
            "weak_slice_antidote",
            "weak_slice_random",
            "high_loss",
            "uncertainty",
            "group_balanced",
            "random",
        ],
    )
    args = parser.parse_args()

    # load_waterbirds expects args.seeds to exist for the background probe seed.
    train, val, test = load_waterbirds(args)
    rows = []
    for seed in args.seeds:
        print(f"seed={seed}: training base {args.objective}", flush=True)
        base_params, base_history = fit_classifier(args, train, seed)
        base_val = evaluate(base_params, train, val, args.alpha, args.ridge, args.eps)
        base_test = evaluate(base_params, train, test, args.alpha, args.ridge, args.eps)
        scored = score_candidate_pool(base_params, train, val, args.alpha, args.ridge, args.eps)
        for budget in args.budgets:
            for selector in args.selectors:
                print(f"seed={seed} budget={budget:.3f} selector={selector}", flush=True)
                rows.append(
                    run_one(
                        args,
                        train,
                        val,
                        test,
                        seed,
                        budget,
                        selector,
                        base_params,
                        base_history,
                        base_val,
                        base_test,
                        scored,
                    )
                )
                partial = {"args": vars(args), "summary": aggregate(rows), "runs": rows}
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps(partial, indent=2), encoding="utf-8")

    summary = aggregate(rows)
    result = {"args": vars(args), "summary": summary, "runs": rows}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    write_summary_csv(Path(args.summary_csv), summary)
    print(json.dumps({"output": str(output), "summary_csv": args.summary_csv, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
