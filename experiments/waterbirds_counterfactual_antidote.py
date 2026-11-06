"""Waterbirds counterfactual Fisher-antidote experiment.

This script tests the interventional version of tail-transversality design.
Instead of adding unpaired validation examples, each candidate action is a
same-label, opposite-background feature-space counterfactual:

    x_cf = x + gamma * (mean_train[y, 1-bg] - mean_train[y, bg]).

This is not a pixel-level segmentation/background-swap benchmark. It is a
controlled frozen-feature proxy for the claim that Fisher-Glass should select
task-preserving nuisance interventions, not generic high-scoring examples.
The selected counterfactual feature examples are added to the training set, the
linear head is retrained, and evaluation is performed on the official
Waterbirds test split.
"""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from waterbirds_fisher_glass_training import (
    GROUP_NAMES,
    SplitData,
    evaluate,
    group_ids,
    load_waterbirds,
    normalize_scores as normalize_train_scores,
    predict_logits,
)
from waterbirds_antidote_acquisition import (
    fit_classifier,
    group_counts,
    mean_ci,
    make_augmented_train,
    score_candidate_pool,
    select_group_balanced,
    sigmoid,
    top_k,
)


def stable_selector_offset(selector: str) -> int:
    return sum((i + 1) * ord(char) for i, char in enumerate(selector))


def refit_background_nuisance(
    train: SplitData,
    val: SplitData,
    test: SplitData,
    seed: int,
) -> tuple[SplitData, SplitData, SplitData, LogisticRegression, dict[str, float]]:
    """Fit a reusable background probe so counterfactual features get nuisance scores."""

    probe = LogisticRegression(max_iter=1000, class_weight="balanced", solver="lbfgs", random_state=seed)
    probe.fit(train.x, train.bg)
    train_scores = probe.decision_function(train.x)
    train_nuisance, norm = normalize_train_scores(train_scores, train_scores)
    val_nuisance = ((probe.decision_function(val.x) - norm["mean"]) / norm["std"]).astype(np.float32)
    test_nuisance = ((probe.decision_function(test.x) - norm["mean"]) / norm["std"]).astype(np.float32)
    return (
        replace(train, nuisance=train_nuisance),
        replace(val, nuisance=val_nuisance),
        replace(test, nuisance=test_nuisance),
        probe,
        norm,
    )


def counterfactual_pool(
    train: SplitData,
    pool: SplitData,
    probe: LogisticRegression,
    nuisance_norm: dict[str, float],
    gamma: float,
) -> tuple[SplitData, dict[str, Any]]:
    """Create same-label, opposite-background feature-space interventions."""

    group_means: dict[tuple[int, int], np.ndarray] = {}
    global_bg_means: dict[int, np.ndarray] = {}
    for label in (0, 1):
        for bg in (0, 1):
            mask = (train.y == label) & (train.bg == bg)
            if mask.any():
                group_means[(label, bg)] = train.x[mask].mean(axis=0)
    for bg in (0, 1):
        mask = train.bg == bg
        global_bg_means[bg] = train.x[mask].mean(axis=0)

    x_cf = np.empty_like(pool.x)
    shift_norms = np.empty(pool.x.shape[0], dtype=np.float32)
    used_group_shift = np.zeros(pool.x.shape[0], dtype=bool)
    for i, (x, label, bg) in enumerate(zip(pool.x, pool.y.astype(int), pool.bg.astype(int), strict=True)):
        src = group_means.get((label, bg))
        dst = group_means.get((label, 1 - bg))
        if src is None or dst is None:
            src = global_bg_means[bg]
            dst = global_bg_means[1 - bg]
        else:
            used_group_shift[i] = True
        shift = dst - src
        x_cf[i] = (x + gamma * shift).astype(np.float32)
        shift_norms[i] = float(np.linalg.norm(gamma * shift))

    bg_cf = (1 - pool.bg.astype(int)).astype(int)
    nuisance_scores = probe.decision_function(x_cf)
    nuisance_cf = ((nuisance_scores - nuisance_norm["mean"]) / nuisance_norm["std"]).astype(np.float32)
    cf = SplitData(
        x=x_cf,
        y=pool.y.copy(),
        bg=bg_cf,
        group=group_ids(pool.y, bg_cf),
        nuisance=nuisance_cf,
    )
    meta = {
        "gamma": gamma,
        "mean_shift_norm": float(shift_norms.mean()),
        "p95_shift_norm": float(np.quantile(shift_norms, 0.95)),
        "used_label_conditional_shift_fraction": float(used_group_shift.mean()),
    }
    return cf, meta


def score_counterfactual_actions(
    params: np.ndarray,
    train: SplitData,
    cf_pool: SplitData,
    tail_info: dict[str, Any],
    eps: float,
) -> dict[str, np.ndarray]:
    train_logits = predict_logits(params, train.x)
    cf_logits = predict_logits(params, cf_pool.x)
    z_cf, _ = normalize_train_scores(train_logits, cf_logits)
    probs = sigmoid(cf_logits)
    losses = -(
        cf_pool.y * np.log(np.clip(probs, 1e-12, 1.0))
        + (1 - cf_pool.y) * np.log(np.clip(1.0 - probs, 1e-12, 1.0))
    )
    beta = float(tail_info["weakest"]["beta"])
    phi = z_cf - beta * cf_pool.nuisance
    tail_gain = np.zeros_like(z_cf, dtype=np.float64)
    tail_weight_sum = 0.0
    for row in tail_info["tail_rows"]:
        lam = float(row["projected_feature_fisher"])
        weight = 1.0 / ((lam + eps) ** 2)
        phi_g = z_cf - float(row["beta"]) * cf_pool.nuisance
        tail_gain += weight * phi_g * phi_g
        tail_weight_sum += weight
    tail_gain = tail_gain / max(tail_weight_sum, 1e-12)
    tail_rank = rank01(tail_gain)
    loss_rank = rank01(losses)
    return {
        "z": z_cf,
        "phi2": (phi * phi).astype(np.float32),
        "tail_transverse_gain": tail_gain.astype(np.float32),
        "raw_fisher": (z_cf * z_cf).astype(np.float32),
        "loss": losses.astype(np.float32),
        "uncertainty": (probs * (1.0 - probs)).astype(np.float32),
        "tail_x_loss": (tail_gain * losses).astype(np.float32),
        "tail_rank_x_loss_rank": (tail_rank * loss_rank).astype(np.float32),
        "tail_x_uncertainty": (tail_gain * probs * (1.0 - probs)).astype(np.float32),
        "low_loss_mask": losses <= float(np.median(losses)),
    }


def rank01(scores: np.ndarray) -> np.ndarray:
    order = np.argsort(np.asarray(scores, dtype=float))
    ranks = np.empty(order.size, dtype=np.float64)
    if order.size <= 1:
        ranks.fill(1.0)
        return ranks
    ranks[order] = np.linspace(0.0, 1.0, order.size)
    return ranks


def select_actions(
    selector: str,
    scores: dict[str, np.ndarray],
    cf_pool: SplitData,
    k: int,
    rng: np.random.Generator,
) -> np.ndarray:
    low_loss_allowed = np.flatnonzero(scores["low_loss_mask"])
    if selector == "fg_tail_transverse_swap":
        return top_k(scores["tail_transverse_gain"], k)
    if selector == "fg_tail_x_loss_swap":
        return top_k(scores["tail_x_loss"], k)
    if selector == "fg_tail_rank_x_loss_rank_swap":
        return top_k(scores["tail_rank_x_loss_rank"], k)
    if selector == "fg_tail_x_uncertainty_swap":
        return top_k(scores["tail_x_uncertainty"], k)
    if selector == "low_loss_tail_transverse_swap":
        return top_k(scores["tail_transverse_gain"], k, low_loss_allowed)
    if selector == "raw_fisher_swap":
        return top_k(scores["raw_fisher"], k)
    if selector == "high_loss_swap":
        return top_k(scores["loss"], k)
    if selector == "uncertainty_swap":
        return top_k(scores["uncertainty"], k)
    if selector == "group_balanced_swap":
        return select_group_balanced(rng, cf_pool.group, k)
    if selector == "random_swap":
        return rng.choice(np.arange(len(cf_pool.y)), size=k, replace=False).astype(int)
    raise ValueError(f"Unknown selector: {selector}")


def run_one(
    args: argparse.Namespace,
    train: SplitData,
    val: SplitData,
    test: SplitData,
    cf_pool: SplitData,
    cf_meta: dict[str, Any],
    seed: int,
    budget_fraction: float,
    selector: str,
    base_params: np.ndarray,
    base_history: dict[str, Any],
    base_val: dict[str, Any],
    base_test: dict[str, Any],
    tail_info: dict[str, Any],
    scores: dict[str, np.ndarray],
) -> dict[str, Any]:
    rng = np.random.default_rng(seed + int(100000 * budget_fraction) + stable_selector_offset(selector))
    k = max(1, int(round(budget_fraction * len(train.y))))
    k = min(k, len(cf_pool.y))
    selected = select_actions(selector, scores, cf_pool, k, rng)
    augmented = make_augmented_train(train, cf_pool, selected)
    params, history = fit_classifier(args, augmented, seed + 1000)
    test_metrics = evaluate(params, augmented, test, args.alpha, args.ridge, args.eps)
    val_metrics = evaluate(params, augmented, val, args.alpha, args.ridge, args.eps)

    selected_tail = scores["tail_transverse_gain"][selected]
    selected_raw = scores["raw_fisher"][selected]
    selected_loss = scores["loss"][selected]
    selected_phi2 = scores["phi2"][selected]
    return {
        "seed": seed,
        "budget_fraction": float(budget_fraction),
        "budget_n": int(selected.size),
        "selector": selector,
        "intervention": "same_label_opposite_background_feature_shift",
        "counterfactual_meta": cf_meta,
        "weakest_validation_slice": tail_info["weakest"],
        "tail_validation_slices": tail_info["tail_rows"],
        "base_history": base_history,
        "history": history,
        "base_val": base_val,
        "base_test": base_test,
        "val": val_metrics,
        "test": test_metrics,
        "acquisition": {
            "selected_counterfactual_group_counts": group_counts(cf_pool.group[selected]),
            "selected_tail_transverse_gain_mean": float(np.mean(selected_tail)) if selected.size else float("nan"),
            "selected_tail_transverse_gain_p95": float(np.quantile(selected_tail, 0.95)) if selected.size else float("nan"),
            "selected_raw_fisher_mean": float(np.mean(selected_raw)) if selected.size else float("nan"),
            "selected_loss_mean": float(np.mean(selected_loss)) if selected.size else float("nan"),
            "selected_uncertainty_mean": float(np.mean(scores["uncertainty"][selected])) if selected.size else float("nan"),
            "selected_phi2_mean": float(np.mean(selected_phi2)) if selected.size else float("nan"),
            "selected_tail_x_loss_mean": float(np.mean(scores["tail_x_loss"][selected])) if selected.size else float("nan"),
            "selected_tail_rank_x_loss_rank_mean": float(np.mean(scores["tail_rank_x_loss_rank"][selected])) if selected.size else float("nan"),
            "selected_low_loss_fraction": float(np.mean(scores["low_loss_mask"][selected])) if selected.size else float("nan"),
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
    keys = sorted({(row["budget_fraction"], row["selector"]) for row in rows})
    for budget, selector in keys:
        items = [row for row in rows if row["budget_fraction"] == budget and row["selector"] == selector]
        summary.append(
            {
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
                "selected_tail_transverse_gain_mean_ci": mean_ci(
                    [row["acquisition"]["selected_tail_transverse_gain_mean"] for row in items]
                ),
                "selected_raw_fisher_mean_ci": mean_ci([row["acquisition"]["selected_raw_fisher_mean"] for row in items]),
                "selected_loss_mean_ci": mean_ci([row["acquisition"]["selected_loss_mean"] for row in items]),
                "selected_tail_x_loss_mean_ci": mean_ci([row["acquisition"]["selected_tail_x_loss_mean"] for row in items]),
                "selected_tail_rank_x_loss_rank_mean_ci": mean_ci(
                    [row["acquisition"]["selected_tail_rank_x_loss_rank_mean"] for row in items]
                ),
                "selected_low_loss_fraction_mean_ci": mean_ci(
                    [row["acquisition"]["selected_low_loss_fraction"] for row in items]
                ),
            }
        )
    return summary


def write_summary_csv(path: Path, summary: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
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
        "selected_tail_transverse_gain",
        "selected_raw_fisher",
        "selected_loss",
        "selected_tail_x_loss",
        "selected_tail_rank_x_loss_rank",
        "selected_low_loss_fraction",
    ]
    metric_map = {
        "test_avg_accuracy": "test_avg_accuracy_mean_ci",
        "test_worst_group_accuracy": "test_worst_group_accuracy_mean_ci",
        "test_fisher_glass_certificate": "test_fisher_glass_certificate_mean_ci",
        "test_inverse_loss_p95": "test_inverse_loss_p95_mean_ci",
        "delta_worst_group_accuracy": "delta_worst_group_accuracy_mean_ci",
        "delta_fisher_glass_certificate": "delta_fisher_glass_certificate_mean_ci",
        "delta_inverse_loss_p95": "delta_inverse_loss_p95_mean_ci",
        "selected_tail_transverse_gain": "selected_tail_transverse_gain_mean_ci",
        "selected_raw_fisher": "selected_raw_fisher_mean_ci",
        "selected_loss": "selected_loss_mean_ci",
        "selected_tail_x_loss": "selected_tail_x_loss_mean_ci",
        "selected_tail_rank_x_loss_rank": "selected_tail_rank_x_loss_rank_mean_ci",
        "selected_low_loss_fraction": "selected_low_loss_fraction_mean_ci",
    }
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            flat = {"budget_fraction": row["budget_fraction"], "selector": row["selector"], "n": row["n"]}
            for out_name, key in metric_map.items():
                flat[out_name] = row[key][0]
            writer.writerow(flat)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/wilds")
    parser.add_argument("--cache", default="build/features/waterbirds_mobilenetv3_small.npz")
    parser.add_argument("--output", default="build/results/waterbirds_counterfactual_antidote.json")
    parser.add_argument("--summary-csv", default="build/results/waterbirds_counterfactual_antidote_summary.csv")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--trainer", choices=["sklearn", "torch"], default="sklearn")
    parser.add_argument("--objective", choices=["erm", "group_dro"], default="erm")
    parser.add_argument("--max-iter", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--fg-alpha", type=float, default=0.75)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--budgets", type=float, nargs="+", default=[0.01, 0.02, 0.05])
    parser.add_argument(
        "--selectors",
        nargs="+",
        default=[
            "fg_tail_transverse_swap",
            "fg_tail_x_loss_swap",
            "fg_tail_rank_x_loss_rank_swap",
            "fg_tail_x_uncertainty_swap",
            "low_loss_tail_transverse_swap",
            "raw_fisher_swap",
            "high_loss_swap",
            "uncertainty_swap",
            "group_balanced_swap",
            "random_swap",
        ],
    )
    args = parser.parse_args()

    train, val, test = load_waterbirds(args)
    train, val, test, probe, nuisance_norm = refit_background_nuisance(train, val, test, args.seeds[0] + 1)
    cf_pool, cf_meta = counterfactual_pool(train, val, probe, nuisance_norm, args.gamma)
    rows = []
    for seed in args.seeds:
        print(f"seed={seed}: training base {args.objective}", flush=True)
        base_params, base_history = fit_classifier(args, train, seed)
        base_val = evaluate(base_params, train, val, args.alpha, args.ridge, args.eps)
        base_test = evaluate(base_params, train, test, args.alpha, args.ridge, args.eps)
        tail_info = score_candidate_pool(base_params, train, val, args.alpha, args.ridge, args.eps)
        action_scores = score_counterfactual_actions(base_params, train, cf_pool, tail_info, args.eps)
        for budget in args.budgets:
            for selector in args.selectors:
                print(f"seed={seed} budget={budget:.3f} selector={selector}", flush=True)
                rows.append(
                    run_one(
                        args,
                        train,
                        val,
                        test,
                        cf_pool,
                        cf_meta,
                        seed,
                        budget,
                        selector,
                        base_params,
                        base_history,
                        base_val,
                        base_test,
                        tail_info,
                        action_scores,
                    )
                )
                output = Path(args.output)
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text(json.dumps({"args": vars(args), "summary": aggregate(rows), "runs": rows}, indent=2), encoding="utf-8")

    summary = aggregate(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"args": vars(args), "summary": summary, "runs": rows}, indent=2), encoding="utf-8")
    write_summary_csv(Path(args.summary_csv), summary)
    print(json.dumps({"output": str(output), "summary_csv": args.summary_csv, "counterfactual_meta": cf_meta, "summary": summary}, indent=2))


if __name__ == "__main__":
    main()
