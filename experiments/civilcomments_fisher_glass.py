"""CivilComments-WILDS Fisher-glass diagnostic.

This script runs a lightweight real benchmark using the official WILDS
CivilComments dataset. It trains two TF-IDF/linear models:

* a toxicity classifier, used for accuracy/loss and as an operational task
  score z(x);
* an identity-any probe, used as the nuisance score b(x).

For identity-by-label subgroups on the WILDS test split, it computes a
low-dimensional feature-Schur Fisher diagnostic

    F_{T|N,g} = E_g[z^2] - E_g[zb]^2 / (E_g[b^2] + ridge),

plus a predictive-Fisher variant weighted by p(1-p). This is intentionally a
diagnostic benchmark, not a full deep robust-learning comparison.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, log_loss
from wilds import get_dataset

from ridge_exact_audit import audit_scalar_environments, schur_from_moments


IDENTITY_FIELDS = [
    "male",
    "female",
    "LGBTQ",
    "christian",
    "muslim",
    "other_religions",
    "black",
    "white",
]


def as_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def cvar(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    threshold = np.quantile(values, alpha)
    tail = values[values >= threshold]
    return float(tail.mean())


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
    weight = weight / max(float(np.sum(weight)), 1e-12)
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


def tail_exponent(values: np.ndarray) -> float | None:
    values = np.sort(np.asarray(values, dtype=float))
    if values.size < 8:
        return None
    n = values.size
    start = max(0, int(0.5 * n))
    x = values[start:]
    ranks = np.arange(start, n)
    surv = (n - ranks) / n
    mask = (x > 0) & (surv > 0)
    if int(mask.sum()) < 4:
        return None
    slope, _ = np.polyfit(np.log(x[mask]), np.log(surv[mask]), 1)
    return float(-slope)


def tolist_text(values: list[str], idx: np.ndarray) -> list[str]:
    out: list[str] = []
    for i in idx:
        value = values[int(i)]
        out.append(value if isinstance(value, str) else "")
    return out


def build_groups(metadata: np.ndarray, y: np.ndarray, fields: list[str], field_to_idx: dict[str, int]) -> list[dict]:
    groups: list[dict] = []
    for field in fields:
        col = field_to_idx[field]
        present = metadata[:, col] == 1
        for label in (0, 1):
            mask = present & (y == label)
            groups.append({"name": f"{field}/y={label}", "mask": mask, "identity": field, "label": label})
    any_col = field_to_idx["identity_any"]
    present_any = metadata[:, any_col] == 1
    for label in (0, 1):
        mask = present_any & (y == label)
        groups.append({"name": f"identity_any/y={label}", "mask": mask, "identity": "identity_any", "label": label})
    return groups


def summarize_basis(
    name: str,
    groups: list[dict],
    y_test: np.ndarray,
    pred_test: np.ndarray,
    losses: np.ndarray,
    z: np.ndarray,
    b: np.ndarray,
    ridge: float,
    eps: float,
    alpha: float,
    min_group_size: int,
) -> dict:
    rows = []
    for group in groups:
        mask = group["mask"]
        n = int(mask.sum())
        if n < min_group_size:
            continue
        f_feature = fisher_schur(z[mask], b[mask], ridge)
        rows.append(
            {
                "group": group["name"],
                "n": n,
                "accuracy": float(accuracy_score(y_test[mask], pred_test[mask])),
                "loss": float(np.mean(losses[mask])),
                "projected_feature_fisher": f_feature["projected"],
                "inverse_feature_loss": float(1.0 / (f_feature["projected"] + eps)),
            }
        )
    inverse = np.array([row["inverse_feature_loss"] for row in rows], dtype=float)
    exact_values = []
    for group in groups:
        mask = group["mask"]
        if int(mask.sum()) >= min_group_size:
            exact_values.append(fisher_schur(z[mask], b[mask], ridge)["exact_projected"])
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/wilds")
    parser.add_argument("--output", default="build/results/civilcomments_fisher_glass_results.json")
    parser.add_argument("--download", action="store_true")
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--max-features", type=int, default=50000)
    parser.add_argument("--min-df", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--audit-bootstrap-replicates", type=int, default=100000)
    parser.add_argument("--audit-confidence", type=float, default=0.95)
    parser.add_argument("--audit-seed", type=int, default=20260729)
    parser.add_argument("--min-group-size", type=int, default=100)
    parser.add_argument("--reserve-r", type=float, default=0.05)
    parser.add_argument("--max-iter", type=int, default=50)
    parser.add_argument("--tol", type=float, default=1e-3)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    dataset = get_dataset(dataset="civilcomments", root_dir=args.root_dir, download=args.download)
    texts = dataset._text_array
    y = as_numpy(dataset._y_array).astype(int)
    metadata = as_numpy(dataset._metadata_array).astype(int)
    split = as_numpy(dataset._split_array).astype(int)
    field_to_idx = {name: i for i, name in enumerate(dataset.metadata_fields)}

    train_idx = np.flatnonzero(split == dataset.split_dict["train"])
    val_idx = np.flatnonzero(split == dataset.split_dict["val"])
    test_idx = np.flatnonzero(split == dataset.split_dict["test"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=args.min_df,
        max_features=args.max_features,
        sublinear_tf=True,
    )
    x_train = vectorizer.fit_transform(tolist_text(texts, train_idx))
    x_test = vectorizer.transform(tolist_text(texts, test_idx))

    y_train = y[train_idx]
    y_test = y[test_idx]

    tox_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=args.max_iter,
        tol=args.tol,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    tox_model.fit(x_train, y_train)

    identity_any_train = metadata[train_idx, field_to_idx["identity_any"]]
    probe_model = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=1e-5,
        max_iter=args.max_iter,
        tol=args.tol,
        class_weight="balanced",
        random_state=args.seed + 1,
        n_jobs=-1,
    )
    probe_model.fit(x_train, identity_any_train)

    train_task_score = tox_model.decision_function(x_train)
    test_task_score = tox_model.decision_function(x_test)
    train_probe_score = probe_model.decision_function(x_train)
    test_probe_score = probe_model.decision_function(x_test)
    z_test, z_norm = normalize(train_task_score, test_task_score)
    b_test, b_norm = normalize(train_probe_score, test_probe_score)

    prob_test = tox_model.predict_proba(x_test)[:, 1]
    pred_test = (prob_test >= 0.5).astype(int)
    test_losses = -(y_test * np.log(np.clip(prob_test, 1e-12, 1.0)) + (1 - y_test) * np.log(np.clip(1.0 - prob_test, 1e-12, 1.0)))
    predictive_weight = prob_test * (1.0 - prob_test)

    groups = build_groups(metadata[test_idx], y_test, IDENTITY_FIELDS, field_to_idx)
    train_identity_score = identity_any_train.astype(float)
    test_identity_score = metadata[test_idx, field_to_idx["identity_any"]].astype(float)
    b_metadata, _ = normalize(train_identity_score, test_identity_score)
    random_scores = rng.normal(size=y_test.size)
    b_random, _ = normalize(random_scores, random_scores)
    b_under = np.zeros_like(z_test)
    over_scores = z_test + 0.01 * rng.normal(size=z_test.size)
    b_over, _ = normalize(over_scores, over_scores)
    b_hidden_loss, _ = normalize(test_losses, test_losses)
    nuisance_ablation = [
        summarize_basis("metadata nuisance", groups, y_test, pred_test, test_losses, z_test, b_metadata, args.ridge, args.eps, args.alpha, args.min_group_size),
        summarize_basis("learned probe nuisance", groups, y_test, pred_test, test_losses, z_test, b_test, args.ridge, args.eps, args.alpha, args.min_group_size),
        summarize_basis("random nuisance", groups, y_test, pred_test, test_losses, z_test, b_random, args.ridge, args.eps, args.alpha, args.min_group_size),
        summarize_basis("under-projected nuisance", groups, y_test, pred_test, test_losses, z_test, b_under, args.ridge, args.eps, args.alpha, args.min_group_size),
        summarize_basis("over-projected nuisance", groups, y_test, pred_test, test_losses, z_test, b_over, args.ridge, args.eps, args.alpha, args.min_group_size),
        summarize_basis("hidden-loss nuisance", groups, y_test, pred_test, test_losses, z_test, b_hidden_loss, args.ridge, args.eps, args.alpha, args.min_group_size),
    ]
    group_rows = []
    for group in groups:
        mask = group["mask"]
        n = int(mask.sum())
        if n < args.min_group_size:
            continue
        f_feature = fisher_schur(z_test[mask], b_test[mask], args.ridge)
        f_predictive = fisher_schur(z_test[mask], b_test[mask], args.ridge, predictive_weight[mask])
        row = {
            "group": group["name"],
            "n": n,
            "accuracy": float(accuracy_score(y_test[mask], pred_test[mask])),
            "loss": float(np.mean(test_losses[mask])),
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
        group_rows.append(row)

    if not group_rows:
        raise RuntimeError("No groups passed the minimum size threshold.")

    audit_groups = [group for group in groups if int(group["mask"].sum()) >= args.min_group_size]
    audit_environments = [
        {
            "name": group["name"],
            "z": z_test[group["mask"]],
            "b": b_test[group["mask"]],
        }
        for group in audit_groups
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
            for environment, group in zip(audit_environments, audit_groups)
        ],
        ridge=args.ridge,
        eps=args.eps,
        alpha=args.alpha,
        bootstrap_replicates=args.audit_bootstrap_replicates,
        confidence=args.audit_confidence,
        seed=args.audit_seed + 1,
    )

    feature_inverse = np.array([row["inverse_feature_loss"] for row in group_rows])
    predictive_inverse = np.array([row["inverse_predictive_loss"] for row in group_rows])
    group_losses = np.array([row["loss"] for row in group_rows])
    group_acc = np.array([row["accuracy"] for row in group_rows])

    # A simple anchor distribution: balanced train examples with identity_any=1.
    identity_train_mask = identity_any_train == 1
    pos_anchor = train_idx[identity_train_mask & (y_train == 1)]
    neg_anchor = train_idx[identity_train_mask & (y_train == 0)]
    anchor_size = int(min(len(pos_anchor), len(neg_anchor), 20000))
    if anchor_size > 0:
        pos_sel = rng.choice(pos_anchor, size=anchor_size, replace=False)
        neg_sel = rng.choice(neg_anchor, size=anchor_size, replace=False)
        anchor_idx = np.concatenate([pos_sel, neg_sel])
        x_anchor = vectorizer.transform(tolist_text(texts, anchor_idx))
        anchor_z_raw = tox_model.decision_function(x_anchor)
        anchor_b_raw = probe_model.decision_function(x_anchor)
        anchor_z = (anchor_z_raw - z_norm["mean"]) / z_norm["std"]
        anchor_b = (anchor_b_raw - b_norm["mean"]) / b_norm["std"]
        anchor_fisher = fisher_schur(anchor_z, anchor_b, args.ridge)["projected"]
    else:
        anchor_fisher = 0.0

    projected = np.array([row["projected_feature_fisher"] for row in group_rows])
    mixed_projected = (1.0 - args.reserve_r) * projected + args.reserve_r * anchor_fisher
    mixed_inverse = 1.0 / (mixed_projected + args.eps)

    worst_by_accuracy = min(group_rows, key=lambda row: row["accuracy"])
    worst_by_fisher = max(group_rows, key=lambda row: row["inverse_feature_loss"])

    result = {
        "schema_version": 2,
        "dataset": "CivilComments-WILDS",
        "model": "TF-IDF bigram + SGD logistic regression",
        "seed": args.seed,
        "n_train": int(train_idx.size),
        "n_val": int(val_idx.size),
        "n_test": int(test_idx.size),
        "max_features": args.max_features,
        "min_df": args.min_df,
        "max_iter": args.max_iter,
        "tol": args.tol,
        "min_group_size": args.min_group_size,
        "n_groups": len(group_rows),
        "ridge": args.ridge,
        "eps": args.eps,
        "alpha": args.alpha,
        "overall_accuracy": float(accuracy_score(y_test, pred_test)),
        "overall_loss": float(log_loss(y_test, prob_test, labels=[0, 1])),
        "group_loss_cvar": cvar(group_losses, args.alpha),
        "worst_group_accuracy": float(group_acc.min()),
        "mean_group_accuracy": float(group_acc.mean()),
        "feature_raw_fisher_mean": float(np.mean([row["raw_feature_fisher"] for row in group_rows])),
        "feature_projected_fisher_min": float(np.min(projected)),
        "feature_projected_fisher_q05": float(np.quantile(projected, 0.05)),
        "feature_inverse_loss_p95": float(np.quantile(feature_inverse, 0.95)),
        "feature_inverse_loss_cvar": cvar(feature_inverse, args.alpha),
        "fisher_glass_certificate": float(1.0 / cvar(feature_inverse, args.alpha)),
        "feature_tail_exponent_beta_hat": tail_exponent(feature_inverse),
        "predictive_fisher_glass_certificate": float(1.0 / cvar(predictive_inverse, args.alpha)),
        "predictive_inverse_loss_p95": float(np.quantile(predictive_inverse, 0.95)),
        "ridge_exact_audit": ridge_exact_audit,
        "predictive_ridge_exact_audit": predictive_ridge_exact_audit,
        "reserve_r": args.reserve_r,
        "reserve_anchor_projected_fisher": float(anchor_fisher),
        "reserve_fisher_glass_certificate": float(1.0 / cvar(mixed_inverse, args.alpha)),
        "reserve_theory_floor": float(args.reserve_r * anchor_fisher),
        "worst_group_by_accuracy": worst_by_accuracy,
        "worst_group_by_inverse_fisher": worst_by_fisher,
        "nuisance_ablation": nuisance_ablation,
        "groups": sorted(group_rows, key=lambda row: row["inverse_feature_loss"], reverse=True),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({k: result[k] for k in [
        "dataset",
        "n_train",
        "n_test",
        "n_groups",
        "overall_accuracy",
        "worst_group_accuracy",
        "overall_loss",
        "group_loss_cvar",
        "fisher_glass_certificate",
        "feature_inverse_loss_p95",
        "feature_tail_exponent_beta_hat",
        "reserve_fisher_glass_certificate",
        "reserve_theory_floor",
        "worst_group_by_accuracy",
        "worst_group_by_inverse_fisher",
    ]}, indent=2))


if __name__ == "__main__":
    main()
