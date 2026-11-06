"""Step 2: validation-only selection, then freeze.

Nothing here touches the test split's *shifted* behaviour.  Using validation
data alone we

1. fit a temperature for the trained model,
2. fit a cross-fitted background probe on the trained representation and use its
   score as the nuisance coordinate ``b``,
3. score every candidate slice, keep only the slices that already look safe

       G_benign = { g : Acc_val(g) >= Acc_val(overall)  and  NLL_val(g) <= median_h NLL_val(h) },

4. select                g* = argmin_{g in G_benign} lambda_min(F^rid_{T|N,g}),
5. pick the Fisher-Glass penalty for the intervention arm as the largest
   validation certificate subject to validation accuracy within
   ``--acc-tolerance`` of ERM,

and write everything -- checkpoint hash, temperature, probe coefficients,
standardisation constants, slice definition, bin edges, ridge/alpha/eps -- into
``frozen_selection.json``.  ``evaluate_shift.py`` refuses to run if those
artifacts changed, which is what makes the held-out result prospective rather
than post hoc.

Two candidate slice families are scored:

``groups``  the four canonical (label, background) groups;
``bins``    within each group, quantile bins of the cross-fitted probe score,
            which isolates a rarer, more nuisance-aligned subpopulation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold

from common import (
    GROUP_NAMES,
    binary_nll,
    cvar,
    file_sha256,
    fisher_schur,
    fit_temperature,
    json_sha256,
    load_metadata,
    log,
    read_json,
    sigmoid,
    write_json,
)

NUM_GROUPS = 4


# --------------------------------------------------------------------------- #
# Nuisance probe
# --------------------------------------------------------------------------- #
def fit_background_probe(
    val_features: np.ndarray,
    val_place: np.ndarray,
    val_group: np.ndarray,
    test_features: np.ndarray,
    seed: int,
    n_splits: int = 5,
    max_iter: int = 3000,
) -> dict:
    """Cross-fitted background probe on the trained representation.

    Out-of-fold scores are used for the validation-side geometry so the probe
    never scores a point it was fit on; a full-fit probe scores the test split.
    Each side is standardised by its own statistics, which keeps ``J_NN ~ 1`` in
    both so the fixed ridge means the same thing on both sides (the Schur
    complement is otherwise scale-free in ``b``).
    """
    mean = val_features.mean(axis=0, keepdims=True)
    std = val_features.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0
    x_val = (val_features - mean) / std
    x_test = (test_features - mean) / std

    oof = np.zeros(x_val.shape[0], dtype=np.float64)
    splitter = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, held_idx in splitter.split(x_val, val_group):
        fold = LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="lbfgs")
        fold.fit(x_val[train_idx], val_place[train_idx])
        oof[held_idx] = fold.decision_function(x_val[held_idx])

    full = LogisticRegression(max_iter=max_iter, class_weight="balanced", solver="lbfgs")
    full.fit(x_val, val_place)
    test_raw = full.decision_function(x_test)

    b_val = (oof - oof.mean()) / (oof.std() or 1.0)
    b_test = (test_raw - test_raw.mean()) / (test_raw.std() or 1.0)
    auc_proxy = float(((oof > 0).astype(int) == val_place).mean())
    return {
        "b_val": b_val,
        "b_test": b_test,
        "oof_mean": float(oof.mean()),
        "oof_std": float(oof.std() or 1.0),
        "test_mean": float(test_raw.mean()),
        "test_std": float(test_raw.std() or 1.0),
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "coef": full.coef_.astype(np.float32),
        "intercept": full.intercept_.astype(np.float32),
        "crossfit_accuracy": auc_proxy,
        "n_splits": n_splits,
    }


# --------------------------------------------------------------------------- #
# Slice families
# --------------------------------------------------------------------------- #
def quantile_bins(values: np.ndarray, mask: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin edges (with open ends) from the masked values."""
    edges = np.quantile(values[mask], np.linspace(0.0, 1.0, n_bins + 1))
    edges = np.asarray(edges, dtype=float)
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def assign_bin(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    return np.clip(np.digitize(values, edges[1:-1]), 0, len(edges) - 2)


def slice_stats(
    name: str,
    kind: str,
    gid: int,
    bin_index: int | None,
    mask: np.ndarray,
    z: np.ndarray,
    b: np.ndarray,
    logits: np.ndarray,
    y: np.ndarray,
    temperature: float,
    ridge: float,
    eps: float,
) -> dict:
    preds = (logits >= 0.0).astype(int)
    nll = binary_nll(logits, y)
    nll_cal = binary_nll(logits / temperature, y)
    probs_cal = sigmoid(logits / temperature)
    weight = probs_cal * (1.0 - probs_cal)

    unweighted = fisher_schur(z[mask], b[mask], ridge)
    calibrated = fisher_schur(z[mask], b[mask], ridge, weight=weight[mask])
    return {
        "name": name,
        "kind": kind,
        "gid": int(gid),
        "bin_index": None if bin_index is None else int(bin_index),
        "n": int(mask.sum()),
        "accuracy": float((preds[mask] == y[mask]).mean()),
        "nll": float(nll[mask].mean()),
        "nll_calibrated": float(nll_cal[mask].mean()),
        "raw_fisher": unweighted["raw"],
        "projected_fisher": unweighted["projected"],
        "projected_over_raw": unweighted["projected_over_raw"],
        "beta": unweighted["beta"],
        "projected_fisher_calibrated": calibrated["projected"],
        "projected_over_raw_calibrated": calibrated["projected_over_raw"],
        "inverse_projected_fisher": float(1.0 / (unweighted["projected"] + eps)),
        "inverse_projected_fisher_calibrated": float(1.0 / (calibrated["projected"] + eps)),
    }


def build_slice_table(
    family: str,
    z: np.ndarray,
    b: np.ndarray,
    logits: np.ndarray,
    y: np.ndarray,
    group: np.ndarray,
    temperature: float,
    args,
) -> tuple[list[dict], dict[int, np.ndarray]]:
    rows: list[dict] = []
    edges_by_group: dict[int, np.ndarray] = {}
    for gid in range(NUM_GROUPS):
        group_mask = group == gid
        if not group_mask.any():
            continue
        if family == "groups":
            rows.append(
                slice_stats(
                    GROUP_NAMES[gid], "group", gid, None, group_mask, z, b, logits, y, temperature, args.ridge, args.eps
                )
            )
            continue
        if int(group_mask.sum()) < args.n_bins * args.min_bin:
            continue
        edges = quantile_bins(b, group_mask, args.n_bins)
        edges_by_group[gid] = edges
        assigned = assign_bin(b, edges)
        for bin_index in range(args.n_bins):
            mask = group_mask & (assigned == bin_index)
            if int(mask.sum()) < args.min_bin:
                continue
            rows.append(
                slice_stats(
                    f"{GROUP_NAMES[gid]} | probe bin {bin_index + 1}/{args.n_bins}",
                    "bin",
                    gid,
                    bin_index,
                    mask,
                    z,
                    b,
                    logits,
                    y,
                    temperature,
                    args.ridge,
                    args.eps,
                )
            )
    return rows, edges_by_group


# The absolute projected Fisher is contaminated by J_TT, the within-slice spread
# of the standardised logit, which varies several-fold across Waterbirds groups.
# The scale-free ratio -- the fraction of task Fisher that survives projecting the
# nuisance out -- is what "identifiability relative to the available signal"
# should mean, so it is the default selection statistic.
SELECTION_STATISTICS = {
    "projected_over_raw": "projected_over_raw_calibrated",
    "projected_fisher": "projected_fisher_calibrated",
}


def apply_rule(rows: list[dict], overall_accuracy: float, statistic: str = "projected_over_raw") -> dict:
    """G_benign filter followed by argmin of the selection statistic."""
    if statistic not in SELECTION_STATISTICS:
        raise ValueError(f"unknown selection statistic: {statistic} (expected {sorted(SELECTION_STATISTICS)})")
    calibrated_key = SELECTION_STATISTICS[statistic]
    if not rows:
        return {
            "selected": None,
            "statistic": statistic,
            "benign_names": [],
            "median_nll_threshold": float("nan"),
            "overall_accuracy_threshold": overall_accuracy,
            "fell_back_to_all_slices": False,
            "median_statistic": float("nan"),
            "factor_below_median_slice": float("nan"),
            "loss_rule_would_select": None,
            "accuracy_rule_would_select": None,
            "calibrated_rule_would_select": None,
            "absolute_fisher_rule_would_select": None,
            "reason": "no candidate slices met the minimum size requirement",
        }
    median_nll = float(np.median([row["nll"] for row in rows]))
    benign = [row for row in rows if row["accuracy"] >= overall_accuracy and row["nll"] <= median_nll]
    pool = benign if benign else rows
    selected = min(pool, key=lambda row: row[statistic])
    median_statistic = float(np.median([row[statistic] for row in rows]))
    return {
        "selected": selected,
        "statistic": statistic,
        "benign_names": [row["name"] for row in benign],
        "median_nll_threshold": median_nll,
        "overall_accuracy_threshold": overall_accuracy,
        "fell_back_to_all_slices": not benign,
        "median_statistic": median_statistic,
        "factor_below_median_slice": float(median_statistic / max(selected[statistic], 1e-12)),
        "loss_rule_would_select": max(rows, key=lambda row: row["nll"])["name"],
        "accuracy_rule_would_select": min(rows, key=lambda row: row["accuracy"])["name"],
        "calibrated_rule_would_select": min(pool, key=lambda row: row[calibrated_key])["name"],
        "absolute_fisher_rule_would_select": min(pool, key=lambda row: row["projected_fisher"])["name"],
    }


# --------------------------------------------------------------------------- #
# Per-run analysis
# --------------------------------------------------------------------------- #
def analyse_run(run: dict, args, meta) -> dict:
    out = Path(args.out_dir) / "runs" / run["tag"]
    scores = np.load(out / "scores.npz")
    train_logits = scores["train_logits"].astype(np.float64)
    val_logits = scores["val_logits"].astype(np.float64)
    test_logits = scores["test_logits"].astype(np.float64)
    val_y, test_y = scores["val_y"], scores["test_y"]
    val_group, test_group = scores["val_group"], scores["test_group"]
    val_rows, test_rows = scores["val_rows"], scores["test_rows"]
    val_place = meta.place[val_rows]

    temperature = fit_temperature(val_logits, val_y)
    probe = fit_background_probe(
        scores["val_features"].astype(np.float64),
        val_place,
        val_group,
        scores["test_features"].astype(np.float64),
        seed=run["seed"],
        n_splits=args.probe_folds,
    )

    mean = float(train_logits.mean())
    std = float(train_logits.std()) or 1.0
    z_val = (val_logits - mean) / std
    z_test = (test_logits - mean) / std

    val_preds = (val_logits >= 0.0).astype(int)
    overall_accuracy = float((val_preds == val_y).mean())

    families = {}
    edges_all = {}
    for family in ("groups", "bins"):
        rows, edges = build_slice_table(
            family, z_val, probe["b_val"], val_logits, val_y, val_group, temperature, args
        )
        families[family] = {"slices": rows, "rule": apply_rule(rows, overall_accuracy, args.selection_statistic)}
        if family == "bins":
            edges_all = {int(gid): edge.tolist() for gid, edge in edges.items()}

    inverse = np.array([row["inverse_projected_fisher"] for row in families["groups"]["slices"]])
    certificate = float(1.0 / cvar(inverse, args.alpha))

    np.savez_compressed(
        out / "diagnostics.npz",
        z_val=z_val,
        z_test=z_test,
        b_val=probe["b_val"],
        b_test=probe["b_test"],
        val_rows=val_rows,
        test_rows=test_rows,
        val_y=val_y,
        test_y=test_y,
        val_group=val_group,
        test_group=test_group,
        probe_coef=probe["coef"],
        probe_intercept=probe["intercept"],
        feature_mean=probe["feature_mean"],
        feature_std=probe["feature_std"],
        temperature=np.array([temperature]),
        logit_mean=np.array([mean]),
        logit_std=np.array([std]),
    )

    record = {
        "tag": run["tag"],
        "arch": run["arch"],
        "method": run["method"],
        "fg_lambda": run["fg_lambda"],
        "seed": run["seed"],
        "checkpoint_sha256": file_sha256(out / "checkpoint.pt"),
        "temperature": temperature,
        "probe_crossfit_accuracy": probe["crossfit_accuracy"],
        "probe_folds": probe["n_splits"],
        "val_overall_accuracy": overall_accuracy,
        "val_overall_nll": float(binary_nll(val_logits, val_y).mean()),
        "val_worst_group_accuracy": float(
            min(
                (val_preds[val_group == g] == val_y[val_group == g]).mean()
                for g in range(NUM_GROUPS)
                if (val_group == g).any()
            )
        ),
        "val_fisher_glass_certificate": certificate,
        "bin_edges_by_group": edges_all,
        "families": families,
    }
    write_json(out / "selection.json", record)
    selected = families["groups"]["rule"]["selected"]
    log(
        f"{run['tag']}: T={temperature:.3f} val_acc={overall_accuracy:.4f} cert={certificate:.4f} "
        f"-> selected slice '{selected['name']}' (acc={selected['accuracy']:.4f} nll={selected['nll']:.4f} "
        f"F={selected['projected_fisher']:.4f} F/raw={selected['projected_over_raw']:.4f}; "
        f"statistic={args.selection_statistic})"
    )
    return record


# --------------------------------------------------------------------------- #
# Cross-run consensus
# --------------------------------------------------------------------------- #
def consensus_group(records: list[dict], family: str = "groups") -> dict | None:
    votes: dict[int, int] = {}
    for record in records:
        selected = record["families"][family]["rule"]["selected"]
        if selected is None:
            continue
        votes[int(selected["gid"])] = votes.get(int(selected["gid"]), 0) + 1
    if not votes:
        return None
    gid = max(votes, key=lambda k: (votes[k], -k))
    return {
        "gid": int(gid),
        "name": GROUP_NAMES[int(gid)],
        "votes": int(votes[gid]),
        "n_runs": len(records),
        "unanimous": votes[gid] == len(records),
        "all_votes": {GROUP_NAMES[int(k)]: int(v) for k, v in sorted(votes.items())},
    }


def choose_fg_lambda(erm: list[dict], fg: list[dict], tolerance: float) -> dict:
    """Largest validation certificate subject to validation accuracy within ``tolerance`` of ERM.

    Selection never looks at shifted test performance.
    """
    per_seed = {}
    for seed in sorted({record["seed"] for record in fg}):
        erm_seed = [r for r in erm if r["seed"] == seed]
        if not erm_seed:
            continue
        floor = erm_seed[0]["val_overall_accuracy"] - tolerance
        candidates = [r for r in fg if r["seed"] == seed]
        eligible = [r for r in candidates if r["val_overall_accuracy"] >= floor]
        pool = eligible or candidates
        best = max(pool, key=lambda r: r["val_fisher_glass_certificate"])
        per_seed[str(seed)] = {
            "chosen_lambda": best["fg_lambda"],
            "chosen_tag": best["tag"],
            "val_accuracy": best["val_overall_accuracy"],
            "val_certificate": best["val_fisher_glass_certificate"],
            "erm_val_accuracy": erm_seed[0]["val_overall_accuracy"],
            "accuracy_floor": floor,
            "n_eligible": len(eligible),
            "fell_back": not eligible,
        }
    lambdas = [entry["chosen_lambda"] for entry in per_seed.values()]
    consensus = float(max(set(lambdas), key=lambdas.count)) if lambdas else float("nan")
    fell_back = [seed for seed, entry in per_seed.items() if entry["fell_back"]]
    if fell_back:
        log(
            f"WARNING: no Fisher-Glass lambda stayed within {tolerance:.3f} of ERM validation accuracy "
            f"for seed(s) {fell_back}. Those runs are NOT accuracy-matched, so the intervention arm "
            f"cannot be reported as 'accuracy-matched' -- try smaller lambdas."
        )
    return {
        "per_seed": per_seed,
        "consensus_lambda": consensus,
        "accuracy_tolerance": tolerance,
        "accuracy_matched": not fell_back,
        "fell_back_seeds": fell_back,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="build/prospective_shift")
    parser.add_argument("--metadata", default="data/wilds/waterbirds_v1.0/metadata.csv")
    parser.add_argument("--training-index", default=None)
    parser.add_argument(
        "--selection-statistic",
        choices=sorted(SELECTION_STATISTICS),
        default="projected_over_raw",
        help="'projected_over_raw' (default): scale-free fraction of task Fisher surviving nuisance "
        "projection. 'projected_fisher': the raw absolute value, which is confounded by the "
        "within-slice logit spread J_TT.",
    )
    parser.add_argument("--n-bins", type=int, default=4, help="quantile bins per group for the refined family")
    parser.add_argument("--min-bin", type=int, default=100, help="minimum validation examples per bin")
    parser.add_argument("--probe-folds", type=int, default=5)
    parser.add_argument("--acc-tolerance", type=float, default=0.01, help="Fisher-Glass source-accuracy budget vs ERM")
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    args = parser.parse_args()

    index_path = Path(args.training_index or Path(args.out_dir) / "training_index.json")
    index = read_json(index_path)
    meta = load_metadata(args.metadata)

    records = [analyse_run(run, args, meta) for run in index["runs"]]
    erm = [r for r in records if r["method"] == "erm"]
    fg = [r for r in records if r["method"] == "fisher_glass"]
    if not erm:
        raise SystemExit("no ERM runs found; the headline model must be ERM")

    frozen = {
        "rule": {
            "benign_set": "Acc_val(g) >= Acc_val(overall) and NLL_val(g) <= median_h NLL_val(h)",
            "selection_statistic": args.selection_statistic,
            "objective": f"argmin_g {args.selection_statistic} on validation",
            "families": ["groups", "bins"],
            "n_bins": args.n_bins,
            "min_bin": args.min_bin,
            "probe": "cross-fitted logistic background probe on the trained pooled representation, validation only",
            "alpha": args.alpha,
            "ridge": args.ridge,
            "eps": args.eps,
        },
        "selected_group": consensus_group(erm, "groups"),
        "selected_bin_family": consensus_group(erm, "bins"),
        "fg_lambda_choice": choose_fg_lambda(erm, fg, args.acc_tolerance) if fg else None,
        "runs": {record["tag"]: record for record in records},
    }
    if frozen["selected_group"] is None:
        raise SystemExit("selection rule produced no slice; check the validation split and --min-bin")
    frozen["frozen_hash"] = json_sha256(
        {
            "rule": frozen["rule"],
            "selected_group": frozen["selected_group"],
            "checkpoints": {r["tag"]: r["checkpoint_sha256"] for r in records},
            "temperatures": {r["tag"]: r["temperature"] for r in records},
        }
    )
    path = write_json(Path(args.out_dir) / "frozen_selection.json", frozen)
    log(f"wrote {path}")
    log(f"selected slice: {frozen['selected_group']['name']} {frozen['selected_group']['all_votes']}")
    if frozen["fg_lambda_choice"]:
        log(f"Fisher-Glass lambda: {frozen['fg_lambda_choice']['consensus_lambda']}")
    log(f"frozen hash: {frozen['frozen_hash']}")


if __name__ == "__main__":
    main()
