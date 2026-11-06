"""Step 6b: does the Fisher geometry predict individual held-out failures beyond loss and confidence?

Bin-level monotonicity can arise from bin boundaries, uneven bin sizes, or any
quantity that happens to be monotone along the binning axis.  This is the
stronger test: for each held-out image that the model got *right* on the
original, predict whether it flips under the opposite-type background swap, and
ask whether the Fisher terms add out-of-sample predictive value over the signals
a practitioner already has.

    baseline  :  NLL, calibrated confidence, margin, standardized logit, group
    + Fisher  :  baseline + beta_g * b   (task score attributable to the audited
                 nuisance direction)  and  |z - beta_g * b|  (transverse residual)

``beta_g = J_TN / (J_NN + ridge)`` is the validation Schur coefficient of the
row's group, so both added terms come straight out of the same 2x2 Fisher block
the certificate is built from.  Note what this does and does not test: because
``beta_g`` is constant within a group and ``group`` is already in the baseline,
the added information is the *per-image nuisance coordinate*, weighted by how
strongly that direction loads on the task score in its slice.  The honest
reading is therefore "does the audited nuisance projection add over conventional
confidence and loss signals", not "does the scalar certificate do so" -- a
slice-level scalar is collinear with the group indicator by construction.

Evaluation is cross-fitted with folds grouped by CUB species, so no species
appears in both the fit and the held-out half.  We report held-out mean
log-likelihood, average precision and ROC AUC, plus an in-sample likelihood-ratio
test for the two added terms.

Reads only artifacts written by ``evaluate_shift.py``; no model is re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import GroupKFold

from common import GROUP_NAMES, log, mean_ci, read_json, write_json

BASELINE = ["nll", "confidence_true", "margin", "z"]
FISHER = ["nuisance_component", "abs_transverse_residual"]


def build_design(data: dict[str, np.ndarray], groups_present: list[int]) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    """Rows: originally-correct predictions under the opposite-type swap."""
    keep = (data["arm_opposite"] == 1) & (data["original_correct"] == 1)
    keep &= np.isfinite(data["z"]) & np.isfinite(data["b"]) & np.isfinite(data["beta"])
    columns = {name: data[name][keep].astype(float) for name in BASELINE}
    beta = data["beta"][keep].astype(float)
    b = data["b"][keep].astype(float)
    z = data["z"][keep].astype(float)
    columns["nuisance_component"] = beta * b
    columns["abs_transverse_residual"] = np.abs(z - beta * b)
    group = data["group"][keep].astype(int)
    for gid in groups_present[:-1]:  # drop one level to avoid a redundant column
        columns[f"group_{gid}"] = (group == gid).astype(float)
    y = 1.0 - data["correct_after"][keep].astype(float)
    species = data["species_code"][keep].astype(int)
    return columns, y, species


def stack(columns: dict[str, np.ndarray], names: list[str]) -> np.ndarray:
    return np.column_stack([columns[name] for name in names])


def fit_predict_crossfit(x: np.ndarray, y: np.ndarray, species: np.ndarray, n_splits: int, seed: int) -> np.ndarray:
    """Out-of-fold probabilities, folds grouped so a species never spans the split."""
    predictions = np.zeros(y.size, dtype=float)
    n_splits = min(n_splits, len(np.unique(species)))
    splitter = GroupKFold(n_splits=max(2, n_splits))
    for train_idx, held_idx in splitter.split(x, y, groups=species):
        mean = x[train_idx].mean(axis=0, keepdims=True)
        std = x[train_idx].std(axis=0, keepdims=True)
        std[std < 1e-9] = 1.0
        model = LogisticRegression(max_iter=5000, random_state=seed)
        model.fit((x[train_idx] - mean) / std, y[train_idx])
        predictions[held_idx] = model.predict_proba((x[held_idx] - mean) / std)[:, 1]
    return predictions


def log_likelihood(y: np.ndarray, p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0 - 1e-12)
    return float(np.sum(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def in_sample_ll(x: np.ndarray, y: np.ndarray, seed: int) -> float:
    mean = x.mean(axis=0, keepdims=True)
    std = x.std(axis=0, keepdims=True)
    std[std < 1e-9] = 1.0
    model = LogisticRegression(max_iter=5000, random_state=seed)
    model.fit((x - mean) / std, y)
    return log_likelihood(y, model.predict_proba((x - mean) / std)[:, 1])


def evaluate_variant(columns: dict, y: np.ndarray, species: np.ndarray, use_group: bool, args) -> dict | None:
    if y.size < 50 or len(np.unique(y)) < 2 or len(np.unique(species)) < 3:
        return None
    group_names = [name for name in columns if name.startswith("group_")] if use_group else []
    x_base = stack(columns, BASELINE + group_names)
    x_full = stack(columns, BASELINE + FISHER + group_names)

    p_base = fit_predict_crossfit(x_base, y, species, args.folds, args.seed)
    p_full = fit_predict_crossfit(x_full, y, species, args.folds, args.seed)
    statistic = 2.0 * (in_sample_ll(x_full, y, args.seed) - in_sample_ll(x_base, y, args.seed))

    return {
        "n_rows": int(y.size),
        "n_species": int(len(np.unique(species))),
        "base_rate": float(y.mean()),
        "heldout_loglik_per_row_baseline": log_likelihood(y, p_base) / y.size,
        "heldout_loglik_per_row_with_fisher": log_likelihood(y, p_full) / y.size,
        "heldout_loglik_gain_per_row": (log_likelihood(y, p_full) - log_likelihood(y, p_base)) / y.size,
        "auprc_baseline": float(average_precision_score(y, p_base)),
        "auprc_with_fisher": float(average_precision_score(y, p_full)),
        "auroc_baseline": float(roc_auc_score(y, p_base)),
        "auroc_with_fisher": float(roc_auc_score(y, p_full)),
        "lr_statistic": float(statistic),
        "lr_df": len(FISHER),
        "lr_p_value": float(stats.chi2.sf(max(statistic, 0.0), df=len(FISHER))),
    }


def analyse_run(tag: str, out_dir: Path, args, selected_gid: int) -> dict | None:
    """Three variants, always all reported.

    ``pooled_with_group``  the most stringent test: the baseline already contains
        the group indicator, which is what the certificate outputs, so this asks
        whether per-image nuisance projection adds beyond knowing the slice.
    ``pooled_no_group``    the practitioner's situation: the subpopulation
        structure is not known in advance.
    ``selected_slice``     restricted to the frozen slice, where the bin-level
        relationship was observed; pooling across groups with very different flip
        rates can bury a within-slice effect.

    All three are computed unconditionally so there is no scope for reporting
    only the favourable one.
    """
    path = out_dir / "runs" / tag / "flip_records.npz"
    if not path.exists():
        log(f"{tag}: no flip_records.npz (rerun evaluate_shift.py)")
        return None
    raw = np.load(path, allow_pickle=True)
    data = {key: raw[key] for key in raw.files if key != "species_vocab"}

    groups_present = sorted(set(int(g) for g in np.unique(data["group"])))
    columns, y, species = build_design(data, groups_present)
    if y.size == 0 or len(np.unique(y)) < 2:
        log(f"{tag}: not enough variation in the flip outcome")
        return None
    group_of_row = data["group"][
        (data["arm_opposite"] == 1)
        & (data["original_correct"] == 1)
        & np.isfinite(data["z"])
        & np.isfinite(data["b"])
        & np.isfinite(data["beta"])
    ].astype(int)

    variants: dict[str, dict] = {}
    for name, use_group, mask in (
        ("pooled_with_group", True, np.ones(y.size, dtype=bool)),
        ("pooled_no_group", False, np.ones(y.size, dtype=bool)),
        ("selected_slice", False, group_of_row == selected_gid),
    ):
        subset = {key: value[mask] for key, value in columns.items()}
        outcome = evaluate_variant(subset, y[mask], species[mask], use_group, args)
        if outcome is not None:
            variants[name] = outcome
            log(
                f"{tag} [{name}]: n={outcome['n_rows']} flips={outcome['base_rate']:.3f} | "
                f"AUPRC {outcome['auprc_baseline']:.4f} -> {outcome['auprc_with_fisher']:.4f} | "
                f"loglik gain/row {outcome['heldout_loglik_gain_per_row']:+.4f} | "
                f"LR p={outcome['lr_p_value']:.3g}"
            )
    return {"tag": tag, "variants": variants} if variants else None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="build/prospective_shift")
    parser.add_argument("--tags", nargs="+", default=None, help="default: the ERM runs")
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frozen = read_json(out_dir / "frozen_selection.json")
    tags = args.tags or [tag for tag, run in frozen["runs"].items() if run["method"] == "erm"]

    selected_gid = int(frozen["selected_group"]["gid"])
    results = [r for r in (analyse_run(tag, out_dir, args, selected_gid) for tag in tags) if r is not None]
    if not results:
        raise SystemExit("no runs analysed; rerun evaluate_shift.py to write flip_records.npz")

    summary = {}
    for name in ("pooled_with_group", "pooled_no_group", "selected_slice"):
        items = [r["variants"][name] for r in results if name in r["variants"]]
        if not items:
            continue
        summary[name] = {
            "n_runs": len(items),
            "n_rows": mean_ci(i["n_rows"] for i in items),
            "flip_base_rate": mean_ci(i["base_rate"] for i in items),
            "heldout_loglik_per_row_baseline": mean_ci(i["heldout_loglik_per_row_baseline"] for i in items),
            "heldout_loglik_per_row_with_fisher": mean_ci(i["heldout_loglik_per_row_with_fisher"] for i in items),
            "heldout_loglik_gain_per_row": mean_ci(i["heldout_loglik_gain_per_row"] for i in items),
            "auprc_baseline": mean_ci(i["auprc_baseline"] for i in items),
            "auprc_with_fisher": mean_ci(i["auprc_with_fisher"] for i in items),
            "auroc_baseline": mean_ci(i["auroc_baseline"] for i in items),
            "auroc_with_fisher": mean_ci(i["auroc_with_fisher"] for i in items),
            # Paired within-seed differences. The separate per-arm CIs above are
            # across-seed spreads and will overlap even when every seed improves;
            # the paired delta is the quantity to quote.
            "delta_auprc": mean_ci(i["auprc_with_fisher"] - i["auprc_baseline"] for i in items),
            "delta_auroc": mean_ci(i["auroc_with_fisher"] - i["auroc_baseline"] for i in items),
            "n_runs_auprc_improved": int(sum(i["auprc_with_fisher"] > i["auprc_baseline"] for i in items)),
            "n_runs_auroc_improved": int(sum(i["auroc_with_fisher"] > i["auroc_baseline"] for i in items)),
            "lr_p_value_max": float(max(i["lr_p_value"] for i in items)),
            "n_runs_lr_significant_05": int(sum(i["lr_p_value"] < 0.05 for i in items)),
        }
    path = write_json(
        out_dir / "predictive_validity.json",
        {
            "args": vars(args),
            "outcome": "flip under the opposite-type background swap, among originally-correct predictions",
            "baseline_features": BASELINE,
            "added_features": FISHER,
            "cross_fitting": "GroupKFold over CUB species",
            "selected_slice": GROUP_NAMES[selected_gid],
            "variant_notes": {
                "pooled_with_group": "all groups; group indicators in the baseline (most stringent: the "
                "baseline already knows the slice, which is what the certificate outputs)",
                "pooled_no_group": "all groups; no group indicator (the practitioner's situation, where the "
                "subpopulation structure is unknown)",
                "selected_slice": "restricted to the frozen selected slice, matching the bin analysis",
            },
            "summary": summary,
            "runs": results,
        },
    )
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
