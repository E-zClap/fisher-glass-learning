"""Step 7: assemble the markdown report and fill in the rebuttal paragraph.

Reads only the artifacts written by the earlier steps, so the numbers in the
report cannot drift from the numbers in the JSON.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import log, mean_ci, read_json

TAB = "|"


def fmt(value: float, digits: int = 4) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{value:.{digits}f}"


def fmt_ci(entry: dict, digits: int = 4) -> str:
    if not isinstance(entry, dict):
        return fmt(entry, digits)
    if entry.get("n", 0) <= 1:
        return fmt(entry["mean"], digits)
    return f"{entry['mean']:.{digits}f} ± {entry['ci95']:.{digits}f}"


def fmt_pct(value: float, digits: int = 1) -> str:
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "n/a"
    return f"{100.0 * value:.{digits}f}%"


def table(header: list[str], rows: list[list[str]]) -> str:
    lines = [TAB + TAB.join(header) + TAB, TAB + TAB.join(["---"] * len(header)) + TAB]
    lines += [TAB + TAB.join(row) + TAB for row in rows]
    return "\n".join(lines)


def average_slice_table(frozen: dict) -> str:
    """Validation slice table averaged over the ERM seeds."""
    erm = [run for run in frozen["runs"].values() if run["method"] == "erm"]
    names = [row["name"] for row in erm[0]["families"]["groups"]["slices"]]
    rows = []
    for name in names:
        entries = [
            row
            for run in erm
            for row in run["families"]["groups"]["slices"]
            if row["name"] == name
        ]
        rows.append(
            [
                name,
                str(entries[0]["n"]),
                fmt_ci(mean_ci(e["accuracy"] for e in entries)),
                fmt_ci(mean_ci(e["nll"] for e in entries)),
                fmt_ci(mean_ci(e["raw_fisher"] for e in entries)),
                fmt_ci(mean_ci(e["projected_fisher"] for e in entries)),
                fmt_ci(mean_ci(e["projected_over_raw"] for e in entries), 3),
                fmt_ci(mean_ci(e["projected_fisher_calibrated"] for e in entries)),
            ]
        )
    return table(
        [
            "Validation slice",
            "n",
            "Accuracy",
            "NLL",
            "Raw Fisher J_TT",
            "Projected Fisher",
            "Projected/raw",
            "Calibrated projected",
        ],
        rows,
    )


def headline_table(shift: dict) -> str:
    rows = []
    for row in shift["summary"]:
        label = row.get("label") or ("ERM" if row["method"] == "erm" else f"{row['method']} (λ={row['fg_lambda']:g})")
        rows.append(
            [
                f"{label} (n={row['n_seeds']})",
                fmt_ci(row["acc_original"]),
                fmt_ci(row["nll_original"]),
                fmt_ci(row["val_projected_fisher"]),
                fmt_ci(row["val_projected_over_raw"], 3),
                fmt_ci(row["val_certificate"]),
                fmt_ci(row["acc_same"]),
                fmt_ci(row["acc_opposite"]),
                fmt_ci(row["delta_nuisance"]),
            ]
        )
    return table(
        [
            "Model (slice is the frozen one throughout)",
            "Source acc.",
            "Source NLL",
            "Projected Fisher",
            "Proj./raw",
            "Val. certificate",
            "Same-type acc.",
            "Opposite-type acc.",
            "Δ_nuisance",
        ],
        rows,
    )


def flip_table(shift: dict) -> str:
    """Failure rates among predictions the source metrics call safe, with denominators."""
    rows = []
    for row in shift["summary"]:
        label = row.get("label") or ("ERM" if row["method"] == "erm" else f"{row['method']} (λ={row['fg_lambda']:g})")
        rows.append(
            [
                f"{label} (n={row['n_seeds']})",
                fmt_ci(row.get("flip_among_correct", float("nan"))),
                fmt_ci(row.get("flip_among_correct_same", float("nan"))),
                fmt_ci(row.get("n_correct", float("nan")), 0),
                fmt_ci(row.get("confident_flip_quantile", float("nan"))),
                fmt_ci(row.get("confident_flip_quantile_same", float("nan"))),
                fmt_ci(row.get("n_confident_correct_quantile", float("nan")), 0),
                fmt_ci(row["confident_flip"]),
                fmt_ci(row["confident_flip_same"]),
                fmt_ci(row.get("n_confident_correct", float("nan")), 0),
            ]
        )
    return table(
        [
            "Model",
            "Flip \\| correct",
            "control",
            "n",
            "Flip \\| confident half",
            "control",
            "n",
            "Flip \\| p≥0.9",
            "control",
            "n",
        ],
        rows,
    )


def per_group_table(shift: dict) -> str:
    erm_runs = [run for run in shift["runs"] if run["method"] == "erm"]
    if not erm_runs:
        return "_no ERM runs_"
    names = [entry["slice"] for entry in erm_runs[0]["by_group"]]
    rows = []
    for name in names:
        entries = [entry for run in erm_runs for entry in run["by_group"] if entry["slice"] == name]
        rows.append(
            [
                name,
                str(entries[0]["n"]),
                fmt_ci(mean_ci(e["acc_original"] for e in entries)),
                fmt_ci(mean_ci(e["acc_same"] for e in entries)),
                fmt_ci(mean_ci(e["acc_opposite"] for e in entries)),
                fmt_ci(mean_ci(e["delta_nuisance"] for e in entries)),
                fmt_ci(mean_ci(e.get("flip_among_correct", float("nan")) for e in entries)),
                fmt_ci(mean_ci(e.get("confident_flip_quantile", float("nan")) for e in entries)),
                fmt_ci(mean_ci(e.get("n_confident_correct_quantile", float("nan")) for e in entries), 0),
                fmt_ci(mean_ci(e["confident_flip"] for e in entries)),
                fmt_ci(mean_ci(e.get("n_confident_correct", float("nan")) for e in entries), 0),
            ]
        )
    return table(
        [
            "Held-out group",
            "n",
            "Original",
            "Same-type",
            "Opposite-type",
            "Δ_nuisance",
            "Flip \\| correct",
            "Flip \\| confident half",
            "n",
            "Flip \\| p≥0.9",
            "n",
        ],
        rows,
    )


def bin_table(shift: dict) -> str:
    erm_runs = [run for run in shift["runs"] if run["method"] == "erm"]
    if not erm_runs or not erm_runs[0]["bins"]:
        return "_no refined bins (validation slice too small for the requested bin count)_"
    names = [entry["slice"] for entry in erm_runs[0]["bins"]]
    rows = []
    for name in names:
        entries = [entry for run in erm_runs for entry in run["bins"] if entry["slice"] == name]
        rows.append(
            [
                name,
                str(entries[0]["n"]),
                fmt_ci(mean_ci(e["validation"]["accuracy"] for e in entries)),
                fmt_ci(mean_ci(e["validation"]["nll"] for e in entries)),
                fmt_ci(mean_ci(e["validation"]["projected_fisher"] for e in entries)),
                fmt_ci(mean_ci(e["validation"]["projected_over_raw"] for e in entries), 3),
                fmt_ci(mean_ci(e["validation_selection_statistic"] for e in entries), 3),
                fmt_ci(mean_ci(e["acc_original"] for e in entries)),
                fmt_ci(mean_ci(e["delta_nuisance"] for e in entries)),
                fmt_ci(mean_ci(e.get("flip_among_correct", float("nan")) for e in entries)),
                fmt_ci(mean_ci(e.get("confident_flip_quantile", float("nan")) for e in entries)),
            ]
        )
    return table(
        [
            "Probe-score bin (validation edges)",
            "n (test)",
            "Val. acc.",
            "Val. NLL",
            "Val. projected Fisher",
            "Val. proj./raw",
            "Val. selection statistic",
            "Held-out original acc.",
            "Δ_nuisance",
            "Flip \\| correct",
            "Flip \\| confident half",
        ],
        rows,
    )


def natural_table(natural: dict) -> str:
    rows = []
    for row in natural["summary"]:
        label = "ERM" if row["method"] == "erm" else f"{row['method']} (λ={row['fg_lambda']:g})"
        rows.append(
            [
                label,
                row["selected_slice"],
                fmt_ci(row["acc_selected"]),
                row["partner_slice"],
                fmt_ci(row["acc_partner"]),
                fmt_ci(row["delta_accuracy"]),
                fmt_ci(row["confident_acc_selected"]),
                fmt_ci(row["confident_acc_partner"]),
            ]
        )
    return table(
        [
            "Model",
            "Selected slice",
            "Accuracy",
            "Same-label / other background",
            "Accuracy",
            "Δ accuracy",
            "Confident acc. (selected)",
            "Confident acc. (partner)",
        ],
        rows,
    )


def rebuttal(frozen: dict, shift: dict, generation: dict) -> str:
    primary = [row for row in shift["summary"] if row.get("primary", True)]
    erm = next((row for row in primary if row["method"] == "erm"), None)
    fg = next((row for row in primary if row["method"] == "fisher_glass"), None)
    if erm is None:
        return "_ERM summary unavailable_"
    slice_name = erm["slice"]
    arch = shift["runs"][0]["arch"] if shift["runs"] else "MobileNetV3-small"
    n_seeds = erm["n_seeds"]
    text = (
        f"We agree that the previous version did not empirically close the diagnostic-to-failure loop. "
        f"We have added a prospective held-out nuisance-intervention experiment on a fully fine-tuned "
        f"{arch} ({n_seeds} seeds). Using validation data only, we select the lowest-Fisher-Glass slice "
        f"among slices with above-average accuracy and below-median loss, and then freeze the checkpoint, "
        f"the temperature, the cross-fitted nuisance probe, the slice definition and every numerical "
        f"constant (frozen hash `{frozen['frozen_hash'][:16]}`). The selected slice is `{slice_name}`, with "
        f"{fmt_pct(erm['acc_original']['mean'], 1)} held-out source accuracy, source NLL "
        f"{fmt(erm['nll_original']['mean'], 3)}, and validation projected-Fisher score "
        f"{fmt(erm['val_projected_fisher']['mean'], 3)} "
        f"({fmt(erm['val_projected_over_raw']['mean'], 3)} of its raw Fisher). "
        f"On held-out paired images, replacing the background with another background of the same type "
        f"changes accuracy to {fmt_pct(erm['acc_same']['mean'], 1)}, whereas changing only the background "
        f"type while preserving the bird foreground and label reduces it to "
        f"{fmt_pct(erm['acc_opposite']['mean'], 1)}, a drop of "
        f"{fmt_pct(erm['delta_nuisance']['mean'], 1)} points relative to the artifact-matched control. "
    )
    # Quote the fixed-threshold flip rate only when its denominator can support it;
    # otherwise fall back to the within-slice confidence quantile.
    n_absolute = erm.get("n_confident_correct", {}).get("mean", 0.0)
    if n_absolute >= 30:
        text += (
            f"Among source-correct predictions with calibrated confidence above "
            f"{shift['args']['confidence_threshold']:.2f} (n={n_absolute:.0f}), "
            f"{fmt_pct(erm['confident_flip']['mean'], 1)} become incorrect after the nuisance flip, against "
            f"{fmt_pct(erm['confident_flip_same']['mean'], 1)} for the same-type control. "
        )
    else:
        text += (
            f"A fixed calibrated-confidence threshold is not usable on this slice (only "
            f"{n_absolute:.0f} predictions clear it), so we report the most-confident "
            f"{shift['args'].get('confident_quantile', 0.5):.0%} of the correct predictions instead "
            f"(n={erm.get('n_confident_correct_quantile', {}).get('mean', float('nan')):.0f}): "
            f"{fmt_pct(erm.get('confident_flip_quantile', {}).get('mean', float('nan')), 1)} become "
            f"incorrect after the nuisance flip, against "
            f"{fmt_pct(erm.get('confident_flip_quantile_same', {}).get('mean', float('nan')), 1)} for the "
            f"same-type control. Across all originally-correct predictions the flip rate is "
            f"{fmt_pct(erm.get('flip_among_correct', {}).get('mean', float('nan')), 1)}. "
        )
    text += "A source-loss-based rule does not select this slice."
    if fg is not None:
        # Never assert a direction without checking it: an earlier version of this
        # template said "raises the certificate" while the numbers showed a fall.
        certificate_up = fg["val_certificate"]["mean"] > erm["val_certificate"]["mean"]
        drop_down = fg["delta_nuisance"]["mean"] < erm["delta_nuisance"]["mean"]
        # fg_lambda_choice is explicitly None when there are no Fisher-Glass runs,
        # so .get(key, {}) would hand back None rather than a dict.
        matched = (frozen.get("fg_lambda_choice") or {}).get("accuracy_matched", False)
        text += (
            f" A Fisher-Glass-regularized model ({fg['label']}, {fg['n_seeds']} seeds, validation accuracy "
            f"{fmt(fg['val_accuracy']['mean'], 4)} vs {fmt(erm['val_accuracy']['mean'], 4)} for ERM"
            f"{'' if matched else ', NOT within the matching tolerance for every seed'}) "
            f"{'raises' if certificate_up else 'lowers'} the validation certificate from "
            f"{fmt(erm['val_certificate']['mean'], 3)} to {fmt(fg['val_certificate']['mean'], 3)} and "
            f"{'reduces' if drop_down else 'increases'} the nuisance-shift drop from "
            f"{fmt_pct(erm['delta_nuisance']['mean'], 1)} to {fmt_pct(fg['delta_nuisance']['mean'], 1)} "
            # flip_among_correct, not the fixed threshold: the two models can have
            # very different numbers of predictions above p>=0.9 on the same slice,
            # which would make that comparison span different populations.
            f"(flip among originally-correct predictions "
            f"{fmt_pct(erm.get('flip_among_correct', {}).get('mean', float('nan')), 1)} -> "
            f"{fmt_pct(fg.get('flip_among_correct', {}).get('mean', float('nan')), 1)})."
        )
        if not (certificate_up and drop_down):
            text += (
                " NOTE: this does not support the intended 'raising the certificate reduces the failure it "
                "predicted' claim -- do not write it up as an intervention result without re-checking."
            )
    text += (
        f" The intervention is pixel-level: it inverts the official Waterbirds composition using the CUB "
        f"mattes, so the bird foreground and label are byte-identical across arms and only background "
        f"pixels change; the mean visibly-inpainted background area is "
        f"{fmt_pct(generation['visible_inpaint_fraction']['mean'], 1)} and is matched across arms "
        f"({fmt_pct(generation['visible_inpaint_by_arm']['same'], 1)} same-type vs "
        f"{fmt_pct(generation['visible_inpaint_by_arm']['opposite'], 1)} opposite-type). "
        f"This provides a trained-model, real-image example in which source accuracy and loss are benign "
        f"but the certificate anticipates a specific held-out failure."
    )
    return text


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="build/prospective_shift")
    parser.add_argument("--report", default="reports/prospective_nuisance_shift_report.md")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frozen = read_json(out_dir / "frozen_selection.json")
    shift = read_json(out_dir / "shift_summary.json")
    generation = read_json(out_dir / "paired_generation.json")
    natural_path = out_dir / "natural_shift_summary.json"
    natural = read_json(natural_path) if natural_path.exists() else None
    predictive_path = out_dir / "predictive_validity.json"
    predictive = read_json(predictive_path) if predictive_path.exists() else None

    selected = frozen["selected_group"]
    bin_group = next((run.get("bin_group") for run in shift["runs"] if run.get("bin_group")), selected["name"])
    bin_group_is_selected = all(run.get("bin_group_is_selected", True) for run in shift["runs"])
    erm_example = next(run for run in frozen["runs"].values() if run["method"] == "erm")
    rule = erm_example["families"]["groups"]["rule"]
    lambda_choice = frozen.get("fg_lambda_choice") or {}

    sections = [
        "# An easy, low-identifiability slice fails under a held-out nuisance intervention",
        "",
        "Prospective Waterbirds experiment: the slice is chosen from validation data alone, everything is "
        "frozen, and only then is a label-preserving background intervention applied to held-out images.",
        "",
        "## What was frozen before any shifted image was scored",
        "",
        f"- **Selection rule**: {frozen['rule']['benign_set']}, then {frozen['rule']['objective']}.",
        f"- **Selected slice** `{selected['name']}` "
        f"({selected['votes']}/{selected['n_runs']} seeds"
        f"{', unanimous' if selected['unanimous'] else ''}; votes: {selected['all_votes']}).",
        f"- **Nuisance coordinate**: {frozen['rule']['probe']}.",
        f"- **Constants**: alpha={frozen['rule']['alpha']}, ridge={frozen['rule']['ridge']}, eps={frozen['rule']['eps']}.",
        f"- **Selection statistic**: `{rule.get('statistic', 'projected_fisher')}`. The absolute projected "
        "Fisher is confounded by the within-slice logit spread `J_TT`, which varies several-fold across "
        "groups; the scale-free ratio is not.",
        f"- **A loss-based rule would instead have selected** `{rule['loss_rule_would_select']}`; "
        f"a worst-accuracy rule would have selected `{rule['accuracy_rule_would_select']}`; "
        f"the calibrated variant selects `{rule['calibrated_rule_would_select']}`; "
        f"the absolute-projected-Fisher variant selects `{rule.get('absolute_fisher_rule_would_select')}`.",
        f"- **Selected slice sits {fmt(rule['factor_below_median_slice'], 2)}x below the median slice's "
        f"selection statistic.**",
        f"- **Eligible (benign) slices**: {rule['benign_names']} — note how many candidates the "
        "accuracy/loss filter left, since the statistic only ranks within that set.",
        f"- **Frozen hash**: `{frozen['frozen_hash']}`.",
        "",
        "## 1. Validation-only slice table (ERM, mean ± 95% CI over seeds)",
        "",
        average_slice_table(frozen),
        "",
        "## 2. Prospective held-out outcome on the frozen slice",
        "",
        "`Δ_nuisance = Acc(same-type swap) − Acc(opposite-type swap)`; the same-type arm pays exactly the "
        "same recomposition cost, so it is the artifact floor.",
        "",
        headline_table(shift),
        "",
        "### 2b. Failure among predictions the source metrics call safe",
        "",
        "All three columns condition on the prediction being **correct on the original image**, so neither "
        "source accuracy nor source loss can explain them. They differ only in how much confidence is "
        "additionally required, and each is shown with its denominator `n`:",
        "",
        "- `Flip | correct` — every originally-correct prediction. Always well populated.",
        f"- `Flip | confident half` — the most-confident {shift['args'].get('confident_quantile', 0.5):.0%} "
        "of correct predictions *within the slice*. Keeps the 'confidently correct' framing with a "
        "guaranteed denominator.",
        f"- `Flip | p≥{shift['args']['confidence_threshold']:.2f}` — fixed calibrated-probability threshold. "
        "**Check `n` before quoting this**: a single global temperature can leave a slice with almost "
        "nothing above the threshold, in which case the number is noise.",
        "",
        flip_table(shift),
        "",
        "## 3. Every held-out group (ERM), including the ones the rule did not select",
        "",
        per_group_table(shift),
        "",
        f"## 4. Refined probe-score bins inside `{bin_group}`"
        + ("" if bin_group_is_selected else " (NOT the selected slice — secondary analysis)"),
        "",
        "Bin edges come from validation; the flip rates are held-out. Accuracy and loss barely vary across "
        "these bins, so a monotone relationship between a validation identifiability statistic and held-out "
        "failure is much stronger evidence than the four-group comparison, where the accuracy/loss filter "
        "leaves only two candidates.",
        "",
        bin_table(shift),
        "",
    ]

    if natural is not None:
        sections += [
            "## 5. Unpaired natural-image cross-check (no image generation)",
            "",
            "Same frozen slice, untouched official test split, compared against the same-label / "
            "opposite-background slice. Causally weaker (different birds) but free of any compositing.",
            "",
            natural_table(natural),
            "",
        ]

    if predictive is not None:
        rows = []
        for name, entry in predictive["summary"].items():
            rows.append(
                [
                    f"{name} (n={entry['n_runs']} runs)",
                    fmt_ci(entry["n_rows"], 0),
                    fmt_ci(entry["flip_base_rate"], 3),
                    fmt_ci(entry["heldout_loglik_per_row_baseline"]),
                    fmt_ci(entry["heldout_loglik_per_row_with_fisher"]),
                    fmt_ci(entry["heldout_loglik_gain_per_row"]),
                    fmt_ci(entry["auprc_baseline"]),
                    fmt_ci(entry["auprc_with_fisher"]),
                    fmt_ci(entry.get("delta_auprc", float("nan")))
                    + f" ({entry.get('n_runs_auprc_improved', 0)}/{entry['n_runs']})",
                    fmt_ci(entry["auroc_baseline"]),
                    fmt_ci(entry["auroc_with_fisher"]),
                    fmt_ci(entry.get("delta_auroc", float("nan")))
                    + f" ({entry.get('n_runs_auroc_improved', 0)}/{entry['n_runs']})",
                ]
            )
        sections += [
            "## 6. Incremental predictive validity at the individual-image level",
            "",
            "Outcome: does an originally-**correct** held-out prediction flip under the opposite-type swap? "
            f"Baseline predictors are {', '.join(predictive['baseline_features'])}; the Fisher arm adds "
            f"{', '.join(predictive['added_features'])}, both built from the same 2x2 block as the "
            "certificate. Cross-fitted with folds grouped by CUB species. All three variants are computed "
            "unconditionally and all are reported:",
            "",
        ]
        sections += [f"- `{name}` — {note}" for name, note in predictive["variant_notes"].items()]
        sections += [
            "",
            table(
                [
                    "Variant",
                    "n rows",
                    "flip rate",
                    "loglik base",
                    "loglik +Fisher",
                    "**loglik gain**",
                    "AUPRC base",
                    "AUPRC +Fisher",
                    "**ΔAUPRC (paired)**",
                    "AUROC base",
                    "AUROC +Fisher",
                    "**ΔAUROC (paired)**",
                ],
                rows,
            ),
            "",
            "> Quote the paired columns. Held-out log-likelihood gain and ΔAUPRC/ΔAUROC are within-seed "
            "differences with the count of seeds that improved; the separate per-arm CIs are across-seed "
            "spreads and overlap even when every seed improves. Likelihood-ratio p-values are in "
            "`predictive_validity.json` but are not worth quoting: at several thousand rows per run the test "
            "detects effects far too small to matter.",
            "",
            "> Scope: `beta_g` is constant within a group, so the Fisher arm contributes the per-image "
            "nuisance coordinate weighted by that slice's loading. A slice-level scalar cannot be tested "
            "this way at all, being collinear with the group indicator by construction.",
            "",
        ]

    sections += [
        "## Intervention quality control",
        "",
        f"- Generated **{generation['n_generated']}** images from **{generation['n_source_images']}** held-out birds "
        f"({generation['args']['draws']} independent background draws per bird per arm).",
        f"- Visibly inpainted background area: mean {fmt_pct(generation['visible_inpaint_fraction']['mean'], 2)}, "
        f"p95 {fmt_pct(generation['visible_inpaint_fraction']['p95'], 2)}, "
        f"max {fmt_pct(generation['visible_inpaint_fraction']['max'], 2)}.",
        f"- Matched across arms: same-type {fmt_pct(generation['visible_inpaint_by_arm']['same'], 2)} vs "
        f"opposite-type {fmt_pct(generation['visible_inpaint_by_arm']['opposite'], 2)}.",
        "- Bird foreground, matte and label are byte-identical across arms; only background pixels differ.",
        "",
        "## Fisher-Glass penalty chosen on validation",
        "",
        (
            (
                ""
                if lambda_choice.get("accuracy_matched", True)
                else "> **Not accuracy-matched.** No lambda stayed within "
                f"{lambda_choice.get('accuracy_tolerance', 0.01):.3f} of ERM validation accuracy for "
                f"seed(s) {lambda_choice.get('fell_back_seeds')}, so the intervention arm below must "
                "not be described as accuracy-matched. Try smaller lambdas.\n\n"
            )
            + f"```\n{lambda_choice}\n```"
        )
        if lambda_choice
        else "_no Fisher-Glass runs_",
        "",
        "## Rebuttal paragraph",
        "",
        "> " + rebuttal(frozen, shift, generation).replace("\n", "\n> "),
        "",
    ]

    path = Path(args.report)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
