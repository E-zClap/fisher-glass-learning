"""Render reviewer-ready Markdown, LaTeX, and rebuttal text from aggregate.json."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

try:  # Package import (tests and ``python -m``).
    from .common import json_sha256, require_current_implementation_manifest
except ImportError:  # Direct stage script.
    from common import json_sha256, require_current_implementation_manifest


def _read_aggregate(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"run aggregate.py first: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("stage") != "five_seed_reporting_aggregate" or payload.get("schema_version") != 1:
        raise RuntimeError(f"unsupported aggregate schema: {path}")
    expected = payload.get("aggregate_sha256")
    unhashed = dict(payload)
    unhashed.pop("aggregate_sha256", None)
    if expected != json_sha256(unhashed):
        raise RuntimeError("aggregate.json content hash mismatch")
    require_current_implementation_manifest(
        payload.get("implementation"),
        label="reporting aggregate",
    )
    return payload


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text.rstrip() + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _fmt(summary: dict[str, Any] | None, digits: int = 3, percent: bool = False) -> str:
    if summary is None:
        return "—"
    scale = 100.0 if percent else 1.0
    suffix = "%" if percent else ""
    return (
        f"{scale * summary['mean']:.{digits}f}{suffix} "
        f"[{scale * summary['lo']:.{digits}f}, {scale * summary['hi']:.{digits}f}]"
    )


def _pm(summary: dict[str, Any] | None, digits: int = 3, percent: bool = False) -> str:
    if summary is None:
        return "--"
    scale = 100.0 if percent else 1.0
    suffix = r"\%" if percent else ""
    return f"{scale * summary['mean']:.{digits}f} $\\pm$ {scale * summary['half_width']:.{digits}f}{suffix}"


def _markdown_table(header: list[str], rows: list[list[str]]) -> str:
    clean = lambda value: str(value).replace("|", "\\|").replace("\n", " ")
    output = ["| " + " | ".join(map(clean, header)) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    output.extend("| " + " | ".join(map(clean, row)) + " |" for row in rows)
    return "\n".join(output)


def _row_map(aggregate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    rows = {row["key"]: row for row in aggregate["main_table"]}
    required = {
        "erm_hidden_fisher",
        "erm_high_loss",
        "erm_random_slice",
        "erm_uniform_gate",
        "erm_random_nuisance",
        "erm_continuation",
        "loss_repair",
        "fisher_repair",
        "combined_repair",
        "oracle_group_dro",
        "oracle_nuisance_probe",
    }
    missing = required.difference(rows)
    if missing:
        raise RuntimeError(f"aggregate is missing main rows: {sorted(missing)}")
    return rows


def _paired(
    aggregate: dict[str, Any], left: str, right: str, metric: str
) -> dict[str, Any]:
    matches = [
        row
        for row in aggregate["paired_comparisons"]
        if row["left"] == left and row["right"] == right and row["metric"] == metric
    ]
    if len(matches) != 1:
        raise RuntimeError(f"expected one paired comparison for {left}, {right}, {metric}")
    return matches[0]


def _difference_claim(summary: dict[str, Any], negative_good: bool, subject: str) -> str:
    if int(summary.get("n", 0)) < 2:
        return f"There are too few seeds to assess {subject}."
    if negative_good and summary["hi"] < 0:
        return f"The paired interval is wholly below zero, supporting {subject}."
    if negative_good and summary["lo"] > 0:
        return f"The paired interval is wholly above zero, contradicting {subject}."
    if not negative_good and summary["lo"] > 0:
        return f"The paired interval is wholly above zero, supporting {subject}."
    if not negative_good and summary["hi"] < 0:
        return f"The paired interval is wholly below zero, contradicting {subject}."
    return f"The paired interval includes zero, so these runs do not establish {subject}."


def _difference_status(summary: dict[str, Any], *, negative_good: bool) -> str:
    """Machine-readable direction used to keep prose fail-closed."""
    if int(summary.get("n", 0)) < 2:
        return "insufficient"
    if negative_good:
        if summary["hi"] < 0:
            return "favorable"
        if summary["lo"] > 0:
            return "unfavorable"
    else:
        if summary["lo"] > 0:
            return "favorable"
        if summary["hi"] < 0:
            return "unfavorable"
    return "inconclusive"


def _discovery_control_verdict(summaries: list[dict[str, Any]]) -> str:
    statuses = [_difference_status(summary, negative_good=True) for summary in summaries]
    if all(status == "favorable" for status in statuses):
        return (
            "Together, the matched loss, random-slice, uniform-gate, random-nuisance, centered, and "
            "predefined-group checks support the claimed hidden-slice discovery result."
        )
    if "unfavorable" in statuses:
        return (
            "At least one decisive control is significantly unfavorable; these runs execute the discovery protocol "
            "but do not establish the claimed unknown-subpopulation advantage."
        )
    return (
        "At least one decisive control is unresolved; these runs execute the discovery protocol but do not yet "
        "establish the full unknown-subpopulation claim."
    )


def _repair_safety_claim(
    certificate_summary: dict[str, Any],
    fisher_worst_group_summary: dict[str, Any],
    combined_worst_group_summary: dict[str, Any],
    fisher_selection: dict[str, Any] | None = None,
    combined_selection: dict[str, Any] | None = None,
) -> str:
    selections = [value for value in (fisher_selection, combined_selection) if value]
    certificate_status = _difference_status(certificate_summary, negative_good=False)
    fisher_worst_status = _difference_status(fisher_worst_group_summary, negative_good=False)
    combined_worst_status = _difference_status(combined_worst_group_summary, negative_good=False)
    if fisher_worst_status == "unfavorable" or combined_worst_status == "unfavorable":
        no_op_note = (
            " At least one arm also contains no-op seeds."
            if any(bool(value.get("any_no_op")) for value in selections)
            else ""
        )
        return (
            "The oracle worst-group interval is significantly unfavorable for at least one Fisher-based repair, so "
            "the certificate gain must be reported as a tradeoff rather than a safe subpopulation repair."
            + no_op_note
        )
    if any(bool(value.get("all_no_op")) for value in selections):
        return (
            "At least one Fisher-based arm selected the exact no-op in every seed, so these runs do not "
            "demonstrate a repair for that arm."
        )
    if any(bool(value.get("any_no_op")) for value in selections):
        return (
            "At least one Fisher-based arm selected the exact no-op in some seeds; the mixed active/no-op "
            "result cannot establish a consistent safe subpopulation repair."
        )
    if certificate_status == "favorable" and fisher_worst_status != "unfavorable":
        return (
            "The fresh certificate improves beyond continuation without a statistically established oracle "
            "worst-group decrease, although the latter remains a post-hoc diagnostic."
        )
    return "The repair evidence is inconclusive and does not establish a safe hidden-slice improvement."


def _repair_selection_suffix(row: dict[str, Any]) -> str:
    selection = row.get("repair_selection")
    if not selection or not selection.get("any_no_op"):
        return ""
    count = int(selection["no_op_seeds"])
    total = int(selection["total_seeds"])
    if selection.get("all_no_op"):
        return f" [NO-OP selected in all {total} seeds]"
    return f" [NO-OP selected in {count}/{total} seeds]"


def _qualified_repair_claim(
    summary: dict[str, Any],
    row: dict[str, Any],
    subject: str,
    *,
    negative_good: bool = False,
) -> str:
    selection = row.get("repair_selection")
    if selection and selection.get("all_no_op"):
        return (
            f"The {row['method']} arm selected the exact no-op in every seed, so it does not establish "
            f"{subject}."
        )
    claim = _difference_claim(summary, negative_good, subject)
    if selection and selection.get("any_no_op"):
        return (
            claim
            + f" The arm selected no-op in {selection['no_op_seeds']}/{selection['total_seeds']} seeds, "
            "so this is mixed active/no-op evidence."
        )
    return claim


def _repair_activity_statement(rows: dict[str, dict[str, Any]]) -> str:
    details = []
    for key in ("loss_repair", "fisher_repair", "combined_repair"):
        selection = rows[key].get("repair_selection")
        if selection:
            details.append(
                f"{rows[key]['method']}: active repair in {selection['active_repair_seeds']}/"
                f"{selection['total_seeds']} seeds"
            )
    return "Repair activity was " + "; ".join(details) + "."


def _main_markdown(aggregate: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    order = (
        "erm_hidden_fisher",
        "erm_high_loss",
        "erm_random_slice",
        "erm_uniform_gate",
        "erm_random_nuisance",
        "erm_continuation",
        "loss_repair",
        "fisher_repair",
        "combined_repair",
        "oracle_group_dro",
        "oracle_nuisance_probe",
    )
    body = []
    for key in order:
        row = rows[key]
        label_use = "No" if not row["group_labels_used"] else f"Yes—{row['group_label_role']}"
        composition = row.get("composition")
        interpretation = (
            f"{composition['dominant_metadata_group']} ({100 * composition['dominant_group_purity']['mean']:.1f}%)"
            if composition
            else "—"
        )
        budget = row.get("accuracy_budget")
        budget_note = ""
        if budget is not None and not budget["all_satisfied"]:
            budget_note = f" ⚠ budget met in {budget['satisfied_seeds']}/{budget['total_seeds']} seeds"
        body.append(
            [
                row["method"] + _repair_selection_suffix(row) + budget_note,
                label_use,
                f"{row['selector']} / {row['nuisance']}",
                _fmt(row["metrics"]["overall_accuracy"], 1, True),
                _fmt(row["metrics"]["oracle_worst_group_accuracy"], 1, True),
                _fmt(row["metrics"]["slice_loss"]),
                _fmt(row["metrics"]["predictive_fisher"]),
                _fmt(row["metrics"]["inverse_fisher"]),
                interpretation,
            ]
        )
    return _markdown_table(
        [
            "Method",
            "Group labels used",
            "Selector / nuisance",
            "Avg. acc.",
            "Oracle worst acc.",
            "Selected-slice loss",
            "Projected Fisher",
            "Inverse Fisher",
            "Post-hoc dominant group",
        ],
        body,
    )


def _paired_markdown(aggregate: dict[str, Any]) -> str:
    body = []
    for row in aggregate["paired_comparisons"]:
        summary = row["paired_difference"]
        body.append(
            [
                row["comparison"],
                row["metric"],
                f"{summary['mean']:.4f}",
                f"[{summary['lo']:.4f}, {summary['hi']:.4f}]",
                row["favorable_direction"],
            ]
        )
    return _markdown_table(
        ["Paired comparison", "Metric", "Mean Δ", "95% CI", "Favorable direction"], body
    )


def _sensitivity_markdown(aggregate: dict[str, Any]) -> str:
    body = []
    for row in aggregate["sensitivity"]:
        body.append(
            [
                row["sensitivity"],
                f"{row['value']:.2f}",
                row["selector"],
                _fmt(row["metrics"]["predictive_fisher"]),
                _fmt(row["metrics"]["unweighted_fisher"]),
                _fmt(row["metrics"]["centered_unweighted_fisher"]),
                _fmt(row["metrics"]["slice_loss"]),
                _fmt(row["metrics"]["slice_accuracy"], 1, True),
                _fmt(row["metrics"]["top_overlap"]),
            ]
        )
    return _markdown_table(
        [
            "Sweep",
            "Value",
            "Selector",
            "Predictive Fisher",
            "Unweighted Fisher",
            "Centered Fisher",
            "Loss",
            "Accuracy",
            "Top overlap",
        ],
        body,
    )


def _frozen_repair_markdown(aggregate: dict[str, Any]) -> str:
    body = []
    for row in aggregate["frozen_slice_repairs"]:
        body.append(
            [
                row["method"] + _repair_selection_suffix(row),
                _fmt(row["metrics"]["slice_loss"]),
                _fmt(row["metrics"]["slice_accuracy"], 1, True),
                _fmt(row["metrics"]["predictive_fisher"]),
                _fmt(row["metrics"]["inverse_fisher"]),
            ]
        )
    return _markdown_table(
        ["Repair", "Original-slice loss", "Original-slice accuracy", "Original-slice Fisher", "Inv.-Fisher"],
        body,
    )


def _rebuttal(aggregate: dict[str, Any], rows: dict[str, dict[str, Any]]) -> str:
    primary = rows["erm_hidden_fisher"]
    loss = rows["erm_high_loss"]
    random_slice = rows["erm_random_slice"]
    uniform_gate = rows["erm_uniform_gate"]
    random_nuisance = rows["erm_random_nuisance"]
    oracle_probe = rows["oracle_nuisance_probe"]
    fisher_difference = _paired(aggregate, "erm_hidden_fisher", "erm_high_loss", "predictive_fisher")["paired_difference"]
    centered_difference = _paired(
        aggregate, "erm_hidden_fisher", "erm_high_loss", "centered_unweighted_fisher"
    )["paired_difference"]
    random_difference = _paired(
        aggregate, "erm_hidden_fisher", "erm_random_slice", "predictive_fisher"
    )["paired_difference"]
    uniform_difference = _paired(
        aggregate, "erm_hidden_fisher", "erm_uniform_gate", "predictive_fisher"
    )["paired_difference"]
    random_nuisance_difference = _paired(
        aggregate, "erm_hidden_fisher", "erm_random_nuisance", "predictive_fisher"
    )["paired_difference"]
    fisher_repair_difference = _paired(
        aggregate, "fisher_repair", "erm_hidden_fisher", "predictive_fisher"
    )["paired_difference"]
    combined_difference = _paired(
        aggregate, "combined_repair", "erm_hidden_fisher", "predictive_fisher"
    )["paired_difference"]
    fisher_worst_group = _paired(
        aggregate, "fisher_repair", "erm_hidden_fisher", "oracle_worst_group_accuracy"
    )["paired_difference"]
    combined_worst_group = _paired(
        aggregate, "combined_repair", "erm_hidden_fisher", "oracle_worst_group_accuracy"
    )["paired_difference"]
    loss_fixed_loss = _paired(
        aggregate,
        "loss_repair_on_original_slice",
        "erm_hidden_fisher",
        "slice_loss",
    )["paired_difference"]
    loss_fixed_fisher = _paired(
        aggregate,
        "loss_repair_on_original_slice",
        "erm_hidden_fisher",
        "predictive_fisher",
    )["paired_difference"]
    fisher_fixed_fisher = _paired(
        aggregate,
        "fisher_repair_on_original_slice",
        "erm_hidden_fisher",
        "predictive_fisher",
    )["paired_difference"]
    fisher_accuracy = _paired(
        aggregate, "fisher_repair", "erm_hidden_fisher", "overall_accuracy"
    )["paired_difference"]
    continuation_difference = _paired(
        aggregate,
        "erm_continuation",
        "erm_hidden_fisher",
        "predictive_fisher",
    )["paired_difference"]
    fisher_beyond_continuation = _paired(
        aggregate,
        "fisher_repair",
        "erm_continuation",
        "predictive_fisher",
    )["paired_difference"]
    loss_beyond_continuation = _paired(
        aggregate,
        "loss_repair_on_original_slice",
        "erm_continuation_on_original_slice",
        "slice_loss",
    )["paired_difference"]
    composition = primary["composition"]
    primary_f = primary["metrics"]["predictive_fisher"]
    primary_unweighted = primary["metrics"]["unweighted_fisher"]
    primary_centered = primary["metrics"]["centered_unweighted_fisher"]
    primary_loss = primary["metrics"]["slice_loss"]
    primary_accuracy = primary["metrics"]["slice_accuracy"]
    primary_confidence = primary["metrics"]["slice_confidence"]
    primary_entropy = primary["metrics"]["slice_entropy"]
    primary_margin = primary["metrics"]["slice_abs_margin"]
    loss_f = loss["metrics"]["predictive_fisher"]
    repair_budgets = {key: rows[key]["accuracy_budget"] for key in ("loss_repair", "fisher_repair", "combined_repair")}
    if all(value and value["all_satisfied"] for value in repair_budgets.values()):
        budget_phrase = (
            "The fixed overall, balanced, and per-class task-label safety envelope was satisfied for all "
            "three repair families in every seed."
        )
    else:
        details = ", ".join(
            f"{key}: {value['satisfied_seeds']}/{value['total_seeds']}"
            for key, value in repair_budgets.items()
            if value is not None
        )
        budget_phrase = (
            f"Task-label safety-envelope satisfaction was {details}; repair claims must be qualified accordingly."
        )

    conclusion_discovery = _difference_claim(
        fisher_difference, True, "a lower-information witness than matched high-loss discovery"
    )
    conclusion_random = _difference_claim(
        random_difference, True, "a lower-information witness than random equal-mass selection"
    )
    conclusion_uniform = _difference_claim(
        uniform_difference, True, "an improvement over the constant/uniform feasible gate"
    )
    conclusion_random_nuisance = _difference_claim(
        random_nuisance_difference,
        True,
        "a lower-information witness under the audited nuisance than under a random nuisance axis",
    )
    conclusion_centered = _difference_claim(
        centered_difference, True, "the centered, unweighted robustness check"
    )
    conclusion_repair = _qualified_repair_claim(
        fisher_beyond_continuation,
        rows["fisher_repair"],
        "a re-discovered hidden-slice improvement beyond matched continued optimization",
    )
    conclusion_loss_repair = _qualified_repair_claim(
        loss_beyond_continuation,
        rows["loss_repair"],
        "an original-slice loss improvement beyond matched continued optimization",
        negative_good=True,
    )
    predefined_difference = aggregate["hidden_vs_predefined"]["paired_difference"]
    if predefined_difference["n"] < 2:
        predefined_claim = "There are too few seeds for a predefined-group comparison claim."
    elif predefined_difference["hi"] < 0:
        predefined_claim = (
            "On average, the hidden slice is lower-Fisher than the per-seed lowest-Fisher predefined group."
        )
    elif predefined_difference["lo"] > 0:
        predefined_claim = "The hidden slice is not lower-Fisher than the weakest predefined group in these runs."
    else:
        predefined_claim = "The paired interval does not resolve whether the hidden slice is lower-Fisher than every predefined group."
    discovery_verdict = _discovery_control_verdict(
        [
            fisher_difference,
            random_difference,
            uniform_difference,
            random_nuisance_difference,
            centered_difference,
            predefined_difference,
        ]
    )
    repair_safety = _repair_safety_claim(
        fisher_beyond_continuation,
        fisher_worst_group,
        combined_worst_group,
        rows["fisher_repair"].get("repair_selection"),
        rows["combined_repair"].get("repair_selection"),
    )
    repair_activity = _repair_activity_statement(rows)
    completion = (
        "five predeclared seeds"
        if aggregate["paper_protocol_complete"]
        else f"the available seeds {aggregate['seeds']} (not the complete five-seed paper protocol)"
    )
    return (
        "**W3. Thank you; we agree that the original real-data experiments evaluated Fisher-Glass only on "
        "predefined metadata groups and therefore did not execute the hidden-slice adversary. We added a "
        f"group-blind, cross-fitted Waterbirds experiment over {completion}.** Background and four-way group "
        "metadata are excluded from discovery, certification, hyperparameter selection, and the three "
        "pseudo-group repairs. The primary adversary observes images, task labels, frozen representations, and "
        "a fixed CLIP water-versus-land background direction; it fits a class-conditional linear gate on discovery "
        "folds and freezes that gate before held-out certification.\n\n"
        f"On the untouched test split, the Fisher selector has projected Fisher {_fmt(primary_f)}, versus "
        f"{_fmt(loss_f)} for the equal-mass high-loss selector. The paired Fisher-minus-loss difference is "
        f"{_fmt(fisher_difference)}. {conclusion_discovery} Its slice accuracy is {_fmt(primary_accuracy, 1, True)} "
        f"and loss is {_fmt(primary_loss)}; the unweighted feature-Schur result is {_fmt(primary_unweighted)}, "
        f"and the centered unweighted result is {_fmt(primary_centered)}. The centered Fisher-minus-loss difference is "
        f"{_fmt(centered_difference)}. {conclusion_centered} Mean confidence, entropy, and absolute calibrated margin "
        f"inside the slice are {_fmt(primary_confidence)}, {_fmt(primary_entropy)}, and {_fmt(primary_margin)}, respectively. "
        f"The held-out effective sample size is {_fmt(primary['metrics']['effective_sample_size'])}, and the transferable "
        f"sigmoid gate's absolute mass error is {_fmt(primary['metrics']['membership_mass_error'])}. "
        f"An equal-mass random slice has Fisher {_fmt(random_slice['metrics']['predictive_fisher'])}, with paired "
        f"learned-minus-random difference {_fmt(random_difference)}. {conclusion_random} A constant/uniform feasible gate "
        f"has Fisher {_fmt(uniform_gate['metrics']['predictive_fisher'])}, with paired difference "
        f"{_fmt(uniform_difference)}. {conclusion_uniform} The random nuisance-axis control has Fisher "
        f"{_fmt(random_nuisance['metrics']['predictive_fisher'])}, with audited-minus-random paired difference "
        f"{_fmt(random_nuisance_difference)}. {conclusion_random_nuisance} {discovery_verdict}\n\n"
        f"Only after these blind results were frozen did we reveal metadata. The primary slice is most enriched for "
        f"{composition['dominant_metadata_group']} (mean soft purity "
        f"{100 * composition['dominant_group_purity']['mean']:.1f}%). Against the lowest-Fisher predefined group in "
        f"each seed, the paired hidden-minus-predefined difference is {_fmt(predefined_difference)}. "
        f"The hidden value is lower in {aggregate['hidden_vs_predefined']['seeds_below_lowest_predefined']}/"
        f"{aggregate['hidden_vs_predefined']['total_seeds']} seeds. {predefined_claim} The Fisher/high-loss top-slice "
        f"Jaccard overlap is {_fmt(aggregate['selector_distinctness']['top_slice_jaccard'])}, showing directly whether "
        f"the two adversaries recover the same examples. The metadata-trained nuisance probe is "
        f"reported separately as an oracle sensitivity analysis (Fisher {_fmt(oracle_probe['metrics']['predictive_fisher'])}); "
        "it is not the primary result and does not select any primary hyperparameter.\n\n"
        f"We then froze the discovered slice, considered loss-only, Fisher-only, and combined repairs under fixed "
        f"overall, balanced, and per-class task-label accuracy budgets, and evaluated both the original gate and a "
        f"seed-matched adversary initialized from scratch for every active repair; an exact no-op was eligible and "
        f"reuses the identical frozen ERM adversary. {repair_activity} "
        f"A matched ERM continuation with the same warm start and extra optimizer steps changes the fresh certificate by "
        f"{_fmt(continuation_difference)}. Relative to that matched control, Fisher repair changes the fresh certificate "
        f"by {_fmt(fisher_beyond_continuation)}, while loss repair changes original-slice loss by "
        f"{_fmt(loss_beyond_continuation)}. {conclusion_loss_repair} "
        f"On the original slice, loss-only repair changes loss by {_fmt(loss_fixed_loss)} and Fisher by "
        f"{_fmt(loss_fixed_fisher)}; Fisher repair changes Fisher by {_fmt(fisher_fixed_fisher)}. Under fresh discovery, Fisher "
        f"repair changes the re-discovered certificate by {_fmt(fisher_repair_difference)} and average accuracy by "
        f"{_fmt(fisher_accuracy, 1, True)}; combined repair changes the certificate by {_fmt(combined_difference)}. "
        f"The oracle worst-group accuracy changes by {_fmt(fisher_worst_group, 1, True)} for Fisher repair and "
        f"{_fmt(combined_worst_group, 1, True)} for combined repair. {budget_phrase} {conclusion_repair} {repair_safety} "
        "Oracle Group DRO is shown only as a labeled-group comparator. The added experiment therefore executes the "
        "missing mechanism end to end, while the control outcomes above determine whether the present run supports—or "
        "fails to support—the stronger novelty and repair claims."
    )


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in value)


def _latex_table(rows: dict[str, dict[str, Any]], seed_count: int) -> str:
    order = (
        "erm_hidden_fisher",
        "erm_high_loss",
        "erm_random_slice",
        "erm_uniform_gate",
        "erm_random_nuisance",
        "erm_continuation",
        "loss_repair",
        "fisher_repair",
        "combined_repair",
        "oracle_group_dro",
        "oracle_nuisance_probe",
    )
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        r"\setlength{\tabcolsep}{4pt}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{lclrrrrrl}",
        r"\toprule",
        r"Method & Groups used & Selector / nuisance & Avg. acc. & Oracle worst & Slice loss & Projected Fisher & Inverse Fisher & Post-hoc interpretation$^\dagger$ \\",
        r"\midrule",
    ]
    for key in order:
        row = rows[key]
        groups = "No" if not row["group_labels_used"] else "Yes (oracle)"
        composition = row.get("composition")
        interpretation = (
            f"{composition['dominant_metadata_group']} ({100 * composition['dominant_group_purity']['mean']:.1f}%)"
            if composition
            else "--"
        )
        lines.append(
            " & ".join(
                [
                    _latex_escape(row["method"] + _repair_selection_suffix(row)),
                    _latex_escape(groups),
                    _latex_escape(f"{row['selector']} / {row['nuisance']}"),
                    _pm(row["metrics"]["overall_accuracy"], 1, True),
                    _pm(row["metrics"]["oracle_worst_group_accuracy"], 1, True),
                    _pm(row["metrics"]["slice_loss"]),
                    _pm(row["metrics"]["predictive_fisher"]),
                    _pm(row["metrics"]["inverse_fisher"]),
                    _latex_escape(interpretation),
                ]
            )
            + r" \\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"}%",
            (
                r"\caption{Metadata-blind Waterbirds hidden-slice discovery and repair. Values are means "
                rf"$\pm$ the 95\% Student-$t$ confidence-interval half-width over {seed_count} paired seeds. The CLIP text "
                r"nuisance is primary. Waterbirds groups are revealed only post hoc; Group DRO is the sole "
                r"labeled-group training comparator. Lower Fisher is favorable for discovery rows, whereas higher "
                r"re-discovered Fisher is favorable after repair. $^\dagger$Metadata interpretation is never used "
                r"by the blind algorithm.}"
            ),
            r"\label{tab:waterbirds-hidden-slice}",
            r"\end{table*}",
        ]
    )
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/results/waterbirds_hidden_slice_v2")
    parser.add_argument("--report", default="reports/waterbirds_hidden_slice_report.md")
    parser.add_argument("--latex", default=None, help="default: OUT_DIR/waterbirds_hidden_slice_table.tex")
    parser.add_argument("--rebuttal", default=None, help="default: OUT_DIR/rebuttal_w3.txt")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    out_dir = Path(args.out_dir)
    aggregate = _read_aggregate(out_dir / "aggregate.json")
    rows = _row_map(aggregate)
    paper_complete = bool(aggregate["paper_protocol_complete"])
    if paper_complete:
        rebuttal = _rebuttal(aggregate, rows)
    else:
        rebuttal = (
            "INCOMPLETE RUN — NOT FOR THE PAPER OR REBUTTAL. The predeclared five-seed protocol is not "
            f"complete; available seeds: {aggregate['seeds']}."
        )

    protocol_warning = ""
    if not paper_complete:
        protocol_warning = (
            "> **Incomplete run:** these artifacts do not contain the predeclared five seeds. "
            "Do not use the numerical text below in the paper or rebuttal.\n\n"
        )
    repair_keys = {"loss_repair", "fisher_repair", "combined_repair"}
    budget_failures = [
        key
        for key, result in aggregate["accuracy_budget"].items()
        if key in repair_keys and not result["all_satisfied"]
    ]
    budget_note = (
        "All three repair families satisfy the fixed overall, balanced, and per-class task-label safety "
        "envelope in every seed."
        if not budget_failures
        else "Task-label safety-envelope failures require qualification: " + ", ".join(budget_failures) + "."
    )
    primary = rows["erm_hidden_fisher"]
    fisher_vs_loss = _paired(aggregate, "erm_hidden_fisher", "erm_high_loss", "predictive_fisher")
    fisher_vs_random = _paired(aggregate, "erm_hidden_fisher", "erm_random_slice", "predictive_fisher")
    fisher_vs_uniform = _paired(aggregate, "erm_hidden_fisher", "erm_uniform_gate", "predictive_fisher")
    fisher_vs_random_nuisance = _paired(
        aggregate,
        "erm_hidden_fisher",
        "erm_random_nuisance",
        "predictive_fisher",
    )
    centered_vs_loss = _paired(
        aggregate, "erm_hidden_fisher", "erm_high_loss", "centered_unweighted_fisher"
    )
    fisher_repair_worst = _paired(
        aggregate, "fisher_repair", "erm_hidden_fisher", "oracle_worst_group_accuracy"
    )
    combined_repair_worst = _paired(
        aggregate, "combined_repair", "erm_hidden_fisher", "oracle_worst_group_accuracy"
    )
    definition = aggregate["primary_definition"]
    report = "\n\n".join(
        [
            "# Waterbirds metadata-blind hidden-slice results",
            protocol_warning
            + f"The primary row uses the fixed CLIP text nuisance, {100 * definition['slice_mass']:.0f}% "
            f"within-class mass, {definition['shrinkage']:.2f} shrinkage, {definition['crossfit_folds']}-fold "
            f"cross-fitting, {definition['gate_restarts']} gate restarts, and no Waterbirds group/background metadata. "
            f"Deterministic Torch algorithms were {'enabled' if definition['deterministic_torch_algorithms'] else 'disabled'}. "
            "Intervals below are over paired seeds; within-seed bootstrap intervals remain in the raw certification artifacts.",
            (
                "## 1. Main result\n\n"
                + _main_markdown(aggregate, rows)
                + "\n\nLower projected Fisher is favorable for discovery selectors; higher re-discovered "
                "Fisher is favorable when evaluating repair. The neutral column heading avoids combining those "
                "opposite directions into one arrow."
            ),
            (
                "## 2. Repairs on the original frozen slice\n\n"
                + _frozen_repair_markdown(aggregate)
                + "\n\nEvery row uses the exact gate discovered against ERM; the main table instead uses a fresh adversary. "
                + _repair_activity_statement(rows)
            ),
            (
                "## 3. Decisive paired comparisons\n\n"
                + _paired_markdown(aggregate)
                + "\n\n"
                + _difference_claim(
                    fisher_vs_loss["paired_difference"],
                    True,
                    "a lower-information witness than matched high-loss discovery",
                )
                + " "
                + _difference_claim(
                    fisher_vs_random["paired_difference"],
                    True,
                    "a lower-information witness than random equal-mass selection",
                )
                + " "
                + _difference_claim(
                    fisher_vs_uniform["paired_difference"],
                    True,
                    "an improvement over the constant/uniform feasible gate",
                )
                + " "
                + _difference_claim(
                    fisher_vs_random_nuisance["paired_difference"],
                    True,
                    "a lower-information witness under the audited nuisance than under a random nuisance axis",
                )
                + " "
                + _difference_claim(
                    centered_vs_loss["paired_difference"],
                    True,
                    "the centered, unweighted robustness check",
                )
            ),
            "## 4. Predeclared mass and shrinkage sensitivity\n\n" + _sensitivity_markdown(aggregate),
            (
                "## 5. Stability, admissibility, and interpretation\n\n"
                f"The primary gate's selection-weighted per-class cross-fold score correlation is "
                f"{_fmt(primary['gate_stability']['score_correlation'])}, mean top-slice overlap is "
                f"{_fmt(primary['gate_stability']['top_overlap'])}, and selected-class agreement is "
                f"{_fmt(primary['gate_stability']['selected_class_agreement'])}. Classes selected in fewer than two "
                f"folds have no gate pair and are excluded from these aggregate stability summaries. Its mean effective sample "
                f"size is {_fmt(primary['metrics']['effective_sample_size'])}; its held-out sigmoid mass is "
                f"{_fmt(primary['metrics']['heldout_membership_mass'])}, with absolute target error "
                f"{_fmt(primary['metrics']['membership_mass_error'])}. Fisher-versus-high-loss test top-slice Jaccard is "
                f"{_fmt(aggregate['selector_distinctness']['top_slice_jaccard'])}; its Student-t interval is clipped to "
                f"the valid [0, 1] range for display. Post hoc, its dominant metadata group is "
                f"{primary['composition']['dominant_metadata_group']} with mean soft purity "
                f"{100 * primary['composition']['dominant_group_purity']['mean']:.1f}%. The paired hidden-minus-lowest-"
                f"predefined Fisher difference is {_fmt(aggregate['hidden_vs_predefined']['paired_difference'])} and is below "
                f"zero in {aggregate['hidden_vs_predefined']['seeds_below_lowest_predefined']}/"
                f"{aggregate['hidden_vs_predefined']['total_seeds']} seeds. Across seeds, the oracle lowest-predefined-group "
                f"Fisher is {_fmt(aggregate['predefined_extrema']['lowest_fisher']['predictive_fisher'])}, and the oracle "
                f"highest-predefined-group loss is {_fmt(aggregate['predefined_extrema']['highest_loss']['loss'])}. "
                f"{_difference_claim(aggregate['hidden_vs_predefined']['paired_difference'], True, 'a slice lower-Fisher than the per-seed weakest predefined group')} "
                f"Fisher repair changes oracle worst-group accuracy by {_fmt(fisher_repair_worst['paired_difference'], 1, True)}; "
                f"combined repair changes it by {_fmt(combined_repair_worst['paired_difference'], 1, True)}. "
                f"{_difference_claim(fisher_repair_worst['paired_difference'], False, 'an oracle worst-group improvement from Fisher repair')} "
                f"{_difference_claim(combined_repair_worst['paired_difference'], False, 'an oracle worst-group improvement from combined repair')} "
                f"{budget_note} {_repair_activity_statement(rows)}"
            ),
            (
                "## 6. Scope of controls\n\n"
                "- **Primary:** CLIP text nuisance; no group labels in discovery, tuning, certification, or repair.\n"
                "- **Random and constant controls:** equal-mass random slices and a constant/uniform feasible gate test whether learned discovery improves on non-optimized selectors.\n"
                "- **Random nuisance:** a frozen random representation axis; negative control.\n"
                "- **Oracle nuisance probe:** background metadata is used only after blind freezing; sensitivity analysis, not primary.\n"
                "- **Oracle Group DRO:** group labels are used for training; labeled-group reference, not the hidden method.\n"
                "- **Predefined groups:** metadata-only post-hoc reference rows.\n"
                "- **Repair cross-fitting:** validation-fold repair diagnostics are descriptive because task-label safety "
                "and the frozen-slice objective select λ (with an exact no-op eligible); the untouched-test "
                "fresh-adversary result is confirmatory."
            ),
            (
                "## 7. Reviewer-ready response\n\n> " + rebuttal.replace("\n", "\n> ")
                if paper_complete
                else "## 7. Reviewer-ready response\n\nUnavailable until all five predeclared seeds complete."
            ),
        ]
    )

    report_path = _atomic_write_text(Path(args.report), report)
    latex_text = (
        _latex_table(rows, len(aggregate["seeds"]))
        if paper_complete
        else (
            "% INCOMPLETE RUN -- NOT FOR PAPER USE.\n"
            f"% Available seeds: {aggregate['seeds']}; required: {list(range(17, 22))}.\n"
        )
    )
    latex_path = _atomic_write_text(
        Path(args.latex) if args.latex else out_dir / "waterbirds_hidden_slice_table.tex",
        latex_text,
    )
    rebuttal_path = _atomic_write_text(
        Path(args.rebuttal) if args.rebuttal else out_dir / "rebuttal_w3.txt",
        rebuttal,
    )
    print(f"wrote {report_path}, {latex_path}, and {rebuttal_path}", flush=True)


if __name__ == "__main__":
    main()
