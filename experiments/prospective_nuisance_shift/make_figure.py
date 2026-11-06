"""Step 6: the two-panel main figure, plus the refined-bin monotonicity panel.

Panel A -- validation selection.  Every candidate slice is plotted as
(projected Fisher, validation accuracy).  The selected slice should sit in the
high-accuracy / weak-Fisher corner, i.e. exactly where accuracy and loss say
"safe" and the certificate says "unidentified".

Panel B -- prospective held-out outcome.  Original / same-type swap /
opposite-type swap accuracy on the frozen slice, for ERM and for the
accuracy-matched Fisher-Glass model.  The story is: original and same-type stay
high, opposite-type drops for ERM, and the drop shrinks after regularization.

The extra figure plots validation inverse projected Fisher against the held-out
background-swap flip rate across the probe-score bins inside the selected slice.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import log, read_json

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9})

ARM_COLORS = {"original": "#4C6EF5", "same": "#38A169", "opposite": "#E03131"}
ARM_LABELS = {"original": "original", "same": "same-type swap", "opposite": "opposite-type swap"}


def method_label(method: str, fg_lambda: float) -> str:
    if method == "erm":
        return "ERM"
    if method == "fisher_glass":
        return f"ERM + FG ($\\lambda$={fg_lambda:g})"
    if method == "group_dro":
        return "Group DRO"
    if method == "group_dro_fisher_glass":
        return f"GDRO + FG ($\\lambda$={fg_lambda:g})"
    return method


def panel_selection(ax, frozen: dict, erm_tag: str) -> None:
    selection = frozen["runs"][erm_tag]
    groups = selection["families"]["groups"]["slices"]
    rule = selection["families"]["groups"]["rule"]
    selected_name = rule["selected"]["name"]
    benign = set(rule["benign_names"])
    accuracy_floor = rule["overall_accuracy_threshold"]
    statistic = rule.get("statistic", "projected_fisher")

    for row in groups:
        x = row[statistic]
        y = row["accuracy"]
        in_benign = row["name"] in benign
        ax.scatter(
            x,
            y,
            s=64,
            marker="o" if in_benign else "s",
            facecolor="#495057" if in_benign else "white",
            edgecolor="#495057",
            zorder=3,
            linewidths=1.0,
        )
        ax.annotate(
            row["name"],
            (x, y),
            textcoords="offset points",
            xytext=(7, -3),
            fontsize=7.5,
            color="#212529",
        )
        if row["name"] == selected_name:
            ax.scatter(x, y, s=320, facecolor="none", edgecolor=ARM_COLORS["opposite"], linewidths=1.8, zorder=4)
            ax.annotate(
                "selected $g^\\star$",
                (x, y),
                textcoords="offset points",
                xytext=(7, 11),
                fontsize=8,
                color=ARM_COLORS["opposite"],
                fontweight="bold",
            )

    ax.axhline(accuracy_floor, color="#adb5bd", linestyle="--", linewidth=0.9, zorder=1)
    ax.annotate(
        f"overall val. accuracy = {accuracy_floor:.3f}",
        (ax.get_xlim()[0], accuracy_floor),
        textcoords="offset points",
        xytext=(4, 4),
        fontsize=7,
        color="#868e96",
    )
    label = (
        "Surviving Fisher fraction $\\lambda_{\\min}(F^{\\mathrm{rid}}_{T\\mid N,g})\\,/\\,J_{TT,g}$, validation"
        if statistic == "projected_over_raw"
        else "Projected (nuisance-Schur) Fisher $\\lambda_{\\min}(F^{\\mathrm{rid}}_{T\\mid N,g})$, validation"
    )
    ax.set_xlabel(label)
    ax.set_ylabel("Validation accuracy")
    ax.set_title("A. Validation-only selection", loc="left", fontweight="bold")
    ax.grid(alpha=0.25, linewidth=0.6)


def panel_outcome(ax, summary: list[dict]) -> None:
    # Only the primary rows: ERM and the pooled validation-chosen Fisher-Glass arm.
    # The per-lambda breakdown rows have inconsistent seed counts.
    primary = [row for row in summary if row.get("primary", True)]
    order = [row for row in primary if row["method"] == "erm"] + [row for row in primary if row["method"] != "erm"]
    labels = [row.get("label") or method_label(row["method"], row["fg_lambda"]) for row in order]
    arms = ["original", "same", "opposite"]
    keys = {"original": "acc_original", "same": "acc_same", "opposite": "acc_opposite"}
    width = 0.26
    positions = np.arange(len(order))

    for i, arm in enumerate(arms):
        means = [row[keys[arm]]["mean"] for row in order]
        errors = [row[keys[arm]]["ci95"] for row in order]
        ax.bar(
            positions + (i - 1) * width,
            means,
            width=width,
            yerr=errors,
            capsize=3,
            color=ARM_COLORS[arm],
            edgecolor="white",
            linewidth=0.6,
            label=ARM_LABELS[arm],
            zorder=3,
        )
        for x, value in zip(positions + (i - 1) * width, means):
            ax.annotate(f"{value:.3f}", (x, value), textcoords="offset points", xytext=(0, 4), ha="center", fontsize=6.8)

    slice_name = order[0]["slice"] if order else "selected slice"
    ax.set_xticks(positions)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Held-out accuracy on the frozen slice")
    ax.set_ylim(0.0, 1.06)
    ax.set_title(f"B. Prospective outcome on '{slice_name}'", loc="left", fontweight="bold")
    ax.legend(frameon=False, ncol=3, loc="lower center", fontsize=7.5)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)


def figure_bins(runs: list[dict], path_stem: Path) -> None:
    points = []
    statistic = "projected_over_raw"
    for run in runs:
        if run["method"] != "erm":
            continue
        for entry in run["bins"]:
            if entry.get("n", 0) == 0:
                continue
            statistic = entry.get("selection_statistic_name", statistic)
            points.append(
                (
                    entry.get("validation_selection_statistic", entry["validation_inverse_projected_fisher"]),
                    # Quantile variant: the fixed-threshold flip rate can have an
                    # empty denominator in a bin, which would plot as noise.
                    entry.get("confident_flip_quantile", entry["confident_flip"]),
                    entry["delta_nuisance"],
                    entry["slice"],
                )
            )
    if not points:
        return
    x = np.array([p[0] for p in points])
    flip = np.array([p[1] for p in points])
    delta = np.array([p[2] for p in points])

    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for ax, values, name in (
        (axes[0], flip, "Flip rate (confident half of correct)"),
        (axes[1], delta, "$\\Delta_{\\mathrm{nuisance}}$"),
    ):
        ax.scatter(x, values, s=34, color=ARM_COLORS["opposite"], alpha=0.85, zorder=3)
        finite = np.isfinite(x) & np.isfinite(values)
        if finite.sum() >= 2:
            slope, intercept = np.polyfit(x[finite], values[finite], 1)
            grid = np.linspace(x[finite].min(), x[finite].max(), 50)
            ax.plot(grid, slope * grid + intercept, color="#495057", linewidth=1.0, linestyle="--", zorder=2)
            rho = float(np.corrcoef(x[finite], values[finite])[0, 1])
            ax.annotate(f"Pearson r = {rho:.2f}", (0.03, 0.92), xycoords="axes fraction", fontsize=8)
        ax.set_xlabel(
            "Validation surviving Fisher fraction per probe-score bin (lower = less identified)"
            if statistic == "projected_over_raw"
            else "Validation $1/\\lambda_{\\min}(F^{\\mathrm{rid}}_{T\\mid N,g})$ per probe-score bin"
        )
        ax.set_ylabel(f"Held-out {name.lower()}")
        ax.grid(alpha=0.25, linewidth=0.6)
    fig.suptitle("Validation identifiability vs held-out background-swap failure, across probe-score bins", fontsize=9)
    fig.tight_layout()
    for suffix in (".pdf", ".png"):
        fig.savefig(path_stem.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {path_stem.with_suffix('.pdf')}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="build/prospective_shift")
    parser.add_argument("--figure-dir", default="tex/figures")
    parser.add_argument("--stem", default="waterbirds_prospective_shift")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frozen = read_json(out_dir / "frozen_selection.json")
    shift = read_json(out_dir / "shift_summary.json")

    erm_tags = sorted(tag for tag, run in frozen["runs"].items() if run["method"] == "erm")
    if not erm_tags:
        raise SystemExit("no ERM run available for panel A")

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 3.8), gridspec_kw={"width_ratios": [1.05, 1.0]})
    panel_selection(axes[0], frozen, erm_tags[0])
    panel_outcome(axes[1], shift["summary"])
    fig.tight_layout()

    figure_dir = Path(args.figure_dir)
    figure_dir.mkdir(parents=True, exist_ok=True)
    for target in (figure_dir / args.stem, out_dir / "previews" / args.stem):
        target.parent.mkdir(parents=True, exist_ok=True)
        for suffix in (".pdf", ".png"):
            fig.savefig(target.with_suffix(suffix), dpi=200, bbox_inches="tight")
    plt.close(fig)
    log(f"wrote {figure_dir / (args.stem + '.pdf')}")

    figure_bins(shift["runs"], figure_dir / f"{args.stem}_bins")


if __name__ == "__main__":
    main()
