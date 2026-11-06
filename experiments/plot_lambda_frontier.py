"""Plot the Waterbirds Fisher-Glass lambda frontier.

This script reads the end-to-end Waterbirds lambda sweep JSON files produced by
`end_to_end_deep_suite.py` and draws the regularization frontier:

* absolute Group DRO + Fisher-Glass metrics as a function of lambda;
* paired changes relative to Group DRO in the same result file;
* certificate-vs-accuracy tradeoff plots.

The paired comparison is important because each lambda file reruns Group DRO on
the same seeds and training settings. The clean claim is therefore:

    Group DRO + Fisher-Glass(lambda) - Group DRO, paired by seed.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

METRICS = [
    "overall_accuracy",
    "worst_group_accuracy",
    "group_loss_cvar",
    "fisher_glass_certificate",
    "inverse_loss_p95",
]

DISPLAY = {
    "overall_accuracy": "Average accuracy",
    "worst_group_accuracy": "Worst-group accuracy",
    "group_loss_cvar": "Loss-CVaR",
    "fisher_glass_certificate": "Fisher-Glass certificate",
    "inverse_loss_p95": "Inverse-Fisher p95",
}


def mean_std(values: list[float]) -> tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    if arr.size <= 1:
        return float(arr.mean()), 0.0
    return float(arr.mean()), float(arr.std(ddof=1))


def paired_diff_ci(values: list[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return mean, 0.0, 0.0
    sd = float(arr.std(ddof=1))
    # t_{0.975,4} for the five-seed sweeps; close enough for this report script.
    tcrit = 2.776 if arr.size == 5 else 1.96
    half_width = tcrit * sd / float(np.sqrt(arr.size))
    return mean, sd, float(half_width)


def load_runs(path: Path) -> tuple[float, list[dict[str, Any]]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    fg_lambda = obj.get("parameters", {}).get("fg_lambda")
    if fg_lambda is None:
        raise ValueError(f"{path} does not contain parameters.fg_lambda")
    runs = [run for run in obj.get("runs", []) if not run.get("skipped")]
    return float(fg_lambda), runs


def summarize_file(path: Path) -> dict[str, Any]:
    fg_lambda, runs = load_runs(path)
    by_key = {(run["method"], int(run["seed"])): run for run in runs}
    seeds = sorted(
        {seed for method, seed in by_key if method == "group_dro"}
        & {seed for method, seed in by_key if method == "group_dro_fisher_glass"}
    )
    if not seeds:
        raise ValueError(f"{path} has no paired group_dro/group_dro_fisher_glass seeds")

    row: dict[str, Any] = {
        "lambda": fg_lambda,
        "path": str(path),
        "n": len(seeds),
        "seeds": seeds,
    }
    for method in ["group_dro", "group_dro_fisher_glass"]:
        for metric in METRICS:
            values = [float(by_key[(method, seed)]["test"][metric]) for seed in seeds]
            mean, sd = mean_std(values)
            row[f"{method}_{metric}_mean"] = mean
            row[f"{method}_{metric}_std"] = sd
            row[f"{method}_{metric}_values"] = values

    for metric in METRICS:
        diffs = [
            float(by_key[("group_dro_fisher_glass", seed)]["test"][metric])
            - float(by_key[("group_dro", seed)]["test"][metric])
            for seed in seeds
        ]
        mean, sd, ci = paired_diff_ci(diffs)
        row[f"delta_{metric}_mean"] = mean
        row[f"delta_{metric}_std"] = sd
        row[f"delta_{metric}_ci95"] = ci
        row[f"delta_{metric}_values"] = diffs
    return row


def discover_files(input_dir: Path, pattern: str) -> list[Path]:
    files = sorted(input_dir.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files matched {input_dir / pattern}")
    return files


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["lambda", "n"]
    for metric in METRICS:
        fields.extend(
            [
                f"group_dro_{metric}_mean",
                f"group_dro_fisher_glass_{metric}_mean",
                f"delta_{metric}_mean",
                f"delta_{metric}_ci95",
            ]
        )
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fields})


def errorbar(ax, x, y, yerr, **kwargs) -> None:
    ax.errorbar(x, y, yerr=yerr, marker="o", capsize=3, linewidth=1.8, markersize=5, **kwargs)


def plot_frontier(rows: list[dict[str, Any]], out_pdf: Path, out_png: Path) -> None:
    lambdas = np.asarray([row["lambda"] for row in rows], dtype=float)
    fg = "group_dro_fisher_glass"

    avg = np.asarray([row[f"{fg}_overall_accuracy_mean"] for row in rows])
    avg_sd = np.asarray([row[f"{fg}_overall_accuracy_std"] for row in rows])
    worst = np.asarray([row[f"{fg}_worst_group_accuracy_mean"] for row in rows])
    worst_sd = np.asarray([row[f"{fg}_worst_group_accuracy_std"] for row in rows])
    cert = np.asarray([row[f"{fg}_fisher_glass_certificate_mean"] for row in rows])
    cert_sd = np.asarray([row[f"{fg}_fisher_glass_certificate_std"] for row in rows])
    inv = np.asarray([row[f"{fg}_inverse_loss_p95_mean"] for row in rows])
    inv_sd = np.asarray([row[f"{fg}_inverse_loss_p95_std"] for row in rows])
    gdro_avg = np.asarray([row["group_dro_overall_accuracy_mean"] for row in rows])
    gdro_worst = np.asarray([row["group_dro_worst_group_accuracy_mean"] for row in rows])
    gdro_cert = np.asarray([row["group_dro_fisher_glass_certificate_mean"] for row in rows])
    gdro_inv = np.asarray([row["group_dro_inverse_loss_p95_mean"] for row in rows])

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(7.4, 4.9),
        gridspec_kw={"wspace": 0.34, "hspace": 0.55},
    )

    ax = axes[0, 0]
    errorbar(ax, lambdas, avg, avg_sd, label="GDRO + FG avg", color="#1f77b4")
    errorbar(ax, lambdas, worst, worst_sd, label="GDRO + FG worst", color="#d62728")
    ax.plot(lambdas, gdro_avg, "--", color="#1f77b4", alpha=0.45, label="GDRO avg")
    ax.plot(lambdas, gdro_worst, "--", color="#d62728", alpha=0.45, label="GDRO worst")
    ax.set_xscale("log")
    ax.set_xlabel("Fisher-Glass lambda")
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy frontier")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    errorbar(ax, lambdas, cert, cert_sd, label="GDRO + FG", color="#2ca02c")
    ax.plot(lambdas, gdro_cert, "--", color="#2ca02c", alpha=0.45, label="GDRO")
    ax.set_xscale("log")
    ax.set_xlabel("Fisher-Glass lambda")
    ax.set_ylabel("Certificate")
    ax.set_title("Information certificate")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    errorbar(ax, lambdas, inv, inv_sd, label="GDRO + FG", color="#9467bd")
    ax.plot(lambdas, gdro_inv, "--", color="#9467bd", alpha=0.45, label="GDRO")
    ax.set_xscale("log")
    ax.set_xlabel("Fisher-Glass lambda")
    ax.set_ylabel("Inverse-Fisher p95")
    ax.set_title("Tail risk")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    scatter = ax.scatter(worst, cert, c=np.log10(lambdas), s=72, cmap="viridis", edgecolor="black", linewidth=0.5)
    for lam, x, y in zip(lambdas, worst, cert):
        dx = -34 if x > 0.55 else 4
        dy = -12 if y > 0.78 else 3
        ax.annotate(f"{lam:g}", (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5)
    ax.set_xlabel("Worst-group accuracy")
    ax.set_ylabel("Certificate")
    ax.set_title("Worst-group vs identifiability")
    ax.margins(x=0.08, y=0.08)
    ax.grid(True, alpha=0.25)
    fig.subplots_adjust(left=0.08, right=0.92, bottom=0.11, top=0.93)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_paired_deltas(rows: list[dict[str, Any]], out_pdf: Path, out_png: Path) -> None:
    lambdas = np.asarray([row["lambda"] for row in rows], dtype=float)
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.6))
    specs = [
        ("delta_worst_group_accuracy_mean", "delta_worst_group_accuracy_ci95", "Worst acc change", "#d62728"),
        ("delta_overall_accuracy_mean", "delta_overall_accuracy_ci95", "Average acc change", "#1f77b4"),
        ("delta_fisher_glass_certificate_mean", "delta_fisher_glass_certificate_ci95", "Certificate change", "#2ca02c"),
        ("delta_inverse_loss_p95_mean", "delta_inverse_loss_p95_ci95", "Inv-Fisher p95 change", "#9467bd"),
    ]
    for ax, (mean_key, ci_key, title, color) in zip(axes, specs):
        y = np.asarray([row[mean_key] for row in rows], dtype=float)
        yerr = np.asarray([row[ci_key] for row in rows], dtype=float)
        errorbar(ax, lambdas, y, yerr, color=color)
        ax.axhline(0.0, color="black", linewidth=0.9, alpha=0.5)
        ax.set_xscale("log")
        ax.set_xlabel("lambda")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("GDRO + FG minus GDRO")
    fig.suptitle("Paired changes relative to Group DRO", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_pdf, bbox_inches="tight")
    fig.savefig(out_png, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="build/results")
    parser.add_argument("--pattern", default="waterbirds_fast_lambda*.json")
    parser.add_argument("--summary-json", default="build/results/waterbirds_lambda_frontier_summary.json")
    parser.add_argument("--summary-csv", default="build/results/waterbirds_lambda_frontier_summary.csv")
    parser.add_argument("--frontier-pdf", default="tex/figures/waterbirds_lambda_frontier.pdf")
    parser.add_argument("--frontier-png", default="build/results/waterbirds_lambda_frontier.png")
    parser.add_argument("--deltas-pdf", default="tex/figures/waterbirds_lambda_frontier_deltas.pdf")
    parser.add_argument("--deltas-png", default="build/results/waterbirds_lambda_frontier_deltas.png")
    args = parser.parse_args()

    files = discover_files(Path(args.input_dir), args.pattern)
    rows = [summarize_file(path) for path in files]
    rows = sorted(rows, key=lambda row: row["lambda"])

    summary_path = Path(args.summary_json)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps({"files": [str(path) for path in files], "rows": rows}, indent=2), encoding="utf-8")
    write_csv(rows, Path(args.summary_csv))
    plot_frontier(rows, Path(args.frontier_pdf), Path(args.frontier_png))
    plot_paired_deltas(rows, Path(args.deltas_pdf), Path(args.deltas_png))

    compact = [
        {
            "lambda": row["lambda"],
            "avg_acc": row["group_dro_fisher_glass_overall_accuracy_mean"],
            "worst_acc": row["group_dro_fisher_glass_worst_group_accuracy_mean"],
            "certificate": row["group_dro_fisher_glass_fisher_glass_certificate_mean"],
            "inverse_p95": row["group_dro_fisher_glass_inverse_loss_p95_mean"],
            "delta_worst": row["delta_worst_group_accuracy_mean"],
            "delta_certificate": row["delta_fisher_glass_certificate_mean"],
        }
        for row in rows
    ]
    print(json.dumps(compact, indent=2))


if __name__ == "__main__":
    main()
