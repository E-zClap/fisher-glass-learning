"""Run and aggregate multi-seed Fisher-Glass diagnostics.

This script strengthens the diagnostic evidence by repeating the synthetic,
CivilComments-WILDS, and Waterbirds diagnostics over multiple seeds and writing
a compact summary with 95% confidence intervals. Individual per-seed outputs are
kept under build/results/multiseed_diagnostics so the reported aggregate can be
audited without overwriting the main single-seed result files.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def mean_ci(values: list[float]) -> list[float]:
    arr = np.asarray(values, dtype=float)
    mean = float(arr.mean())
    if arr.size <= 1:
        return [mean, 0.0]
    return [mean, float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size))]


def optional_mean_ci(values: list[float | None]) -> list[float] | None:
    available = [float(value) for value in values if value is not None]
    return mean_ci(available) if available else None


def optional_min(values: list[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return min(available) if available else None


def optional_max(values: list[float | None]) -> float | None:
    available = [float(value) for value in values if value is not None]
    return max(available) if available else None


def mode(values: list[str]) -> str:
    return Counter(values).most_common(1)[0][0]


def run_command(command: list[str]) -> None:
    print(" ".join(command), flush=True)
    subprocess.run(command, check=True)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def aggregate_synthetic(records: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "n": len(records),
        "accuracy_mean_ci": mean_ci([row["main"]["accuracy_mean"] for row in records]),
        "accuracy_worst_5pct_mean_ci": mean_ci([row["main"]["accuracy_worst_5pct"] for row in records]),
        "loss_mean_ci": mean_ci([row["main"]["loss_mean"] for row in records]),
        "loss_cvar_mean_ci": mean_ci([row["main"]["loss_cvar"] for row in records]),
        "raw_fisher_min_mean_ci": mean_ci([row["main"]["raw_fisher_min"] for row in records]),
        "projected_fisher_q01_mean_ci": mean_ci([row["main"]["projected_fisher_q01"] for row in records]),
        "fisher_glass_certificate_mean_ci": mean_ci([row["main"]["fisher_glass_certificate"] for row in records]),
        "tail_exponent_beta_hat_mean_ci": mean_ci([row["main"]["tail_exponent_beta_hat"] for row in records]),
        "reserve_certificate_mean_ci": mean_ci([row["reserve"]["fisher_glass_certificate"] for row in records]),
        "portfolio_beta_hat_mean_ci": {
            str(k): mean_ci(
                [
                    next(port for port in row["portfolio"] if int(port["K"]) == k)["tail_exponent_beta_hat"]
                    for row in records
                ]
            )
            for k in (1, 2, 3)
        },
    }


def aggregate_real(records: list[dict[str, Any]]) -> dict[str, Any]:
    if any(row.get("ridge_exact_audit", {}).get("schema_version") != 2 for row in records):
        raise RuntimeError(
            "legacy real-data outputs do not contain the W2 audit; rerun without --skip-run"
        )
    audits = [row["ridge_exact_audit"]["aggregate"] for row in records]
    return {
        "n": len(records),
        "overall_accuracy_mean_ci": mean_ci([row["overall_accuracy"] for row in records]),
        "worst_group_accuracy_mean_ci": mean_ci([row["worst_group_accuracy"] for row in records]),
        "group_loss_cvar_mean_ci": mean_ci([row["group_loss_cvar"] for row in records]),
        "fisher_glass_certificate_mean_ci": mean_ci([row["fisher_glass_certificate"] for row in records]),
        "exact_fisher_glass_certificate_mean_ci": optional_mean_ci(
            [audit["exact_certificate"] for audit in audits]
        ),
        "exact_fisher_glass_certificate_min": optional_min(
            [audit["exact_certificate"] for audit in audits]
        ),
        "conservative_lower_certificate_mean_ci": optional_mean_ci(
            [audit["conservative_lower_certificate"] for audit in audits]
        ),
        "conservative_lower_certificate_min": optional_min(
            [audit["conservative_lower_certificate"] for audit in audits]
        ),
        "minimum_nuisance_eigenvalue": optional_min(
            [audit["min_nuisance_eigenvalue"] for audit in audits]
        ),
        "maximum_ridge_to_exact_gap": optional_max(
            [audit["max_operator_gap"] for audit in audits]
        ),
        "audit_statuses": sorted({audit["status"] for audit in audits}),
        "audit_by_seed": [
            {
                "seed": row["seed"],
                "minimum_nuisance_eigenvalue": audit["min_nuisance_eigenvalue"],
                "maximum_ridge_to_exact_gap": audit["max_operator_gap"],
                "ridge_certificate": audit["ridge_certificate"],
                "exact_certificate": audit["exact_certificate"],
                "conservative_lower_certificate": audit["conservative_lower_certificate"],
                "status": audit["status"],
            }
            for row, audit in zip(records, audits)
        ],
        "inverse_loss_p95_mean_ci": mean_ci([row["feature_inverse_loss_p95"] for row in records]),
        "reserve_certificate_mean_ci": mean_ci([row["reserve_fisher_glass_certificate"] for row in records]),
        "weakest_slice_mode": mode([row["worst_group_by_inverse_fisher"]["group"] for row in records]),
        "worst_accuracy_slice_mode": mode([row["worst_group_by_accuracy"]["group"] for row in records]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/wilds")
    parser.add_argument("--out-dir", default="build/results/multiseed_diagnostics")
    parser.add_argument("--summary-output", default="build/results/multiseed_fisher_glass_diagnostics.json")
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 12, 13, 14, 15])
    parser.add_argument("--waterbirds-seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--civilcomments-seeds", type=int, nargs="+", default=[13, 14, 15, 16, 17])
    parser.add_argument("--synthetic-n-env", type=int, default=300000)
    parser.add_argument("--skip-run", action="store_true", help="Only aggregate existing per-seed outputs.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    synthetic_records = []
    civil_records = []
    water_records = []

    for seed in args.seeds:
        seed_dir = out_dir / f"synthetic_seed_{seed}"
        output = seed_dir / "synthetic_fisher_glass_results.json"
        if not args.skip_run:
            run_command(
                [
                    sys.executable,
                    "experiments/synthetic_fisher_glass.py",
                    "--seed",
                    str(seed),
                    "--n-env",
                    str(args.synthetic_n_env),
                    "--out-dir",
                    str(seed_dir),
                    "--fig-dir",
                    str(out_dir / "figures"),
                ]
            )
        synthetic_records.append(read_json(output))

    for seed in args.civilcomments_seeds:
        output = out_dir / f"civilcomments_seed_{seed}.json"
        if not args.skip_run:
            run_command(
                [
                    sys.executable,
                    "experiments/civilcomments_fisher_glass.py",
                    "--root-dir",
                    args.root_dir,
                    "--seed",
                    str(seed),
                    "--max-features",
                    "50000",
                    "--output",
                    str(output),
                ]
            )
        civil_records.append(read_json(output))

    for seed in args.waterbirds_seeds:
        output = out_dir / f"waterbirds_seed_{seed}.json"
        if not args.skip_run:
            run_command(
                [
                    sys.executable,
                    "experiments/waterbirds_fisher_glass.py",
                    "--root-dir",
                    args.root_dir,
                    "--seed",
                    str(seed),
                    "--batch-size",
                    "64",
                    "--output",
                    str(output),
                ]
            )
        water_records.append(read_json(output))

    summary = {
        "seeds": {
            "synthetic": args.seeds,
            "civilcomments": args.civilcomments_seeds,
            "waterbirds": args.waterbirds_seeds,
        },
        "synthetic": aggregate_synthetic(synthetic_records),
        "civilcomments": aggregate_real(civil_records),
        "waterbirds": aggregate_real(water_records),
        "individual_outputs_dir": str(out_dir),
    }

    output_path = Path(args.summary_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
