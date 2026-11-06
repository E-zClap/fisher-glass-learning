"""Regenerate Table D.7 and paired seed changes from retained results (stdlib only)."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

HERE = Path(__file__).resolve().parent
VARIANTS = ("pooled_no_group", "pooled_with_group", "selected_slice")
METRICS = (
    "n_rows", "n_species", "base_rate",
    "heldout_loglik_per_row_baseline", "heldout_loglik_per_row_with_fisher",
    "heldout_loglik_gain_per_row", "auprc_baseline", "auprc_with_fisher",
    "delta_auprc", "auroc_baseline", "auroc_with_fisher", "delta_auroc",
)


def mean_ci(values):
    values = list(values)
    return {
        "mean": statistics.mean(values),
        "nominal_normal_95_half_width": 1.96 * statistics.stdev(values) / math.sqrt(len(values)),
        "n_seeds": len(values),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=HERE / "predictive_validity.json")
    parser.add_argument("--output-dir", type=Path, default=HERE)
    args = parser.parse_args()
    payload = args.input.read_bytes()
    data = json.loads(payload)
    if data["args"]["folds"] != 5 or data["args"]["seed"] != 0:
        raise ValueError("Unexpected auxiliary fit settings")
    if sorted(r["tag"] for r in data["runs"]) != [f"erm_lam0_seed{s}" for s in range(22, 27)]:
        raise ValueError("Expected exactly ERM seeds 22--26")
    rows = []
    for variant in VARIANTS:
        for run in sorted(data["runs"], key=lambda r: r["tag"]):
            item = run["variants"][variant]
            row = {"variant": variant, "seed": int(run["tag"].rsplit("seed", 1)[1])}
            row.update({key: item[key] for key in METRICS if not key.startswith("delta_")})
            for metric in ("auprc", "auroc"):
                row[f"delta_{metric}"] = item[f"{metric}_with_fisher"] - item[f"{metric}_baseline"]
            reconstructed = item["heldout_loglik_per_row_with_fisher"] - item["heldout_loglik_per_row_baseline"]
            if not math.isclose(reconstructed, row["heldout_loglik_gain_per_row"], abs_tol=1e-14):
                raise ValueError(f"Gain mismatch: {run['tag']} / {variant}")
            rows.append(row)
    summary = {
        "source_sha256": hashlib.sha256(payload).hexdigest(),
        "aggregation": "Equal-weight mean across training seeds; paired differences within each seed",
        "uncertainty": "Nominal normal-approximation 95% half-width = 1.96 * sample SD / sqrt(5)",
        "variants": {},
    }
    table = ["analysis,mean_rows,flip_rate,loglik_gain_nats_per_donor_row,delta_auprc,n_auprc_improved,delta_auroc,n_auroc_improved"]
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        result = {metric: mean_ci(row[metric] for row in selected) for metric in METRICS}
        for metric in ("auprc", "auroc"):
            result[f"n_{metric}_improved"] = sum(row[f"delta_{metric}"] > 0 for row in selected)
        for key, expected in data["summary"][variant].items():
            mapped = "base_rate" if key == "flip_base_rate" else key
            if mapped in result and isinstance(expected, dict):
                if not math.isclose(result[mapped]["mean"], expected["mean"], abs_tol=1e-12):
                    raise ValueError(f"Archived mean mismatch: {variant}/{key}")
                if not math.isclose(result[mapped]["nominal_normal_95_half_width"], expected["ci95"], abs_tol=1e-12):
                    raise ValueError(f"Archived interval mismatch: {variant}/{key}")
        summary["variants"][variant] = result
        table.append(
            f"{variant},{result['n_rows']['mean']:.0f},{result['base_rate']['mean']:.3f},"
            f"{result['heldout_loglik_gain_per_row']['mean']:+.3f},"
            f"{result['delta_auprc']['mean']:+.3f},{result['n_auprc_improved']}/5,"
            f"{result['delta_auroc']['mean']:+.3f},{result['n_auroc_improved']}/5"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "paired_per_seed.csv").open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=("variant", "seed") + METRICS)
        writer.writeheader()
        writer.writerows(rows)
    (args.output_dir / "paired_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (args.output_dir / "table_d9.csv").write_text("\n".join(table) + "\n", encoding="utf-8")
    print("Verified all archived means and nominal normal intervals; regenerated 15 paired rows and Table D.7.")


if __name__ == "__main__":
    main()
