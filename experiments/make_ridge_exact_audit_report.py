"""Collect W2 ridge/exact audits into JSON, CSV, and Markdown tables."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


FIELDS = [
    "dataset",
    "model",
    "method",
    "seed",
    "split",
    "lambda_fg",
    "schur_ridge",
    "eps",
    "alpha",
    "confidence",
    "sampling_replicates",
    "sampling_method",
    "selection",
    "minimum_nuisance_eigenvalue",
    "minimum_nuisance_eigenvalue_lcb",
    "maximum_ridge_to_exact_gap",
    "maximum_ridge_to_exact_gap_ucb",
    "ridge_certificate",
    "exact_certificate",
    "conservative_lower_certificate",
    "certificate_gap",
    "status",
    "reasons",
    "source",
]


def _audit_row(
    audit: dict[str, Any],
    *,
    dataset: str,
    model: str,
    method: str,
    seed: int | None,
    split: str,
    source: Path,
    lambda_fg: float | None = None,
    selection: str | None = None,
) -> dict[str, Any]:
    aggregate = audit["aggregate"]
    sampling = audit.get("sampling_audit", {})
    return {
        "dataset": dataset,
        "model": model,
        "method": method,
        "seed": seed,
        "split": split,
        "lambda_fg": lambda_fg,
        "schur_ridge": audit.get("ridge"),
        "eps": audit.get("eps"),
        "alpha": audit.get("alpha"),
        "confidence": sampling.get("confidence"),
        "sampling_replicates": sampling.get("replicates"),
        "sampling_method": sampling.get("method"),
        "selection": selection,
        "minimum_nuisance_eigenvalue": aggregate["min_nuisance_eigenvalue"],
        "minimum_nuisance_eigenvalue_lcb": aggregate["min_nuisance_eigenvalue_lcb"],
        "maximum_ridge_to_exact_gap": aggregate["max_operator_gap"],
        "maximum_ridge_to_exact_gap_ucb": aggregate["max_operator_gap_ucb"],
        "ridge_certificate": aggregate["ridge_certificate"],
        "exact_certificate": aggregate["exact_certificate"],
        "conservative_lower_certificate": aggregate["conservative_lower_certificate"],
        "certificate_gap": aggregate["certificate_gap"],
        "status": aggregate["status"],
        "reasons": ";".join(aggregate["reasons"]),
        "source": str(source),
    }


def collect_rows(path: Path, include_candidates: bool = False) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    if "ridge_exact_audit" in payload:
        rows.append(
            _audit_row(
                payload["ridge_exact_audit"],
                dataset=str(payload.get("dataset", "unknown")),
                model=str(payload.get("model", payload.get("classifier", "unknown"))),
                method="diagnostic",
                seed=payload.get("seed"),
                split="test",
                source=path,
            )
        )
        return rows

    if "runs" not in payload:
        return rows

    parameters = payload.get("parameters", {})
    dataset = str(payload.get("dataset", "multi-benchmark"))
    model = str(payload.get("backbone", payload.get("classifier", "unknown")))
    for run in payload["runs"]:
        if run.get("skipped"):
            continue
        selection = run.get("selection")
        if selection == "candidate" and not include_candidates:
            continue
        run_dataset = str(run.get("benchmark", dataset))
        run_model = str(run.get("backbone", model))
        for split in ("val", "test"):
            metrics = run.get(split, {})
            audit = metrics.get("ridge_exact_audit")
            if audit is None:
                continue
            rows.append(
                _audit_row(
                    audit,
                    dataset=run_dataset,
                    model=run_model,
                    method=str(run.get("method", "unknown")),
                    seed=run.get("seed"),
                    split=split,
                    source=path,
                    lambda_fg=run.get("lambda_fg", parameters.get("fg_lambda")),
                    selection=selection,
                )
            )
    return rows


def aggregate_rows(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if row["split"] == "test":
            groups[
                (
                    row["dataset"],
                    row["model"],
                    row["method"],
                    row["lambda_fg"],
                    row["schur_ridge"],
                    row["eps"],
                    row["alpha"],
                    row["confidence"],
                    row["sampling_replicates"],
                )
            ].append(row)

    summaries = []
    for group_key, items in sorted(groups.items(), key=lambda item: tuple(str(v) for v in item[0])):
        (
            dataset,
            model,
            method,
            lambda_fg,
            schur_ridge,
            eps,
            alpha,
            confidence,
            sampling_replicates,
        ) = group_key

        def values(field: str) -> list[float]:
            return [float(item[field]) for item in items if item[field] is not None]

        def mean(field: str) -> float | None:
            available = values(field)
            return sum(available) / len(available) if available else None

        def minimum(field: str) -> float | None:
            available = values(field)
            return min(available) if available else None

        def maximum(field: str) -> float | None:
            available = values(field)
            return max(available) if available else None

        summaries.append(
            {
                "dataset": dataset,
                "model": model,
                "method": method,
                "lambda_fg": lambda_fg,
                "schur_ridge": schur_ridge,
                "eps": eps,
                "alpha": alpha,
                "confidence": confidence,
                "sampling_replicates": sampling_replicates,
                "n_runs": len(items),
                "minimum_nuisance_eigenvalue": minimum("minimum_nuisance_eigenvalue"),
                "minimum_nuisance_eigenvalue_lcb": minimum(
                    "minimum_nuisance_eigenvalue_lcb"
                ),
                "maximum_ridge_to_exact_gap": maximum("maximum_ridge_to_exact_gap"),
                "maximum_ridge_to_exact_gap_ucb": maximum(
                    "maximum_ridge_to_exact_gap_ucb"
                ),
                "ridge_certificate_mean": mean("ridge_certificate"),
                "exact_certificate_mean": mean("exact_certificate"),
                "exact_certificate_min": minimum("exact_certificate"),
                "conservative_lower_certificate_mean": mean("conservative_lower_certificate"),
                "conservative_lower_certificate_min": minimum(
                    "conservative_lower_certificate"
                ),
                "statuses": sorted({str(item["status"]) for item in items}),
            }
        )
    return summaries


def _format(value: Any) -> str:
    if value is None:
        return "--"
    if isinstance(value, float):
        return f"{value:.6g}"
    return str(value)


def markdown_report(rows: list[dict[str, Any]], summaries: list[dict[str, Any]]) -> str:
    out = [
        "# Ridge-to-exact W2 numerical audit",
        "",
        "All certificates are computed per run before aggregation. The conservative value uses the "
        "reported Bonferroni simultaneous Gaussian/Wald confidence level, conditional on the fitted "
        "model and fixed evaluated environment set. Confidence is simultaneous within each run, not "
        "jointly across all seeds and methods in this report.",
        "",
        "## Per-run test audits",
        "",
        "| Dataset | Method | lambda_FG | Seed | Schur ridge | Conf. | min J_NN (LCB) | max gap (UCB) | ridge I | exact I | lower I | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in rows:
        if row["split"] != "test":
            continue
        out.append(
            "| {dataset} | {method} | {lambda_fg} | {seed} | {schur_ridge} | {confidence} | "
            "{minimum_nuisance_eigenvalue} ({minimum_nuisance_eigenvalue_lcb}) | "
            "{maximum_ridge_to_exact_gap} ({maximum_ridge_to_exact_gap_ucb}) | "
            "{ridge_certificate} | {exact_certificate} | {conservative_lower_certificate} | "
            "{status} |".format(
                **{key: _format(value) for key, value in row.items()}
            )
        )

    out += [
        "",
        "## Across-seed summaries",
        "",
        "Means are descriptive only; the minimum lower certificate is the fail-closed multi-run quantity.",
        "",
        "| Dataset | Method | lambda_FG | n | Schur ridge | min J_NN (LCB) | max gap (UCB) | ridge mean | exact mean | exact min | lower mean | lower min | Status |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for row in summaries:
        out.append(
            "| {dataset} | {method} | {lambda_fg} | {n_runs} | {schur_ridge} | "
            "{minimum_nuisance_eigenvalue} ({minimum_nuisance_eigenvalue_lcb}) | "
            "{maximum_ridge_to_exact_gap} ({maximum_ridge_to_exact_gap_ucb}) | "
            "{ridge_certificate_mean} | {exact_certificate_mean} | {exact_certificate_min} | "
            "{conservative_lower_certificate_mean} | {conservative_lower_certificate_min} | "
            "{statuses} |".format(
                **{key: _format(value) for key, value in row.items()}
            )
        )
    out.append("")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", type=Path, nargs="+", help="Diagnostic/training JSON result files")
    parser.add_argument("--output-dir", type=Path, default=Path("build/results/ridge_exact_audit"))
    parser.add_argument("--include-candidates", action="store_true")
    args = parser.parse_args()

    rows = []
    for path in args.inputs:
        rows.extend(collect_rows(path, args.include_candidates))
    if not rows:
        raise RuntimeError("No ridge_exact_audit records were found in the supplied result files")
    summaries = aggregate_rows(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.output_dir / "ridge_exact_audit.json"
    csv_path = args.output_dir / "ridge_exact_audit.csv"
    markdown_path = args.output_dir / "ridge_exact_audit.md"
    json_path.write_text(
        json.dumps({"schema_version": 2, "runs": rows, "summary": summaries}, indent=2),
        encoding="utf-8",
    )
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    markdown_path.write_text(markdown_report(rows, summaries), encoding="utf-8")
    print(f"wrote {json_path}")
    print(f"wrote {csv_path}")
    print(f"wrote {markdown_path}")


if __name__ == "__main__":
    main()
