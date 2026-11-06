"""Recompute the two supporting numerical sentences from public retained records.

From a release checkout: python reanalysis/supporting_contrasts/recompute.py
From another checkout: pass --artifact-root PATH/TO/artifacts.
Requires scipy (included in the public release requirements); performs no training.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from statistics import mean, stdev

from scipy.stats import t


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def same_number(left: float, right: float, label: str) -> None:
    require(math.isclose(left, right, rel_tol=1e-12, abs_tol=1e-14), label)


def projected(moment: dict, ridge: float) -> float:
    require(moment["j_nn"] > 0, "The retained nuisance moment must be positive")
    return max(moment["j_tt"] - moment["j_tn"] ** 2 / (moment["j_nn"] + ridge), 0.0)


def recompute(artifact_root: Path) -> dict:
    source_hashes = {}

    def read(relative: str) -> dict:
        contents = (artifact_root / relative).read_bytes()
        source_hashes[relative] = hashlib.sha256(contents).hexdigest()
        return json.loads(contents)

    hidden_rows = []
    for seed in range(17, 22):
        prefix = f"hidden_slice_corrected_v2/runs/seed_{seed}"
        discovery = read(f"{prefix}/discovery.json")
        blind = read(f"{prefix}/blind_test_results.json")
        posthoc = read(f"{prefix}/posthoc.json")
        require(all(record["seed"] == seed for record in (discovery, blind, posthoc)), "Hidden seed mismatch")
        protocol = discovery["protocol"]
        config = {"epsilon": protocol["epsilon"], "ridge": protocol["ridge"], "shrinkage": protocol["shrinkage"]}
        require(config == {"epsilon": 1e-8, "ridge": 1e-3, "shrinkage": 0.1}, "Unexpected hidden Fisher configuration")
        hidden = blind["arms"]["erm"]["fisher_selector"]
        groups = posthoc["predefined_group_reference"]["rows"]
        require(len(groups) == 4, "Expected all four predefined groups")
        for row in [hidden, *groups]:
            require(row["fisher_config"] == config, "Fisher configurations differ")
            same_number(projected(row["predictive"], config["ridge"]), row["predictive"]["projected_fisher"], "Hidden or predefined Schur moment mismatch")
        weakest = min(groups, key=lambda row: row["predictive"]["projected_fisher"])
        require(hidden["class_id"] == weakest["class_id"], "Reference task class differs")
        require(hidden["n_class"] == weakest["class_reference_n"], "Reference class population differs")
        require(all(row["shrinkage_reference"] == "full task class, matching hidden-slice certification" for row in groups), "Unexpected pooling reference")
        hidden_fisher = hidden["predictive"]["projected_fisher"]
        weakest_fisher = weakest["predictive"]["projected_fisher"]
        hidden_rows.append({
            "seed": seed,
            "hidden_projected_predictive_information": hidden_fisher,
            "minimum_predefined_projected_predictive_information": weakest_fisher,
            "minimum_predefined_group": weakest["group"],
            "difference_hidden_minus_minimum": hidden_fisher - weakest_fisher,
            "class_reference_n": hidden["n_class"],
            "model_temperature": discovery["base_model"]["temperature"],
        })
    differences = [row["difference_hidden_minus_minimum"] for row in hidden_rows]
    n = len(differences)
    standard_error = stdev(differences) / math.sqrt(n)
    critical_value = float(t.ppf(0.975, n - 1))
    half_width = critical_value * standard_error
    center = mean(differences)

    civil_rows = []
    for seed in range(13, 18):
        record = read(f"w2_audit/diagnostics/civilcomments_seed_{seed}.json")
        require(record["seed"] == seed and record["dataset"] == "CivilComments-WILDS", "CivilComments cohort mismatch")
        audit = record["ridge_exact_audit"]
        environments = audit["environments"]
        require((audit["ridge"], audit["eps"], audit["alpha"], len(environments)) == (0.001, 1e-8, 0.95, 18), "Unexpected CivilComments audit configuration")
        require(audit["environment_weighting"] == "uniform", "Expected uniform environment weights")
        # The 5% tail is contained in the worst environment (18 uniform groups).
        require(0 < 1 - audit["alpha"] <= 1 / len(environments), "CVaR no longer reduces to the maximum inverse term")
        ridge_values = [projected(row, audit["ridge"]) for row in environments]
        exact_values = [projected(row, 0.0) for row in environments]
        for row, ridge_value, exact_value in zip(environments, ridge_values, exact_values):
            same_number(ridge_value, row["ridge_information"], "CivilComments ridge moment mismatch")
            same_number(exact_value, row["exact_information"], "CivilComments exact moment mismatch")
        ridge_certificate = min(ridge_values) + audit["eps"]
        exact_certificate = min(exact_values) + audit["eps"]
        same_number(ridge_certificate, audit["aggregate"]["ridge_certificate"], "CivilComments ridge certificate mismatch")
        same_number(exact_certificate, audit["aggregate"]["exact_certificate"], "CivilComments exact certificate mismatch")
        civil_rows.append({"seed": seed, "ridge_certificate": ridge_certificate, "unregularized_certificate": exact_certificate})

    return {
        "schema_version": 1,
        "scope": "Post hoc arithmetic reanalysis of public retained records; no new training or raw example data.",
        "hidden_vs_predefined": {
            "definition": "Hidden projected predictive information minus the minimum predefined-group value computed separately within each seed.",
            "model": "ERM",
            "nuisance": protocol["primary_nuisance"],
            "fisher_config": config,
            "pooling_reference": "Full task class, matching hidden-slice certification",
            "n_seeds": n,
            "seeds_below_predefined_minimum": sum(value < 0 for value in differences),
            "mean_paired_difference": center,
            "standard_error": standard_error,
            "interval": {"method": "Two-sided Student-t over paired seeds", "confidence": 0.95, "degrees_of_freedom": n - 1, "critical_value": critical_value, "half_width": half_width, "lower": center - half_width, "upper": center + half_width},
            "per_seed": hidden_rows,
        },
        "civilcomments_certificates": {
            "definition": "Mean over the five feature-Schur ridge audit seeds; each certificate is the reciprocal uniform 0.95-CVaR of stabilized inverse information.",
            "ridge": 0.001,
            "reciprocal_epsilon": 1e-8,
            "n_seeds": len(civil_rows),
            "mean_ridge_certificate": mean(row["ridge_certificate"] for row in civil_rows),
            "mean_unregularized_certificate": mean(row["unregularized_certificate"] for row in civil_rows),
            "per_seed": civil_rows,
        },
        "source_sha256_relative_to_artifact_root": source_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    result = recompute(args.artifact_root)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output = args.output_dir / "results.json"
    output.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    hidden = result["hidden_vs_predefined"]
    civil = result["civilcomments_certificates"]
    print(f"Hidden paired difference: {hidden['mean_paired_difference']:.12f}; 95% Student-t [{hidden['interval']['lower']:.12f}, {hidden['interval']['upper']:.12f}]")
    print(f"CivilComments means: ridge {civil['mean_ridge_certificate']:.12f}; unregularized {civil['mean_unregularized_certificate']:.12f}")
    print(f"Wrote {output.name}; recorded {len(result['source_sha256_relative_to_artifact_root'])} source hashes.")


if __name__ == "__main__":
    main()
