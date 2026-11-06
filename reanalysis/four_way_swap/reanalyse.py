#!/usr/bin/env python3
"""Replay the post hoc four-summary comparison from retained JSON records.

Python 3.10+ standard library only. No training, inference, or private data.
Default layout: <repository>/reanalysis/four_way_swap/reanalyse.py and
<repository>/artifacts/{prospective_v1,prospective_v2}/.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import statistics


GROUPS = ("landbird/land", "landbird/water", "waterbird/land", "waterbird/water")
VAL_N = (467, 466, 133, 133)
TEST_N = (2255, 2255, 642, 642)
COHORTS = (
    ("primary", "prospective_v2", tuple(range(22, 27)), "projected_over_raw"),
    ("development", "prospective_v1", tuple(range(17, 22)), "projected_fisher"),
)
SUMMARIES = ("A", "P", "H", "C")
RULES = (
    ("minimum_raw", "all_four"),
    ("minimum_projected_largest_inverse_cvar_active", "all_four"),
    ("largest_inverse_contribution", "eligible"),
)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context, atol=1e-12):
    require(math.isfinite(actual) and math.isfinite(expected), f"{context}: nonfinite value")
    require(math.isclose(actual, expected, rel_tol=1e-12, abs_tol=atol),
            f"{context}: {actual!r} != {expected!r}")


def mean_ci(values):
    values = list(values)
    require(len(values) == 5, "Seed-level aggregation requires exactly five values")
    return {"mean": statistics.mean(values),
            "ci95_half_width": 1.96 * statistics.stdev(values) / math.sqrt(len(values)),
            "n_seed_pairs": len(values)}


def ranks(values):
    """Average ranks; Fraction outcomes keep exact ties without float tolerances."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    result = [0.0] * len(values)
    first = 0
    while first < len(order):
        stop = first + 1
        while stop < len(order) and values[order[stop]] == values[order[first]]:
            stop += 1
        for i in order[first:stop]:
            result[i] = (first + 1 + stop) / 2
        first = stop
    return result


def spearman(x, y):
    require(len(x) == len(y) == 10, "Descriptive correlations require exactly ten models")
    x, y = ranks(x), ranks(y)
    xbar, ybar = statistics.mean(x), statistics.mean(y)
    numerator = math.fsum((a - xbar) * (b - ybar) for a, b in zip(x, y))
    denominator = math.sqrt(math.fsum((a - xbar) ** 2 for a in x)
                            * math.fsum((b - ybar) ** 2 for b in y))
    require(denominator > 0, "Cannot correlate a constant rank vector")
    return numerator / denominator


def upper_cvar(values, alpha):
    """Exact discrete upper-tail integral for equally weighted atoms."""
    mass = 1.0 - alpha
    require(0 < mass <= 1, "Invalid CVaR tail mass")
    remaining, integral = mass, 0.0
    for value in sorted(values, reverse=True):
        take = min(remaining, 1.0 / len(values))
        integral += take * value
        remaining -= take
        if remaining <= 0:
            break
    close(remaining, 0.0, "CVaR tail covered")
    return integral / mass


def count_fraction(accuracy, denominator, context):
    require(type(denominator) is int and denominator > 0, f"{context}: invalid denominator")
    require(math.isfinite(accuracy) and 0 <= accuracy <= 1, f"{context}: invalid accuracy")
    count = round(accuracy * denominator)
    # This is integer recovery, not an arbitrary tolerance for rank ties.
    close(accuracy * denominator, count, f"{context}: non-integral success count", atol=1e-8)
    return count, Fraction(count, denominator)


def unique_min(rows, key, context):
    minimum = min(row[key] for row in rows)
    selected = [row for row in rows if row[key] == minimum]
    require(len(selected) == 1, f"{context}: group-selection tie requires an explicit policy")
    return selected[0]["name"]


def load_record(root, relative, provenance):
    raw = (root / relative).read_bytes()
    provenance.append({"path": "artifacts/" + relative,
                       "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)})
    # Retained files contain NaN/Infinity in unused confidence/bin fields.
    # Python accepts these; every numerical field consumed below is validated.
    return json.loads(raw)


def verify_geometry(run, eps, alpha, statistic):
    tag = run["tag"]
    family = run["families"]["groups"]
    rows = sorted(family["slices"], key=lambda g: g["gid"])
    require(len(rows) == 4, f"{tag}: need all four validation groups")
    require([g["gid"] for g in rows] == list(range(4)), f"{tag}: duplicate/missing group IDs")
    require(tuple(g["name"] for g in rows) == GROUPS, f"{tag}: group names disagree")
    correct = 0
    for row, n in zip(rows, VAL_N):
        require(row["n"] == n and row["kind"] == "group", f"{tag}: validation group size/kind")
        raw, projected = row["raw_fisher"], row["projected_fisher"]
        require(math.isfinite(raw) and math.isfinite(projected) and 0 <= projected <= raw,
                f"{tag}/{row['name']}: invalid feature-Schur information")
        require(math.isfinite(row["nll"]) and row["nll"] >= 0, f"{tag}: invalid NLL")
        close(row["inverse_projected_fisher"], 1 / (projected + eps), f"{tag}: inverse")
        close(row["projected_over_raw"], projected / raw, f"{tag}: surviving fraction")
        count, _ = count_fraction(row["accuracy"], n, f"{tag}/{row['name']}: validation")
        correct += count
    close(run["val_overall_accuracy"], correct / sum(VAL_N), f"{tag}: validation total")
    losses = [1 / (g["projected_fisher"] + eps) for g in rows]
    summary = {"A": statistics.mean(g["raw_fisher"] for g in rows),
               "P": statistics.mean(g["projected_fisher"] for g in rows),
               "H": 1 / statistics.mean(losses), "C": 1 / upper_cvar(losses, alpha)}
    close(summary["C"], min(g["projected_fisher"] for g in rows) + eps, f"{tag}: C reduction")
    close(summary["C"], run["val_fisher_glass_certificate"], f"{tag}: stored certificate")
    median_nll = statistics.median(g["nll"] for g in rows)
    eligible = [g for g in rows if g["accuracy"] >= run["val_overall_accuracy"] and g["nll"] <= median_nll]
    rule = family["rule"]
    require(rule["statistic"] == statistic, f"{tag}: headline statistic changed")
    require(set(rule["benign_names"]) == {g["name"] for g in eligible}, f"{tag}: eligibility mismatch")
    require(rule["fell_back_to_all_slices"] == (not eligible), f"{tag}: eligibility fallback mismatch")
    close(rule["median_nll_threshold"], median_nll, f"{tag}: median NLL")
    close(rule["overall_accuracy_threshold"], run["val_overall_accuracy"], f"{tag}: accuracy threshold")
    require(rule["selected"]["name"] == unique_min(eligible or rows, statistic, tag),
            f"{tag}: original headline selection mismatch")
    return rows, eligible, losses, summary


def verify_selection(frozen, seeds, statistic):
    runs, rule, choice = frozen["runs"], frozen["rule"], frozen["fg_lambda_choice"]
    require(len(runs) == 20, "Expected five ERM plus fifteen FG candidate records")
    close(rule["alpha"], 0.95, "alpha")
    close(rule["ridge"], 1e-3, "nuisance ridge")
    close(rule["eps"], 1e-8, "reciprocal stabilization")
    require(rule["selection_statistic"] == statistic, "Cohort headline statistic changed")
    require(set(choice["per_seed"]) == set(map(str, seeds)), "FG seed set changed")
    close(choice["accuracy_tolerance"], 0.01, "FG accuracy tolerance")
    frozen_payload = {"rule": rule, "selected_group": frozen["selected_group"],
                      "checkpoints": {tag: r["checkpoint_sha256"] for tag, r in runs.items()},
                      "temperatures": {tag: r["temperature"] for tag, r in runs.items()}}
    digest = hashlib.sha256(json.dumps(frozen_payload, sort_keys=True, default=str).encode()).hexdigest()
    require(digest == frozen["frozen_hash"], "Frozen protocol/checkpoint hash mismatch")
    geometry = {}
    for tag, run in runs.items():
        require(tag == run["tag"] and run["seed"] in seeds, f"{tag}: invalid run identity")
        geometry[tag] = verify_geometry(run, rule["eps"], rule["alpha"], statistic)
    selected, fallbacks = [], []
    for seed in seeds:
        candidates = [r for r in runs.values() if r["seed"] == seed]
        erm = [r for r in candidates if r["method"] == "erm"]
        fg = [r for r in candidates if r["method"] == "fisher_glass"]
        require(len(candidates) == 4 and len(erm) == 1 and len(fg) == 3, f"seed {seed}: candidate count")
        require(erm[0]["fg_lambda"] == 0 and {r["fg_lambda"] for r in fg} == {0.003, 0.01, 0.03},
                f"seed {seed}: candidate coefficients")
        floor = erm[0]["val_overall_accuracy"] - choice["accuracy_tolerance"]
        eligible = [r for r in fg if r["val_overall_accuracy"] >= floor]
        chosen = choice["per_seed"][str(seed)]
        winner = max(eligible or fg, key=lambda r: r["val_fisher_glass_certificate"])
        require(chosen["chosen_tag"] == winner["tag"], f"seed {seed}: stored FG selection differs")
        require(chosen["n_eligible"] == len(eligible) and chosen["fell_back"] == (not eligible),
                f"seed {seed}: stored FG eligibility/fallback differs")
        for key, value in (("accuracy_floor", floor), ("val_accuracy", winner["val_overall_accuracy"]),
                           ("val_certificate", winner["val_fisher_glass_certificate"]),
                           ("erm_val_accuracy", erm[0]["val_overall_accuracy"]),
                           ("chosen_lambda", winner["fg_lambda"])):
            close(chosen[key], value, f"seed {seed}: {key}")
        if not eligible:
            fallbacks.append(seed)
        selected.extend((erm[0]["tag"], chosen["chosen_tag"]))
    require(set(map(int, choice["fell_back_seeds"])) == set(fallbacks), "Fallback seed list mismatch")
    require(choice["accuracy_matched"] == (not fallbacks), "Accuracy-matching flag mismatch")
    return selected, fallbacks, geometry


def analyse_cohort(root, cohort, directory, seeds, statistic, provenance):
    frozen = load_record(root, f"{directory}/frozen_selection.json", provenance)
    shifted = load_record(root, f"{directory}/shift_summary.json", provenance)
    generation = load_record(root, f"{directory}/paired_generation.json", provenance)
    selected, fallbacks, geometry = verify_selection(frozen, seeds, statistic)
    require(len(shifted["runs"]) == 10, f"{cohort}: expected exactly ten intervention records")
    shifts = {r["tag"]: r for r in shifted["runs"]}
    require(len(shifts) == 10 and set(shifts) == set(selected), f"{cohort}: intervention/selection join mismatch")
    draws = generation["args"]["draws"]
    require(type(draws) is int and draws == 2, f"{cohort}: expected two draws per arm/source")
    require(generation["args"]["groups"] == list(range(4)) and not generation["args"]["only_selected_group"],
            f"{cohort}: incomplete generation group coverage")
    require(generation["n_source_images"] == sum(TEST_N), f"{cohort}: source total mismatch")
    require(generation["n_generated"] == 2 * draws * sum(TEST_N), f"{cohort}: generation total mismatch")
    require(generation["groups"] == {name: 2 * draws * n for name, n in zip(GROUPS, TEST_N)},
            f"{cohort}: generation group counts mismatch")
    models, groups, rule_values, shares = [], [], defaultdict(list), []
    exact_outcomes = {"D_max_pp": [], "W_opp": []}
    for tag in selected:
        run, shift = frozen["runs"][tag], shifts[tag]
        rows, eligible, losses, summary = geometry[tag]
        for key in ("seed", "method", "fg_lambda", "arch"):
            require(run[key] == shift[key], f"{tag}: inconsistent {key}")
        require(shift["frozen_hash"] == frozen["frozen_hash"], f"{tag}: intervention frozen hash")
        for key in ("val_fisher_glass_certificate", "val_overall_accuracy", "temperature"):
            close(shift[key], run[key], f"{tag}: intervention {key}")
        by_group = {g["slice"]: g for g in shift["by_group"]}
        require(len(shift["by_group"]) == len(by_group) == 4 and set(by_group) == set(GROUPS),
                f"{tag}: incomplete/duplicate intervention groups")
        drops, opposites = {}, {}
        for gid, (row, n) in enumerate(zip(rows, TEST_N)):
            name, outcome = row["name"], by_group[row["name"]]
            require(outcome["n"] == n, f"{tag}/{name}: source count mismatch")
            denominator = n * draws
            same_count, same = count_fraction(outcome["acc_same"], denominator, f"{tag}/{name}: same")
            opp_count, opposite = count_fraction(outcome["acc_opposite"], denominator, f"{tag}/{name}: opposite")
            drop = same - opposite
            close(float(drop), outcome["delta_nuisance"], f"{tag}/{name}: signed drop")
            for key in ("acc_same", "acc_opposite", "delta_nuisance"):
                boot = outcome["bootstrap_source"][key]
                require(boot["n_clusters"] == n, f"{tag}/{name}: bootstrap source count")
                close(boot["point"], outcome[key], f"{tag}/{name}: bootstrap point")
            drops[name], opposites[name] = drop, opposite
            groups.append({"cohort": cohort, "seed": run["seed"], "tag": tag, "group": name,
                           "validation_n": row["n"], "raw": row["raw_fisher"],
                           "projected": row["projected_fisher"], "inverse": losses[gid],
                           "inverse_share": losses[gid] / sum(losses), "eligible": row in eligible,
                           "source_n": n, "draws_per_arm": draws, "trials_per_arm": denominator,
                           "same_correct": same_count, "opposite_correct": opp_count,
                           "signed_drop_fraction": str(drop), "signed_drop_pp": 100 * float(drop)})
        dmax, wopp = max(drops.values()), min(opposites.values())
        exact_outcomes["D_max_pp"].append(dmax * 100)
        exact_outcomes["W_opp"].append(wopp)
        is_fg = run["method"] == "fisher_glass"
        models.append({"cohort": cohort, "seed": run["seed"], "arm": "selected_FG" if is_fg else "ERM",
                       "tag": tag, "fg_lambda": run["fg_lambda"],
                       "accuracy_budget_fallback": bool(is_fg and run["seed"] in fallbacks),
                       "checkpoint_sha256": run["checkpoint_sha256"], **summary,
                       "mean_inverse": statistics.mean(losses),
                       "D_max_pp": 100 * float(dmax), "D_max_fraction": str(dmax),
                       "W_opp": float(wopp), "W_opp_fraction": str(wopp)})
        if not is_fg:
            require(eligible, f"{tag}: eligible group rule has no candidates")
            raw_min = unique_min(rows, "raw_fisher", tag)
            projected_min = unique_min(rows, "projected_fisher", tag)
            inverse_max = max(rows, key=lambda g: g["inverse_projected_fisher"])["name"]
            cvar_active = [g["name"] for g in rows if g["inverse_projected_fisher"] == max(losses)]
            require(cvar_active == [projected_min] and inverse_max == projected_min,
                    f"{tag}: projected/inverse/CVaR selectors disagree")
            eligible_inverse = max(eligible, key=lambda g: g["inverse_projected_fisher"])["name"]
            for rule, name in zip(RULES, (raw_min, projected_min, eligible_inverse)):
                rule_values[rule].append({"seed": run["seed"], "group": name,
                                          "signed_drop_pp": 100 * float(drops[name])})
            shares.append(max(losses) / sum(losses))
    panel_a = []
    for arm in ("ERM", "selected_FG", "FG_minus_ERM"):
        for summary in SUMMARIES:
            values = []
            for seed in seeds:
                pair = {m["arm"]: m for m in models if m["seed"] == seed}
                require(set(pair) == {"ERM", "selected_FG"}, f"{cohort}/{seed}: incomplete pair")
                values.append(pair["selected_FG"][summary] - pair["ERM"][summary]
                              if arm == "FG_minus_ERM" else pair[arm][summary])
            panel_a.append({"cohort": cohort, "arm": arm, "summary": summary, **mean_ci(values)})
    panel_b = [{"cohort": cohort, "rule": rule, "candidates": candidates,
                "group_counts": dict(sorted(Counter(v["group"] for v in values).items())),
                **mean_ci(v["signed_drop_pp"] for v in values), "per_seed": values}
               for (rule, candidates), values in rule_values.items()]
    panel_c = [{"cohort": cohort, "summary": key, "n_models": 10, "n_seed_pairs": 5,
                **{outcome: spearman([m[key] for m in models], exact)
                   for outcome, exact in exact_outcomes.items()}} for key in SUMMARIES]
    ties = {}
    for outcome, values in exact_outcomes.items():
        tied = defaultdict(list)
        for model, value in zip(models, values):
            tied[value].append(model["tag"])
        ties[outcome] = [{"exact_value": str(value), "models": tags}
                         for value, tags in sorted(tied.items()) if len(tags) > 1]
    metadata = {"cohort": cohort, "artifact_directory": directory, "seeds": list(seeds),
                "original_headline_statistic": statistic, "original_headline_group": frozen["selected_group"]["name"],
                "accuracy_budget_fallback_seeds": fallbacks, "frozen_hash": frozen["frozen_hash"],
                "candidate_models_checked": 20, "candidate_group_records_checked": 80,
                "analysed_models": len(models), "analysed_group_records": len(groups),
                "held_out_sources_per_model": sum(TEST_N), "draws_per_source_per_arm": draws,
                "largest_inverse_share_ERM": mean_ci(shares), "outcome_ties": ties}
    return metadata, panel_a, panel_b, panel_c, models, groups


def write_csv(path, rows):
    require(rows, f"No rows for {path.name}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        for row in rows:
            writer.writerow({k: json.dumps(v, sort_keys=True) if isinstance(v, (dict, list)) else v
                             for k, v in row.items()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-root", type=Path, default=Path(__file__).resolve().parents[2] / "artifacts")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    provenance, metadata, panel_a, panel_b, panel_c, models, groups = [], [], [], [], [], [], []
    for cohort in COHORTS:
        result = analyse_cohort(args.artifact_root, *cohort, provenance)
        metadata.append(result[0])
        for target, rows in zip((panel_a, panel_b, panel_c, models, groups), result[1:]):
            target.extend(rows)
    require(len(models) == 20 and len(groups) == 80, "Combined model/group counts changed")
    payload = {"analysis": "post_hoc_four_way_swap", "schema_version": 1,
               "definitions": {"A": "mean_g raw_fisher", "P": "mean_g projected_fisher",
                               "H": "1 / mean_g (1 / (projected_fisher + epsilon))",
                               "C": "1 / upper_CVaR_0.95_uniform(1 / (projected_fisher + epsilon))",
                               "D_max_pp": "100 * max_g (accuracy_same_g - accuracy_opposite_g)",
                               "W_opp": "min_g accuracy_opposite_g"},
               "parameters": {"groups": list(GROUPS), "group_weight": 0.25, "ridge": 0.001,
                              "epsilon": 1e-8, "alpha": 0.95,
                              "ci": "mean +/- 1.96 * sample_sd / sqrt(5); paired changes within seed",
                              "spearman": "Pearson correlation of average ranks; exact rational outcome ties; descriptive only"},
               "cohorts": metadata, "panel_a": panel_a, "panel_b": panel_b, "panel_c": panel_c,
               "provenance": {"source_repository": "https://github.com/E-zClap/fisher-glass-learning",
                              "reference_source_commit": "7f129365db05c69917383eeca9e1af2eeae4873f",
                              "inputs": provenance,
                              "analysis_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}}
    # All checks finish before creating or replacing any output.
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "results.json").write_text(
        json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8", newline="\n")
    for filename, rows in (("panel_a.csv", panel_a), ("panel_b.csv", panel_b), ("panel_c.csv", panel_c),
                           ("models.csv", models), ("groups.csv", groups)):
        write_csv(args.output_dir / filename, rows)
    print("Verified 40 candidate models, 160 candidate validation groups, 20 selected models, and 80 intervention groups.")
    for cohort in metadata:
        print(f"{cohort['cohort']}: FG fallback seeds {cohort['accuracy_budget_fallback_seeds']}; "
              f"largest inverse share {100 * cohort['largest_inverse_share_ERM']['mean']:.3f}%")
    print("Wrote results.json and five CSV files.")


if __name__ == "__main__":
    main()
