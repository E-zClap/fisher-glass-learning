"""Aggregate the frozen five-seed hidden-slice experiment.

This stage consumes only completed discovery, repair, blind certification, and
post-hoc artifacts.  It never refits a model or selector.  The four tabular
outputs intentionally separate headline rows, paired seed comparisons, and
predeclared sensitivity analyses.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.stats import t as student_t

try:  # Package import (tests and ``python -m``).
    from .common import (
        atomic_write_json,
        file_sha256,
        json_sha256,
        require_current_scientific_implementation,
        require_json_artifact_sha256,
    )
    from .metrics import FisherMetricConfig, evaluate_class_slice
except ImportError:  # Direct stage script.
    from common import (
        atomic_write_json,
        file_sha256,
        json_sha256,
        require_current_scientific_implementation,
        require_json_artifact_sha256,
    )
    from metrics import FisherMetricConfig, evaluate_class_slice


SCHEMA_VERSION = 1
PAPER_SEEDS = (17, 18, 19, 20, 21)
GROUP_NAMES = ("landbird/land", "landbird/water", "waterbird/land", "waterbird/water")
REPAIR_ARMS = ("loss_repair", "fisher_repair", "combined_repair")
CONTINUATION_ARMS = ("erm_continuation",)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _summary(values: list[float], confidence: float) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a non-empty finite vector")
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if array.size > 1 else 0.0
    se = float(std / np.sqrt(array.size))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, df=array.size - 1)) if array.size > 1 else 0.0
    half_width = critical * se
    return {
        "n": int(array.size),
        "mean": mean,
        "std": std,
        "se": se,
        "lo": mean - half_width,
        "hi": mean + half_width,
        "half_width": half_width,
        "values": array.tolist(),
    }


def _bounded_summary(
    values: list[float],
    confidence: float,
    *,
    lower: float,
    upper: float,
) -> dict[str, Any]:
    """Student-t summary with displayed endpoints clipped to known bounds.

    The unbounded endpoints are retained so clipping cannot be mistaken for a
    different inferential procedure.  This is primarily used for overlap and
    correlation diagnostics, whose ordinary small-sample t intervals can
    otherwise display impossible values.
    """
    if not lower < upper:
        raise ValueError("summary bounds must be ordered")
    result = _summary(values, confidence)
    result["unbounded_lo"] = result["lo"]
    result["unbounded_hi"] = result["hi"]
    result["lo"] = max(float(lower), float(result["lo"]))
    result["hi"] = min(float(upper), float(result["hi"]))
    result["bounds"] = [float(lower), float(upper)]
    result["interval_note"] = "Student-t endpoints clipped to the metric's parameter bounds"
    return result


def _optional_summary(values: list[float | None], confidence: float) -> dict[str, Any] | None:
    finite = [float(value) for value in values if value is not None and np.isfinite(value)]
    return _summary(finite, confidence) if finite else None


def _load_seed(out_dir: Path, seed: int) -> dict[str, Any]:
    seed_dir = out_dir / "runs" / f"seed_{seed}"
    paths = {
        "discovery": seed_dir / "discovery.json",
        "repair": seed_dir / "repair.json",
        "blind_test": seed_dir / "blind_test_results.json",
        "posthoc": seed_dir / "posthoc.json",
    }
    payload = {name: _read_json(path) for name, path in paths.items()}
    for name, artifact in payload.items():
        if int(artifact.get("seed", -1)) != seed:
            raise RuntimeError(f"{paths[name]} has the wrong seed")
        if int(artifact.get("schema_version", -1)) != 1:
            raise RuntimeError(f"unsupported schema in {paths[name]}")
    if payload["discovery"].get("metadata_accessed") is not False:
        raise RuntimeError(f"seed {seed}: discovery was not metadata blind")
    if payload["repair"].get("metadata_accessed") is not False:
        raise RuntimeError(f"seed {seed}: repair was not metadata blind")
    if payload["blind_test"].get("metadata_accessed") is not False:
        raise RuntimeError(f"seed {seed}: certification was not metadata blind")
    if payload["posthoc"].get("metadata_revealed_after_blind_freeze") is not True:
        raise RuntimeError(f"seed {seed}: metadata reveal ordering is not certified")
    require_current_scientific_implementation(
        payload["discovery"],
        label=f"seed {seed} discovery",
    )
    require_json_artifact_sha256(
        payload["discovery"],
        "frozen_blind_sha256",
        label=f"seed {seed} discovery",
    )
    require_json_artifact_sha256(
        payload["repair"],
        "frozen_repair_sha256",
        label=f"seed {seed} repair",
    )
    require_json_artifact_sha256(
        payload["blind_test"],
        "blind_test_sha256",
        label=f"seed {seed} blind certification",
    )
    require_json_artifact_sha256(
        payload["posthoc"],
        "posthoc_sha256",
        label=f"seed {seed} post-hoc",
    )
    if payload["repair"]["discovery_artifact_sha256"] != file_sha256(paths["discovery"]):
        raise RuntimeError(f"seed {seed}: repair/discovery hash mismatch")
    if payload["blind_test"]["discovery_artifact_sha256"] != file_sha256(paths["discovery"]):
        raise RuntimeError(f"seed {seed}: certification/discovery hash mismatch")
    if payload["blind_test"]["repair_artifact_sha256"] != file_sha256(paths["repair"]):
        raise RuntimeError(f"seed {seed}: certification/repair hash mismatch")
    if payload["posthoc"]["blind_test_sha256"] != payload["blind_test"]["blind_test_sha256"]:
        raise RuntimeError(f"seed {seed}: post-hoc/blind certification hash mismatch")
    score_path = seed_dir / "blind_test_scores.npz"
    if not score_path.is_file() or file_sha256(score_path) != payload["blind_test"]["score_artifact_sha256"]:
        raise RuntimeError(f"seed {seed}: blind score artifact hash mismatch")
    oracle_score_path = Path(payload["posthoc"]["oracle_score_artifact"])
    if not oracle_score_path.is_file() or file_sha256(oracle_score_path) != payload["posthoc"]["oracle_score_artifact_sha256"]:
        raise RuntimeError(f"seed {seed}: oracle score artifact hash mismatch")
    return {
        "seed": seed,
        **payload,
        "paths": {name: str(path) for name, path in paths.items()},
        "hashes": {name: file_sha256(path) for name, path in paths.items()},
        "score_path": str(score_path),
    }


def _mean_stability(crossfit: dict[str, Any] | None) -> dict[str, float | str] | None:
    if not crossfit:
        return None
    aggregate = crossfit.get("aggregate", {})
    by_class = aggregate.get("stability", {})
    selected_counts = aggregate.get("selected_class_counts", {})
    fold_count = len(crossfit.get("folds", []))
    if not selected_counts or sum(int(value) for value in selected_counts.values()) != fold_count:
        raise RuntimeError("cross-fit stability is missing complete selected-class counts")
    selected_rows = [
        (row, int(selected_counts.get(str(class_id), 0)))
        for class_id, row in ((int(key), value) for key, value in by_class.items())
        if int(selected_counts.get(str(class_id), 0)) > 0 and int(row.get("pairs", 0)) > 0
    ]
    if not selected_rows:
        return None
    selected_total = sum(count for _, count in selected_rows)

    def weighted(key: str) -> float:
        return float(sum(float(row[key]) * count for row, count in selected_rows) / selected_total)

    return {
        "score_correlation": weighted("score_correlation_mean"),
        "top_overlap": weighted("top_overlap_mean"),
        "score_correlation_min": float(
            np.min([row.get("score_correlation_min", 1.0) for row, _ in selected_rows])
        ),
        "top_overlap_min": float(np.min([row.get("top_overlap_min", 1.0) for row, _ in selected_rows])),
        "selected_class_agreement": float(max(int(value) for value in selected_counts.values()) / fold_count),
        "definition": (
            "per-class selected-gate stability weighted by selected-fold counts; "
            "classes selected in fewer than two folds have no gate pair and are excluded"
        ),
    }


def _uniform_slice_metrics(context: dict[str, Any]) -> dict[str, Any]:
    """Evaluate the constant-membership gate on the Fisher-selected class.

    A zero-weight linear gate with bias ``logit(target_mass)`` induces these
    uniform normalized weights, so this is both a feasible adversary and a
    necessary optimization sanity check.
    """
    blind = context["blind_test"]
    discovery = context["discovery"]
    selected = blind["arms"]["erm"]["fisher_selector"]
    class_id = int(selected["class_id"])
    with np.load(Path(context["score_path"]), allow_pickle=False) as archive:
        required = {"y", "erm_task", "b_vl", "erm_logits"}
        missing = required.difference(archive.files)
        if missing:
            raise RuntimeError(f"uniform-gate comparator is missing score arrays: {sorted(missing)}")
        labels = np.asarray(archive["y"], dtype=np.int64)
        task = np.asarray(archive["erm_task"], dtype=np.float64)
        nuisance = np.asarray(archive["b_vl"], dtype=np.float64)
        logits = np.asarray(archive["erm_logits"], dtype=np.float64)
    mask = labels == class_id
    count = int(mask.sum())
    if count < 1:
        raise RuntimeError(f"uniform-gate selected class {class_id} is absent")
    q = np.full(count, 1.0 / float(count), dtype=np.float64)
    protocol = discovery["protocol"]
    result = evaluate_class_slice(
        task=task[mask],
        nuisance=nuisance[mask],
        logits=logits[mask],
        labels=labels[mask],
        q=q,
        class_id=class_id,
        temperature=float(discovery["base_model"]["temperature"]),
        config=FisherMetricConfig(
            float(protocol["shrinkage"]),
            float(protocol["ridge"]),
            float(protocol["epsilon"]),
        ),
        hard_mass=float(protocol["slice_mass"]),
    )
    target = float(protocol["slice_mass"])
    result["selector_objective"] = "constant_uniform"
    result["gate_transfer"] = {
        "target_membership_mass": target,
        "heldout_mean_membership": target,
        "absolute_mass_error": 0.0,
        "membership_min": target,
        "membership_max": target,
    }
    return result


def _random_slice_metrics(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "class_id": int(payload["class_id"]),
        "predictive": {
            "projected_fisher": float(payload["predictive_fisher"]["mean"]),
            "inverse_fisher": float(payload["inverse_fisher"]["mean"]),
        },
        "unweighted": {"projected_fisher": float(payload["unweighted_fisher"]["mean"])},
        "weighted_loss": float(payload["weighted_loss"]["mean"]),
        "weighted_accuracy": float(payload["weighted_accuracy"]["mean"]),
    }


def _seed_method(context: dict[str, Any], key: str) -> dict[str, Any]:
    discovery = context["discovery"]
    repair = context["repair"]
    blind = context["blind_test"]
    posthoc = context["posthoc"]
    erm = blind["arms"]["erm"]

    if key == "erm_hidden_fisher":
        return {
            "slice": erm["fisher_selector"],
            "overall": erm["overall"],
            "group": posthoc["model_group_metrics"]["erm"],
            "composition": posthoc["compositions"]["erm_fisher"],
            "stability": _mean_stability(discovery["crossfit"]["primary_fisher"]),
            "accuracy_budget": None,
        }
    if key == "erm_high_loss":
        return {
            "slice": erm["high_loss_selector"],
            "overall": erm["overall"],
            "group": posthoc["model_group_metrics"]["erm"],
            "composition": posthoc["compositions"]["erm_high_loss"],
            "stability": _mean_stability(discovery["crossfit"]["high_loss"]),
            "accuracy_budget": None,
        }
    if key == "erm_random_slice":
        return {
            "slice": _random_slice_metrics(erm["random_slice_baseline"]),
            "overall": erm["overall"],
            "group": posthoc["model_group_metrics"]["erm"],
            "composition": None,
            "stability": None,
            "accuracy_budget": None,
        }
    if key == "erm_uniform_gate":
        return {
            "slice": _uniform_slice_metrics(context),
            "overall": erm["overall"],
            "group": posthoc["model_group_metrics"]["erm"],
            "composition": None,
            "stability": None,
            "accuracy_budget": None,
        }
    if key == "erm_random_nuisance":
        return {
            "slice": erm["random_nuisance_selector"],
            "overall": erm["overall"],
            "group": posthoc["model_group_metrics"]["erm"],
            "composition": posthoc["compositions"]["erm_random_nuisance"],
            "stability": _mean_stability(discovery["crossfit"]["random_nuisance"]),
            "accuracy_budget": None,
        }
    if key in (*CONTINUATION_ARMS, *REPAIR_ARMS):
        arm = blind["arms"][key]
        selection = repair["selected_arms"][key]["selection"]
        return {
            "slice": arm["rediscovered_fisher_selector"],
            "overall": arm["overall"],
            "group": posthoc["model_group_metrics"][key],
            "composition": posthoc["compositions"][key],
            "stability": _mean_stability(repair["selected_arms"][key]["fresh_adversary"]["crossfit"]),
            "accuracy_budget": bool(arm["accuracy_budget_satisfied"]),
            "spec": arm["spec"],
            "selected_is_no_op": bool(selection.get("selected_is_no_op", False)),
        }
    if key == "oracle_group_dro":
        oracle = posthoc["oracle_group_dro"]
        return {
            "slice": oracle["rediscovered_hidden_slice"],
            "overall": oracle["overall"],
            "group": oracle["group_metrics"],
            "composition": oracle["composition"],
            "stability": _mean_stability(oracle["crossfit"]),
            "accuracy_budget": None,
        }
    if key == "oracle_nuisance_probe":
        oracle = posthoc["oracle_nuisance_sensitivity"]
        return {
            "slice": oracle["hidden_slice"],
            "overall": erm["overall"],
            "group": posthoc["model_group_metrics"]["erm"],
            "composition": oracle["composition"],
            "stability": _mean_stability(oracle["crossfit"]),
            "accuracy_budget": None,
        }
    raise KeyError(key)


METHOD_SPECS = (
    {
        "key": "erm_hidden_fisher",
        "method": "ERM",
        "selector": "hidden Fisher",
        "nuisance": "CLIP text",
        "group_labels_used": False,
        "group_label_role": "none",
        "role": "primary metadata-free discovery",
    },
    {
        "key": "erm_high_loss",
        "method": "ERM",
        "selector": "high loss",
        "nuisance": "CLIP text for Fisher evaluation",
        "group_labels_used": False,
        "group_label_role": "none",
        "role": "group-blind loss-discovery baseline",
    },
    {
        "key": "erm_random_slice",
        "method": "ERM",
        "selector": "random equal-mass slice",
        "nuisance": "CLIP text for evaluation",
        "group_labels_used": False,
        "group_label_role": "none",
        "role": "random-slice baseline",
    },
    {
        "key": "erm_uniform_gate",
        "method": "ERM",
        "selector": "constant/uniform class gate",
        "nuisance": "CLIP text for evaluation",
        "group_labels_used": False,
        "group_label_role": "none",
        "role": "feasible-adversary optimization sanity check",
    },
    {
        "key": "erm_random_nuisance",
        "method": "ERM",
        "selector": "hidden Fisher",
        "nuisance": "random axis",
        "group_labels_used": False,
        "group_label_role": "none",
        "role": "negative nuisance control",
    },
    {
        "key": "erm_continuation",
        "method": "Matched ERM continuation",
        "selector": "re-discovered hidden Fisher",
        "nuisance": "CLIP text",
        "group_labels_used": False,
        "group_label_role": "none",
        "role": "matched additional-optimization control",
    },
    {
        "key": "loss_repair",
        "method": "Loss pseudo-group repair",
        "selector": "re-discovered hidden Fisher",
        "nuisance": "CLIP text",
        "group_labels_used": False,
        "group_label_role": "none",
        "role": "group-blind repair",
    },
    {
        "key": "fisher_repair",
        "method": "Hidden-Fisher repair",
        "selector": "re-discovered hidden Fisher",
        "nuisance": "CLIP text",
        "group_labels_used": False,
        "group_label_role": "none",
        "role": "group-blind repair",
    },
    {
        "key": "combined_repair",
        "method": "Combined repair",
        "selector": "re-discovered hidden Fisher",
        "nuisance": "CLIP text",
        "group_labels_used": False,
        "group_label_role": "none",
        "role": "group-blind repair",
    },
    {
        "key": "oracle_group_dro",
        "method": "Oracle Group DRO",
        "selector": "re-discovered hidden Fisher",
        "nuisance": "CLIP text",
        "group_labels_used": True,
        "group_label_role": "model training",
        "role": "oracle labeled-group comparator",
    },
    {
        "key": "oracle_nuisance_probe",
        "method": "ERM",
        "selector": "hidden Fisher",
        "nuisance": "oracle metadata probe",
        "group_labels_used": True,
        "group_label_role": "nuisance probe only",
        "role": "oracle-nuisance sensitivity; not primary",
    },
)


METRIC_EXTRACTORS: dict[str, Callable[[dict[str, Any]], float | None]] = {
    "overall_accuracy": lambda row: row["overall"]["accuracy"],
    "overall_loss": lambda row: row["overall"]["loss"],
    "oracle_worst_group_accuracy": lambda row: row["group"]["worst_group_accuracy"],
    "predictive_fisher": lambda row: row["slice"]["predictive"]["projected_fisher"],
    "inverse_fisher": lambda row: row["slice"]["predictive"]["inverse_fisher"],
    "unweighted_fisher": lambda row: row["slice"]["unweighted"]["projected_fisher"],
    "centered_unweighted_fisher": lambda row: row["slice"].get("centered_unweighted", {}).get("projected_fisher"),
    "centered_projected_over_raw": lambda row: row["slice"].get("centered_unweighted", {}).get("projected_over_raw"),
    "centered_task_nuisance_correlation": lambda row: row["slice"].get("centered_unweighted", {}).get("correlation"),
    "slice_loss": lambda row: row["slice"]["weighted_loss"],
    "slice_accuracy": lambda row: row["slice"]["weighted_accuracy"],
    "slice_confidence": lambda row: row["slice"].get("mean_confidence"),
    "slice_entropy": lambda row: row["slice"].get("mean_entropy"),
    "slice_abs_margin": lambda row: row["slice"].get("mean_abs_margin"),
    "effective_sample_size": lambda row: row["slice"].get("constraints", {}).get("effective_sample_size"),
    "heldout_membership_mass": lambda row: row["slice"].get("gate_transfer", {}).get("heldout_mean_membership"),
    "membership_mass_error": lambda row: row["slice"].get("gate_transfer", {}).get("absolute_mass_error"),
}


def _aggregate_composition(rows: list[dict[str, Any]], confidence: float) -> dict[str, Any] | None:
    compositions = [row["composition"] for row in rows if row.get("composition")]
    if not compositions:
        return None
    weighted = {
        group: _summary([entry["weighted_group_composition"][group] for entry in compositions], confidence)
        for group in GROUP_NAMES
    }
    dominant = max(GROUP_NAMES, key=lambda group: weighted[group]["mean"])
    seed_dominants = [entry["dominant_metadata_group"] for entry in compositions]
    return {
        "weighted_group_composition": weighted,
        "dominant_metadata_group": dominant,
        "dominant_group_purity": weighted[dominant],
        "dominant_group_seed_agreement": float(np.mean([name == dominant for name in seed_dominants])),
        "metadata_role": "post-hoc interpretation only",
    }


def _aggregate_main(contexts: list[dict[str, Any]], confidence: float) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    main_rows = []
    per_seed: dict[str, list[dict[str, Any]]] = {}
    for spec in METHOD_SPECS:
        seed_rows = [_seed_method(context, spec["key"]) for context in contexts]
        per_seed[spec["key"]] = seed_rows
        metrics = {
            name: _optional_summary([extractor(row) for row in seed_rows], confidence)
            for name, extractor in METRIC_EXTRACTORS.items()
        }
        stability_rows = [row["stability"] for row in seed_rows if row.get("stability")]
        stability = None
        if stability_rows:
            stability = {
                "score_correlation": _bounded_summary(
                    [float(row["score_correlation"]) for row in stability_rows],
                    confidence,
                    lower=-1.0,
                    upper=1.0,
                ),
                "top_overlap": _bounded_summary(
                    [float(row["top_overlap"]) for row in stability_rows],
                    confidence,
                    lower=0.0,
                    upper=1.0,
                ),
                "score_correlation_min": _bounded_summary(
                    [float(row["score_correlation_min"]) for row in stability_rows],
                    confidence,
                    lower=-1.0,
                    upper=1.0,
                ),
                "top_overlap_min": _bounded_summary(
                    [float(row["top_overlap_min"]) for row in stability_rows],
                    confidence,
                    lower=0.0,
                    upper=1.0,
                ),
                "selected_class_agreement": _bounded_summary(
                    [float(row["selected_class_agreement"]) for row in stability_rows],
                    confidence,
                    lower=0.0,
                    upper=1.0,
                ),
                "definition": stability_rows[0]["definition"],
            }
        budget_values = [row["accuracy_budget"] for row in seed_rows if row["accuracy_budget"] is not None]
        accuracy_budget = None
        if budget_values:
            accuracy_budget = {
                "satisfied_seeds": int(sum(budget_values)),
                "total_seeds": len(budget_values),
                "all_satisfied": bool(all(budget_values)),
                "per_seed": {str(contexts[i]["seed"]): bool(seed_rows[i]["accuracy_budget"]) for i in range(len(contexts))},
            }
        repair_selection = None
        if spec["key"] in REPAIR_ARMS:
            no_op_flags = [bool(row.get("selected_is_no_op", False)) for row in seed_rows]
            repair_selection = {
                "no_op_seeds": int(sum(no_op_flags)),
                "active_repair_seeds": int(len(no_op_flags) - sum(no_op_flags)),
                "total_seeds": len(no_op_flags),
                "any_no_op": bool(any(no_op_flags)),
                "all_no_op": bool(all(no_op_flags)),
                "per_seed": {
                    str(contexts[index]["seed"]): no_op_flags[index]
                    for index in range(len(contexts))
                },
            }
        main_rows.append(
            {
                **spec,
                "metrics": metrics,
                "gate_stability": stability,
                "composition": _aggregate_composition(seed_rows, confidence),
                "accuracy_budget": accuracy_budget,
                "repair_selection": repair_selection,
                "selected_specs": {
                    str(context["seed"]): seed_rows[index].get("spec")
                    for index, context in enumerate(contexts)
                    if seed_rows[index].get("spec") is not None
                },
            }
        )
    return main_rows, per_seed


def _frozen_slice_repair_rows(
    contexts: list[dict[str, Any]],
    confidence: float,
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    """Summarize every repair on the exact gate discovered against ERM."""
    summaries: list[dict[str, Any]] = []
    per_seed: dict[str, list[dict[str, Any]]] = {}
    names = {entry["key"]: entry["method"] for entry in METHOD_SPECS}
    for arm in (*CONTINUATION_ARMS, *REPAIR_ARMS):
        virtual_key = f"{arm}_on_original_slice"
        rows = [
            {
                "slice": context["blind_test"]["arms"][arm]["frozen_original_fisher_selector"],
                "overall": context["blind_test"]["arms"][arm]["overall"],
                "group": context["posthoc"]["model_group_metrics"][arm],
                "selected_is_no_op": bool(
                    context["repair"]["selected_arms"][arm]["selection"].get(
                        "selected_is_no_op",
                        False,
                    )
                ),
            }
            for context in contexts
        ]
        per_seed[virtual_key] = rows
        summaries.append(
            {
                "key": virtual_key,
                "method": names[arm],
                "selector": "original ERM-discovered Fisher gate (frozen)",
                "group_labels_used": False,
                "metrics": {
                    name: _optional_summary([extractor(row) for row in rows], confidence)
                    for name, extractor in METRIC_EXTRACTORS.items()
                },
                "repair_selection": (
                    None
                    if arm in CONTINUATION_ARMS
                    else {
                        "no_op_seeds": int(sum(row["selected_is_no_op"] for row in rows)),
                        "active_repair_seeds": int(
                            len(rows) - sum(row["selected_is_no_op"] for row in rows)
                        ),
                        "total_seeds": len(rows),
                        "any_no_op": bool(any(row["selected_is_no_op"] for row in rows)),
                        "all_no_op": bool(all(row["selected_is_no_op"] for row in rows)),
                    }
                ),
            }
        )
    return summaries, per_seed


def _paired_row(
    *,
    comparison: str,
    metric: str,
    left_key: str,
    right_key: str,
    per_seed: dict[str, list[dict[str, Any]]],
    seeds: list[int],
    confidence: float,
    favorable_direction: str,
) -> dict[str, Any]:
    extractor = METRIC_EXTRACTORS[metric]
    left = np.asarray([extractor(row) for row in per_seed[left_key]], dtype=np.float64)
    right = np.asarray([extractor(row) for row in per_seed[right_key]], dtype=np.float64)
    if not np.all(np.isfinite(left)) or not np.all(np.isfinite(right)):
        raise ValueError(f"paired comparison {comparison} contains a missing value")
    difference = left - right
    return {
        "comparison": comparison,
        "metric": metric,
        "left": left_key,
        "right": right_key,
        "difference_definition": "left minus right",
        "favorable_direction": favorable_direction,
        "paired_difference": _summary(difference.tolist(), confidence),
        "per_seed": [
            {"seed": seed, "left": float(left[i]), "right": float(right[i]), "difference": float(difference[i])}
            for i, seed in enumerate(seeds)
        ],
    }


def _paired_comparisons(
    per_seed: dict[str, list[dict[str, Any]]], seeds: list[int], confidence: float
) -> list[dict[str, Any]]:
    rows = [
        _paired_row(
            comparison="Fisher selector versus high-loss selector",
            metric="predictive_fisher",
            left_key="erm_hidden_fisher",
            right_key="erm_high_loss",
            per_seed=per_seed,
            seeds=seeds,
            confidence=confidence,
            favorable_direction="negative (Fisher finds lower information)",
        ),
        _paired_row(
            comparison="Fisher selector versus high-loss selector",
            metric="unweighted_fisher",
            left_key="erm_hidden_fisher",
            right_key="erm_high_loss",
            per_seed=per_seed,
            seeds=seeds,
            confidence=confidence,
            favorable_direction="negative (robustness check)",
        ),
        _paired_row(
            comparison="Fisher selector versus high-loss selector",
            metric="centered_unweighted_fisher",
            left_key="erm_hidden_fisher",
            right_key="erm_high_loss",
            per_seed=per_seed,
            seeds=seeds,
            confidence=confidence,
            favorable_direction="negative (centered shift-artifact check)",
        ),
        _paired_row(
            comparison="Fisher selector versus high-loss selector",
            metric="slice_loss",
            left_key="erm_hidden_fisher",
            right_key="erm_high_loss",
            per_seed=per_seed,
            seeds=seeds,
            confidence=confidence,
            favorable_direction="negative or near zero (not merely a high-loss slice)",
        ),
        _paired_row(
            comparison="Fisher selector versus random equal-mass slice",
            metric="predictive_fisher",
            left_key="erm_hidden_fisher",
            right_key="erm_random_slice",
            per_seed=per_seed,
            seeds=seeds,
            confidence=confidence,
            favorable_direction="negative (learned adversary finds lower information than random)",
        ),
        _paired_row(
            comparison="Fisher selector versus random equal-mass slice",
            metric="unweighted_fisher",
            left_key="erm_hidden_fisher",
            right_key="erm_random_slice",
            per_seed=per_seed,
            seeds=seeds,
            confidence=confidence,
            favorable_direction="negative (unweighted optimization sanity check)",
        ),
        _paired_row(
            comparison="Fisher selector versus constant/uniform class gate",
            metric="predictive_fisher",
            left_key="erm_hidden_fisher",
            right_key="erm_uniform_gate",
            per_seed=per_seed,
            seeds=seeds,
            confidence=confidence,
            favorable_direction="negative (learned adversary beats a trivial feasible gate)",
        ),
        _paired_row(
            comparison="Fisher selector versus constant/uniform class gate",
            metric="unweighted_fisher",
            left_key="erm_hidden_fisher",
            right_key="erm_uniform_gate",
            per_seed=per_seed,
            seeds=seeds,
            confidence=confidence,
            favorable_direction="negative (unweighted optimization sanity check)",
        ),
        _paired_row(
            comparison="CLIP-nuisance Fisher selector versus random-nuisance Fisher selector",
            metric="predictive_fisher",
            left_key="erm_hidden_fisher",
            right_key="erm_random_nuisance",
            per_seed=per_seed,
            seeds=seeds,
            confidence=confidence,
            favorable_direction=(
                "negative (the audited nuisance exposes a lower-information slice than a random axis)"
            ),
        ),
    ]
    for arm in (*REPAIR_ARMS, "oracle_group_dro"):
        label = METHOD_SPECS[[entry["key"] for entry in METHOD_SPECS].index(arm)]["method"]
        for metric, direction in (
            ("predictive_fisher", "positive (repair raises worst hidden Fisher)"),
            ("inverse_fisher", "negative (repair lowers inverse Fisher)"),
            ("slice_loss", "negative for loss repair; interpret jointly with Fisher"),
            ("overall_accuracy", "nonnegative; any decrease must remain within the fixed accuracy budget"),
            ("oracle_worst_group_accuracy", "positive"),
        ):
            rows.append(
                _paired_row(
                    comparison=f"{label} versus ERM re-discovery",
                    metric=metric,
                    left_key=arm,
                    right_key="erm_hidden_fisher",
                    per_seed=per_seed,
                    seeds=seeds,
                    confidence=confidence,
                    favorable_direction=direction,
                )
            )
    for arm in CONTINUATION_ARMS:
        label = METHOD_SPECS[[entry["key"] for entry in METHOD_SPECS].index(arm)]["method"]
        for metric in ("predictive_fisher", "inverse_fisher", "slice_loss", "overall_accuracy"):
            rows.append(
                _paired_row(
                    comparison=f"{label} versus ERM re-discovery",
                    metric=metric,
                    left_key=arm,
                    right_key="erm_hidden_fisher",
                    per_seed=per_seed,
                    seeds=seeds,
                    confidence=confidence,
                    favorable_direction="near zero (matched optimization diagnostic)",
                )
            )
    for arm in REPAIR_ARMS:
        label = METHOD_SPECS[[entry["key"] for entry in METHOD_SPECS].index(arm)]["method"]
        for metric, direction in (
            ("predictive_fisher", "positive (repair exceeds matched continued optimization)"),
            ("inverse_fisher", "negative (repair exceeds matched continued optimization)"),
            ("overall_accuracy", "nonnegative; any decrease must remain within the fixed accuracy budget"),
        ):
            rows.append(
                _paired_row(
                    comparison=f"{label} versus matched ERM continuation",
                    metric=metric,
                    left_key=arm,
                    right_key="erm_continuation",
                    per_seed=per_seed,
                    seeds=seeds,
                    confidence=confidence,
                    favorable_direction=direction,
                )
            )
        for metric, direction in (
            ("predictive_fisher", "positive (fixed-slice gain beyond continued optimization)"),
            ("slice_loss", "negative (fixed-slice loss reduction beyond continued optimization)"),
        ):
            rows.append(
                _paired_row(
                    comparison=f"{label} versus matched ERM continuation on original frozen slice",
                    metric=metric,
                    left_key=f"{arm}_on_original_slice",
                    right_key="erm_continuation_on_original_slice",
                    per_seed=per_seed,
                    seeds=seeds,
                    confidence=confidence,
                    favorable_direction=direction,
                )
            )
    for arm in (*CONTINUATION_ARMS, *REPAIR_ARMS):
        label = METHOD_SPECS[[entry["key"] for entry in METHOD_SPECS].index(arm)]["method"]
        virtual_key = f"{arm}_on_original_slice"
        directions = (
            ("predictive_fisher", "positive (the fixed slice gains information)"),
            ("inverse_fisher", "negative (the fixed-slice certificate improves)"),
            ("slice_loss", "negative (the fixed slice incurs less loss)"),
            ("slice_accuracy", "positive (the fixed slice gains accuracy)"),
        )
        if arm in CONTINUATION_ARMS:
            directions = tuple((metric, "near zero (matched optimization diagnostic)") for metric, _ in directions)
        for metric, direction in directions:
            rows.append(
                _paired_row(
                    comparison=f"{label} on original frozen slice versus ERM",
                    metric=metric,
                    left_key=virtual_key,
                    right_key="erm_hidden_fisher",
                    per_seed=per_seed,
                    seeds=seeds,
                    confidence=confidence,
                    favorable_direction=direction,
                )
            )
    return rows


def _hidden_vs_predefined(
    contexts: list[dict[str, Any]],
    primary_rows: list[dict[str, Any]],
    confidence: float,
) -> dict[str, Any]:
    """Paired comparison with the weakest predefined group in each seed.

    Metadata enters only here, after all blind artifacts have been frozen.  The
    per-seed minimum avoids manufacturing a comparison by choosing one group
    from the aggregate means.
    """
    hidden = np.asarray(
        [METRIC_EXTRACTORS["predictive_fisher"](row) for row in primary_rows],
        dtype=np.float64,
    )
    predefined = []
    weakest_groups = []
    for context in contexts:
        rows = context["posthoc"]["predefined_group_reference"]["rows"]
        if {row["group"] for row in rows} != set(GROUP_NAMES):
            raise RuntimeError(
                f"seed {context['seed']}: predefined-group reference is incomplete"
            )
        weakest = min(
            rows,
            key=lambda row: float(row["predictive"]["projected_fisher"]),
        )
        predefined.append(float(weakest["predictive"]["projected_fisher"]))
        weakest_groups.append(str(weakest["group"]))
    known = np.asarray(predefined, dtype=np.float64)
    if hidden.size != known.size or not np.all(np.isfinite(hidden)):
        raise RuntimeError("hidden/predefined Fisher comparison is not finite and paired")
    difference = hidden - known
    success = difference < 0.0
    return {
        "comparison": "hidden Fisher slice versus per-seed lowest-Fisher predefined group",
        "metric": "predictive_fisher",
        "difference_definition": "hidden slice minus lowest predefined group",
        "favorable_direction": "negative (hidden slice is finer/worse than every predefined group)",
        "paired_difference": _summary(difference.tolist(), confidence),
        "seeds_below_lowest_predefined": int(success.sum()),
        "total_seeds": int(success.size),
        "per_seed": [
            {
                "seed": int(context["seed"]),
                "hidden_fisher": float(hidden[index]),
                "lowest_predefined_fisher": float(known[index]),
                "lowest_predefined_group": weakest_groups[index],
                "difference": float(difference[index]),
            }
            for index, context in enumerate(contexts)
        ],
        "metadata_role": "oracle post-hoc reference only",
    }


def _selector_distinctness(contexts: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    rows = [context["blind_test"]["arms"]["erm"]["fisher_vs_high_loss"] for context in contexts]
    correlations = [
        float(row["score_correlation_within_shared_class"])
        for row in rows
        if row.get("score_correlation_within_shared_class") is not None
    ]
    return {
        "comparison": "hidden-Fisher selector versus high-loss selector on untouched test examples",
        "top_slice_jaccard": _bounded_summary(
            [float(row["top_slice_jaccard"]) for row in rows],
            confidence,
            lower=0.0,
            upper=1.0,
        ),
        "score_correlation_same_class_only": (
            _bounded_summary(correlations, confidence, lower=-1.0, upper=1.0)
            if correlations
            else None
        ),
        "same_selected_class_seeds": int(sum(bool(row["same_selected_class"]) for row in rows)),
        "total_seeds": len(rows),
        "per_seed": [
            {"seed": int(context["seed"]), **row}
            for context, row in zip(contexts, rows)
        ],
    }


def _crossfit_values(crossfit: dict[str, Any]) -> dict[str, float | None]:
    aggregate = crossfit["aggregate"]
    stability = _mean_stability(crossfit)
    return {
        "hidden_certificate": float(aggregate["hidden_certificate"]),
        "inverse_fisher": float(aggregate["inverse_fisher_mean"]),
        "predictive_fisher": float(aggregate["predictive_fisher_mean"]),
        "unweighted_fisher": float(aggregate["unweighted_fisher_mean"]),
        "centered_unweighted_fisher": float(aggregate["centered_unweighted_fisher_mean"]),
        "centered_projected_over_raw": float(aggregate["centered_projected_over_raw_mean"]),
        "centered_task_nuisance_correlation": float(aggregate["centered_task_nuisance_correlation_mean"]),
        "slice_loss": float(aggregate["weighted_loss_mean"]),
        "slice_accuracy": float(aggregate["weighted_accuracy_mean"]),
        "score_correlation": stability["score_correlation"] if stability else None,
        "top_overlap": stability["top_overlap"] if stability else None,
        "selected_class_agreement": stability["selected_class_agreement"] if stability else None,
    }


def _sensitivity(contexts: list[dict[str, Any]], confidence: float) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    mass_values = sorted(
        {key for context in contexts for key in context["discovery"]["crossfit"]["mass_sensitivity"]},
        key=float,
    )
    for mass in mass_values:
        for selector in ("fisher", "high_loss"):
            per_seed = [
                _crossfit_values(context["discovery"]["crossfit"]["mass_sensitivity"][mass][selector])
                for context in contexts
            ]
            rows.append(
                {
                    "sensitivity": "slice_mass",
                    "value": float(mass),
                    "selector": selector,
                    "split": "validation cross-fit",
                    "metrics": {
                        key: _optional_summary([entry[key] for entry in per_seed], confidence)
                        for key in per_seed[0]
                    },
                }
            )
    shrinkage_values = sorted(
        {key for context in contexts for key in context["discovery"]["crossfit"]["shrinkage_sensitivity"]},
        key=float,
    )
    for shrinkage in shrinkage_values:
        per_seed = [
            _crossfit_values(context["discovery"]["crossfit"]["shrinkage_sensitivity"][shrinkage])
            for context in contexts
        ]
        rows.append(
            {
                "sensitivity": "shrinkage",
                "value": float(shrinkage),
                "selector": "fisher",
                "split": "validation cross-fit",
                "metrics": {
                    key: _optional_summary([entry[key] for entry in per_seed], confidence)
                    for key in per_seed[0]
                },
            }
        )
    return rows


def _predefined_groups(contexts: list[dict[str, Any]], confidence: float) -> list[dict[str, Any]]:
    result = []
    for group in GROUP_NAMES:
        rows = []
        for context in contexts:
            matches = [
                row for row in context["posthoc"]["predefined_group_reference"]["rows"] if row["group"] == group
            ]
            if len(matches) != 1:
                raise RuntimeError(f"seed {context['seed']}: expected one predefined row for {group}")
            rows.append(matches[0])
        result.append(
            {
                "group": group,
                "metadata_role": "oracle post-hoc reference only",
                "metrics": {
                    "predictive_fisher": _summary([row["predictive"]["projected_fisher"] for row in rows], confidence),
                    "inverse_fisher": _summary([row["predictive"]["inverse_fisher"] for row in rows], confidence),
                    "unweighted_fisher": _summary([row["unweighted"]["projected_fisher"] for row in rows], confidence),
                    "loss": _summary([row["weighted_loss"] for row in rows], confidence),
                    "accuracy": _summary([row["weighted_accuracy"] for row in rows], confidence),
                },
            }
        )
    return result


def _predefined_extrema(contexts: list[dict[str, Any]], confidence: float) -> dict[str, Any]:
    lowest = [context["posthoc"]["predefined_group_reference"]["lowest_fisher_group"] for context in contexts]
    highest = [context["posthoc"]["predefined_group_reference"]["highest_loss_group"] for context in contexts]

    def counts(rows: list[dict[str, Any]]) -> dict[str, int]:
        return {
            group: int(sum(row["group"] == group for row in rows))
            for group in GROUP_NAMES
        }

    return {
        "metadata_role": "oracle post-hoc reference only",
        "lowest_fisher": {
            "group_counts": counts(lowest),
            "predictive_fisher": _summary(
                [float(row["predictive"]["projected_fisher"]) for row in lowest],
                confidence,
            ),
        },
        "highest_loss": {
            "group_counts": counts(highest),
            "loss": _summary([float(row["weighted_loss"]) for row in highest], confidence),
        },
    }


def _flatten_main(row: dict[str, Any]) -> dict[str, Any]:
    flat: dict[str, Any] = {
        key: row[key]
        for key in ("key", "method", "selector", "nuisance", "group_labels_used", "group_label_role", "role")
    }
    for metric, summary in row["metrics"].items():
        for suffix in ("mean", "lo", "hi", "n"):
            flat[f"{metric}_{suffix}"] = "" if summary is None else summary[suffix]
    stability = row.get("gate_stability") or {}
    for metric in ("score_correlation", "top_overlap"):
        summary = stability.get(metric)
        flat[f"gate_{metric}_mean"] = "" if summary is None else summary["mean"]
        flat[f"gate_{metric}_lo"] = "" if summary is None else summary["lo"]
        flat[f"gate_{metric}_hi"] = "" if summary is None else summary["hi"]
    composition = row.get("composition")
    flat["dominant_metadata_group"] = composition["dominant_metadata_group"] if composition else ""
    flat["dominant_group_purity_mean"] = composition["dominant_group_purity"]["mean"] if composition else ""
    for group in GROUP_NAMES:
        key = group.replace("/", "_")
        flat[f"composition_{key}_mean"] = (
            composition["weighted_group_composition"][group]["mean"] if composition else ""
        )
    budget = row.get("accuracy_budget")
    flat["accuracy_budget_all_satisfied"] = "" if budget is None else budget["all_satisfied"]
    flat["accuracy_budget_satisfied_seeds"] = "" if budget is None else budget["satisfied_seeds"]
    repair_selection = row.get("repair_selection")
    flat["repair_no_op_seeds"] = "" if repair_selection is None else repair_selection["no_op_seeds"]
    flat["repair_active_seeds"] = (
        "" if repair_selection is None else repair_selection["active_repair_seeds"]
    )
    return flat


def _flatten_paired(row: dict[str, Any]) -> dict[str, Any]:
    summary = row["paired_difference"]
    return {
        "comparison": row["comparison"],
        "metric": row["metric"],
        "left": row["left"],
        "right": row["right"],
        "difference_definition": row["difference_definition"],
        "favorable_direction": row["favorable_direction"],
        "n": summary["n"],
        "mean_difference": summary["mean"],
        "ci95_lo": summary["lo"],
        "ci95_hi": summary["hi"],
        "standard_error": summary["se"],
    }


def _flatten_sensitivity(row: dict[str, Any]) -> dict[str, Any]:
    flat = {key: row[key] for key in ("sensitivity", "value", "selector", "split")}
    for metric, summary in row["metrics"].items():
        for suffix in ("mean", "lo", "hi", "n"):
            flat[f"{metric}_{suffix}"] = "" if summary is None else summary[suffix]
    return flat


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/results/waterbirds_hidden_slice_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(PAPER_SEEDS))
    parser.add_argument("--confidence", type=float, default=0.95)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if not 0.0 < args.confidence < 1.0:
        raise ValueError("--confidence must lie in (0, 1)")
    if not np.isclose(args.confidence, 0.95):
        raise ValueError("the frozen reporting protocol requires --confidence 0.95")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("--seeds contains duplicates")
    seeds = sorted(args.seeds)
    out_dir = Path(args.out_dir)
    contexts = [_load_seed(out_dir, seed) for seed in seeds]
    protocol_hashes = {json_sha256(context["discovery"]["protocol"]) for context in contexts}
    if len(protocol_hashes) != 1:
        raise RuntimeError("discovery protocol differs across seeds")

    main_rows, per_seed = _aggregate_main(contexts, args.confidence)
    frozen_slice_repairs, frozen_per_seed = _frozen_slice_repair_rows(
        contexts,
        args.confidence,
    )
    per_seed.update(frozen_per_seed)
    paired_rows = _paired_comparisons(per_seed, seeds, args.confidence)
    sensitivity_rows = _sensitivity(contexts, args.confidence)
    predefined_groups = _predefined_groups(contexts, args.confidence)
    predefined_extrema = _predefined_extrema(contexts, args.confidence)
    hidden_vs_predefined = _hidden_vs_predefined(
        contexts,
        per_seed["erm_hidden_fisher"],
        args.confidence,
    )
    selector_distinctness = _selector_distinctness(contexts, args.confidence)

    csv_payloads = {
        "main_table.csv": [_flatten_main(row) for row in main_rows],
        "paired_comparisons.csv": [_flatten_paired(row) for row in paired_rows],
        "sensitivity.csv": [_flatten_sensitivity(row) for row in sensitivity_rows],
    }
    csv_paths: dict[str, Path] = {}
    for name, rows in csv_payloads.items():
        if not rows:
            raise RuntimeError(f"refusing to write empty {name}")
        path = out_dir / name
        csv_paths[name] = _atomic_write_csv(path, list(rows[0]), rows)

    accuracy_budget = {
        row["key"]: row["accuracy_budget"] for row in main_rows if row["accuracy_budget"] is not None
    }
    aggregate = {
        "schema_version": SCHEMA_VERSION,
        "stage": "five_seed_reporting_aggregate",
        "seeds": seeds,
        "paper_protocol_seeds": list(PAPER_SEEDS),
        "paper_protocol_complete": tuple(seeds) == PAPER_SEEDS,
        "confidence": args.confidence,
        "interval_convention": "two-sided Student-t interval over paired experiment seeds",
        "implementation": contexts[0]["discovery"]["protocol"]["implementation"],
        "primary_definition": {
            "model": "ERM",
            "selector": "class-conditional linear hidden-Fisher gate",
            "nuisance": "fixed CLIP water-background minus land-background text direction",
            "slice_mass": contexts[0]["discovery"]["protocol"]["slice_mass"],
            "shrinkage": contexts[0]["discovery"]["protocol"]["shrinkage"],
            "crossfit_folds": contexts[0]["discovery"]["protocol"]["crossfit_folds"],
            "gate_restarts": contexts[0]["discovery"]["protocol"]["gate"]["restarts"],
            "deterministic_torch_algorithms": contexts[0]["discovery"]["protocol"]["deterministic_torch_algorithms"],
            "group_metadata_used": False,
        },
        "main_table": main_rows,
        "frozen_slice_repairs": frozen_slice_repairs,
        "paired_comparisons": paired_rows,
        "sensitivity": sensitivity_rows,
        "predefined_groups": predefined_groups,
        "predefined_extrema": predefined_extrema,
        "hidden_vs_predefined": hidden_vs_predefined,
        "selector_distinctness": selector_distinctness,
        "gate_stability": [
            {"key": row["key"], "method": row["method"], "stability": row["gate_stability"]}
            for row in main_rows
            if row["gate_stability"] is not None
        ],
        "composition": [
            {"key": row["key"], "method": row["method"], "composition": row["composition"]}
            for row in main_rows
            if row["composition"] is not None
        ],
        "accuracy_budget": accuracy_budget,
        "metadata_boundary": {
            "primary_rows_metadata_free": all(
                not row["group_labels_used"]
                for row in main_rows
                if row["role"] not in {"oracle labeled-group comparator", "oracle-nuisance sensitivity; not primary"}
            ),
            "metadata_revealed_after_blind_freeze_all_seeds": all(
                context["posthoc"]["metadata_revealed_after_blind_freeze"] for context in contexts
            ),
            "oracle_rows": ["oracle_group_dro", "oracle_nuisance_probe", "predefined_groups"],
        },
        "inputs": {
            str(context["seed"]): {"paths": context["paths"], "sha256": context["hashes"]}
            for context in contexts
        },
        "csv_outputs": {
            name: {"path": str(path), "sha256": file_sha256(path)} for name, path in csv_paths.items()
        },
    }
    aggregate["aggregate_sha256"] = json_sha256(aggregate)
    path = atomic_write_json(out_dir / "aggregate.json", aggregate)
    print(f"wrote {path} and {', '.join(str(path) for path in csv_paths.values())}", flush=True)


if __name__ == "__main__":
    main()
