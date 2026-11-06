"""Reveal Waterbirds metadata only after every blind lambda result is frozen.

This stage is an audit, not a model-selection stage.  It reports y-by-background
performance for the frozen ERM reference and all matched-grid candidates, and contrasts the group-free rule with a
group-supervised validation comparator restricted to the same one-standard-error
risk set.  No head, gate, task score, certificate, or primary selection is
changed.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

from experiment_common import (
    DEFAULT_SEEDS,
    FROZEN_ERM_REFERENCE_KEY,
    SCHEMA_VERSION,
    atomic_write_json,
    blind_input_hashes,
    canonical_probability_weights,
    canonical_protocol_hash,
    file_sha256,
    group_metrics,
    implementation_manifest,
    join_sealed,
    json_artifact_sha256,
    json_sha256,
    lambda_key,
    load_blind_npz,
    load_sealed_metadata,
    read_json,
    require_current_implementation,
    require_json_artifact_sha256,
    task_metrics,
    validate_lambdas,
)
from blind_protocol import make_model_coordinates
from compat import BooleanOptionalAction
from gates import FrozenLinearGate, admissible_weights
from heads import FrozenHead, apply_feature_normalizer
from metrics import FisherMetricConfig, evaluate_class_slice


SELECTION_STAGE = "metadata_blind_lambda_selection"
BLIND_TEST_STAGE = "blind_heldout_test_certification"
POSTHOC_STAGE = "posthoc_group_label_audit"
POLICY_SOURCES = {
    "frozen_erm_reference": "frozen_reference",
    "matched_erm_lambda_0": "lambda_grid",
    "fixed_0.03": "lambda_grid",
    "validation_risk_minimizer": "lambda_grid",
    "group_free": "lambda_grid",
    "group_supervised_val_comparator": "lambda_grid",
}


def _same_lambda(left: float, right: float) -> bool:
    return bool(np.isclose(float(left), float(right), rtol=0.0, atol=1e-12))


def _candidate_for_lambda(candidates: list[dict[str, Any]], coefficient: float) -> dict[str, Any]:
    matches = [row for row in candidates if _same_lambda(row["lambda_fg"], coefficient)]
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one candidate for lambda={coefficient:g}")
    return matches[0]


def _evaluation_records(
    frozen_reference: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    records: list[tuple[str, Mapping[str, Any]]] = [
        (FROZEN_ERM_REFERENCE_KEY, frozen_reference)
    ]
    records.extend((lambda_key(candidate["lambda_fg"]), candidate) for candidate in candidates)
    prefixes = [prefix for prefix, _record in records]
    if len(prefixes) != len(set(prefixes)):
        raise RuntimeError("frozen-reference and lambda-grid score prefixes are not unique")
    return records


def _expected_validation_keys(
    frozen_reference: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> set[str]:
    keys = {"row_id", "y", "nuisance", "fold_assignment"}
    for prefix, _record in _evaluation_records(frozen_reference, candidates):
        keys.update({f"{prefix}__logits", f"{prefix}__task", f"{prefix}__nll"})
    return keys


def _expected_test_keys(
    frozen_reference: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> set[str]:
    keys = {"row_id", "y", "nuisance"}
    for prefix, _record in _evaluation_records(frozen_reference, candidates):
        keys.update(
            {
                f"{prefix}__logits",
                f"{prefix}__task",
                f"{prefix}__q",
                f"{prefix}__membership",
            }
        )
    return keys


def _load_exact_npz(path: Path, expected: set[str]) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        actual = set(archive.files)
        if actual != expected:
            raise RuntimeError(
                f"{path} has the wrong schema; missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    n = arrays["row_id"].size
    if n == 0 or any(array.ndim != 1 or array.size != n for array in arrays.values()):
        raise RuntimeError(f"{path} must contain non-empty aligned vectors")
    row_id = np.asarray(arrays["row_id"], dtype=np.int64)
    if np.unique(row_id).size != n:
        raise RuntimeError(f"{path} contains duplicate row ids")
    return arrays


def _validate_validation_arrays(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    frozen_reference: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    for name, array in arrays.items():
        if name != "row_id" and not np.all(np.isfinite(array)):
            raise RuntimeError(f"{path}: {name} contains non-finite values")
    labels = np.asarray(arrays["y"], dtype=np.int64)
    if not np.all(np.isin(labels, (0, 1))):
        raise RuntimeError(f"{path}: labels must be binary")
    for prefix, _record in _evaluation_records(frozen_reference, candidates):
        if not np.all(np.isfinite(arrays[f"{prefix}__logits"])):
            raise RuntimeError(f"{path}: non-finite logits for {prefix}")


def _validate_test_arrays(
    path: Path,
    arrays: Mapping[str, np.ndarray],
    frozen_reference: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> None:
    labels = np.asarray(arrays["y"], dtype=np.int64)
    if not np.all(np.isin(labels, (0, 1))):
        raise RuntimeError(f"{path}: labels must be binary")
    for name, array in arrays.items():
        if name.endswith("__membership"):
            continue
        if not np.all(np.isfinite(array)):
            raise RuntimeError(f"{path}: {name} contains non-finite values")
    for prefix, candidate in _evaluation_records(frozen_reference, candidates):
        class_id = int(candidate["final_gate"]["class_value"])
        class_mask = labels == class_id
        membership = np.asarray(arrays[f"{prefix}__membership"], dtype=np.float64)
        q = np.asarray(arrays[f"{prefix}__q"], dtype=np.float64)
        if not np.all(np.isfinite(membership[class_mask])) or not np.all(np.isnan(membership[~class_mask])):
            raise RuntimeError(f"{path}: {prefix} membership does not follow its class-conditional schema")
        if np.any(membership[class_mask] < 0.0) or np.any(membership[class_mask] > 1.0):
            raise RuntimeError(f"{path}: {prefix} membership is outside [0, 1]")
        if np.any(q < -1e-12) or not np.allclose(q[~class_mask], 0.0, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{path}: {prefix} has inadmissible out-of-class weights")
        if not np.isclose(q.sum(), 1.0, rtol=0.0, atol=1e-8):
            raise RuntimeError(f"{path}: {prefix} weights do not sum to one")


def _fisher_metric_config(protocol: Mapping[str, Any]) -> FisherMetricConfig:
    payload = protocol.get("fisher_metric")
    if not isinstance(payload, Mapping):
        raise RuntimeError("selection protocol lacks its Fisher metric configuration")
    config = FisherMetricConfig(
        shrinkage=float(payload["shrinkage"]),
        ridge=float(payload["ridge"]),
        epsilon=float(payload["epsilon"]),
    )
    config.validate()
    return config


def _validate_blind_json_metrics(
    selection: Mapping[str, Any],
    blind_test: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
) -> None:
    """Reconcile JSON task/certificate summaries with the frozen score archive."""

    labels = np.asarray(arrays["y"], dtype=np.int64)
    nuisance = np.asarray(arrays["nuisance"], dtype=np.float64)
    temperature = float(selection["normalization"]["base_temperature"])
    metric_config = _fisher_metric_config(selection["protocol"])
    selection_records = _evaluation_records(
        selection[FROZEN_ERM_REFERENCE_KEY], selection["candidates"]
    )
    blind_records: dict[str, Mapping[str, Any]] = {
        FROZEN_ERM_REFERENCE_KEY: blind_test[FROZEN_ERM_REFERENCE_KEY]
    }
    blind_records.update(
        {
            lambda_key(record["lambda_fg"]): record
            for record in blind_test["candidates"]
        }
    )
    for prefix, selection_record in selection_records:
        blind_record = blind_records[prefix]
        logits = np.asarray(arrays[f"{prefix}__logits"], dtype=np.float64)
        task = np.asarray(arrays[f"{prefix}__task"], dtype=np.float64)
        expected_task = (
            logits - float(selection_record["task_mean"])
        ) / float(selection_record["task_std"])
        if not np.allclose(task, expected_task, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"blind test {prefix} task coordinates are inconsistent")

        recomputed_task = task_metrics(logits, labels, temperature)
        recorded_task = blind_record["task_metrics"]
        for field in (
            "n",
            "nll",
            "nll_standard_error",
            "accuracy",
            "balanced_accuracy",
            "minimum_class_accuracy",
        ):
            if not np.isclose(
                float(recorded_task[field]),
                float(recomputed_task[field]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(f"blind test {prefix} task metric {field} is inconsistent")
        for class_id in (0, 1):
            if not np.isclose(
                float(recorded_task["per_class_accuracy"][str(class_id)]),
                float(recomputed_task["per_class_accuracy"][str(class_id)]),
                rtol=1e-10,
                atol=1e-12,
            ):
                raise RuntimeError(
                    f"blind test {prefix} class-{class_id} accuracy is inconsistent"
                )

        gate = selection_record["final_gate"]
        class_id = int(gate["class_value"])
        class_mask = labels == class_id
        q = np.asarray(arrays[f"{prefix}__q"], dtype=np.float64)[class_mask]
        recomputed_hidden = evaluate_class_slice(
            task=task[class_mask],
            nuisance=nuisance[class_mask],
            logits=logits[class_mask],
            labels=labels[class_mask],
            q=q,
            class_id=class_id,
            temperature=temperature,
            config=metric_config,
            hard_mass=float(gate["constraints"]["target_mass"]),
        )
        recorded_hidden = blind_record["hidden_certificate"]
        expected_fisher = float(recomputed_hidden["predictive"]["projected_fisher"])
        expected_inverse = float(recomputed_hidden["predictive"]["inverse_fisher"])
        for name, observed, expected in (
            ("certificate", recorded_hidden["certificate"], expected_fisher),
            (
                "predictive.projected_fisher",
                recorded_hidden["predictive"]["projected_fisher"],
                expected_fisher,
            ),
            (
                "predictive.inverse_fisher",
                recorded_hidden["predictive"]["inverse_fisher"],
                expected_inverse,
            ),
        ):
            if not np.isclose(float(observed), expected, rtol=1e-9, atol=1e-12):
                raise RuntimeError(f"blind test {prefix} hidden metric {name} is inconsistent")


def _validate_selection_core(
    selection: dict[str, Any],
    *,
    seed: int,
    current_inputs: Mapping[str, str],
    validation_score_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, np.ndarray]]:
    label = f"seed {seed} selection"
    if selection.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"{label} has an unsupported schema")
    if selection.get("stage") != SELECTION_STAGE or selection.get("seed") != seed:
        raise RuntimeError(f"{label} has the wrong stage or seed")
    if selection.get("metadata_accessed") is not False or selection.get("test_data_accessed") is not False:
        raise RuntimeError(f"{label} violates the blind-selection contract")
    require_current_implementation(selection.get("implementation_manifest"), label=label)
    require_json_artifact_sha256(selection, "frozen_selection_sha256", label=label)
    if selection.get("input_hashes") != dict(current_inputs):
        raise RuntimeError(f"{label} does not match current train/validation inputs")
    protocol = selection.get("protocol")
    frozen_reference = selection.get(FROZEN_ERM_REFERENCE_KEY)
    candidates = selection.get("candidates")
    decision = selection.get("selection")
    if not isinstance(protocol, Mapping) or not isinstance(candidates, list) or not candidates:
        raise RuntimeError(f"{label} is missing its protocol or candidates")
    if not isinstance(frozen_reference, Mapping):
        raise RuntimeError(f"{label} is missing its frozen ERM reference")
    if not isinstance(decision, Mapping):
        raise RuntimeError(f"{label} is missing its decision")
    lambdas = validate_lambdas(protocol["lambdas"], fixed_lambda=float(protocol["fixed_lambda"]))
    if len(lambdas) != len(candidates):
        raise RuntimeError(f"{label} candidate count differs from the lambda grid")
    if frozen_reference.get("reference_key") != FROZEN_ERM_REFERENCE_KEY:
        raise RuntimeError(f"{label} frozen ERM reference has the wrong reference key")
    if frozen_reference.get("role") != "frozen_pre_continuation_erm_reference":
        raise RuntimeError(f"{label} frozen ERM reference has the wrong role")
    for forbidden in ("lambda_fg", "candidate_index"):
        if frozen_reference.get(forbidden) is not None:
            raise RuntimeError(
                f"{label} frozen ERM reference must not masquerade as a lambda-grid candidate"
            )
    for field in (
        "head",
        "task_mean",
        "task_std",
        "validation_metrics",
        "crossfit",
        "certificate_scores",
        "final_gate",
    ):
        if field not in frozen_reference:
            raise RuntimeError(f"{label} frozen ERM reference lacks {field}")
    base_model = selection.get("base_model")
    if not isinstance(base_model, Mapping) or frozen_reference["head"] != base_model.get("head"):
        raise RuntimeError(f"{label} frozen ERM reference differs from the base model")
    warm_start_sha256 = base_model.get("head_sha256")
    if not isinstance(warm_start_sha256, str) or frozen_reference.get(
        "warm_start_sha256"
    ) != warm_start_sha256:
        raise RuntimeError(f"{label} frozen ERM reference has the wrong warm-start digest")
    for field in ("task_mean", "task_std"):
        if not np.isclose(
            float(frozen_reference[field]),
            float(base_model[field]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"{label} frozen ERM reference {field} differs from the base model")
    for index, (candidate, expected_lambda) in enumerate(zip(candidates, lambdas)):
        if candidate.get("candidate_index") != index or not _same_lambda(candidate["lambda_fg"], expected_lambda):
            raise RuntimeError(f"{label} candidate ordering differs from the lambda grid")
        expected_role = "matched_erm_continuation" if expected_lambda == 0.0 else "fisher_repair"
        if candidate.get("role") != expected_role or candidate.get(
            "warm_start_sha256"
        ) != warm_start_sha256:
            raise RuntimeError(f"{label} candidate {index} violates matched-grid identity")
        for field in ("head", "task_mean", "task_std", "validation_metrics", "final_gate"):
            if field not in candidate:
                raise RuntimeError(f"{label} candidate {index} lacks {field}")
    selected_index = int(decision["selected_candidate_index"])
    if selected_index < 0 or selected_index >= len(candidates):
        raise RuntimeError(f"{label} selected index is invalid")
    if not _same_lambda(decision["selected_lambda"], candidates[selected_index]["lambda_fg"]):
        raise RuntimeError(f"{label} selected index and lambda disagree")
    risk_eligible = [float(value) for value in decision.get("risk_eligible_lambdas", [])]
    if not risk_eligible or any(not any(_same_lambda(value, item) for item in lambdas) for value in risk_eligible):
        raise RuntimeError(f"{label} has an invalid one-standard-error risk set")
    eligible = [float(value) for value in decision.get("eligible_lambdas", [])]
    if not eligible or any(not any(_same_lambda(value, item) for item in lambdas) for value in eligible):
        raise RuntimeError(f"{label} has an invalid primary eligible set")
    if len(eligible) != len(risk_eligible) or any(
        not any(_same_lambda(value, item) for item in risk_eligible) for value in eligible
    ):
        raise RuntimeError(f"{label} primary eligible set differs from its one-standard-error risk set")

    if not validation_score_path.is_file():
        raise FileNotFoundError(validation_score_path)
    if file_sha256(validation_score_path) != selection.get("validation_score_artifact_sha256"):
        raise RuntimeError(f"{label} validation score archive is missing or changed")
    arrays = _load_exact_npz(
        validation_score_path, _expected_validation_keys(frozen_reference, candidates)
    )
    _validate_validation_arrays(validation_score_path, arrays, frozen_reference, candidates)
    return candidates, arrays


def _validate_blind_test_core(
    blind_test: dict[str, Any],
    *,
    seed: int,
    selection: dict[str, Any],
    selection_path: Path,
    current_inputs: Mapping[str, str],
    score_path: Path,
) -> dict[str, np.ndarray]:
    label = f"seed {seed} blind test"
    if blind_test.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"{label} has an unsupported schema")
    if blind_test.get("stage") != BLIND_TEST_STAGE or blind_test.get("seed") != seed:
        raise RuntimeError(f"{label} has the wrong stage or seed")
    if blind_test.get("metadata_accessed") is not False:
        raise RuntimeError(f"{label} violates the metadata-blind contract")
    if blind_test.get("test_data_accessed_after_freeze") is not True:
        raise RuntimeError(f"{label} does not attest post-freeze test access")
    require_current_implementation(blind_test.get("implementation_manifest"), label=label)
    require_json_artifact_sha256(blind_test, "blind_test_sha256", label=label)
    if blind_test.get("input_hashes") != dict(current_inputs):
        raise RuntimeError(f"{label} does not match current blind inputs")
    if blind_test.get("selection_artifact_sha256") != file_sha256(selection_path):
        raise RuntimeError(f"{label} is not linked to the current selection artifact")
    if blind_test.get("frozen_selection_sha256") != selection["frozen_selection_sha256"]:
        raise RuntimeError(f"{label} carries the wrong frozen-selection digest")
    if blind_test.get("validation_score_artifact_sha256") != selection["validation_score_artifact_sha256"]:
        raise RuntimeError(f"{label} carries the wrong validation-score digest")
    if not score_path.is_file() or file_sha256(score_path) != blind_test.get("score_artifact_sha256"):
        raise RuntimeError(f"{label} score archive is missing or changed")

    selection_candidates = selection["candidates"]
    selection_reference = selection[FROZEN_ERM_REFERENCE_KEY]
    test_reference = blind_test.get(FROZEN_ERM_REFERENCE_KEY)
    if not isinstance(test_reference, Mapping):
        raise RuntimeError(f"{label} is missing its frozen ERM reference result")
    if test_reference.get("source") != "frozen_reference":
        raise RuntimeError(f"{label} frozen ERM reference has the wrong source identity")
    if not isinstance(test_reference.get("task_metrics"), Mapping) or not isinstance(
        test_reference.get("hidden_certificate"), Mapping
    ):
        raise RuntimeError(f"{label} frozen ERM reference result is incomplete")
    if blind_test.get("selected_candidate_source") != "lambda_grid":
        raise RuntimeError(f"{label} does not identify the selected policy as a lambda-grid candidate")
    test_candidates = blind_test.get("candidates")
    if not isinstance(test_candidates, list) or len(test_candidates) != len(selection_candidates):
        raise RuntimeError(f"{label} has the wrong candidate count")
    for expected, observed in zip(selection_candidates, test_candidates):
        if (
            observed.get("source") != "lambda_grid"
            or observed.get("candidate_index") != expected["candidate_index"]
            or not _same_lambda(observed["lambda_fg"], expected["lambda_fg"])
        ):
            raise RuntimeError(f"{label} candidate identities differ from selection")
        if not isinstance(observed.get("task_metrics"), Mapping) or not isinstance(
            observed.get("hidden_certificate"), Mapping
        ):
            raise RuntimeError(f"{label} candidate result is incomplete")
    arrays = _load_exact_npz(
        score_path, _expected_test_keys(selection_reference, selection_candidates)
    )
    _validate_test_arrays(score_path, arrays, selection_reference, selection_candidates)
    _validate_blind_json_metrics(selection, blind_test, arrays)
    return arrays


def _require_all_blind_results_frozen(
    data_dir: Path,
    out_dir: Path,
    seeds: list[int],
    *,
    smoke: bool,
) -> tuple[
    dict[int, tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], Path, Path]],
    dict[str, Any],
    dict[str, Any],
]:
    """Validate the complete multi-seed blind chain before metadata access."""

    manifest_path = data_dir / "preparation_manifest.json"
    manifest = read_json(manifest_path)
    selection_inputs = blind_input_hashes(data_dir, include_test=False)
    all_inputs = blind_input_hashes(data_dir, include_test=True)
    for name in ("train_blind", "val_blind", "test_blind"):
        expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        if all_inputs[name] != expected:
            raise RuntimeError(f"{name}.npz does not match preparation_manifest.json")

    contexts: dict[
        int,
        tuple[dict[str, Any], dict[str, Any], dict[str, np.ndarray], dict[str, np.ndarray], Path, Path],
    ] = {}
    frozen_selection_cohort: Mapping[str, Any] | None = None
    for seed in seeds:
        seed_dir = out_dir / "runs" / f"seed_{seed}"
        selection_path = seed_dir / "selection.json"
        validation_score_path = seed_dir / "validation_scores.npz"
        blind_test_path = seed_dir / "blind_test.json"
        blind_score_path = seed_dir / "blind_test_scores.npz"
        if not selection_path.is_file() or not blind_test_path.is_file():
            raise FileNotFoundError(
                f"all requested blind stages must finish before metadata access; seed {seed} is incomplete"
            )
        selection = read_json(selection_path)
        candidates, validation_arrays = _validate_selection_core(
            selection,
            seed=seed,
            current_inputs=selection_inputs,
            validation_score_path=validation_score_path,
        )
        blind_test = read_json(blind_test_path)
        certification = blind_test.get("certification_protocol")
        expected_complete = tuple(int(value) for value in seeds) == DEFAULT_SEEDS
        if not isinstance(certification, Mapping):
            raise RuntimeError(f"seed {seed}: blind test cohort attestation is absent")
        expected_certification_sha256 = canonical_protocol_hash(
            {"seed": seed, "certification_protocol": certification}
        )
        if blind_test.get("certification_protocol_sha256") != expected_certification_sha256:
            raise RuntimeError(f"seed {seed}: certification protocol digest is inconsistent")
        if certification.get("cohort_seeds") != [int(value) for value in seeds]:
            raise RuntimeError(f"seed {seed}: blind test used a different reveal cohort")
        if certification.get("complete_default_cohort_frozen_before_test") is not expected_complete:
            raise RuntimeError(f"seed {seed}: blind test has an invalid complete-cohort attestation")
        if certification.get("smoke") is not bool(smoke):
            raise RuntimeError(f"seed {seed}: blind test smoke attestation differs from post-hoc mode")
        selection_cohort = certification.get("selection_cohort")
        if not isinstance(selection_cohort, Mapping):
            raise RuntimeError(f"seed {seed}: certification lacks its frozen selection cohort")
        cohort_payload = dict(selection_cohort)
        cohort_digest = cohort_payload.pop("sha256", None)
        if (
            cohort_digest != canonical_protocol_hash(cohort_payload)
            or certification.get("selection_cohort_sha256") != cohort_digest
            or selection_cohort.get("seeds") != [int(value) for value in seeds]
        ):
            raise RuntimeError(f"seed {seed}: frozen selection cohort digest is inconsistent")
        cohort_entries = selection_cohort.get("artifacts")
        if not isinstance(cohort_entries, list) or any(
            not isinstance(entry, Mapping) for entry in cohort_entries
        ):
            raise RuntimeError(f"seed {seed}: frozen selection cohort entries are absent")
        own_entries = [entry for entry in cohort_entries if int(entry.get("seed", -1)) == seed]
        if len(own_entries) != 1 or own_entries[0] != {
            "seed": int(seed),
            "selection_artifact_sha256": file_sha256(selection_path),
            "frozen_selection_sha256": selection["frozen_selection_sha256"],
            "validation_score_artifact_sha256": file_sha256(validation_score_path),
        }:
            raise RuntimeError(f"seed {seed}: frozen selection cohort entry is stale")
        if frozen_selection_cohort is None:
            frozen_selection_cohort = selection_cohort
        elif json_sha256(frozen_selection_cohort) != json_sha256(selection_cohort):
            raise RuntimeError("blind test seeds were certified against different selection cohorts")
        test_arrays = _validate_blind_test_core(
            blind_test,
            seed=seed,
            selection=selection,
            selection_path=selection_path,
            current_inputs=all_inputs,
            score_path=blind_score_path,
        )
        if len(candidates) != len(blind_test["candidates"]):
            raise RuntimeError(f"seed {seed}: blind candidate grid changed")
        contexts[seed] = (
            selection,
            blind_test,
            validation_arrays,
            test_arrays,
            selection_path,
            blind_test_path,
        )
    if frozen_selection_cohort is None:
        raise RuntimeError("the frozen selection cohort is empty")
    expected_selection_entries = [
        {
            "seed": int(seed),
            "selection_artifact_sha256": file_sha256(contexts[seed][4]),
            "frozen_selection_sha256": contexts[seed][0]["frozen_selection_sha256"],
            "validation_score_artifact_sha256": file_sha256(
                out_dir / "runs" / f"seed_{seed}" / "validation_scores.npz"
            ),
        }
        for seed in seeds
    ]
    if frozen_selection_cohort.get("artifacts") != expected_selection_entries:
        raise RuntimeError("the frozen selection cohort does not exactly match all requested seeds")
    blind_cohort = {
        "seeds": [int(seed) for seed in seeds],
        "selection_cohort_sha256": str(frozen_selection_cohort["sha256"]),
        "artifacts": [
            {
                "seed": int(seed),
                "blind_test_artifact_sha256": file_sha256(contexts[seed][5]),
                "blind_test_sha256": contexts[seed][1]["blind_test_sha256"],
                "blind_test_score_artifact_sha256": file_sha256(
                    out_dir / "runs" / f"seed_{seed}" / "blind_test_scores.npz"
                ),
            }
            for seed in seeds
        ],
    }
    blind_cohort["sha256"] = canonical_protocol_hash(blind_cohort)
    return contexts, manifest, blind_cohort


def _verify_score_rows(
    arrays: Mapping[str, np.ndarray],
    split: Any,
    *,
    label: str,
) -> None:
    if not np.array_equal(np.asarray(arrays["row_id"], dtype=np.int64), np.asarray(split.row_id, dtype=np.int64)):
        raise RuntimeError(f"{label} row ids do not match the prepared blind split")
    if not np.array_equal(np.asarray(arrays["y"], dtype=np.int64), np.asarray(split.y, dtype=np.int64)):
        raise RuntimeError(f"{label} labels do not match the prepared blind split")


def _verify_blind_score_semantics(
    selection: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    split: Any,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    """Bind every blind-test score vector to its frozen head and gate."""

    normalization = selection["normalization"]
    features = apply_feature_normalizer(
        split.h,
        np.asarray(normalization["feature_mean"], dtype=np.float32),
        np.asarray(normalization["feature_std"], dtype=np.float32),
    )
    nuisance = (
        np.asarray(split.b_vl, dtype=np.float64)
        - float(normalization["nuisance_mean"])
    ) / float(normalization["nuisance_std"])
    if not np.allclose(
        np.asarray(arrays["nuisance"], dtype=np.float64),
        nuisance,
        rtol=1e-10,
        atol=1e-10,
    ):
        raise RuntimeError(f"{label} nuisance coordinate does not match the prepared blind split")

    labels = np.asarray(split.y, dtype=np.int64)
    temperature = float(normalization["base_temperature"])
    expected: dict[str, np.ndarray] = {
        "row_id": np.asarray(split.row_id, dtype=np.int64),
        "y": labels,
        "nuisance": np.asarray(nuisance, dtype=np.float64),
    }
    for prefix, record in _evaluation_records(
        selection[FROZEN_ERM_REFERENCE_KEY], selection["candidates"]
    ):
        head = FrozenHead.from_dict(record["head"])
        bundle = make_model_coordinates(
            head,
            features,
            labels,
            nuisance,
            task_mean=float(record["task_mean"]),
            task_std=float(record["task_std"]),
            temperature=temperature,
        )
        if not np.allclose(
            np.asarray(arrays[f"{prefix}__logits"], dtype=np.float64),
            bundle.logits,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(f"{label} {prefix} logits do not match its frozen head")
        if not np.allclose(
            np.asarray(arrays[f"{prefix}__task"], dtype=np.float64),
            bundle.task,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(f"{label} {prefix} task coordinate does not match its frozen scale")
        expected[f"{prefix}__logits"] = np.asarray(bundle.logits, dtype=np.float64)
        expected[f"{prefix}__task"] = np.asarray(bundle.task, dtype=np.float64)

        gate = FrozenLinearGate.from_dict(record["final_gate"])
        class_mask = labels == gate.class_value
        class_features = torch.as_tensor(
            bundle.features[class_mask], dtype=torch.float32
        )
        membership = (
            gate.membership(class_features)
            .detach()
            .cpu()
            .numpy()
            .astype(np.float64)
        )
        q_tensor, _diagnostics = admissible_weights(
            torch.as_tensor(membership, dtype=torch.float64), gate.constraints
        )
        q = canonical_probability_weights(
            q_tensor.detach().cpu().numpy(), name=f"{label} {prefix} gate weights"
        )
        recorded_membership = np.asarray(
            arrays[f"{prefix}__membership"], dtype=np.float64
        )[class_mask]
        if not np.allclose(
            recorded_membership, membership, rtol=1e-7, atol=1e-8
        ):
            raise RuntimeError(f"{label} {prefix} membership does not match its frozen gate")
        expected_q = np.zeros(labels.size, dtype=np.float64)
        expected_q[class_mask] = q
        if not np.allclose(
            np.asarray(arrays[f"{prefix}__q"], dtype=np.float64),
            expected_q,
            rtol=1e-7,
            atol=1e-10,
        ):
            raise RuntimeError(f"{label} {prefix} weights do not match its frozen gate")
        expected_membership = np.full(labels.size, np.nan, dtype=np.float64)
        expected_membership[class_mask] = membership
        expected[f"{prefix}__q"] = expected_q
        expected[f"{prefix}__membership"] = expected_membership
    return expected


def _derived_groups(
    labels: np.ndarray,
    joined: Mapping[str, np.ndarray],
    *,
    label: str,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    background = np.asarray(joined["background"], dtype=np.int64)
    recorded = np.asarray(joined["group"], dtype=np.int64)
    if not np.all(np.isin(background, (0, 1))):
        raise RuntimeError(f"{label} background values are not binary")
    derived = 2 * y + background
    if not np.array_equal(recorded, derived):
        raise RuntimeError(f"{label} sealed group is inconsistent with y-by-background")
    return derived


def _validate_sealed_arrays(sealed: Mapping[str, np.ndarray]) -> None:
    row_id = np.asarray(sealed["row_id"])
    if row_id.dtype.kind not in "iu" or row_id.ndim != 1 or row_id.size == 0:
        raise RuntimeError("sealed metadata row ids must be a non-empty integer vector")
    for name, values in sealed.items():
        array = np.asarray(values)
        if array.ndim != 1 or array.size != row_id.size:
            raise RuntimeError(f"sealed metadata field {name!r} is not aligned with row_id")
    for name, allowed in (("background", (0, 1)), ("group", (0, 1, 2, 3))):
        array = np.asarray(sealed[name])
        if array.dtype.kind not in "iu" or not np.all(np.isin(array, allowed)):
            raise RuntimeError(f"sealed metadata field {name!r} has invalid values")


def _group_audit(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    metrics = group_metrics(logits, labels, groups)
    prediction = (np.asarray(logits, dtype=np.float64) >= 0.0).astype(np.int64)
    metrics["average_accuracy"] = float(np.mean(prediction == np.asarray(labels, dtype=np.int64)))
    metrics["mean_group_accuracy"] = float(np.mean([row["accuracy"] for row in metrics["groups"]]))
    metrics["definition"] = "four post-hoc y-by-background groups"
    return metrics


def _run_seed(
    args: argparse.Namespace,
    *,
    seed: int,
    context: tuple[
        dict[str, Any],
        dict[str, Any],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        Path,
        Path,
    ],
    val: Any,
    test: Any,
    val_groups: np.ndarray,
    test_groups: np.ndarray,
    sealed_sha256: str,
    blind_cohort: Mapping[str, Any],
) -> dict[str, Any]:
    selection, blind_test, validation_arrays, test_arrays, selection_path, blind_test_path = context
    seed_dir = Path(args.out_dir) / "runs" / f"seed_{seed}"
    output_path = seed_dir / "posthoc.json"
    protocol = {
        "selection_artifact_sha256": file_sha256(selection_path),
        "frozen_selection_sha256": selection["frozen_selection_sha256"],
        "validation_score_artifact_sha256": selection["validation_score_artifact_sha256"],
        "blind_test_artifact_sha256": file_sha256(blind_test_path),
        "blind_test_sha256": blind_test["blind_test_sha256"],
        "blind_test_score_artifact_sha256": blind_test["score_artifact_sha256"],
        "sealed_metadata_sha256": sealed_sha256,
        "cohort_seeds": [int(value) for value in args.seeds],
        "complete_default_cohort_frozen_before_metadata": bool(
            tuple(int(value) for value in args.seeds) == DEFAULT_SEEDS
        ),
        "blind_cohort": dict(blind_cohort),
        "blind_cohort_sha256": str(blind_cohort["sha256"]),
        "smoke": bool(args.smoke),
        "comparator_candidate_set": (
            "selection.selection.eligible_lambdas (the frozen one-standard-error risk set)"
        ),
        "comparator_tie_break": "smallest lambda_fg among equal validation worst-group accuracies",
        "candidate_schema": (
            "frozen_erm_reference is audited separately; comparator and selected policies use only "
            "the matched lambda grid"
        ),
        "metadata_role": "post-hoc reporting and explicitly labeled group-supervised validation comparator only",
    }
    protocol_sha256 = canonical_protocol_hash({"seed": seed, "posthoc_protocol": protocol})
    existing_result: dict[str, Any] | None = None
    if args.resume and output_path.is_file():
        existing = read_json(output_path)
        if existing.get("stage") != POSTHOC_STAGE or existing.get("seed") != seed:
            raise RuntimeError(f"seed {seed}: malformed post-hoc artifact")
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"seed {seed}: unsupported post-hoc schema")
        if existing.get("posthoc_protocol_sha256") != protocol_sha256:
            raise RuntimeError(f"seed {seed}: refusing to resume under a changed post-hoc protocol")
        if existing.get("metadata_accessed_after_blind_freeze") is not True:
            raise RuntimeError(f"seed {seed}: post-hoc artifact lacks the metadata timing attestation")
        require_current_implementation(existing.get("implementation_manifest"), label=f"seed {seed} post-hoc")
        require_json_artifact_sha256(existing, "posthoc_sha256", label=f"seed {seed} post-hoc")
        existing_reference = existing.get(FROZEN_ERM_REFERENCE_KEY)
        if not isinstance(existing_reference, Mapping) or existing_reference.get("source") != "frozen_reference":
            raise RuntimeError(f"seed {seed}: post-hoc frozen ERM reference is absent or mislabeled")
        existing_policies = existing.get("policies")
        if not isinstance(existing_policies, Mapping) or set(existing_policies) != set(POLICY_SOURCES):
            raise RuntimeError(f"seed {seed}: post-hoc policy schema is incomplete")
        for policy_key, source in POLICY_SOURCES.items():
            if existing_policies[policy_key].get("source") != source:
                raise RuntimeError(f"seed {seed}: post-hoc policy {policy_key} has the wrong source")
        print(
            f"seed {seed}: post-hoc artifact is linked; recomputing its deterministic group audit",
            flush=True,
        )
        existing_result = existing

    blind_reference = blind_test[FROZEN_ERM_REFERENCE_KEY]
    frozen_reference_audit = {
        "source": "frozen_reference",
        "validation_group_metrics": _group_audit(
            validation_arrays[f"{FROZEN_ERM_REFERENCE_KEY}__logits"], val.y, val_groups
        ),
        "test_group_metrics": _group_audit(
            test_arrays[f"{FROZEN_ERM_REFERENCE_KEY}__logits"], test.y, test_groups
        ),
        "blind_test_candidate": {
            "task_metrics": blind_reference["task_metrics"],
            "hidden_certificate": blind_reference["hidden_certificate"],
        },
    }

    blind_by_index = {int(row["candidate_index"]): row for row in blind_test["candidates"]}
    candidate_audits: list[dict[str, Any]] = []
    for candidate in selection["candidates"]:
        index = int(candidate["candidate_index"])
        coefficient = float(candidate["lambda_fg"])
        prefix = lambda_key(coefficient)
        blind_candidate = blind_by_index[index]
        candidate_audits.append(
            {
                "source": "lambda_grid",
                "candidate_index": index,
                "lambda_fg": coefficient,
                "validation_group_metrics": _group_audit(
                    validation_arrays[f"{prefix}__logits"], val.y, val_groups
                ),
                "test_group_metrics": _group_audit(test_arrays[f"{prefix}__logits"], test.y, test_groups),
                "blind_test_candidate": {
                    "task_metrics": blind_candidate["task_metrics"],
                    "hidden_certificate": blind_candidate["hidden_certificate"],
                },
            }
        )

    decision = selection["selection"]
    comparator_lambdas = sorted(float(value) for value in decision["eligible_lambdas"])
    comparator_pool = [
        row
        for row in candidate_audits
        if any(_same_lambda(row["lambda_fg"], value) for value in comparator_lambdas)
    ]
    if not comparator_pool:
        raise RuntimeError(f"seed {seed}: group-supervised comparator has an empty predeclared candidate set")
    comparator = sorted(
        comparator_pool,
        key=lambda row: (-float(row["validation_group_metrics"]["worst_group_accuracy"]), row["lambda_fg"]),
    )[0]

    def policy(
        key: str,
        candidate: Mapping[str, Any],
        *,
        source: str,
        basis: str,
        group_labels_used: bool,
    ) -> dict[str, Any]:
        if source not in {"frozen_reference", "lambda_grid"}:
            raise ValueError(f"unsupported policy source {source!r}")
        if candidate.get("source") != source:
            raise RuntimeError(f"seed {seed}: policy {key} source/candidate mismatch")
        result: dict[str, Any] = {
            "key": key,
            "source": source,
            "lambda_fg": (
                None if source == "frozen_reference" else float(candidate["lambda_fg"])
            ),
            "selection_basis": basis,
            "group_labels_used_for_selection": bool(group_labels_used),
            "validation_group_metrics": candidate["validation_group_metrics"],
            "test_group_metrics": candidate["test_group_metrics"],
            "blind_test_candidate": candidate["blind_test_candidate"],
        }
        return result

    fixed_lambda = float(selection["protocol"]["fixed_lambda"])
    matched_zero = _candidate_for_lambda(candidate_audits, 0.0)
    fixed_candidate = _candidate_for_lambda(candidate_audits, fixed_lambda)
    risk_candidate = _candidate_for_lambda(
        candidate_audits, float(decision["validation_risk_minimizer_lambda"])
    )
    group_free_candidate = _candidate_for_lambda(
        candidate_audits, float(decision["selected_lambda"])
    )
    policies = {
        "frozen_erm_reference": policy(
            "frozen_erm_reference",
            frozen_reference_audit,
            source="frozen_reference",
            basis="predeclared frozen ERM reference; no continuation and no repair",
            group_labels_used=False,
        ),
        "matched_erm_lambda_0": policy(
            "matched_erm_lambda_0",
            matched_zero,
            source="lambda_grid",
            basis="matched continuation candidate with lambda=0 and no Fisher penalty",
            group_labels_used=False,
        ),
        "fixed_0.03": policy(
            "fixed_0.03",
            fixed_candidate,
            source="lambda_grid",
            basis=f"predeclared fixed lambda={fixed_lambda:g} reference on the hidden-slice penalty scale",
            group_labels_used=False,
        ),
        "validation_risk_minimizer": policy(
            "validation_risk_minimizer",
            risk_candidate,
            source="lambda_grid",
            basis="minimum ordinary validation NLL; task labels only",
            group_labels_used=False,
        ),
        "group_free": policy(
            "group_free",
            group_free_candidate,
            source="lambda_grid",
            basis=(
                "primary frozen rule: highest Bonferroni-adjusted cross-fit hidden-certificate lower "
                "score within the one-standard-error validation-NLL risk set; ties choose smaller lambda"
            ),
            group_labels_used=False,
        ),
        "group_supervised_val_comparator": policy(
            "group_supervised_val_comparator",
            comparator,
            source="lambda_grid",
            basis=(
                "post-hoc group-supervised comparator maximizing validation worst-group accuracy over "
                "the same one-standard-error risk set as the primary rule; ties choose the smallest lambda"
            ),
            group_labels_used=True,
        ),
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": POSTHOC_STAGE,
        "seed": seed,
        "metadata_accessed_after_blind_freeze": True,
        "primary_selection_unchanged": True,
        "implementation_manifest": implementation_manifest(),
        "selection_artifact_sha256": file_sha256(selection_path),
        "frozen_selection_sha256": selection["frozen_selection_sha256"],
        "blind_test_artifact_sha256": file_sha256(blind_test_path),
        "blind_test_sha256": blind_test["blind_test_sha256"],
        "posthoc_protocol": protocol,
        "posthoc_protocol_sha256": protocol_sha256,
        FROZEN_ERM_REFERENCE_KEY: frozen_reference_audit,
        "candidates": candidate_audits,
        "policies": policies,
        "comparator_disclosure": (
            "group_supervised_val_comparator uses validation y-by-background labels. It is not an upper "
            "bound, does not alter the primary group-free choice, and uses no test group label to choose "
            "a policy."
        ),
    }
    result["posthoc_sha256"] = json_artifact_sha256(result, "posthoc_sha256")
    if existing_result is not None:
        if json_sha256(existing_result) != json_sha256(result):
            raise RuntimeError(
                f"seed {seed}: resumed post-hoc content differs from its deterministic recomputation"
            )
        print(f"seed {seed}: post-hoc group audit already complete and verified", flush=True)
        return existing_result
    atomic_write_json(output_path, result)
    print(f"seed {seed}: wrote post-hoc group-label audit {output_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="build/results/waterbirds_hidden_slice_v2/prepared")
    parser.add_argument("--out-dir", default="build/results/waterbirds_group_free_lambda_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="permit an incomplete seed cohort and attest the metadata reveal as non-reportable",
    )
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be unique")
    if tuple(int(value) for value in args.seeds) != DEFAULT_SEEDS and not args.smoke:
        raise ValueError(
            "metadata reveal requires the complete default seed cohort; "
            "use --smoke only for explicitly non-reportable plumbing checks"
        )
    data_dir = Path(args.data_dir)
    contexts, preparation_manifest, blind_cohort = _require_all_blind_results_frozen(
        data_dir,
        Path(args.out_dir),
        args.seeds,
        smoke=bool(args.smoke),
    )

    # Blind split/score alignment is checked for all seeds before metadata is
    # opened.  Reading these exact-schema archives cannot reveal background.
    val = load_blind_npz(data_dir / "val_blind.npz")
    test = load_blind_npz(data_dir / "test_blind.npz")
    for seed, context in contexts.items():
        _verify_score_rows(context[2], val, label=f"seed {seed} validation scores")
        _verify_score_rows(context[3], test, label=f"seed {seed} blind test scores")
        recomputed_test_arrays = _verify_blind_score_semantics(
            context[0],
            context[3],
            test,
            label=f"seed {seed} blind test scores",
        )
        _validate_test_arrays(
            Path(context[1]["score_artifact"]),
            recomputed_test_arrays,
            context[0][FROZEN_ERM_REFERENCE_KEY],
            context[0]["candidates"],
        )
        _validate_blind_json_metrics(context[0], context[1], recomputed_test_arrays)
        contexts[seed] = (
            context[0],
            context[1],
            context[2],
            recomputed_test_arrays,
            context[4],
            context[5],
        )

    sealed_path = data_dir / "sealed_metadata.npz"
    expected_sealed_sha256 = preparation_manifest.get("artifacts", {}).get("sealed_metadata", {}).get("sha256")
    if not isinstance(expected_sealed_sha256, str):
        raise RuntimeError("preparation_manifest.json does not attest sealed_metadata.npz")
    sealed = load_sealed_metadata(sealed_path, expected_sha256=expected_sealed_sha256)
    _validate_sealed_arrays(sealed)
    val_metadata = join_sealed(val.row_id, sealed)
    test_metadata = join_sealed(test.row_id, sealed)
    val_groups = _derived_groups(val.y, val_metadata, label="validation")
    test_groups = _derived_groups(test.y, test_metadata, label="test")

    for seed in args.seeds:
        _run_seed(
            args,
            seed=seed,
            context=contexts[seed],
            val=val,
            test=test,
            val_groups=val_groups,
            test_groups=test_groups,
            sealed_sha256=expected_sealed_sha256,
            blind_cohort=blind_cohort,
        )


if __name__ == "__main__":
    main()
