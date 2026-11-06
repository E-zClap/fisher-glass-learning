"""Evaluate the frozen ERM reference and matched lambda grid on blind Waterbirds test data.

The stage is deliberately fail closed.  It validates every requested seed,
selection digest, validation-score archive, scientific source manifest, and
train/validation input hash before it even hashes ``test_blind.npz``.  The
Waterbirds background/group metadata is never opened here.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np
import torch

from experiment_common import (
    DEFAULT_SEEDS,
    FROZEN_ERM_REFERENCE_KEY,
    SCHEMA_VERSION,
    atomic_savez_compressed,
    atomic_write_json,
    blind_input_hashes,
    canonical_probability_weights,
    canonical_protocol_hash,
    crossfit_certificate_scores,
    derive_seed,
    file_sha256,
    implementation_manifest,
    json_artifact_sha256,
    json_sha256,
    lambda_key,
    load_blind_npz,
    read_json,
    require_current_implementation,
    require_json_artifact_sha256,
    risk_certificate_choice,
    task_metrics,
    validate_lambdas,
)
from blind_protocol import evaluate_gate, make_model_coordinates
from common import stratified_kfold_indices
from compat import BooleanOptionalAction
from gates import FrozenLinearGate, admissible_weights
from heads import FrozenHead, apply_feature_normalizer, per_example_loss
from metrics import FisherMetricConfig, bootstrap_fixed_gate_fisher, evaluate_class_slice


SELECTION_STAGE = "metadata_blind_lambda_selection"
BLIND_TEST_STAGE = "blind_heldout_test_certification"


def _finite_scalar(value: Any, name: str, *, positive: bool = False) -> float:
    result = float(value)
    if not np.isfinite(result) or (positive and result <= 0.0):
        qualifier = "positive " if positive else ""
        raise ValueError(f"{name} must be a finite {qualifier}scalar")
    return result


def _fisher_config(protocol: Mapping[str, Any]) -> FisherMetricConfig:
    payload = protocol.get("fisher_metric")
    if payload is None:
        payload = protocol.get("metric")
    if payload is None:
        payload = protocol.get("gate")
    if not isinstance(payload, Mapping):
        # Forward-compatible with an earlier flat draft of the protocol.
        payload = protocol
    config = FisherMetricConfig(
        shrinkage=_finite_scalar(payload["shrinkage"], "shrinkage"),
        ridge=_finite_scalar(payload["ridge"], "ridge", positive=True),
        epsilon=_finite_scalar(payload["epsilon"], "epsilon", positive=True),
    )
    config.validate()
    return config


def _evaluation_records(
    frozen_reference: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> list[tuple[str, Mapping[str, Any]]]:
    """Return the frozen reference and matched lambda grid without aliasing lambda zero."""

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


def _load_validation_scores(
    path: Path,
    frozen_reference: Mapping[str, Any],
    candidates: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    expected = _expected_validation_keys(frozen_reference, candidates)
    with np.load(path, allow_pickle=False) as archive:
        actual = set(archive.files)
        if actual != expected:
            raise RuntimeError(
                f"{path} has the wrong schema; missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    n = arrays["row_id"].size
    if any(array.ndim != 1 or array.size != n for array in arrays.values()):
        raise RuntimeError(f"{path} must contain aligned one-dimensional arrays")
    if np.unique(np.asarray(arrays["row_id"], dtype=np.int64)).size != n:
        raise RuntimeError(f"{path} contains duplicate row ids")
    if np.asarray(arrays["row_id"]).dtype.kind not in "iu" or np.asarray(arrays["y"]).dtype.kind not in "iu":
        raise RuntimeError(f"{path} row ids and labels must use integer dtypes")
    for name, array in arrays.items():
        if name != "row_id" and not np.all(np.isfinite(array)):
            raise RuntimeError(f"{path}: {name} contains non-finite values")
    return arrays


def _stratified_fold_assignment(labels: np.ndarray, folds: int, seed: int) -> np.ndarray:
    assignment = np.full(np.asarray(labels).size, -1, dtype=np.int64)
    for fold_index, (_discover, heldout) in enumerate(
        stratified_kfold_indices(labels, folds, seed)
    ):
        if np.any(assignment[heldout] != -1):
            raise RuntimeError("recomputed cross-fit held-out folds overlap")
        assignment[heldout] = fold_index
    if np.any(assignment < 0):
        raise RuntimeError("recomputed cross-fit held-out folds do not cover validation")
    return assignment


def _require_nested_close(
    recorded: Any,
    recomputed: Any,
    *,
    label: str,
) -> None:
    """Compare a replayed held-out metric tree with modest device tolerance."""

    if isinstance(recomputed, Mapping):
        if not isinstance(recorded, Mapping) or set(recorded) != set(recomputed):
            raise RuntimeError(f"{label} has a different metric schema")
        for key, value in recomputed.items():
            _require_nested_close(recorded[key], value, label=f"{label}.{key}")
        return
    if recomputed is None or isinstance(recomputed, (str, bool)):
        if recorded != recomputed:
            raise RuntimeError(f"{label} is inconsistent")
        return
    if isinstance(recomputed, (int, np.integer)):
        if int(recorded) != int(recomputed):
            raise RuntimeError(f"{label} is inconsistent")
        return
    try:
        matches = np.isclose(
            float(recorded), float(recomputed), rtol=5e-5, atol=1e-8
        )
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{label} is not numeric") from exc
    if not bool(matches):
        raise RuntimeError(f"{label} is inconsistent with replayed validation evidence")


def _validate_crossfit_evidence(
    record: Mapping[str, Any],
    bundle: Any,
    fold_pairs: list[tuple[np.ndarray, np.ndarray]],
    metric_config: FisherMetricConfig,
    crossfit_gate_seed: int,
    gate_config_seed: int,
    gate_protocol: Mapping[str, Any],
    *,
    label: str,
) -> None:
    """Replay every selected fold gate on its deterministic held-out rows."""

    crossfit = record.get("crossfit")
    if not isinstance(crossfit, Mapping):
        raise RuntimeError(f"{label} lacks cross-fit evidence")
    if crossfit.get("heldout_evaluation_device") != "cpu":
        raise RuntimeError(f"{label} cross-fit evidence was not evaluated on CPU")
    rows = crossfit.get("folds")
    if not isinstance(rows, list) or len(rows) != len(fold_pairs):
        raise RuntimeError(f"{label} has the wrong number of cross-fit evidence rows")
    config = crossfit.get("config")
    if not isinstance(config, Mapping) or not isinstance(config.get("constraints"), Mapping):
        raise RuntimeError(f"{label} lacks its cross-fit gate configuration")
    expected_config = {
        "objective": "fisher",
        "steps": int(gate_protocol["steps"]),
        "learning_rate": float(gate_protocol["learning_rate"]),
        "restarts": int(gate_protocol["restarts"]),
        "weight_decay": float(gate_protocol["weight_decay"]),
        "mass_penalty_strength": float(gate_protocol["mass_penalty_strength"]),
        "eta": float(gate_protocol["shrinkage"]),
        "ridge": float(gate_protocol["ridge"]),
        "eps": float(gate_protocol["epsilon"]),
        "use_predictive_fisher": True,
        "seed": int(gate_config_seed),
    }
    for field, expected in expected_config.items():
        observed = config.get(field)
        if isinstance(expected, float):
            matches = np.isclose(float(observed), expected, rtol=0.0, atol=1e-12)
        else:
            matches = observed == expected
        if not bool(matches):
            raise RuntimeError(f"{label} cross-fit configuration {field} is inconsistent")
    expected_constraints = {
        "target_mass": float(gate_protocol["slice_mass"]),
        "minimum_ess_fraction": float(gate_protocol["minimum_ess_fraction"]),
        "max_weight_multiplier": float(gate_protocol["maximum_weight_multiplier"]),
    }
    for field, expected in expected_constraints.items():
        if not np.isclose(
            float(config["constraints"].get(field, np.nan)),
            expected,
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(f"{label} cross-fit constraint {field} is inconsistent")
    for fold_index, (row, (discover_indices, heldout_indices)) in enumerate(
        zip(rows, fold_pairs)
    ):
        if not isinstance(row, Mapping) or int(row.get("fold", -1)) != fold_index:
            raise RuntimeError(f"{label} fold evidence is not canonically ordered")
        if (
            int(row.get("n_cert", -1)) != heldout_indices.size
            or int(row.get("n_discover", -1)) != discover_indices.size
        ):
            raise RuntimeError(f"{label} fold {fold_index} row counts are inconsistent")
        gate_payload = row.get("gate")
        if not isinstance(gate_payload, Mapping):
            raise RuntimeError(f"{label} fold {fold_index} lacks its frozen selected gate")
        gate = FrozenLinearGate.from_dict(gate_payload)
        if (
            gate.feature_dim != bundle.features.shape[1]
            or
            gate.objective != config.get("objective")
            or dict(gate_payload.get("constraints", {})) != dict(config["constraints"])
        ):
            raise RuntimeError(f"{label} fold {fold_index} gate/configuration differs")
        discovery = row.get("discovery")
        if not isinstance(discovery, Mapping) or int(
            discovery.get("selected_class", -1)
        ) != gate.class_value:
            raise RuntimeError(f"{label} fold {fold_index} gate/discovery class differs")
        if discovery.get("objective") != config["objective"]:
            raise RuntimeError(f"{label} fold {fold_index} discovery objective differs")
        expected_discovery_seed = derive_seed(
            crossfit_gate_seed, str(config["objective"]), "fold", fold_index
        )
        if int(discovery.get("seed", -1)) != expected_discovery_seed:
            raise RuntimeError(f"{label} fold {fold_index} discovery seed is inconsistent")
        replayed, _arrays = evaluate_gate(
            gate,
            bundle.take(heldout_indices),
            metric_config,
            device="cpu",
        )
        _require_nested_close(
            row.get("heldout"),
            replayed,
            label=f"{label}.fold[{fold_index}].heldout",
        )


def _validate_selection(
    selection: dict[str, Any],
    *,
    seed: int,
    current_inputs: Mapping[str, str],
    validation_score_path: Path,
    validation_split: Any | None = None,
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
        raise RuntimeError(f"{label} does not match the current train/validation inputs")

    protocol = selection.get("protocol")
    normalization = selection.get("normalization")
    frozen_reference = selection.get(FROZEN_ERM_REFERENCE_KEY)
    candidates = selection.get("candidates")
    decision = selection.get("selection")
    if not isinstance(protocol, Mapping) or not isinstance(normalization, Mapping):
        raise RuntimeError(f"{label} is missing its protocol or normalization")
    if not isinstance(frozen_reference, Mapping):
        raise RuntimeError(f"{label} is missing its frozen ERM reference")
    if not isinstance(candidates, list) or not candidates or not isinstance(decision, Mapping):
        raise RuntimeError(f"{label} is missing candidates or its frozen decision")
    lambdas = validate_lambdas(protocol["lambdas"], fixed_lambda=float(protocol["fixed_lambda"]))
    if len(candidates) != len(lambdas):
        raise RuntimeError(f"{label} candidate count differs from its predeclared grid")

    feature_mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(normalization["feature_std"], dtype=np.float32)
    if (
        feature_mean.ndim != 2
        or feature_mean.shape[0] != 1
        or feature_mean.shape[1] == 0
        or feature_std.shape != feature_mean.shape
        or not np.all(np.isfinite(feature_mean))
        or not np.all(np.isfinite(feature_std))
        or np.any(feature_std <= 0.0)
    ):
        raise RuntimeError(f"{label} has invalid feature normalization")
    _finite_scalar(normalization["nuisance_mean"], "nuisance_mean")
    _finite_scalar(normalization["nuisance_std"], "nuisance_std", positive=True)
    _finite_scalar(normalization["base_temperature"], "base_temperature", positive=True)
    metric_config = _fisher_config(protocol)

    # The frozen ERM reference is a separately identified reporting baseline.
    # It is deliberately not a lambda-grid candidate and never enters
    # ``risk_certificate_choice`` below.
    if frozen_reference.get("reference_key") != FROZEN_ERM_REFERENCE_KEY:
        raise RuntimeError(f"{label} frozen ERM reference has the wrong reference key")
    if frozen_reference.get("role") != "frozen_pre_continuation_erm_reference":
        raise RuntimeError(f"{label} frozen ERM reference has the wrong role")
    if "excluded from selection" not in str(
        frozen_reference.get("certificate_score_role", "")
    ):
        raise RuntimeError(f"{label} frozen ERM reference lacks its diagnostic-only score role")
    for forbidden in ("lambda_fg", "candidate_index"):
        if frozen_reference.get(forbidden) is not None:
            raise RuntimeError(
                f"{label} frozen ERM reference must not masquerade as a lambda-grid candidate"
            )
    reference_head = FrozenHead.from_dict(frozen_reference["head"])
    reference_gate = FrozenLinearGate.from_dict(frozen_reference["final_gate"])
    feature_dim = int(feature_mean.shape[1])
    if reference_head.weight.size != feature_dim or reference_gate.feature_dim != feature_dim:
        raise RuntimeError(f"{label} frozen ERM reference has the wrong feature dimension")
    _finite_scalar(frozen_reference["task_mean"], "frozen reference task_mean")
    _finite_scalar(
        frozen_reference["task_std"], "frozen reference task_std", positive=True
    )
    for field in ("validation_metrics", "crossfit", "certificate_scores"):
        if not isinstance(frozen_reference.get(field), Mapping):
            raise RuntimeError(f"{label} frozen ERM reference lacks {field}")
    if len(frozen_reference["crossfit"].get("folds", [])) != int(protocol["folds"]):
        raise RuntimeError(f"{label} frozen ERM reference has the wrong cross-fit fold count")
    reference_scores = crossfit_certificate_scores(
        frozen_reference["crossfit"],
        confidence=float(protocol["confidence"]),
        comparisons=len(lambdas),
    )
    if json_sha256(frozen_reference["certificate_scores"]) != json_sha256(reference_scores):
        raise RuntimeError(f"{label} frozen ERM reference certificate scores are inconsistent")
    base_model = selection.get("base_model")
    if not isinstance(base_model, Mapping) or not isinstance(base_model.get("head"), Mapping):
        raise RuntimeError(f"{label} is missing its frozen base model")
    if json_sha256(frozen_reference["head"]) != json_sha256(base_model["head"]):
        raise RuntimeError(f"{label} frozen ERM reference head differs from the base model")
    warm_start_sha256 = json_sha256(base_model["head"])
    if base_model.get("head_sha256") != warm_start_sha256:
        raise RuntimeError(f"{label} base-model head digest is inconsistent")
    if frozen_reference.get("warm_start_sha256") != warm_start_sha256:
        raise RuntimeError(f"{label} frozen ERM reference carries the wrong warm-start digest")
    for field in ("task_mean", "task_std"):
        if not np.isclose(
            float(frozen_reference[field]),
            float(base_model[field]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise RuntimeError(
                f"{label} frozen ERM reference {field} differs from the base model"
            )

    matched_randomness = selection.get("matched_randomness")
    if not isinstance(matched_randomness, Mapping):
        raise RuntimeError(f"{label} lacks its matched-randomness record")
    expected_randomness = {
        "train_partition_seed": derive_seed(
            seed, "group-free-lambda", "train-partition"
        ),
        "candidate_head_training_seed": derive_seed(
            seed, "group-free-lambda", "fisher-repair-head"
        ),
        "candidate_crossfit_gate_seed": derive_seed(
            seed, "group-free-lambda", "candidate-crossfit-gates"
        ),
        "validation_crossfit_partition_seed": derive_seed(
            seed, "group-free-lambda", "validation-crossfit-partition"
        ),
        "candidate_final_gate_seed": derive_seed(
            seed, "group-free-lambda", "candidate-final-gates"
        ),
    }
    for field, expected_seed in expected_randomness.items():
        if int(matched_randomness.get(field, -1)) != expected_seed:
            raise RuntimeError(f"{label} matched randomness {field} is inconsistent")
    crossfit_gate_seed = expected_randomness["candidate_crossfit_gate_seed"]
    crossfit_partition_seed = expected_randomness[
        "validation_crossfit_partition_seed"
    ]
    gate_config_seed = derive_seed(
        seed, "group-free-lambda", "candidate-gate-config"
    )

    matched_training_seed: int | None = None
    matched_head_config_sha256: str | None = None
    for index, (candidate, expected_lambda) in enumerate(zip(candidates, lambdas)):
        if not isinstance(candidate, dict):
            raise RuntimeError(f"{label} candidate {index} is malformed")
        if candidate.get("candidate_index") != index:
            raise RuntimeError(f"{label} candidate indices are not canonical")
        coefficient = _finite_scalar(candidate["lambda_fg"], "lambda_fg")
        if coefficient < 0.0 or not np.isclose(coefficient, expected_lambda, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{label} candidate grid is inconsistent")
        expected_role = "matched_erm_continuation" if coefficient == 0.0 else "fisher_repair"
        if candidate.get("role") != expected_role:
            raise RuntimeError(f"{label} candidate {index} has the wrong matched-grid role")
        if candidate.get("warm_start_sha256") != warm_start_sha256:
            raise RuntimeError(f"{label} candidate {index} does not share the frozen warm start")
        training = candidate.get("training")
        if not isinstance(training, Mapping):
            raise RuntimeError(f"{label} candidate {index} lacks its matched-training record")
        training_config = training.get("config")
        training_final = training.get("final")
        if not isinstance(training_config, Mapping) or not isinstance(training_final, Mapping):
            raise RuntimeError(f"{label} candidate {index} has a malformed training record")
        expected_mode = "erm" if coefficient == 0.0 else "fisher_repair"
        if training_config.get("mode") != expected_mode:
            raise RuntimeError(f"{label} candidate {index} used the wrong training mode")
        repair = training_config.get("repair")
        if coefficient == 0.0:
            if repair is not None:
                raise RuntimeError(f"{label} matched lambda=0 must use ordinary ERM continuation")
        elif (
            not isinstance(repair, Mapping)
            or repair.get("name") != "fisher_repair"
            or not np.isclose(
                float(repair.get("lambda_fisher", np.nan)),
                coefficient,
                rtol=0.0,
                atol=1e-12,
            )
            or not np.isclose(
                float(repair.get("lambda_loss", np.nan)),
                0.0,
                rtol=0.0,
                atol=1e-12,
            )
        ):
            raise RuntimeError(f"{label} candidate {index} Fisher repair does not match its lambda")
        if float(training_final.get("optimization_steps_run", 0.0)) <= 0.0:
            raise RuntimeError(f"{label} candidate {index} received no continuation optimization")
        if training.get("warm_start_sha256") != warm_start_sha256:
            raise RuntimeError(f"{label} candidate {index} training used the wrong warm start")
        head_config = training_config.get("head")
        if not isinstance(head_config, Mapping):
            raise RuntimeError(f"{label} candidate {index} lacks its head-training configuration")
        training_seed = int(training["head_training_seed"])
        head_config_sha256 = json_sha256(head_config)
        if matched_training_seed is None:
            matched_training_seed = training_seed
            matched_head_config_sha256 = head_config_sha256
        elif (
            training_seed != matched_training_seed
            or head_config_sha256 != matched_head_config_sha256
        ):
            raise RuntimeError(
                f"{label} candidate {index} violates matched training randomness/configuration"
            )
        head = FrozenHead.from_dict(candidate["head"])
        gate = FrozenLinearGate.from_dict(candidate["final_gate"])
        feature_dim = int(feature_mean.shape[1])
        if head.weight.size != feature_dim or gate.feature_dim != feature_dim:
            raise RuntimeError(f"{label} candidate {index} has the wrong feature dimension")
        _finite_scalar(candidate["task_mean"], "task_mean")
        _finite_scalar(candidate["task_std"], "task_std", positive=True)
        if not isinstance(candidate.get("validation_metrics"), Mapping):
            raise RuntimeError(f"{label} candidate {index} lacks validation task metrics")
        if not isinstance(candidate.get("crossfit"), Mapping):
            raise RuntimeError(f"{label} candidate {index} lacks cross-fitted discovery")
        if not isinstance(candidate.get("certificate_scores"), Mapping):
            raise RuntimeError(f"{label} candidate {index} lacks certificate scores")
        if len(candidate["crossfit"].get("folds", [])) != int(protocol["folds"]):
            raise RuntimeError(f"{label} candidate {index} has the wrong cross-fit fold count")
        recomputed_scores = crossfit_certificate_scores(
            candidate["crossfit"],
            confidence=float(protocol["confidence"]),
            comparisons=len(lambdas),
        )
        if json_sha256(candidate["certificate_scores"]) != json_sha256(recomputed_scores):
            raise RuntimeError(f"{label} candidate {index} certificate scores are inconsistent")

    eligible = [float(value) for value in decision.get("eligible_lambdas", [])]
    risk_eligible = [float(value) for value in decision.get("risk_eligible_lambdas", [])]
    if not eligible or any(value not in lambdas for value in eligible):
        raise RuntimeError(f"{label} has an invalid final eligible set")
    if not risk_eligible or any(value not in lambdas for value in risk_eligible):
        raise RuntimeError(f"{label} has an invalid risk-eligible set")
    selected_index = int(decision["selected_candidate_index"])
    if selected_index < 0 or selected_index >= len(candidates):
        raise RuntimeError(f"{label} selected candidate index is out of range")
    if not np.isclose(
        float(decision["selected_lambda"]),
        float(candidates[selected_index]["lambda_fg"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(f"{label} selected lambda and index disagree")
    if len(eligible) != len(risk_eligible) or any(value not in risk_eligible for value in eligible):
        raise RuntimeError(f"{label} primary eligible set differs from its one-standard-error risk set")
    recomputed_choice = risk_certificate_choice(candidates)
    expected_eligible = [float(row["lambda_fg"]) for row in recomputed_choice["eligible"]]
    if eligible != expected_eligible or risk_eligible != expected_eligible:
        raise RuntimeError(f"{label} frozen eligible set does not implement the declared one-SE rule")
    expected_selected = recomputed_choice["selected"]
    expected_risk_lambda = float(recomputed_choice["risk_minimizer"]["lambda_fg"])
    for field in ("risk_best_lambda", "validation_risk_minimizer_lambda"):
        if not np.isclose(float(decision[field]), expected_risk_lambda, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"{label} frozen {field} is inconsistent with validation NLL")
    if selected_index != int(expected_selected["candidate_index"]):
        raise RuntimeError(f"{label} frozen decision does not maximize the declared certificate score")
    if not np.isclose(
        float(decision["risk_ceiling"]),
        float(recomputed_choice["risk_ceiling"]),
        rtol=1e-12,
        atol=1e-15,
    ):
        raise RuntimeError(f"{label} frozen one-SE risk ceiling is inconsistent")

    if not validation_score_path.is_file():
        raise FileNotFoundError(validation_score_path)
    expected_score_sha = selection.get("validation_score_artifact_sha256")
    if file_sha256(validation_score_path) != expected_score_sha:
        raise RuntimeError(f"{label} validation score archive is missing or changed")
    scores = _load_validation_scores(validation_score_path, frozen_reference, candidates)
    labels = np.asarray(scores["y"], dtype=np.int64)
    if not np.all(np.isin(labels, (0, 1))):
        raise RuntimeError(f"{label} validation score labels are not binary")
    prepared_features: np.ndarray | None = None
    if validation_split is not None:
        if not np.array_equal(
            np.asarray(scores["row_id"], dtype=np.int64),
            np.asarray(validation_split.row_id, dtype=np.int64),
        ):
            raise RuntimeError(f"{label} validation score row ids do not match val_blind.npz")
        if not np.array_equal(labels, np.asarray(validation_split.y, dtype=np.int64)):
            raise RuntimeError(f"{label} validation score labels do not match val_blind.npz")
        prepared_features = apply_feature_normalizer(
            validation_split.h,
            feature_mean,
            feature_std,
        )
        expected_nuisance = (
            np.asarray(validation_split.b_vl, dtype=np.float64)
            - float(normalization["nuisance_mean"])
        ) / float(normalization["nuisance_std"])
        if not np.allclose(
            np.asarray(scores["nuisance"], dtype=np.float64),
            expected_nuisance,
            rtol=1e-10,
            atol=1e-10,
        ):
            raise RuntimeError(f"{label} validation nuisance scores do not match val_blind.npz")
    folds = np.asarray(scores["fold_assignment"])
    if folds.dtype.kind not in "iu" or not np.array_equal(folds, folds.astype(np.int64)):
        raise RuntimeError(f"{label} cross-fit assignments must be integer-valued")
    expected_folds = set(range(int(protocol["folds"])))
    if set(np.asarray(folds, dtype=np.int64).tolist()) != expected_folds:
        raise RuntimeError(f"{label} cross-fit assignments do not cover every declared fold")
    temperature = float(normalization["base_temperature"])
    for prefix, candidate in _evaluation_records(frozen_reference, candidates):
        logits = np.asarray(scores[f"{prefix}__logits"], dtype=np.float64)
        validation_bundle = None
        if prepared_features is not None:
            candidate_head = FrozenHead.from_dict(candidate["head"])
            expected_logits = candidate_head.logits(prepared_features)
            if not np.allclose(logits, expected_logits, rtol=1e-10, atol=1e-10):
                raise RuntimeError(
                    f"{label} {prefix} validation logits do not match its frozen head"
                )
            if int(candidate["crossfit"].get("partition_seed", -1)) != crossfit_partition_seed:
                raise RuntimeError(
                    f"{label} {prefix} cross-fit partition seed is inconsistent"
                )
            fold_pairs = stratified_kfold_indices(
                labels, int(protocol["folds"]), crossfit_partition_seed
            )
            expected_assignment = _stratified_fold_assignment(
                labels, int(protocol["folds"]), crossfit_partition_seed
            )
            if not np.array_equal(np.asarray(folds, dtype=np.int64), expected_assignment):
                raise RuntimeError(
                    f"{label} {prefix} fold assignment does not match its deterministic partition"
                )
            validation_bundle = make_model_coordinates(
                candidate_head,
                prepared_features,
                labels,
                np.asarray(scores["nuisance"], dtype=np.float64),
                task_mean=float(candidate["task_mean"]),
                task_std=float(candidate["task_std"]),
                temperature=temperature,
            )
        task = np.asarray(scores[f"{prefix}__task"], dtype=np.float64)
        nll = np.asarray(scores[f"{prefix}__nll"], dtype=np.float64)
        expected_task = (logits - float(candidate["task_mean"])) / float(candidate["task_std"])
        expected_nll = per_example_loss(logits, labels, temperature)
        if not np.allclose(task, expected_task, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"{label} {prefix} task coordinates do not match its frozen scale")
        if np.any(nll < 0.0) or not np.allclose(nll, expected_nll, rtol=1e-10, atol=1e-10):
            raise RuntimeError(f"{label} {prefix} per-example NLL is inconsistent")
        recomputed = task_metrics(logits, labels, temperature)
        recorded = candidate["validation_metrics"]
        for field in ("n", "nll", "nll_standard_error", "accuracy", "balanced_accuracy", "minimum_class_accuracy"):
            if not np.isclose(float(recorded[field]), float(recomputed[field]), rtol=1e-10, atol=1e-12):
                raise RuntimeError(f"{label} {prefix} validation metric {field} is inconsistent")
        if validation_bundle is not None:
            _validate_crossfit_evidence(
                candidate,
                validation_bundle,
                fold_pairs,
                metric_config,
                crossfit_gate_seed,
                gate_config_seed,
                protocol["gate"],
                label=f"{label} {prefix}",
            )
    return candidates, scores


def _require_all_selections_frozen(
    data_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> tuple[dict[int, tuple[dict[str, Any], Path]], str, dict[str, Any]]:
    """Finish the complete selection audit before touching the test archive."""

    manifest_path = data_dir / "preparation_manifest.json"
    manifest = read_json(manifest_path)
    current_inputs = blind_input_hashes(data_dir, include_test=False)
    for name in ("train_blind", "val_blind"):
        expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        if current_inputs[name] != expected:
            raise RuntimeError(f"{name}.npz does not match preparation_manifest.json")
    validation_split = load_blind_npz(data_dir / "val_blind.npz")

    contexts: dict[int, tuple[dict[str, Any], Path]] = {}
    for seed in seeds:
        seed_dir = out_dir / "runs" / f"seed_{seed}"
        selection_path = seed_dir / "selection.json"
        validation_score_path = seed_dir / "validation_scores.npz"
        if not selection_path.is_file():
            raise FileNotFoundError(
                f"all requested selections must be frozen before test access; missing {selection_path}"
            )
        selection = read_json(selection_path)
        _, validation_scores = _validate_selection(
            selection,
            seed=seed,
            current_inputs=current_inputs,
            validation_score_path=validation_score_path,
            validation_split=validation_split,
        )
        contexts[seed] = (selection, selection_path)

    selection_cohort = {
        "seeds": [int(seed) for seed in seeds],
        "artifacts": [
            {
                "seed": int(seed),
                "selection_artifact_sha256": file_sha256(contexts[seed][1]),
                "frozen_selection_sha256": contexts[seed][0]["frozen_selection_sha256"],
                "validation_score_artifact_sha256": file_sha256(
                    out_dir / "runs" / f"seed_{seed}" / "validation_scores.npz"
                ),
            }
            for seed in seeds
        ],
    }
    selection_cohort["sha256"] = canonical_protocol_hash(selection_cohort)

    # This is intentionally below the complete multi-seed validation loop.
    test_path = data_dir / "test_blind.npz"
    test_sha256 = file_sha256(test_path)
    expected_test_sha256 = manifest.get("artifacts", {}).get("test_blind", {}).get("sha256")
    if test_sha256 != expected_test_sha256:
        raise RuntimeError("test_blind.npz does not match preparation_manifest.json")
    return contexts, test_sha256, selection_cohort


def _make_weights(membership: np.ndarray, gate: FrozenLinearGate) -> np.ndarray:
    q, _ = admissible_weights(torch.as_tensor(membership, dtype=torch.float64), gate.constraints)
    return canonical_probability_weights(q.detach().cpu().numpy(), name="certification gate weights")


def _evaluate_gate(
    gate: FrozenLinearGate,
    bundle: Any,
    metric_config: FisherMetricConfig,
    *,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Transfer a frozen validation gate and bootstrap without refitting it."""

    class_mask = np.asarray(bundle.labels == gate.class_value)
    indices = np.flatnonzero(class_mask)
    if indices.size < 2:
        raise RuntimeError(f"test split has fewer than two rows for task class {gate.class_value}")
    features = torch.as_tensor(bundle.features[indices], dtype=torch.float32)
    membership = gate.membership(features).detach().cpu().numpy().astype(np.float64)
    q = _make_weights(membership, gate)
    metrics = evaluate_class_slice(
        task=bundle.task[indices],
        nuisance=bundle.nuisance[indices],
        logits=bundle.logits[indices],
        labels=bundle.labels[indices],
        q=q,
        class_id=gate.class_value,
        temperature=bundle.temperature,
        config=metric_config,
        hard_mass=gate.constraints.target_mass,
    )
    metrics["selector_objective"] = gate.objective
    metrics["certificate"] = float(metrics["predictive"]["projected_fisher"])
    metrics["gate_transfer"] = {
        "target_membership_mass": float(gate.constraints.target_mass),
        "heldout_mean_membership": float(membership.mean()),
        "absolute_mass_error": float(abs(membership.mean() - gate.constraints.target_mass)),
        "membership_min": float(membership.min()),
        "membership_max": float(membership.max()),
    }
    metrics["bootstrap_predictive_fisher"] = bootstrap_fixed_gate_fisher(
        membership=membership,
        task=bundle.task[indices],
        nuisance=bundle.nuisance[indices],
        logits=bundle.logits[indices],
        labels=bundle.labels[indices],
        class_id=gate.class_value,
        make_weights=lambda values: _make_weights(values, gate),
        config=metric_config,
        temperature=bundle.temperature,
        hard_mass=gate.constraints.target_mass,
        replicates=bootstrap_replicates,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    full_q = np.zeros(bundle.labels.size, dtype=np.float64)
    full_membership = np.full(bundle.labels.size, np.nan, dtype=np.float64)
    full_q[indices] = q
    full_membership[indices] = membership
    return metrics, {
        "q": full_q,
        "membership": full_membership,
    }


def _run_seed(
    args: argparse.Namespace,
    *,
    test: Any,
    seed: int,
    frozen: tuple[dict[str, Any], Path],
    test_sha256: str,
    selection_cohort: Mapping[str, Any],
) -> dict[str, Any]:
    selection, selection_path = frozen
    seed_dir = Path(args.out_dir) / "runs" / f"seed_{seed}"
    score_path = seed_dir / "blind_test_scores.npz"
    output_path = seed_dir / "blind_test.json"
    validation_score_path = seed_dir / "validation_scores.npz"
    certification_protocol = {
        "selection_artifact_sha256": file_sha256(selection_path),
        "frozen_selection_sha256": selection["frozen_selection_sha256"],
        "validation_score_artifact_sha256": file_sha256(validation_score_path),
        "test_blind_sha256": test_sha256,
        "bootstrap_replicates": int(args.bootstrap_replicates),
        "bootstrap_confidence": float(args.bootstrap_confidence),
        "cohort_seeds": [int(value) for value in args.seeds],
        "complete_default_cohort_frozen_before_test": bool(
            tuple(int(value) for value in args.seeds) == DEFAULT_SEEDS
        ),
        "selection_cohort": dict(selection_cohort),
        "selection_cohort_sha256": str(selection_cohort["sha256"]),
        "smoke": bool(args.smoke),
        "gate_handling": "each full-validation gate is transferred once; no test-time fitting",
        "candidate_schema": (
            "frozen_erm_reference is a diagnostic frozen-reference root; candidates is the matched "
            "lambda grid used for selection"
        ),
        "metadata_contract": "no Waterbirds background or group metadata is opened",
    }
    certification_protocol_sha256 = canonical_protocol_hash(
        {"seed": seed, "certification_protocol": certification_protocol}
    )

    if args.resume and output_path.is_file():
        existing = read_json(output_path)
        if existing.get("stage") != BLIND_TEST_STAGE or existing.get("seed") != seed:
            raise RuntimeError(f"seed {seed}: malformed blind test artifact")
        if existing.get("schema_version") != SCHEMA_VERSION:
            raise RuntimeError(f"seed {seed}: unsupported blind test schema")
        if existing.get("certification_protocol_sha256") != certification_protocol_sha256:
            raise RuntimeError(f"seed {seed}: refusing to resume under a changed certification protocol")
        if existing.get("metadata_accessed") is not False:
            raise RuntimeError(f"seed {seed}: blind test artifact violates the metadata contract")
        require_current_implementation(existing.get("implementation_manifest"), label=f"seed {seed} blind test")
        require_json_artifact_sha256(existing, "blind_test_sha256", label=f"seed {seed} blind test")
        existing_reference = existing.get(FROZEN_ERM_REFERENCE_KEY)
        if not isinstance(existing_reference, Mapping) or existing_reference.get("source") != "frozen_reference":
            raise RuntimeError(f"seed {seed}: blind test frozen ERM reference is absent or mislabeled")
        existing_candidates = existing.get("candidates")
        if not isinstance(existing_candidates, list) or len(existing_candidates) != len(
            selection["candidates"]
        ):
            raise RuntimeError(f"seed {seed}: blind test lambda grid is incomplete")
        for expected, observed in zip(selection["candidates"], existing_candidates):
            if (
                observed.get("source") != "lambda_grid"
                or observed.get("candidate_index") != expected.get("candidate_index")
                or not np.isclose(
                    float(observed.get("lambda_fg", np.nan)),
                    float(expected["lambda_fg"]),
                    rtol=0.0,
                    atol=1e-12,
                )
            ):
                raise RuntimeError(f"seed {seed}: blind test lambda-grid identity mismatch")
        if not score_path.is_file() or file_sha256(score_path) != existing.get("score_artifact_sha256"):
            raise RuntimeError(f"seed {seed}: blind test score archive is missing or changed")
        with np.load(score_path, allow_pickle=False) as archive:
            expected_keys = _expected_test_keys(
                selection[FROZEN_ERM_REFERENCE_KEY], selection["candidates"]
            )
            if set(archive.files) != expected_keys:
                raise RuntimeError(f"seed {seed}: blind test score archive has the wrong schema")
        print(f"seed {seed}: blind test certification already complete", flush=True)
        return existing

    normalization = selection["normalization"]
    feature_mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(normalization["feature_std"], dtype=np.float32)
    test_h = apply_feature_normalizer(test.h, feature_mean, feature_std)
    nuisance = (
        np.asarray(test.b_vl, dtype=np.float64) - float(normalization["nuisance_mean"])
    ) / float(normalization["nuisance_std"])
    temperature = float(normalization["base_temperature"])
    metric_config = _fisher_config(selection["protocol"])

    score_arrays: dict[str, np.ndarray] = {
        "row_id": np.asarray(test.row_id, dtype=np.int64),
        "y": np.asarray(test.y, dtype=np.int64),
        "nuisance": nuisance.astype(np.float64),
    }

    def evaluate_record(
        record: Mapping[str, Any],
        *,
        prefix: str,
        source: str,
        candidate_index: Optional[int] = None,
        coefficient: Optional[float] = None,
    ) -> dict[str, Any]:
        head = FrozenHead.from_dict(record["head"])
        gate = FrozenLinearGate.from_dict(record["final_gate"])
        bundle = make_model_coordinates(
            head,
            test_h,
            test.y,
            nuisance,
            task_mean=float(record["task_mean"]),
            task_std=float(record["task_std"]),
            temperature=temperature,
        )
        hidden, gate_arrays = _evaluate_gate(
            gate,
            bundle,
            metric_config,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=derive_seed(seed, "lambda-test-bootstrap", prefix),
        )
        score_arrays[f"{prefix}__logits"] = np.asarray(bundle.logits, dtype=np.float64)
        score_arrays[f"{prefix}__task"] = np.asarray(bundle.task, dtype=np.float64)
        score_arrays[f"{prefix}__q"] = gate_arrays["q"]
        score_arrays[f"{prefix}__membership"] = gate_arrays["membership"]
        result: dict[str, Any] = {
            "source": source,
            "task_metrics": task_metrics(bundle.logits, test.y, temperature),
            "hidden_certificate": hidden,
        }
        if candidate_index is not None:
            result["candidate_index"] = int(candidate_index)
        if coefficient is not None:
            result["lambda_fg"] = float(coefficient)
        return result

    frozen_reference_result = evaluate_record(
        selection[FROZEN_ERM_REFERENCE_KEY],
        prefix=FROZEN_ERM_REFERENCE_KEY,
        source="frozen_reference",
    )
    candidate_results: list[dict[str, Any]] = []
    for candidate in selection["candidates"]:
        coefficient = float(candidate["lambda_fg"])
        prefix = lambda_key(coefficient)
        candidate_results.append(
            evaluate_record(
                candidate,
                prefix=prefix,
                source="lambda_grid",
                candidate_index=int(candidate["candidate_index"]),
                coefficient=coefficient,
            )
        )

    atomic_savez_compressed(score_path, **score_arrays)
    current_inputs = blind_input_hashes(args.data_dir, include_test=True)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": BLIND_TEST_STAGE,
        "seed": seed,
        "metadata_accessed": False,
        "test_data_accessed_after_freeze": True,
        "implementation_manifest": implementation_manifest(),
        "input_hashes": current_inputs,
        "selection_artifact_sha256": file_sha256(selection_path),
        "frozen_selection_sha256": selection["frozen_selection_sha256"],
        "validation_score_artifact_sha256": file_sha256(validation_score_path),
        "certification_protocol": certification_protocol,
        "certification_protocol_sha256": certification_protocol_sha256,
        "test_blind_sha256": test_sha256,
        "score_artifact": str(score_path),
        "score_artifact_sha256": file_sha256(score_path),
        FROZEN_ERM_REFERENCE_KEY: frozen_reference_result,
        "candidates": candidate_results,
        "selected_lambda": float(selection["selection"]["selected_lambda"]),
        "selected_candidate_source": "lambda_grid",
        "freeze_statement": (
            "The frozen ERM reference, every matched-grid head, and every full-validation hidden-slice gate "
            "were frozen before any test array was "
            "deserialized or evaluated. Earlier preflight access was limited to opaque hashing and manifest "
            "attestation. No background or y-by-background group labels were available to this stage."
        ),
    }
    result["blind_test_sha256"] = json_artifact_sha256(result, "blind_test_sha256")
    atomic_write_json(output_path, result)
    print(f"seed {seed}: wrote blind test certification {output_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="build/results/waterbirds_hidden_slice_v2/prepared")
    parser.add_argument("--out-dir", default="build/results/waterbirds_group_free_lambda_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="permit an incomplete seed cohort and attest the resulting blind artifact as non-reportable",
    )
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be unique")
    if tuple(int(value) for value in args.seeds) != DEFAULT_SEEDS and not args.smoke:
        raise ValueError(
            "blind test reveal requires the complete default seed cohort; "
            "use --smoke only for explicitly non-reportable plumbing checks"
        )
    if args.bootstrap_replicates < 1:
        raise ValueError("bootstrap_replicates must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap_confidence must lie in (0, 1)")
    data_dir = Path(args.data_dir)
    contexts, test_sha256, selection_cohort = _require_all_selections_frozen(
        data_dir, Path(args.out_dir), args.seeds
    )
    test = load_blind_npz(data_dir / "test_blind.npz")
    for seed in args.seeds:
        _run_seed(
            args,
            test=test,
            seed=seed,
            frozen=contexts[seed],
            test_sha256=test_sha256,
            selection_cohort=selection_cohort,
        )


if __name__ == "__main__":
    main()
