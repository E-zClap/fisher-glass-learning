"""Select the Fisher-Glass coefficient without Waterbirds group metadata.

This is the only model-selection stage in the targeted Waterbirds audit.  It
uses the blind training archive to fit an ERM head, calibrate it, and discover
an Appendix-B Fisher slice. Candidate repairs optimize on the training fit
partition and use the separate training-calibration partition only for
task-label checkpoint selection. The untouched official validation split is then
used for ordinary task risk and candidate-specific, cross-fitted hidden-slice
certificates.  Neither ``test_blind.npz`` nor ``sealed_metadata.npz`` is opened
by this module.

The predeclared rule retains candidates within one standard error of the best
validation NLL, then maximizes a Bonferroni-adjusted cross-fit certificate
lower score, breaking exact ties toward the smaller coefficient. Overall,
balanced, and per-class accuracy screens plus paired improvement versus the
matched ERM continuation are retained as diagnostics; they do not alter the
primary choice. The pre-continuation ERM head is retained separately as a
frozen deployment reference and never enters the lambda-selection family.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import numpy as np
import torch

from experiment_common import (
    ALGORITHM_VERSION,
    DEFAULT_LAMBDAS,
    DEFAULT_SEEDS,
    FROZEN_ERM_REFERENCE_KEY,
    SCHEMA_VERSION,
    array_sha256,
    atomic_savez_compressed,
    atomic_write_json,
    blind_input_hashes,
    canonical_probability_weights,
    canonical_protocol_hash,
    configure_torch_determinism,
    crossfit_certificate_scores,
    derive_seed,
    file_sha256,
    fit_temperature,
    implementation_manifest,
    json_artifact_sha256,
    json_sha256,
    lambda_key,
    load_blind_npz,
    paired_inverse_diagnostic,
    read_json,
    risk_certificate_choice,
    require_current_implementation,
    require_json_artifact_sha256,
    stratified_partition_indices,
    task_metrics,
    validate_lambdas,
    verify_preparation_artifact,
)
from blind_protocol import (
    crossfit_selector,
    discover_final,
    make_gate_config,
    make_model_coordinates,
)
from common import stratified_kfold_indices
from compat import BooleanOptionalAction
from gates import apply_frozen_gate
from heads import (
    HeadConfig,
    RepairSpec,
    apply_feature_normalizer,
    fit_feature_normalizer,
    fit_scalar_normalizer,
    resolve_device,
    train_head,
)


STAGE = "metadata_blind_lambda_selection"


def _validate_prepared_inputs(data_dir: Path) -> Tuple[Any, Dict[str, Any], Dict[str, str]]:
    """Load the blind training input and verify both blind input artifacts.

    Validation is hash-checked here but deliberately not deserialized.  The
    per-seed routine opens it only after every candidate head has been frozen.
    """

    manifest_path = data_dir / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing preparation manifest: {manifest_path}")
    manifest = read_json(manifest_path)
    if not isinstance(manifest, dict):
        raise RuntimeError("preparation manifest is not a JSON object")

    verify_preparation_artifact(data_dir, "train_blind.npz")
    verify_preparation_artifact(data_dir, "val_blind.npz")
    train = load_blind_npz(data_dir / "train_blind.npz")

    configuration_keys = ("blindness_contract", "dataset", "representation", "nuisance")
    if all(key in manifest for key in configuration_keys):
        from experiment_common import json_sha256

        configuration = {key: manifest[key] for key in configuration_keys}
        expected = manifest.get("configuration_sha256")
        if expected is not None and expected != json_sha256(configuration):
            raise RuntimeError("preparation manifest configuration hash mismatch")
    return train, manifest, blind_input_hashes(data_dir, include_test=False)


def _partition_payload(train: Any, parts: Mapping[str, np.ndarray]) -> Dict[str, Any]:
    joined = np.concatenate([np.asarray(parts[name], dtype=np.int64) for name in parts])
    if joined.size != len(train) or np.unique(joined).size != len(train):
        raise RuntimeError("training partitions are not disjoint and complete")
    if int(joined.min()) != 0 or int(joined.max()) != len(train) - 1:
        raise RuntimeError("training partitions do not cover all positional indices")
    result: Dict[str, Any] = {}
    for name, raw in parts.items():
        index = np.asarray(raw, dtype=np.int64)
        result[name] = {
            "positions": index,
            "row_id_sha256": array_sha256(train.row_id[index]),
            "n": int(index.size),
            "label_counts": {str(class_id): int(np.sum(train.y[index] == class_id)) for class_id in (0, 1)},
        }
    return result


def _head_config(args: argparse.Namespace, *, candidates: bool) -> HeadConfig:
    return HeadConfig(
        steps=args.head_steps,
        learning_rate=args.head_lr,
        weight_decay=args.head_weight_decay,
        gradient_clip=args.gradient_clip,
        log_every=max(1, args.head_steps // 12),
        early_stopping_patience=args.early_stopping_patience if candidates else 0,
        early_stopping_eval_every=args.early_stopping_eval_every,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_checkpoint=True,
    )


def _gate_config(args: argparse.Namespace, *, seed: int):
    return make_gate_config(
        objective="fisher",
        slice_mass=args.slice_mass,
        minimum_ess_fraction=args.ess_fraction,
        max_weight_multiplier=args.cap_multiplier,
        restarts=args.gate_restarts,
        steps=args.gate_steps,
        learning_rate=args.gate_lr,
        shrinkage=args.shrinkage,
        ridge=args.ridge,
        epsilon=args.epsilon,
        seed=seed,
        weight_decay=args.gate_weight_decay,
        mass_penalty_strength=args.gate_mass_penalty,
    )


def _protocol(args: argparse.Namespace, lambdas: List[float]) -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "algorithm_version": ALGORITHM_VERSION,
        "analysis_status": (
            "post-hoc matched-control sensitivity designed after v1 Waterbirds group/test outcomes "
            "were inspected; not an independent held-out confirmation"
        ),
        "lambdas": lambdas,
        "fixed_lambda": float(args.fixed_lambda),
        "confidence": float(args.confidence),
        "folds": int(args.folds),
        "training_partition_fractions": {
            "fit": float(args.fit_fraction),
            "calibration": float(args.calibration_fraction),
            "gate_discovery": float(args.gate_discovery_fraction),
        },
        "head": {
            "steps": int(args.head_steps),
            "learning_rate": float(args.head_lr),
            "weight_decay": float(args.head_weight_decay),
            "gradient_clip": float(args.gradient_clip),
            "early_stopping_patience": int(args.early_stopping_patience),
            "early_stopping_eval_every": int(args.early_stopping_eval_every),
            "early_stopping_min_delta": float(args.early_stopping_min_delta),
            "candidate_checkpoint_split": "blind training calibration partition",
            "candidate_checkpoint_metric": "binary NLL at temperature 1; step zero is eligible",
        },
        "gate": {
            "objective": "Appendix-B predictive Fisher",
            "steps": int(args.gate_steps),
            "learning_rate": float(args.gate_lr),
            "restarts": int(args.gate_restarts),
            "weight_decay": float(args.gate_weight_decay),
            "mass_penalty_strength": float(args.gate_mass_penalty),
            "slice_mass": float(args.slice_mass),
            "minimum_ess_fraction": float(args.ess_fraction),
            "maximum_weight_multiplier": float(args.cap_multiplier),
            "shrinkage": float(args.shrinkage),
            "ridge": float(args.ridge),
            "epsilon": float(args.epsilon),
        },
        "fisher_metric": {
            "shrinkage": float(args.shrinkage),
            "ridge": float(args.ridge),
            "epsilon": float(args.epsilon),
        },
        "repair": {
            "objective": "ERM plus lambda times dimensionless inverse Fisher on a frozen train-audit slice",
            "warm_start": "same frozen ERM head for every lambda",
            "candidate_seed_matching": "one identical head-training seed for every lambda, including zero",
            "matched_training_includes_lambda_zero": True,
            "fisher_normalization": "F_initial/(F_current+epsilon), equal to one at the warm start",
            "lambda_zero": "matched ERM continuation under the same optimizer and checkpoint rule",
            "frozen_erm_reference": (
                "pre-continuation base ERM head; reported separately and excluded from selection"
            ),
            "frozen_reference_in_candidate_grid": False,
            "temperature": "one ERM calibration temperature frozen across every candidate",
            "validation_used_for_candidate_training": False,
        },
        "selection": {
            "ordinary_risk": "mean fixed-temperature validation binary NLL",
            "risk_envelope": "within one per-example standard error of the validation-NLL minimizer",
            "certificate": "reciprocal one-sided upper score on mean held-out inverse Fisher",
            "simultaneous_correction": "Bonferroni over the predeclared lambda grid",
            "ranking": (
                "highest Bonferroni-adjusted cross-fit certificate lower score; exact ties choose the smaller lambda"
            ),
            "crossfit_score_scope": (
                "validation ranking only; overlapping gate-training folds mean this is not presented as a "
                "formal independent-fold confidence interval"
            ),
            "crossfit_evidence": (
                "held-out metrics are evaluated on CPU; the selected gate for every fold and the deterministic "
                "fold assignment are archived so certification can replay each ranking statistic from validation rows"
            ),
            "accuracy_diagnostic_tolerance": float(args.accuracy_diagnostic_tolerance),
            "accuracy_diagnostic_role": "diagnostic only; does not alter selection",
            "paired_inverse_diagnostic_family": (
                f"Bonferroni over the {len(lambdas) - 1} predeclared positive-lambda versus matched-ERM comparisons"
            ),
            "paired_inverse_diagnostic_role": (
                "diagnostic only; overlapping discovery folds preclude a formal independent-fold test"
            ),
            "diagnostics": [
                "overall_accuracy",
                "balanced_accuracy",
                "both_per_class_accuracies",
                "paired_inverse_fisher",
            ],
            "group_metadata_used": False,
        },
        "deterministic_torch_algorithms": bool(args.deterministic),
    }


def _resume_if_valid(
    output_path: Path,
    scores_path: Path,
    *,
    seed: int,
    protocol_sha256: str,
    input_hashes: Mapping[str, str],
    lambdas: List[float],
) -> Any:
    if not output_path.is_file():
        return None
    existing = read_json(output_path)
    label = f"seed {seed} lambda selection"
    if existing.get("schema_version") != SCHEMA_VERSION:
        raise RuntimeError(f"refusing to resume {label}: unsupported schema")
    if existing.get("stage") != STAGE or existing.get("seed") != seed:
        raise RuntimeError(f"refusing to resume malformed {label} artifact")
    if existing.get("metadata_accessed") is not False or existing.get("test_data_accessed") is not False:
        raise RuntimeError(f"refusing to resume {label}: blindness contract is not attested")
    if existing.get("protocol_sha256") != protocol_sha256:
        raise RuntimeError(f"refusing to resume {label}: protocol changed")
    if existing.get("input_hashes") != dict(input_hashes):
        raise RuntimeError(f"refusing to resume {label}: blind inputs changed")
    require_current_implementation(existing.get("implementation_manifest"), label=label)
    require_json_artifact_sha256(existing, "frozen_selection_sha256", label=label)
    if not scores_path.is_file():
        raise RuntimeError(f"refusing to resume {label}: validation score artifact is missing")
    score_hash = file_sha256(scores_path)
    if existing.get("validation_score_artifact_sha256") != score_hash:
        raise RuntimeError(f"refusing to resume {label}: validation score artifact changed")
    expected_keys = {"row_id", "y", "nuisance", "fold_assignment"}
    expected_keys.update(
        {
            f"{FROZEN_ERM_REFERENCE_KEY}__logits",
            f"{FROZEN_ERM_REFERENCE_KEY}__task",
            f"{FROZEN_ERM_REFERENCE_KEY}__nll",
        }
    )
    for value in lambdas:
        key = lambda_key(value)
        expected_keys.update({f"{key}__logits", f"{key}__task", f"{key}__nll"})
    with np.load(scores_path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise RuntimeError(f"refusing to resume {label}: validation score schema changed")
    recorded_lambdas = [float(row["lambda_fg"]) for row in existing.get("candidates", [])]
    if recorded_lambdas != lambdas:
        raise RuntimeError(f"refusing to resume {label}: candidate grid changed")
    reference = existing.get("frozen_erm_reference")
    if not isinstance(reference, Mapping) or reference.get("reference_key") != FROZEN_ERM_REFERENCE_KEY:
        raise RuntimeError(f"refusing to resume {label}: frozen ERM reference is missing or malformed")
    candidates = existing.get("candidates", [])
    if candidates[0].get("role") != "matched_erm_continuation":
        raise RuntimeError(f"refusing to resume {label}: lambda zero is not matched ERM continuation")
    warm_start_sha256 = existing.get("base_model", {}).get("head_sha256")
    if not isinstance(warm_start_sha256, str) or any(
        row.get("warm_start_sha256") != warm_start_sha256 for row in candidates
    ):
        raise RuntimeError(f"refusing to resume {label}: candidate warm starts are not matched")
    print(f"seed {seed}: group-free lambda selection already complete", flush=True)
    return existing


def _accuracy_diagnostic(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any], tolerance: float
) -> Dict[str, Any]:
    candidate_class = candidate["per_class_accuracy"]
    baseline_class = baseline["per_class_accuracy"]
    floors = {
        "accuracy": float(baseline["accuracy"] - tolerance),
        "balanced_accuracy": float(baseline["balanced_accuracy"] - tolerance),
        "per_class_accuracy": {
            str(class_id): float(baseline_class[str(class_id)] - tolerance) for class_id in (0, 1)
        },
    }
    checks = {
        "overall": bool(candidate["accuracy"] >= floors["accuracy"]),
        "balanced": bool(candidate["balanced_accuracy"] >= floors["balanced_accuracy"]),
        "class_0": bool(candidate_class["0"] >= floors["per_class_accuracy"]["0"]),
        "class_1": bool(candidate_class["1"] >= floors["per_class_accuracy"]["1"]),
    }
    return {"satisfied": bool(all(checks.values())), "checks": checks, "floors": floors}


def _crossfit_fold_assignment(labels: np.ndarray, folds: int, seed: int) -> np.ndarray:
    assignment = np.full(np.asarray(labels).size, -1, dtype=np.int16)
    for fold_index, (_, heldout) in enumerate(stratified_kfold_indices(labels, folds, seed)):
        if np.any(assignment[heldout] != -1):
            raise RuntimeError("cross-fit held-out folds overlap")
        assignment[heldout] = fold_index
    if np.any(assignment < 0):
        raise RuntimeError("cross-fit held-out folds do not cover validation")
    return assignment


def _run_seed(
    args: argparse.Namespace,
    train: Any,
    preparation_manifest: Mapping[str, Any],
    input_hashes: Mapping[str, str],
    protocol: Mapping[str, Any],
    lambdas: List[float],
    seed: int,
    device: str,
) -> Dict[str, Any]:
    seed_dir = Path(args.out_dir) / "runs" / f"seed_{seed}"
    output_path = seed_dir / "selection.json"
    scores_path = seed_dir / "validation_scores.npz"
    protocol_sha256 = canonical_protocol_hash({"seed": int(seed), "protocol": protocol})
    if args.resume:
        resumed = _resume_if_valid(
            output_path,
            scores_path,
            seed=seed,
            protocol_sha256=protocol_sha256,
            input_hashes=input_hashes,
            lambdas=lambdas,
        )
        if resumed is not None:
            return resumed

    partition_seed = derive_seed(seed, "group-free-lambda", "train-partition")
    parts = stratified_partition_indices(
        train.y,
        {
            "fit": args.fit_fraction,
            "calibration": args.calibration_fraction,
            "gate_discovery": args.gate_discovery_fraction,
        },
        seed=partition_seed,
    )
    fit_idx = parts["fit"]
    calibration_idx = parts["calibration"]
    gate_discovery_idx = parts["gate_discovery"]

    feature_mean, feature_std = fit_feature_normalizer(train.h[fit_idx])
    train_h = apply_feature_normalizer(train.h, feature_mean, feature_std)
    nuisance_mean, nuisance_std = fit_scalar_normalizer(train.b_vl[fit_idx])
    train_b = ((np.asarray(train.b_vl, dtype=np.float64) - nuisance_mean) / nuisance_std).astype(np.float64)

    base_training = train_head(
        train_h[fit_idx],
        train.y[fit_idx],
        seed=derive_seed(seed, "group-free-lambda", "erm-head"),
        device=device,
        config=_head_config(args, candidates=False),
    )
    base_head = base_training.head
    base_temperature = fit_temperature(
        base_head.logits(train_h[calibration_idx]),
        train.y[calibration_idx],
    )
    base_task_mean, base_task_std = fit_scalar_normalizer(base_head.logits(train_h[fit_idx]))

    # This train-only audit slice is the sole slice used in candidate training.
    # The official validation split is still untouched at this point.
    audit_bundle = make_model_coordinates(
        base_head,
        train_h[gate_discovery_idx],
        train.y[gate_discovery_idx],
        train_b[gate_discovery_idx],
        task_mean=base_task_mean,
        task_std=base_task_std,
        temperature=base_temperature,
    )
    audit_gate_seed = derive_seed(seed, "group-free-lambda", "train-audit-gate")
    audit_gate_config = _gate_config(args, seed=audit_gate_seed)
    audit_gate, audit_discovery = discover_final(
        audit_bundle,
        config=audit_gate_config,
        device=device,
        seed=audit_gate_seed,
    )
    fit_application = apply_frozen_gate(
        audit_gate,
        torch.as_tensor(train_h[fit_idx], dtype=torch.float32),
        torch.as_tensor(train.y[fit_idx], dtype=torch.long),
    )
    frozen_q_fit = canonical_probability_weights(
        fit_application.full_q.detach().cpu().numpy(),
        name="train-audit gate transferred to training fit partition",
    )
    if np.any(frozen_q_fit[train.y[fit_idx] != audit_gate.class_value] > 1e-12):
        raise RuntimeError("transferred gate has nonzero weight outside its selected task class")

    candidate_head_seed = derive_seed(seed, "group-free-lambda", "fisher-repair-head")
    candidate_config = _head_config(args, candidates=True)
    warm_start_sha256 = json_sha256(base_head.to_dict())
    frozen_reference_head: Dict[str, Any] = {
        "reference_key": FROZEN_ERM_REFERENCE_KEY,
        "role": "frozen_pre_continuation_erm_reference",
        "head": base_head,
        "task_mean": float(base_task_mean),
        "task_std": float(base_task_std),
        "warm_start_sha256": warm_start_sha256,
        "training": {
            "source": "base_model",
            "additional_optimization_steps": 0,
        },
    }
    frozen_candidate_heads: List[Dict[str, Any]] = []
    for candidate_index, lambda_fg in enumerate(lambdas):
        print(
            f"seed {seed}: train candidate {candidate_index + 1}/{len(lambdas)} "
            f"lambda_FG={lambda_fg:g}",
            flush=True,
        )
        repair = (
            None
            if lambda_fg == 0.0
            else RepairSpec("fisher_repair", lambda_fisher=float(lambda_fg))
        )
        trained = train_head(
            train_h[fit_idx],
            train.y[fit_idx],
            seed=candidate_head_seed,
            device=device,
            config=candidate_config,
            initial=base_head,
            repair=repair,
            slice_weights=frozen_q_fit,
            nuisance=train_b[fit_idx],
            selected_class=audit_gate.class_value,
            temperature=base_temperature,
            shrinkage=args.shrinkage,
            ridge=args.ridge,
            epsilon=args.epsilon,
            checkpoint_features=train_h[calibration_idx],
            checkpoint_labels=train.y[calibration_idx],
            normalize_fisher_penalty=True,
        )
        expected_mode = "erm" if lambda_fg == 0.0 else "fisher_repair"
        if trained.config.get("mode") != expected_mode:
            raise RuntimeError(
                f"lambda={lambda_fg:g} used training mode {trained.config.get('mode')!r}; "
                f"expected {expected_mode!r}"
            )
        if float(trained.final.get("optimization_steps_run", 0.0)) <= 0.0:
            raise RuntimeError(f"lambda={lambda_fg:g} did not receive matched continuation training")
        head = trained.head
        task_mean, task_std = fit_scalar_normalizer(head.logits(train_h[fit_idx]))
        training = {
            "config": trained.config,
            "history": trained.history,
            "final": trained.final,
            "head_training_seed": int(candidate_head_seed),
            "warm_start_sha256": warm_start_sha256,
        }
        frozen_candidate_heads.append(
            {
                "candidate_index": int(candidate_index),
                "lambda_fg": float(lambda_fg),
                "role": (
                    "matched_erm_continuation"
                    if lambda_fg == 0.0
                    else "fisher_repair"
                ),
                "warm_start_sha256": warm_start_sha256,
                "head": head,
                "task_mean": float(task_mean),
                "task_std": float(task_std),
                "training": training,
            }
        )

    # Candidate training is now complete.  Only at this boundary is the blind
    # validation archive deserialized for group-free model selection.
    val = load_blind_npz(Path(args.data_dir) / "val_blind.npz")
    if train.feature_dim != val.feature_dim:
        raise ValueError("train and validation feature dimensions differ")
    if np.intersect1d(train.row_id, val.row_id).size:
        raise ValueError("blind train and validation row ids overlap")
    val_h = apply_feature_normalizer(val.h, feature_mean, feature_std)
    # BlindSplit buffers are deliberately read-only.  Cross-fitting sends
    # labels through ``torch.as_tensor``; a private writable copy avoids
    # PyTorch's non-writable-NumPy warning without changing any values.
    val_y = np.array(val.y, dtype=np.int64, copy=True)
    val_b = ((np.asarray(val.b_vl, dtype=np.float64) - nuisance_mean) / nuisance_std).astype(np.float64)

    crossfit_seed = derive_seed(seed, "group-free-lambda", "candidate-crossfit-gates")
    crossfit_partition_seed = derive_seed(seed, "group-free-lambda", "validation-crossfit-partition")
    final_gate_seed = derive_seed(seed, "group-free-lambda", "candidate-final-gates")
    comparisons = len(lambdas)
    candidates: List[Dict[str, Any]] = []
    score_arrays: Dict[str, np.ndarray] = {
        "row_id": np.asarray(val.row_id, dtype=np.int64),
        "y": val_y,
        "nuisance": np.asarray(val_b, dtype=np.float64),
        "fold_assignment": _crossfit_fold_assignment(val_y, args.folds, crossfit_partition_seed),
    }
    def evaluate_arm(
        frozen_arm: Mapping[str, Any],
        *,
        score_prefix: str,
        score_comparisons: int,
    ) -> Dict[str, Any]:
        head = frozen_arm["head"]
        task_mean = float(frozen_arm["task_mean"])
        task_std = float(frozen_arm["task_std"])
        val_logits = head.logits(val_h)
        val_bundle = make_model_coordinates(
            head,
            val_h,
            val_y,
            val_b,
            task_mean=task_mean,
            task_std=task_std,
            temperature=base_temperature,
        )
        validation_metrics = task_metrics(val_logits, val_y, base_temperature)
        fresh_config = _gate_config(args, seed=derive_seed(seed, "group-free-lambda", "candidate-gate-config"))
        crossfit = crossfit_selector(
            val_bundle,
            config=fresh_config,
            folds=args.folds,
            reference_features=train_h[gate_discovery_idx],
            reference_labels=train.y[gate_discovery_idx],
            device=device,
            seed=crossfit_seed,
            partition_seed=crossfit_partition_seed,
            include_fold_gates=True,
        )
        if int(crossfit.get("partition_seed", -1)) != crossfit_partition_seed:
            raise RuntimeError("candidate cross-fit did not preserve the matched fold partition")
        scores = crossfit_certificate_scores(
            crossfit,
            confidence=args.confidence,
            comparisons=score_comparisons,
        )
        final_gate, final_discovery = discover_final(
            val_bundle,
            config=fresh_config,
            device=device,
            seed=final_gate_seed,
        )
        score_arrays[f"{score_prefix}__logits"] = np.asarray(val_logits, dtype=np.float64)
        score_arrays[f"{score_prefix}__task"] = np.asarray(val_bundle.task, dtype=np.float64)
        score_arrays[f"{score_prefix}__nll"] = np.asarray(
            -(
                val_y * np.log(np.clip(val_bundle.probability, 1e-12, 1.0))
                + (1 - val_y) * np.log(np.clip(1.0 - val_bundle.probability, 1e-12, 1.0))
            ),
            dtype=np.float64,
        )
        evaluated = {
            "role": str(frozen_arm["role"]),
            "warm_start_sha256": str(frozen_arm["warm_start_sha256"]),
            "head": head.to_dict(),
            "task_mean": task_mean,
            "task_std": task_std,
            "validation_metrics": validation_metrics,
            "crossfit": crossfit,
            "certificate_scores": scores,
            "final_gate": final_gate.to_dict(),
            "final_discovery": final_discovery,
            "training": frozen_arm["training"],
        }
        return evaluated

    print(f"seed {seed}: validate frozen ERM reference", flush=True)
    frozen_erm_reference = {
        "reference_key": FROZEN_ERM_REFERENCE_KEY,
        "certificate_score_role": (
            f"diagnostic comparability only; adjusted on the {comparisons}-candidate scale "
            "but excluded from selection"
        ),
        **evaluate_arm(
            frozen_reference_head,
            score_prefix=FROZEN_ERM_REFERENCE_KEY,
            score_comparisons=comparisons,
        ),
    }

    for frozen_candidate in frozen_candidate_heads:
        candidate_index = int(frozen_candidate["candidate_index"])
        lambda_fg = float(frozen_candidate["lambda_fg"])
        print(
            f"seed {seed}: validate candidate {candidate_index + 1}/{len(lambdas)} "
            f"lambda_FG={lambda_fg:g}",
            flush=True,
        )
        candidates.append(
            {
                "candidate_index": int(candidate_index),
                "lambda_fg": float(lambda_fg),
                "paired_vs_matched_erm_diagnostic": None,
                **evaluate_arm(
                    frozen_candidate,
                    score_prefix=lambda_key(lambda_fg),
                    score_comparisons=comparisons,
                ),
            }
        )

    baseline = candidates[0]
    if float(baseline["lambda_fg"]) != 0.0:
        raise RuntimeError("lambda zero is not the first candidate")
    zero_difference = [0.0 for _ in baseline["crossfit"]["folds"]]
    baseline["paired_vs_matched_erm_diagnostic"] = {
        "fold_difference_candidate_minus_baseline": zero_difference,
        "marginal_upper_score": None,
        "bonferroni_upper_score": None,
        "diagnostic_upper_score_below_zero": False,
        "favorable_direction": "negative",
        "reason": "self-comparison",
        "interpretation": "diagnostic only; does not alter selection",
    }
    for row in candidates[1:]:
        row["paired_vs_matched_erm_diagnostic"] = paired_inverse_diagnostic(
            row["crossfit"],
            baseline["crossfit"],
            confidence=args.confidence,
            comparisons=max(1, comparisons - 1),
        )

    primary_choice = risk_certificate_choice(candidates)
    risk_minimizer = primary_choice["risk_minimizer"]
    risk_ceiling = float(primary_choice["risk_ceiling"])
    risk_eligible = primary_choice["eligible"]
    accuracy_diagnostics: Dict[str, Any] = {}
    for row in candidates:
        status = _accuracy_diagnostic(
            row["validation_metrics"],
            baseline["validation_metrics"],
            args.accuracy_diagnostic_tolerance,
        )
        accuracy_diagnostics[lambda_key(row["lambda_fg"])] = status
    if not accuracy_diagnostics[lambda_key(0.0)]["satisfied"]:
        raise RuntimeError("matched ERM anchor failed its self-consistency accuracy diagnostic")
    diagnostic_accuracy_safe = [
        row for row in candidates if accuracy_diagnostics[lambda_key(row["lambda_fg"])]["satisfied"]
    ]
    # Primary rule: one-SE task risk, then the adjusted held-out certificate
    # score. Accuracy and paired-difference checks remain visible diagnostics.
    selected = primary_choice["selected"]

    selection = {
        "risk_best_lambda": float(risk_minimizer["lambda_fg"]),
        "risk_ceiling": risk_ceiling,
        "risk_eligible_lambdas": [float(row["lambda_fg"]) for row in risk_eligible],
        "eligible_lambdas": [float(row["lambda_fg"]) for row in risk_eligible],
        "diagnostic_accuracy_safe_lambdas": [
            float(row["lambda_fg"]) for row in diagnostic_accuracy_safe
        ],
        "matched_erm_in_risk_envelope": bool(baseline in risk_eligible),
        "validation_risk_minimizer_lambda": float(risk_minimizer["lambda_fg"]),
        "selected_lambda": float(selected["lambda_fg"]),
        "selected_candidate_index": int(selected["candidate_index"]),
        "one_standard_error": float(risk_minimizer["validation_metrics"]["nll_standard_error"]),
        "accuracy_diagnostic_tolerance": float(args.accuracy_diagnostic_tolerance),
        "accuracy_diagnostics_by_lambda": accuracy_diagnostics,
        "paired_inverse_diagnostic_does_not_select": True,
        "criterion": (
            "one-SE validation-NLL envelope; then maximum Bonferroni-adjusted cross-fit certificate lower "
            "score with smaller-lambda tie break"
        ),
        "metadata_used": False,
    }

    atomic_savez_compressed(scores_path, **score_arrays)
    score_artifact_sha256 = file_sha256(scores_path)
    implementation = implementation_manifest()
    source_manifest = {
        "path": str(Path(args.data_dir) / "preparation_manifest.json"),
        "sha256": input_hashes["preparation_manifest"],
        "configuration_sha256": preparation_manifest.get("configuration_sha256"),
    }
    result: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": STAGE,
        "seed": int(seed),
        "metadata_accessed": False,
        "test_data_accessed": False,
        "input_hashes": dict(input_hashes),
        "source_preparation_manifest": source_manifest,
        "implementation_manifest": implementation,
        "protocol": dict(protocol),
        "protocol_sha256": protocol_sha256,
        "partition": _partition_payload(train, parts),
        "normalization": {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "nuisance_mean": float(nuisance_mean),
            "nuisance_std": float(nuisance_std),
            "base_temperature": float(base_temperature),
        },
        "base_model": {
            "head": base_head.to_dict(),
            "head_sha256": warm_start_sha256,
            "task_mean": float(base_task_mean),
            "task_std": float(base_task_std),
            "training": {
                "config": base_training.config,
                "history": base_training.history,
                "final": base_training.final,
            },
            "calibration_metrics": task_metrics(
                base_head.logits(train_h[calibration_idx]),
                train.y[calibration_idx],
                base_temperature,
            ),
        },
        "train_audit_gate": {
            "gate": audit_gate.to_dict(),
            "discovery": audit_discovery,
            "selected_class": int(audit_gate.class_value),
            "transfer_to_fit": {
                "sum_weight": float(frozen_q_fit.sum()),
                "effective_sample_size": float(
                    fit_application.diagnostics.effective_sample_size.detach().cpu()
                ),
                "maximum_weight": float(fit_application.diagnostics.maximum_weight.detach().cpu()),
            },
        },
        "matched_randomness": {
            "train_partition_seed": int(partition_seed),
            "candidate_head_training_seed": int(candidate_head_seed),
            "candidate_crossfit_gate_seed": int(crossfit_seed),
            "validation_crossfit_partition_seed": int(crossfit_partition_seed),
            "candidate_final_gate_seed": int(final_gate_seed),
        },
        "frozen_erm_reference": frozen_erm_reference,
        "candidates": candidates,
        "selection": selection,
        "validation_score_artifact": str(scores_path),
        "validation_score_artifact_sha256": score_artifact_sha256,
        "freeze_statement": (
            "The frozen pre-continuation ERM reference, every matched-continuation candidate head, validation "
            "score, cross-fitted certificate, chosen lambda, and transferable full-validation gate were frozen "
            "using only blind train/validation arrays. The reference was excluded from lambda selection. "
            "Waterbirds groups, backgrounds, the blind test split, and sealed metadata remained unopened."
        ),
    }
    result["frozen_selection_sha256"] = json_artifact_sha256(result, "frozen_selection_sha256")
    atomic_write_json(output_path, result)
    print(
        f"seed {seed}: selected lambda_FG={selection['selected_lambda']:g} "
        f"(risk-best={selection['risk_best_lambda']:g}); "
        f"artifact={output_path}",
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="build/results/waterbirds_hidden_slice_v2/prepared")
    parser.add_argument("--out-dir", default="build/results/waterbirds_group_free_lambda_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--lambdas", type=float, nargs="+", default=list(DEFAULT_LAMBDAS))
    parser.add_argument("--fixed-lambda", type=float, default=0.03)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--fit-fraction", type=float, default=0.70)
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--gate-discovery-fraction", "--audit-fraction", dest="gate_discovery_fraction", type=float, default=0.15)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--accuracy-diagnostic-tolerance", type=float, default=0.02)
    parser.add_argument("--slice-mass", type=float, default=0.10)
    parser.add_argument("--ess-fraction", type=float, default=0.80)
    parser.add_argument("--cap-multiplier", type=float, default=2.0)
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--head-steps", type=int, default=1200)
    parser.add_argument("--head-lr", type=float, default=1e-2)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gradient-clip", type=float, default=10.0)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-eval-every", type=int, default=25)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    parser.add_argument("--gate-steps", type=int, default=1000)
    parser.add_argument("--gate-lr", type=float, default=3e-2)
    parser.add_argument("--gate-restarts", type=int, default=10)
    parser.add_argument("--gate-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-mass-penalty", type=float, default=10.0)
    parser.add_argument("--deterministic", action=BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True)
    return parser


def _validate_args(args: argparse.Namespace) -> List[float]:
    if not np.isclose(float(args.fixed_lambda), 0.03, rtol=0.0, atol=1e-12):
        raise ValueError("the fixed reference policy is predeclared as lambda_FG=0.03")
    lambdas = validate_lambdas(args.lambdas, fixed_lambda=args.fixed_lambda)
    fractions = np.asarray(
        [args.fit_fraction, args.calibration_fraction, args.gate_discovery_fraction],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(fractions)) or np.any(fractions <= 0.0):
        raise ValueError("training partition fractions must be finite and positive")
    if not np.isclose(float(fractions.sum()), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError("fit, calibration, and gate-discovery fractions must sum to one")
    if args.folds < 2:
        raise ValueError("folds must be at least two")
    if not np.isclose(float(args.confidence), 0.95, rtol=0.0, atol=1e-12):
        raise ValueError("the certificate score aggregation is predeclared at confidence=0.95")
    if args.accuracy_diagnostic_tolerance < 0.0 or args.accuracy_diagnostic_tolerance > 1.0:
        raise ValueError("accuracy diagnostic tolerance must lie in [0, 1]")
    if not 0.0 < args.slice_mass < 1.0:
        raise ValueError("slice mass must lie in (0, 1)")
    if not 0.0 < args.ess_fraction <= 1.0 or args.cap_multiplier < 1.0:
        raise ValueError("invalid gate ESS/cap constraints")
    if not 0.0 <= args.shrinkage < 1.0 or args.ridge <= 0.0 or args.epsilon <= 0.0:
        raise ValueError("invalid Fisher geometry configuration")
    if args.head_steps < 1 or args.head_lr <= 0.0 or args.head_weight_decay < 0.0:
        raise ValueError("invalid head optimizer configuration")
    if args.gradient_clip < 0.0:
        raise ValueError("gradient clip must be nonnegative")
    if args.early_stopping_patience < 0 or args.early_stopping_eval_every < 1:
        raise ValueError("invalid early-stopping configuration")
    if args.early_stopping_min_delta < 0.0:
        raise ValueError("early-stopping minimum delta must be nonnegative")
    if args.gate_steps < 0 or args.gate_lr <= 0.0 or args.gate_restarts < 1:
        raise ValueError("invalid gate optimizer configuration")
    if args.gate_weight_decay < 0.0 or args.gate_mass_penalty < 0.0:
        raise ValueError("gate penalties must be nonnegative")
    if not args.seeds or len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be a nonempty list without duplicates")
    return lambdas


def main() -> None:
    args = build_parser().parse_args()
    lambdas = _validate_args(args)
    configure_torch_determinism(args.deterministic)
    device = str(resolve_device(args.device))
    data_dir = Path(args.data_dir)
    train, manifest, input_hashes = _validate_prepared_inputs(data_dir)
    protocol = _protocol(args, lambdas)
    for seed in args.seeds:
        _run_seed(
            args,
            train,
            manifest,
            input_hashes,
            protocol,
            lambdas,
            int(seed),
            device,
        )


if __name__ == "__main__":
    main()
