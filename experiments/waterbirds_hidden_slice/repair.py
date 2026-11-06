"""Repair a frozen hidden slice, then reinitialize and cross-fit the adversary.

The repair stage opens only the blind training/validation artifacts and the
already-frozen discovery record.  Candidate strengths are selected by a
task-label-only safety envelope (overall, balanced, and per-class validation
accuracy) and a declared frozen-slice objective.  No Waterbirds group statistic
is available to this process.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compat import BooleanOptionalAction
from blind_protocol import (
    crossfit_selector,
    discover_final,
    evaluate_gate,
    make_gate_config,
    make_model_coordinates,
    overall_metrics,
)
from common import (
    atomic_write_json,
    canonical_probability_weights,
    configure_torch_determinism,
    derive_seed,
    file_sha256,
    fit_temperature,
    json_artifact_sha256,
    json_sha256,
    load_blind_npz,
    require_current_scientific_implementation,
    require_json_artifact_sha256,
)
from gates import FrozenLinearGate, apply_frozen_gate
from metrics import FisherMetricConfig
from heads import (
    FrozenHead,
    HeadConfig,
    RepairSpec,
    apply_feature_normalizer,
    fit_scalar_normalizer,
    resolve_device,
    train_head,
)


SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_seed_inputs(data_dir: Path, out_dir: Path, seed: int):
    discovery_path = out_dir / "runs" / f"seed_{seed}" / "discovery.json"
    if not discovery_path.is_file():
        raise FileNotFoundError(f"run discovery before repair: {discovery_path}")
    discovery = _read_json(discovery_path)
    if discovery.get("seed") != seed or discovery.get("stage") != "metadata_blind_discovery":
        raise RuntimeError("discovery artifact has the wrong seed or stage")
    if discovery.get("metadata_accessed") is not False or discovery.get("test_data_accessed") is not False:
        raise RuntimeError("discovery artifact does not satisfy the blind-stage contract")
    require_current_scientific_implementation(discovery, label=f"seed {seed} discovery")
    require_json_artifact_sha256(discovery, "frozen_blind_sha256", label=f"seed {seed} discovery")
    input_hashes = {
        "train_blind": file_sha256(data_dir / "train_blind.npz"),
        "val_blind": file_sha256(data_dir / "val_blind.npz"),
        "preparation_manifest": file_sha256(data_dir / "preparation_manifest.json"),
    }
    if discovery.get("input_hashes") != input_hashes:
        raise RuntimeError("current prepared inputs do not match the frozen discovery")
    train = load_blind_npz(data_dir / "train_blind.npz")
    val = load_blind_npz(data_dir / "val_blind.npz")
    return train, val, discovery, discovery_path, input_hashes


def _candidate_specs(args: argparse.Namespace) -> dict[str, list[RepairSpec]]:
    return {
        "loss_repair": [RepairSpec("no_op")]
        + [RepairSpec("loss_repair", lambda_loss=value) for value in sorted(set(args.loss_lambdas))],
        "fisher_repair": [RepairSpec("no_op")]
        + [RepairSpec("fisher_repair", lambda_fisher=value) for value in sorted(set(args.fisher_lambdas))],
        "combined_repair": [
            RepairSpec("no_op"),
        ]
        + [
            RepairSpec("combined_repair", lambda_loss=loss_value, lambda_fisher=fisher_value)
            for loss_value in sorted(set(args.combined_loss_lambdas))
            for fisher_value in sorted(set(args.combined_fisher_lambdas))
        ],
    }


def _task_label_validation_metrics(bundle) -> dict[str, Any]:
    """Overall metrics plus safeguards that use task labels, never metadata."""

    metrics: dict[str, Any] = overall_metrics(bundle)
    predictions = (np.asarray(bundle.logits) >= 0).astype(np.int64)
    labels = np.asarray(bundle.labels, dtype=np.int64)
    per_class: dict[str, float] = {}
    for class_id in (0, 1):
        mask = labels == class_id
        if not np.any(mask):
            raise ValueError(f"validation split is missing task class {class_id}")
        per_class[str(class_id)] = float((predictions[mask] == labels[mask]).mean())
    metrics["per_class_accuracy"] = per_class
    metrics["balanced_accuracy"] = float(np.mean(list(per_class.values())))
    metrics["minimum_class_accuracy"] = float(min(per_class.values()))
    return metrics


def _candidate_penalty_size(row: dict[str, Any], args: argparse.Namespace) -> float:
    spec = RepairSpec(**row["spec"])
    if spec.name == "no_op":
        return 0.0
    max_loss = max(args.loss_lambdas + args.combined_loss_lambdas)
    max_fisher = max(args.fisher_lambdas + args.combined_fisher_lambdas)
    return float(spec.lambda_loss / max_loss + spec.lambda_fisher / max_fisher)


def _safety_status(
    row: dict[str, Any],
    base_metrics: dict[str, Any],
    args: argparse.Namespace,
) -> dict[str, Any]:
    metrics = row["validation_metrics"]
    overall_floor = float(base_metrics["accuracy"] - args.accuracy_budget)
    balanced_floor = float(base_metrics["balanced_accuracy"] - args.balanced_accuracy_budget)
    class_floors = {
        str(class_id): float(
            base_metrics["per_class_accuracy"][str(class_id)] - args.per_class_accuracy_budget
        )
        for class_id in (0, 1)
    }
    checks = {
        "overall": bool(metrics["accuracy"] >= overall_floor),
        "balanced": bool(metrics["balanced_accuracy"] >= balanced_floor),
        "class_0": bool(metrics["per_class_accuracy"]["0"] >= class_floors["0"]),
        "class_1": bool(metrics["per_class_accuracy"]["1"] >= class_floors["1"]),
    }
    return {
        "satisfied": bool(all(checks.values())),
        "checks": checks,
        "floors": {
            "overall_accuracy": overall_floor,
            "balanced_accuracy": balanced_floor,
            "per_class_accuracy": class_floors,
        },
    }


def _select_candidate(
    arm: str,
    rows: list[dict[str, Any]],
    base_metrics: dict[str, Any],
    base_slice: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Select on a metadata-blind safety envelope and frozen-slice criterion."""

    if arm not in {"loss_repair", "fisher_repair", "combined_repair"}:
        raise ValueError(f"unknown repair arm: {arm}")
    for row in rows:
        row["task_label_safety"] = _safety_status(row, base_metrics, args)
    eligible = [row for row in rows if row["task_label_safety"]["satisfied"]]
    if not eligible:
        raise RuntimeError("no repair candidate satisfies the task-label safety envelope; no-op is missing")

    pareto_indices: list[int] | None = None
    if arm == "loss_repair":
        selected = min(
            eligible,
            key=lambda row: (
                row["validation_original_slice"]["weighted_calibrated_loss"],
                _candidate_penalty_size(row, args),
                row["candidate_index"],
            ),
        )
        criterion = "minimum fixed-temperature calibrated loss on the frozen original slice"
    elif arm == "fisher_repair":
        selected = max(
            eligible,
            key=lambda row: (
                row["validation_original_slice"]["predictive"]["projected_fisher"],
                -_candidate_penalty_size(row, args),
                -row["candidate_index"],
            ),
        )
        criterion = "maximum fixed-temperature predictive Fisher on the frozen original slice"
    else:
        def dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
            left_loss = left["validation_original_slice"]["weighted_calibrated_loss"]
            right_loss = right["validation_original_slice"]["weighted_calibrated_loss"]
            left_fisher = left["validation_original_slice"]["predictive"]["projected_fisher"]
            right_fisher = right["validation_original_slice"]["predictive"]["projected_fisher"]
            return bool(
                left_loss <= right_loss
                and left_fisher >= right_fisher
                and (left_loss < right_loss or left_fisher > right_fisher)
            )

        frontier = [
            row
            for row in eligible
            if not any(dominates(other, row) for other in eligible if other is not row)
        ]
        base_loss = float(base_slice["weighted_calibrated_loss"])
        base_fisher = float(base_slice["predictive"]["projected_fisher"])

        def joint_key(row: dict[str, Any]) -> tuple[float, float, float, int]:
            loss = float(row["validation_original_slice"]["weighted_calibrated_loss"])
            fisher = float(row["validation_original_slice"]["predictive"]["projected_fisher"])
            loss_gain = (base_loss - loss) / max(abs(base_loss), 1e-12)
            fisher_gain = (fisher - base_fisher) / max(abs(base_fisher), 1e-12)
            return (
                min(loss_gain, fisher_gain),
                0.5 * (loss_gain + fisher_gain),
                -_candidate_penalty_size(row, args),
                -int(row["candidate_index"]),
            )

        selected = max(frontier, key=joint_key)
        pareto_indices = [int(row["candidate_index"]) for row in frontier]
        criterion = (
            "Pareto frontier in frozen-slice loss/Fisher, then maximum worst relative improvement; "
            "ties prefer the smaller normalized coefficient"
        )

    selected_safety = selected["task_label_safety"]
    return selected, {
        "base_validation_metrics": base_metrics,
        "accuracy_budget": args.accuracy_budget,
        "balanced_accuracy_budget": args.balanced_accuracy_budget,
        "per_class_accuracy_budget": args.per_class_accuracy_budget,
        "accuracy_floor": selected_safety["floors"]["overall_accuracy"],
        "balanced_accuracy_floor": selected_safety["floors"]["balanced_accuracy"],
        "per_class_accuracy_floor": selected_safety["floors"]["per_class_accuracy"],
        "eligible_candidates": len(eligible),
        "accuracy_budget_satisfied": bool(selected_safety["satisfied"]),
        "selected_is_no_op": selected["spec"]["name"] == "no_op",
        "selected_candidate_index": int(selected["candidate_index"]),
        "pareto_frontier_candidate_indices": pareto_indices,
        "criterion": criterion,
        "rule": (
            "First require overall, balanced, and both per-class task-label accuracies to remain within their "
            "fixed budgets. Then optimize the arm-specific frozen-slice criterion without metadata."
        ),
    }


def _run_seed(args: argparse.Namespace, seed: int, device: str) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    train, val, discovery, discovery_path, input_hashes = _load_seed_inputs(data_dir, out_dir, seed)
    seed_dir = out_dir / "runs" / f"seed_{seed}"
    output_path = seed_dir / "repair.json"
    repair_protocol = {
        "algorithm_version": "repair-hardening-v2",
        "discovery_frozen_blind_sha256": discovery["frozen_blind_sha256"],
        "discovery_artifact_sha256": file_sha256(discovery_path),
        "input_hashes": input_hashes,
        "head": {
            "steps": args.head_steps,
            "learning_rate": args.head_lr,
            "weight_decay": args.head_weight_decay,
            "early_stopping_patience": args.early_stopping_patience,
            "early_stopping_eval_every": args.early_stopping_eval_every,
            "early_stopping_min_delta": args.early_stopping_min_delta,
            "checkpoint_split": "task-label-only held-out training calibration partition",
            "checkpoint_metric": "binary NLL at temperature 1; step zero is eligible",
        },
        "loss_lambdas": sorted(set(args.loss_lambdas)),
        "fisher_lambdas": sorted(set(args.fisher_lambdas)),
        "combined_loss_lambdas": sorted(set(args.combined_loss_lambdas)),
        "combined_fisher_lambdas": sorted(set(args.combined_fisher_lambdas)),
        "accuracy_budget": args.accuracy_budget,
        "balanced_accuracy_budget": args.balanced_accuracy_budget,
        "per_class_accuracy_budget": args.per_class_accuracy_budget,
        "selection": (
            "task-label-only overall/balanced/per-class safety envelope, then frozen-slice loss, Fisher, "
            "or joint Pareto criterion; an exact no-op candidate is always included"
        ),
        "fisher_penalty": "dimensionless F_initial/(F_current+epsilon); equals one at warm start",
        "temperature_convention": (
            "the base model's independently calibrated temperature is frozen for Fisher training, validation "
            "selection, fresh-adversary discovery, and later test certification for every repair arm"
        ),
        "matched_erm_continuation": (
            "same warm start, fresh AdamW optimizer, checkpoint split, early-stopping rule, and maximum "
            "additional steps with both penalties zero"
        ),
        "fresh_adversary_seed_matching": (
            "original ERM, no-op, continuation, and all repair arms use identical gate initialization, "
            "cross-fit partition, fold, and final-discovery seeds"
        ),
        "fresh_adversary_seeds": {
            "gate_config": derive_seed(seed, "primary-gate"),
            "crossfit": derive_seed(seed, "fisher-crossfit"),
            "final": derive_seed(seed, "fisher-final"),
            "partition": derive_seed(seed, "validation-crossfit-partition"),
        },
        "deterministic_torch_algorithms": bool(args.deterministic),
        "selection_uses_group_metadata": False,
    }
    repair_protocol_sha = json_sha256({"seed": seed, "repair_protocol": repair_protocol})
    if args.resume and output_path.is_file():
        existing = _read_json(output_path)
        if existing.get("repair_protocol_sha256") != repair_protocol_sha:
            raise RuntimeError(f"refusing to resume seed {seed}: repair protocol changed")
        if existing.get("seed") != seed or existing.get("stage") != "metadata_blind_repair_and_rediscovery":
            raise RuntimeError(f"refusing to resume malformed repair artifact for seed {seed}")
        if existing.get("metadata_accessed") is not False or existing.get("test_data_accessed") is not False:
            raise RuntimeError(f"refusing to resume non-blind repair artifact for seed {seed}")
        require_json_artifact_sha256(existing, "frozen_repair_sha256", label=f"seed {seed} repair")
        if existing.get("discovery_artifact_sha256") != file_sha256(discovery_path):
            raise RuntimeError(f"refusing to resume seed {seed}: discovery artifact changed")
        if existing.get("input_hashes") != input_hashes:
            raise RuntimeError(f"refusing to resume seed {seed}: prepared blind inputs changed")
        print(f"seed {seed}: repair already complete", flush=True)
        return existing

    normalization = discovery["normalization"]
    feature_mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(normalization["feature_std"], dtype=np.float32)
    train_h = apply_feature_normalizer(train.h, feature_mean, feature_std)
    val_h = apply_feature_normalizer(val.h, feature_mean, feature_std)
    train_b = (np.asarray(train.b_vl, dtype=np.float64) - normalization["vl_nuisance_mean"]) / normalization["vl_nuisance_std"]
    val_b = (np.asarray(val.b_vl, dtype=np.float64) - normalization["vl_nuisance_mean"]) / normalization["vl_nuisance_std"]
    fit_idx = np.asarray(discovery["partition"]["fit_positions"], dtype=np.int64)
    calibration_idx = np.asarray(discovery["partition"]["calibration_positions"], dtype=np.int64)
    base_head = FrozenHead.from_dict(discovery["base_model"]["head"])
    base_temperature = float(discovery["base_model"]["temperature"])

    original_gate = FrozenLinearGate.from_dict(discovery["frozen_selectors"]["primary_fisher"]["gate"])
    fit_application = apply_frozen_gate(
        original_gate,
        torch.as_tensor(train_h[fit_idx], dtype=torch.float32),
        torch.as_tensor(train.y[fit_idx], dtype=torch.long),
    )
    frozen_q_fit = canonical_probability_weights(
        fit_application.full_q.detach().cpu().numpy(),
        name="transferred repair slice",
    )
    fit_class_mask = train.y[fit_idx] == original_gate.class_value
    if np.any(frozen_q_fit[~fit_class_mask] > 0.0):
        raise RuntimeError("transferred repair slice has weight outside its selected class")

    protocol = discovery["protocol"]
    metric_config = FisherMetricConfig(protocol["shrinkage"], protocol["ridge"], protocol["epsilon"])
    base_bundle = make_model_coordinates(
        base_head,
        val_h,
        val.y,
        val_b,
        task_mean=float(normalization["task_mean"]),
        task_std=float(normalization["task_std"]),
        temperature=base_temperature,
    )
    base_validation_metrics = _task_label_validation_metrics(base_bundle)
    base_slice_metrics, _ = evaluate_gate(original_gate, base_bundle, metric_config)
    head_config = HeadConfig(
        steps=args.head_steps,
        learning_rate=args.head_lr,
        weight_decay=args.head_weight_decay,
        log_every=max(1, args.head_steps // 12),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_eval_every=args.early_stopping_eval_every,
        early_stopping_min_delta=args.early_stopping_min_delta,
        restore_best_checkpoint=True,
    )
    cv_partition_seed = derive_seed(seed, "validation-crossfit-partition")
    # Use the *same* adversary initialization and fold seeds as the original
    # ERM discovery for every repaired model and the matched continuation.
    # Consequently an exact no-op reproduces the ERM adversary exactly, while
    # between-arm differences cannot be attributed to non-convex gate luck.
    matched_gate_config_seed = derive_seed(seed, "primary-gate")
    matched_crossfit_seed = derive_seed(seed, "fisher-crossfit")
    matched_final_seed = derive_seed(seed, "fisher-final")
    candidates_by_arm: dict[str, list[dict[str, Any]]] = {}
    candidate_objects: dict[tuple[str, str, float, float], dict[str, Any]] = {}
    for arm, specs in _candidate_specs(args).items():
        rows = []
        for candidate_index, spec in enumerate(specs):
            if spec.name == "no_op":
                candidate_head = base_head
                task_mean = float(normalization["task_mean"])
                task_std = float(normalization["task_std"])
                val_bundle = base_bundle
                refit_temperature = base_temperature
                training_payload = {
                    "config": {
                        "mode": "no_op",
                        "head": asdict(head_config),
                        "repair": asdict(spec),
                        "checkpoint_metric": "not_applicable",
                        "fisher_penalty": None,
                    },
                    "history": [],
                    "final": {
                        "step": 0.0,
                        "optimization_steps_run": 0.0,
                        "selected_exact_warm_start": 1.0,
                    },
                }
            else:
                trained = train_head(
                    train_h[fit_idx],
                    train.y[fit_idx],
                    seed=derive_seed(seed, arm, spec.lambda_loss, spec.lambda_fisher, "head"),
                    device=device,
                    config=head_config,
                    initial=base_head,
                    repair=spec,
                    slice_weights=frozen_q_fit,
                    nuisance=train_b[fit_idx],
                    selected_class=original_gate.class_value,
                    temperature=base_temperature,
                    shrinkage=protocol["shrinkage"],
                    ridge=protocol["ridge"],
                    epsilon=protocol["epsilon"],
                    checkpoint_features=train_h[calibration_idx],
                    checkpoint_labels=train.y[calibration_idx],
                    normalize_fisher_penalty=True,
                )
                candidate_head = trained.head
                calibration_logits = candidate_head.logits(train_h[calibration_idx])
                refit_temperature = fit_temperature(calibration_logits, train.y[calibration_idx])
                task_mean, task_std = fit_scalar_normalizer(candidate_head.logits(train_h[fit_idx]))
                val_bundle = make_model_coordinates(
                    candidate_head,
                    val_h,
                    val.y,
                    val_b,
                    task_mean=task_mean,
                    task_std=task_std,
                    temperature=base_temperature,
                )
                training_payload = {
                    "config": trained.config,
                    "history": trained.history,
                    "final": trained.final,
                }
            validation_original_slice, _ = evaluate_gate(original_gate, val_bundle, metric_config)
            row = {
                "candidate_index": candidate_index,
                "spec": asdict(spec),
                "temperature": base_temperature,
                "refit_temperature_diagnostic_only": refit_temperature,
                "task_mean": task_mean,
                "task_std": task_std,
                "validation_metrics": _task_label_validation_metrics(val_bundle),
                "validation_original_slice": validation_original_slice,
                "training_final": training_payload["final"],
            }
            key = (arm, spec.name, spec.lambda_loss, spec.lambda_fisher)
            candidate_objects[key] = {
                "head": candidate_head,
                "training": training_payload,
                "bundle": val_bundle,
                "row": row,
            }
            rows.append(row)
        candidates_by_arm[arm] = rows

    selected_arms: dict[str, Any] = {}
    for arm, rows in candidates_by_arm.items():
        selected_row, selection = _select_candidate(
            arm,
            rows,
            base_validation_metrics,
            base_slice_metrics,
            args,
        )
        spec = RepairSpec(**selected_row["spec"])
        selected_object = candidate_objects[(arm, spec.name, spec.lambda_loss, spec.lambda_fisher)]
        fresh_config = make_gate_config(
            objective="fisher",
            slice_mass=protocol["slice_mass"],
            minimum_ess_fraction=protocol["minimum_ess_fraction"],
            max_weight_multiplier=protocol["maximum_weight_multiplier"],
            restarts=protocol["gate"]["restarts"],
            steps=protocol["gate"]["steps"],
            learning_rate=protocol["gate"]["learning_rate"],
            shrinkage=protocol["shrinkage"],
            ridge=protocol["ridge"],
            epsilon=protocol["epsilon"],
            seed=matched_gate_config_seed,
            weight_decay=protocol["gate"]["weight_decay"],
            mass_penalty_strength=protocol["gate"]["mass_penalty_strength"],
        )
        if spec.name == "no_op":
            # This model is bit-for-bit the base head.  Reuse its already
            # cross-fitted/frozen adversary so the no-op result is an exact
            # baseline, not another draw from a non-convex optimizer.
            fresh_crossfit = discovery["crossfit"]["primary_fisher"]
            final_gate = original_gate
            final_discovery = discovery["frozen_selectors"]["primary_fisher"]["discovery"]
            initialization_statement = (
                "exact no-op: reused the matched original ERM adversary and frozen gate"
            )
        else:
            fresh_crossfit = crossfit_selector(
                selected_object["bundle"],
                config=fresh_config,
                folds=protocol["crossfit_folds"],
                reference_features=train_h[calibration_idx],
                reference_labels=train.y[calibration_idx],
                device=device,
                seed=matched_crossfit_seed,
                partition_seed=cv_partition_seed,
            )
            final_gate, final_discovery = discover_final(
                selected_object["bundle"],
                config=fresh_config,
                device=device,
                seed=matched_final_seed,
            )
            initialization_statement = (
                "new parameters with initialization/fold seeds matched to ERM and every other arm"
            )
        selected_arms[arm] = {
            "spec": asdict(spec),
            "selection": selection,
            "head": selected_object["head"].to_dict(),
            "temperature": selected_row["temperature"],
            "task_mean": selected_row["task_mean"],
            "task_std": selected_row["task_std"],
            "validation_metrics": selected_row["validation_metrics"],
            "validation_original_slice": selected_row["validation_original_slice"],
            "refit_temperature_diagnostic_only": selected_row["refit_temperature_diagnostic_only"],
            "training": selected_object["training"],
            "fresh_adversary": {
                "initialization": initialization_statement,
                "crossfit": fresh_crossfit,
                "gate": final_gate.to_dict(),
                "full_validation_discovery": final_discovery,
            },
        }

    # Matched optimization control: identical warm start and extra optimizer
    # schedule, but no pseudo-slice loss or Fisher penalty.
    continued = train_head(
        train_h[fit_idx],
        train.y[fit_idx],
        seed=derive_seed(seed, "erm-continuation", "head"),
        device=device,
        config=head_config,
        initial=base_head,
        checkpoint_features=train_h[calibration_idx],
        checkpoint_labels=train.y[calibration_idx],
    )
    continued_refit_temperature = fit_temperature(
        continued.head.logits(train_h[calibration_idx]),
        train.y[calibration_idx],
    )
    continued_task_mean, continued_task_std = fit_scalar_normalizer(
        continued.head.logits(train_h[fit_idx])
    )
    continued_bundle = make_model_coordinates(
        continued.head,
        val_h,
        val.y,
        val_b,
        task_mean=continued_task_mean,
        task_std=continued_task_std,
        temperature=base_temperature,
    )
    continued_metrics = _task_label_validation_metrics(continued_bundle)
    continued_config = make_gate_config(
        objective="fisher",
        slice_mass=protocol["slice_mass"],
        minimum_ess_fraction=protocol["minimum_ess_fraction"],
        max_weight_multiplier=protocol["maximum_weight_multiplier"],
        restarts=protocol["gate"]["restarts"],
        steps=protocol["gate"]["steps"],
        learning_rate=protocol["gate"]["learning_rate"],
        shrinkage=protocol["shrinkage"],
        ridge=protocol["ridge"],
        epsilon=protocol["epsilon"],
        seed=matched_gate_config_seed,
        weight_decay=protocol["gate"]["weight_decay"],
        mass_penalty_strength=protocol["gate"]["mass_penalty_strength"],
    )
    continued_crossfit = crossfit_selector(
        continued_bundle,
        config=continued_config,
        folds=protocol["crossfit_folds"],
        reference_features=train_h[calibration_idx],
        reference_labels=train.y[calibration_idx],
        device=device,
        seed=matched_crossfit_seed,
        partition_seed=cv_partition_seed,
    )
    continued_gate, continued_discovery = discover_final(
        continued_bundle,
        config=continued_config,
        device=device,
        seed=matched_final_seed,
    )
    continued_safety = _safety_status(
        {"validation_metrics": continued_metrics},
        base_validation_metrics,
        args,
    )
    selected_arms["erm_continuation"] = {
        "spec": {"name": "erm_continuation", "lambda_loss": 0.0, "lambda_fisher": 0.0},
        "selection": {
            "base_validation_metrics": base_validation_metrics,
            "accuracy_budget": args.accuracy_budget,
            "balanced_accuracy_budget": args.balanced_accuracy_budget,
            "per_class_accuracy_budget": args.per_class_accuracy_budget,
            "accuracy_floor": continued_safety["floors"]["overall_accuracy"],
            "balanced_accuracy_floor": continued_safety["floors"]["balanced_accuracy"],
            "per_class_accuracy_floor": continued_safety["floors"]["per_class_accuracy"],
            "accuracy_budget_satisfied": bool(continued_safety["satisfied"]),
            "task_label_safety": continued_safety,
            "rule": "Matched warm-start/checkpoint control; no coefficient selection.",
        },
        "head": continued.head.to_dict(),
        "temperature": base_temperature,
        "refit_temperature_diagnostic_only": continued_refit_temperature,
        "task_mean": continued_task_mean,
        "task_std": continued_task_std,
        "validation_metrics": continued_metrics,
        "training": {
            "config": continued.config,
            "history": continued.history,
            "final": continued.final,
        },
        "fresh_adversary": {
            "initialization": (
                "new parameters with initialization/fold seeds matched to ERM and every repair arm"
            ),
            "crossfit": continued_crossfit,
            "gate": continued_gate.to_dict(),
            "full_validation_discovery": continued_discovery,
        },
    }

    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "metadata_blind_repair_and_rediscovery",
        "seed": seed,
        "metadata_accessed": False,
        "test_data_accessed": False,
        "discovery_artifact": str(discovery_path),
        "discovery_artifact_sha256": file_sha256(discovery_path),
        "discovery_frozen_blind_sha256": discovery["frozen_blind_sha256"],
        "input_hashes": input_hashes,
        "repair_protocol": repair_protocol,
        "repair_protocol_sha256": repair_protocol_sha,
        "original_slice_transfer": {
            "selected_class": int(original_gate.class_value),
            "effective_sample_size": float(fit_application.diagnostics.effective_sample_size),
            "maximum_weight": float(fit_application.diagnostics.maximum_weight),
            "sum_weight": float(frozen_q_fit.sum()),
            "gate": original_gate.to_dict(),
        },
        "candidates": candidates_by_arm,
        "selected_arms": selected_arms,
        "freeze_statement": (
            "Repair coefficients were selected inside predeclared overall, balanced, and per-class task-label "
            "accuracy budgets using a metadata-blind frozen-slice objective; an exact no-op was eligible. Each "
            "active repair received a newly initialized, seed-matched cross-fitted Fisher adversary and final "
            "validation-trained gate; an exact no-op reused the identical frozen ERM adversary. test_blind.npz "
            "and sealed_metadata.npz remained unopened."
        ),
    }
    result["frozen_repair_sha256"] = json_artifact_sha256(result, "frozen_repair_sha256")
    atomic_write_json(output_path, result)
    violations = [arm for arm, value in selected_arms.items() if not value["selection"]["accuracy_budget_satisfied"]]
    print(
        f"seed {seed}: froze repairs and fresh adversaries"
        + (f"; WARNING accuracy budget failed for {violations}" if violations else ""),
        flush=True,
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="build/results/waterbirds_hidden_slice_v2/prepared")
    parser.add_argument("--out-dir", default="build/results/waterbirds_hidden_slice_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--loss-lambdas", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--fisher-lambdas", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--combined-loss-lambdas", type=float, nargs="+", default=[0.1, 0.3, 1.0])
    parser.add_argument("--combined-fisher-lambdas", type=float, nargs="+", default=[0.01, 0.03, 0.1])
    parser.add_argument("--accuracy-budget", type=float, default=0.02)
    parser.add_argument("--balanced-accuracy-budget", type=float, default=0.02)
    parser.add_argument("--per-class-accuracy-budget", type=float, default=0.02)
    parser.add_argument("--head-steps", type=int, default=1200)
    parser.add_argument("--head-lr", type=float, default=1e-2)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--early-stopping-patience", type=int, default=12)
    parser.add_argument("--early-stopping-eval-every", type=int, default=25)
    parser.add_argument("--early-stopping-min-delta", type=float, default=1e-5)
    parser.add_argument("--deterministic", action=BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_torch_determinism(args.deterministic)
    if min(args.accuracy_budget, args.balanced_accuracy_budget, args.per_class_accuracy_budget) < 0:
        raise ValueError("all task-label accuracy budgets must be non-negative")
    if args.early_stopping_patience < 0 or args.early_stopping_eval_every < 1:
        raise ValueError("early-stopping patience must be non-negative and eval frequency positive")
    if args.early_stopping_min_delta < 0:
        raise ValueError("early_stopping_min_delta must be non-negative")
    for values, name in (
        (args.loss_lambdas, "loss_lambdas"),
        (args.fisher_lambdas, "fisher_lambdas"),
        (args.combined_loss_lambdas, "combined_loss_lambdas"),
        (args.combined_fisher_lambdas, "combined_fisher_lambdas"),
    ):
        if not values or any(value <= 0 for value in values):
            raise ValueError(f"{name} must contain positive values")
    device = str(resolve_device(args.device))
    for seed in args.seeds:
        _run_seed(args, seed, device)


if __name__ == "__main__":
    main()
