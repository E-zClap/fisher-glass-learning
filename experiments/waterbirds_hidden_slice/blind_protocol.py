"""Reusable group-blind discovery, cross-fitting, and evaluation workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from itertools import combinations
from typing import Any

import numpy as np
import torch

from common import canonical_probability_weights, derive_seed, stratified_kfold_indices
from gates import (
    ClassDiscoveryResult,
    DiscoveryResult,
    FrozenLinearGate,
    GateConstraints,
    GateOptimizationConfig,
    apply_frozen_gate,
    discover_class_conditional,
)
from heads import FrozenHead, accuracy, fit_scalar_normalizer, per_example_loss, probabilities
from metrics import FisherMetricConfig, evaluate_class_slice


@dataclass(frozen=True)
class ModelCoordinates:
    features: np.ndarray
    labels: np.ndarray
    nuisance: np.ndarray
    logits: np.ndarray
    task: np.ndarray
    probability: np.ndarray
    predictive_weight: np.ndarray
    loss: np.ndarray
    temperature: float

    def take(self, indices: np.ndarray) -> "ModelCoordinates":
        index = np.asarray(indices, dtype=np.int64)
        return ModelCoordinates(
            features=self.features[index],
            labels=self.labels[index],
            nuisance=self.nuisance[index],
            logits=self.logits[index],
            task=self.task[index],
            probability=self.probability[index],
            predictive_weight=self.predictive_weight[index],
            loss=self.loss[index],
            temperature=self.temperature,
        )


def make_model_coordinates(
    head: FrozenHead,
    features: np.ndarray,
    labels: np.ndarray,
    nuisance: np.ndarray,
    *,
    task_mean: float,
    task_std: float,
    temperature: float,
) -> ModelCoordinates:
    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    nuisance = np.asarray(nuisance, dtype=np.float64)
    logits = head.logits(features)
    if labels.shape != logits.shape or nuisance.shape != logits.shape:
        raise ValueError("features, labels, and nuisance do not align")
    task = ((logits - task_mean) / task_std).astype(np.float64)
    probability = probabilities(logits, temperature)
    return ModelCoordinates(
        features=features,
        labels=labels,
        nuisance=nuisance,
        logits=logits,
        task=task,
        probability=probability,
        predictive_weight=probability * (1.0 - probability),
        loss=per_example_loss(logits, labels),
        temperature=float(temperature),
    )


def fit_task_scale(head: FrozenHead, fit_features: np.ndarray) -> tuple[float, float]:
    return fit_scalar_normalizer(head.logits(fit_features))


def overall_metrics(bundle: ModelCoordinates) -> dict[str, float]:
    calibrated_loss = per_example_loss(bundle.logits, bundle.labels, bundle.temperature)
    confidence = np.maximum(bundle.probability, 1.0 - bundle.probability)
    entropy = -(
        np.clip(bundle.probability, 1e-12, 1.0) * np.log(np.clip(bundle.probability, 1e-12, 1.0))
        + np.clip(1.0 - bundle.probability, 1e-12, 1.0)
        * np.log(np.clip(1.0 - bundle.probability, 1e-12, 1.0))
    )
    return {
        "accuracy": accuracy(bundle.logits, bundle.labels),
        "loss": float(bundle.loss.mean()),
        "calibrated_loss": float(calibrated_loss.mean()),
        "mean_confidence": float(confidence.mean()),
        "mean_entropy": float(entropy.mean()),
    }


def random_nuisance_axis(feature_dim: int, seed: int) -> np.ndarray:
    if feature_dim < 1:
        raise ValueError("feature_dim must be positive")
    rng = np.random.default_rng(seed)
    axis = rng.normal(size=feature_dim)
    norm = float(np.linalg.norm(axis))
    if norm == 0:
        raise RuntimeError("random nuisance axis had zero norm")
    return (axis / norm).astype(np.float32)


def normalize_nuisance(
    fit_values: np.ndarray,
    *others: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], tuple[float, float]]:
    mean, std = fit_scalar_normalizer(fit_values)
    normalized = tuple(((np.asarray(values, dtype=np.float64) - mean) / std).astype(np.float64) for values in (fit_values, *others))
    return normalized, (mean, std)


def make_gate_config(
    *,
    objective: str,
    slice_mass: float,
    minimum_ess_fraction: float,
    max_weight_multiplier: float,
    restarts: int,
    steps: int,
    learning_rate: float,
    shrinkage: float,
    ridge: float,
    epsilon: float,
    seed: int,
    weight_decay: float = 1e-4,
    mass_penalty_strength: float = 10.0,
) -> GateOptimizationConfig:
    return GateOptimizationConfig(
        objective=objective,
        constraints=GateConstraints(
            target_mass=slice_mass,
            minimum_ess_fraction=minimum_ess_fraction,
            max_weight_multiplier=max_weight_multiplier,
        ),
        restarts=restarts,
        steps=steps,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        mass_penalty_strength=mass_penalty_strength,
        eta=shrinkage,
        ridge=ridge,
        eps=epsilon,
        use_predictive_fisher=True,
        seed=seed,
    )


def _torch_bundle(bundle: ModelCoordinates, device: str) -> dict[str, torch.Tensor]:
    return {
        "features": torch.as_tensor(bundle.features, dtype=torch.float32, device=device),
        "labels": torch.as_tensor(bundle.labels, dtype=torch.long, device=device),
        "task": torch.as_tensor(bundle.task, dtype=torch.float32, device=device),
        "nuisance": torch.as_tensor(bundle.nuisance, dtype=torch.float32, device=device),
        "predictive_weight": torch.as_tensor(bundle.predictive_weight, dtype=torch.float32, device=device),
        "loss": torch.as_tensor(bundle.loss, dtype=torch.float32, device=device),
    }


def _discover(bundle: ModelCoordinates, config: GateOptimizationConfig, device: str) -> DiscoveryResult:
    tensors = _torch_bundle(bundle, device)
    if config.objective == "fisher":
        return discover_class_conditional(
            tensors["features"],
            tensors["labels"],
            config=config,
            task_coordinate=tensors["task"],
            nuisance_coordinate=tensors["nuisance"],
            predictive_weight=tensors["predictive_weight"],
        )
    return discover_class_conditional(
        tensors["features"],
        tensors["labels"],
        config=config,
        losses=tensors["loss"],
    )


def _diagnostic_number(value) -> float:
    return float(torch.as_tensor(value).detach().cpu())


def class_discovery_to_dict(result: ClassDiscoveryResult, *, include_gate: bool = True) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "class_id": int(result.class_value),
        "best_restart": int(result.best_restart),
        "objective_score": float(result.objective_score),
        "restart_scores": result.restart_scores.detach().cpu().tolist(),
        "projected_fisher": result.projected_fisher,
        "inverse_fisher": result.inverse_fisher,
        "centered_projected_fisher": result.centered_projected_fisher,
        "weighted_loss": result.weighted_loss,
        "constraints": {
            "target_membership_mass": result.weight_diagnostics.membership.target_mass,
            "observed_membership_mass": _diagnostic_number(result.weight_diagnostics.membership.observed_mass),
            "nominal_count": result.weight_diagnostics.nominal_count,
            "minimum_ess": result.weight_diagnostics.minimum_ess,
            "effective_sample_size": _diagnostic_number(result.weight_diagnostics.effective_sample_size),
            "weight_cap": result.weight_diagnostics.cap,
            "maximum_weight": _diagnostic_number(result.weight_diagnostics.maximum_weight),
        },
    }
    if include_gate:
        payload["gate"] = result.gate.to_dict()
    return payload


def discovery_to_dict(result: DiscoveryResult, *, include_gates: bool = True) -> dict[str, Any]:
    return {
        "objective": result.objective,
        "seed": int(result.seed),
        "selected_class": int(result.selected.class_value),
        "selected": class_discovery_to_dict(result.selected, include_gate=include_gates),
        "per_class": [class_discovery_to_dict(item, include_gate=include_gates) for item in result.per_class],
    }


def evaluate_gate(
    gate: FrozenLinearGate,
    bundle: ModelCoordinates,
    metric_config: FisherMetricConfig,
    *,
    device: str = "cpu",
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    features = torch.as_tensor(bundle.features, dtype=torch.float32, device=device)
    labels = torch.as_tensor(bundle.labels, dtype=torch.long, device=device)
    application = apply_frozen_gate(gate, features, labels)
    indices = application.indices.detach().cpu().numpy().astype(np.int64)
    q = canonical_probability_weights(
        application.q.detach().cpu().numpy(),
        name="held-out gate weights",
    )
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
    metrics["gate_class"] = int(gate.class_value)
    metrics["membership_mass"] = _diagnostic_number(application.diagnostics.membership.observed_mass)
    artifacts = {
        "indices": indices,
        "membership": application.membership.detach().cpu().numpy().astype(np.float64),
        "q": q,
        "full_q": np.zeros(bundle.labels.size, dtype=np.float64),
    }
    artifacts["full_q"][indices] = q
    return metrics, artifacts


def _pairwise_stability(gates: list[FrozenLinearGate], features: np.ndarray) -> dict[str, float | int]:
    if len(gates) < 2:
        return {"pairs": 0, "score_correlation_mean": 1.0, "top_overlap_mean": 1.0}
    score_vectors = []
    top_sets = []
    features_t = torch.as_tensor(features, dtype=torch.float32)
    for gate in gates:
        score = gate.membership(features_t).detach().cpu().numpy()
        score_vectors.append(score)
        k = max(1, int(np.ceil(gate.constraints.target_mass * score.size)))
        top_sets.append(set(np.argpartition(score, -k)[-k:].tolist()))
    correlations: list[float] = []
    overlaps: list[float] = []
    for left, right in combinations(range(len(gates)), 2):
        a, b = score_vectors[left], score_vectors[right]
        if float(a.std()) < 1e-12 or float(b.std()) < 1e-12:
            correlation = 1.0 if np.allclose(a, b) else 0.0
        else:
            correlation = float(np.corrcoef(a, b)[0, 1])
        union = top_sets[left] | top_sets[right]
        overlap = len(top_sets[left] & top_sets[right]) / max(len(union), 1)
        correlations.append(correlation)
        overlaps.append(float(overlap))
    return {
        "pairs": len(correlations),
        "score_correlation_mean": float(np.mean(correlations)),
        "score_correlation_min": float(np.min(correlations)),
        "top_overlap_mean": float(np.mean(overlaps)),
        "top_overlap_min": float(np.min(overlaps)),
    }


def crossfit_selector(
    bundle: ModelCoordinates,
    *,
    config: GateOptimizationConfig,
    folds: int,
    reference_features: np.ndarray,
    reference_labels: np.ndarray,
    device: str,
    seed: int,
    partition_seed: int | None = None,
    include_fold_gates: bool = False,
) -> dict[str, Any]:
    """Discover on complementary folds and certify on untouched held-out folds."""
    resolved_partition_seed = seed if partition_seed is None else int(partition_seed)
    fold_indices = stratified_kfold_indices(bundle.labels, folds, resolved_partition_seed)
    fold_rows: list[dict[str, Any]] = []
    candidate_gates_by_class: dict[int, list[FrozenLinearGate]] = {0: [], 1: []}
    selected_gates_by_class: dict[int, list[FrozenLinearGate]] = {0: [], 1: []}
    selected_classes: list[int] = []
    for fold_index, (discover_idx, cert_idx) in enumerate(fold_indices):
        print(
            "gate cross-fit "
            f"objective={config.objective} mass={config.constraints.target_mass:.3g} "
            f"shrinkage={config.eta:.3g} fold={fold_index + 1}/{folds} "
            f"discover/cert={discover_idx.size}/{cert_idx.size}",
            flush=True,
        )
        fold_config = replace(config, seed=derive_seed(seed, config.objective, "fold", fold_index))
        result = _discover(bundle.take(discover_idx), fold_config, device)
        for class_result in result.per_class:
            candidate_gates_by_class[int(class_result.class_value)].append(class_result.gate)
        selected_gates_by_class[int(result.gate.class_value)].append(result.gate)
        heldout_device = "cpu" if include_fold_gates else device
        heldout, _ = evaluate_gate(
            result.gate,
            bundle.take(cert_idx),
            FisherMetricConfig(fold_config.eta, fold_config.ridge, fold_config.eps),
            device=heldout_device,
        )
        selected_classes.append(int(result.gate.class_value))
        fold_row = {
            "fold": fold_index,
            "n_discover": int(discover_idx.size),
            "n_cert": int(cert_idx.size),
            "discovery": discovery_to_dict(result, include_gates=False),
            "heldout": heldout,
        }
        if include_fold_gates:
            # The group-free lambda selector persists these gates so its
            # held-out Fisher values can be recomputed from the archived
            # validation rows before the test split is opened.
            fold_row["gate"] = result.gate.to_dict()
        fold_rows.append(fold_row)

    inverse = np.asarray([row["heldout"]["predictive"]["inverse_fisher"] for row in fold_rows], dtype=float)
    fisher = np.asarray([row["heldout"]["predictive"]["projected_fisher"] for row in fold_rows], dtype=float)
    unweighted = np.asarray([row["heldout"]["unweighted"]["projected_fisher"] for row in fold_rows], dtype=float)
    centered_unweighted = np.asarray(
        [row["heldout"]["centered_unweighted"]["projected_fisher"] for row in fold_rows],
        dtype=float,
    )
    centered_ratio = np.asarray(
        [row["heldout"]["centered_unweighted"]["projected_over_raw"] for row in fold_rows],
        dtype=float,
    )
    centered_correlation = np.asarray(
        [row["heldout"]["centered_unweighted"]["correlation"] for row in fold_rows],
        dtype=float,
    )
    loss = np.asarray([row["heldout"]["weighted_loss"] for row in fold_rows], dtype=float)
    accuracy_values = np.asarray([row["heldout"]["weighted_accuracy"] for row in fold_rows], dtype=float)
    stability = {}
    candidate_stability = {}
    reference_features = np.asarray(reference_features, dtype=np.float32)
    reference_labels = np.asarray(reference_labels, dtype=np.int64)
    for class_id in (0, 1):
        mask = reference_labels == class_id
        stability[str(class_id)] = _pairwise_stability(
            selected_gates_by_class[class_id],
            reference_features[mask],
        )
        candidate_stability[str(class_id)] = _pairwise_stability(
            candidate_gates_by_class[class_id],
            reference_features[mask],
        )
    selected_counts = {str(class_id): int(selected_classes.count(class_id)) for class_id in (0, 1)}
    payload = {
        "config": {
            **asdict(config),
            "constraints": asdict(config.constraints),
        },
        "partition_seed": resolved_partition_seed,
        "folds": fold_rows,
        "aggregate": {
            "hidden_certificate": float(1.0 / inverse.mean()),
            "inverse_fisher_mean": float(inverse.mean()),
            "predictive_fisher_mean": float(fisher.mean()),
            "predictive_fisher_std": float(fisher.std(ddof=1)) if fisher.size > 1 else 0.0,
            "unweighted_fisher_mean": float(unweighted.mean()),
            "centered_unweighted_fisher_mean": float(centered_unweighted.mean()),
            "centered_projected_over_raw_mean": float(centered_ratio.mean()),
            "centered_task_nuisance_correlation_mean": float(centered_correlation.mean()),
            "weighted_loss_mean": float(loss.mean()),
            "weighted_accuracy_mean": float(accuracy_values.mean()),
            "selected_class_counts": selected_counts,
            "stability": stability,
            "candidate_stability_diagnostic": candidate_stability,
        },
    }
    if include_fold_gates:
        payload["heldout_evaluation_device"] = "cpu"
    return payload


def discover_final(
    bundle: ModelCoordinates,
    *,
    config: GateOptimizationConfig,
    device: str,
    seed: int,
) -> tuple[FrozenLinearGate, dict[str, Any]]:
    print(
        "gate final fit "
        f"objective={config.objective} mass={config.constraints.target_mass:.3g} "
        f"shrinkage={config.eta:.3g} rows={len(bundle.labels)}",
        flush=True,
    )
    final_config = replace(config, seed=seed)
    result = _discover(bundle, final_config, device)
    return result.gate, discovery_to_dict(result, include_gates=True)
