"""Group-blind slice metrics and held-out uncertainty summaries."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Callable

import numpy as np
import torch
from scipy.stats import t as student_t

try:  # Package import (tests and ``python -m``).
    from .common import canonical_probability_weights
    from .geometry import centered_scalar_diagnostic, shrunken_scalar_fisher
    from .heads import per_example_loss, probabilities
except ImportError:  # Direct stage scripts: ``python experiments/.../certify.py``.
    from common import canonical_probability_weights
    from geometry import centered_scalar_diagnostic, shrunken_scalar_fisher
    from heads import per_example_loss, probabilities


@dataclass(frozen=True)
class FisherMetricConfig:
    shrinkage: float = 0.1
    ridge: float = 1e-3
    epsilon: float = 1e-8

    def validate(self) -> None:
        if not 0.0 <= self.shrinkage < 1.0:
            raise ValueError("shrinkage must lie in [0, 1)")
        if self.ridge <= 0 or self.epsilon <= 0:
            raise ValueError("ridge and epsilon must be positive")


def _as_vector(values: np.ndarray, name: str, dtype=np.float64) -> np.ndarray:
    result = np.asarray(values, dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if not np.all(np.isfinite(result)):
        raise ValueError(f"{name} contains non-finite values")
    return result


def entropy_from_probability(probability: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(probability, dtype=np.float64), 1e-12, 1.0 - 1e-12)
    return -(p * np.log(p) + (1.0 - p) * np.log(1.0 - p))


def effective_sample_size(q: np.ndarray) -> float:
    q = _as_vector(q, "q")
    denominator = float(np.sum(q * q))
    return float(1.0 / denominator) if denominator > 0 else 0.0


def weighted_mean(values: np.ndarray, q: np.ndarray) -> float:
    values = _as_vector(values, "values")
    q = _as_vector(q, "q")
    if values.shape != q.shape:
        raise ValueError("values and q must align")
    return float(np.sum(q * values))


def _fisher_to_dict(result) -> dict[str, float | str | None]:
    j_tt = float(result.moments.j_tt.detach().cpu())
    j_tn = float(result.moments.j_tn.detach().cpu())
    j_nn = float(result.moments.j_nn.detach().cpu())
    projected = float(result.projected.detach().cpu())
    exact = None if j_nn <= 1e-12 else max(j_tt - (j_tn * j_tn) / j_nn, 0.0)
    return {
        "j_tt": j_tt,
        "j_tn": j_tn,
        "j_nn": j_nn,
        "raw_projected_fisher": float(result.raw_projected.detach().cpu()),
        "projected_fisher": projected,
        "inverse_fisher": float(result.inverse.detach().cpu()),
        "ridge_beta": float(result.beta.detach().cpu()),
        "exact_projected_fisher": exact,
        "ridge_to_exact_gap": None if exact is None else projected - exact,
        "exact_status": "singular_nuisance" if exact is None else "ok",
    }


def evaluate_class_slice(
    *,
    task: np.ndarray,
    nuisance: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    q: np.ndarray,
    class_id: int,
    temperature: float,
    config: FisherMetricConfig,
    hard_mass: float = 0.1,
) -> dict:
    """Evaluate one frozen class-conditional soft slice.

    Arrays are already restricted to the selected class and ``q`` sums to one.
    The function contains no metadata argument by construction.
    """
    config.validate()
    if class_id not in (0, 1):
        raise ValueError("class_id must be 0 or 1")
    if not 0.0 < hard_mass <= 1.0:
        raise ValueError("hard_mass must lie in (0, 1]")
    task = _as_vector(task, "task")
    nuisance = _as_vector(nuisance, "nuisance")
    logits = _as_vector(logits, "logits")
    labels = _as_vector(labels, "labels", dtype=np.int64)
    q = canonical_probability_weights(q)
    n = task.size
    if any(value.size != n for value in (nuisance, logits, labels, q)):
        raise ValueError("all slice arrays must align")
    if n == 0 or not np.all(labels == class_id):
        raise ValueError("evaluate_class_slice expects a non-empty single-class sample")

    probability = probabilities(logits, temperature)
    predictive_weight = probability * (1.0 - probability)
    tensor_kwargs = {"dtype": torch.float64}
    task_t = torch.as_tensor(task, **tensor_kwargs)
    nuisance_t = torch.as_tensor(nuisance, **tensor_kwargs)
    q_t = torch.as_tensor(q, **tensor_kwargs)
    predictive_t = torch.as_tensor(predictive_weight, **tensor_kwargs)

    predictive = shrunken_scalar_fisher(
        task_t,
        nuisance_t,
        q_t,
        predictive_weight=predictive_t,
        eta=config.shrinkage,
        ridge=config.ridge,
        eps=config.epsilon,
    )
    unweighted = shrunken_scalar_fisher(
        task_t,
        nuisance_t,
        q_t,
        eta=config.shrinkage,
        ridge=config.ridge,
        eps=config.epsilon,
    )
    centered = centered_scalar_diagnostic(
        task_t,
        nuisance_t,
        q_t,
        eta=config.shrinkage,
        ridge=config.ridge,
        eps=config.epsilon,
    )

    losses = per_example_loss(logits, labels)
    calibrated_losses = per_example_loss(logits, labels, temperature)
    predictions = (logits >= 0).astype(np.int64)
    hard_count = max(1, int(np.ceil(hard_mass * n)))
    top = np.argpartition(q, -hard_count)[-hard_count:]
    return {
        "class_id": int(class_id),
        "n_class": int(n),
        "constraints": {
            "sum": float(q.sum()),
            "effective_sample_size": effective_sample_size(q),
            "max_weight": float(q.max()),
            "nonzero_weights": int(np.count_nonzero(q > 0)),
        },
        "predictive": _fisher_to_dict(predictive),
        "unweighted": _fisher_to_dict(unweighted),
        "centered_unweighted": {
            **_fisher_to_dict(centered.fisher),
            "projected_over_raw": float(centered.projected_over_raw.detach().cpu()),
            "correlation": float(centered.correlation.detach().cpu()),
        },
        "weighted_accuracy": weighted_mean((predictions == labels).astype(float), q),
        "weighted_loss": weighted_mean(losses, q),
        "weighted_calibrated_loss": weighted_mean(calibrated_losses, q),
        "mean_probability": weighted_mean(probability, q),
        "mean_confidence": weighted_mean(np.maximum(probability, 1.0 - probability), q),
        "mean_entropy": weighted_mean(entropy_from_probability(probability), q),
        "mean_abs_margin": weighted_mean(np.abs(logits / temperature), q),
        "hard_top_weight_accuracy_diagnostic": float((predictions[top] == labels[top]).mean()),
        "hard_top_mass": float(hard_mass),
        "fisher_config": asdict(config),
    }


def paired_seed_interval(differences: np.ndarray, confidence: float = 0.95) -> dict[str, float | int]:
    values = _as_vector(differences, "differences")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    # The protocol fixes 95%; avoid an optional scipy dependency for other
    # levels by using a normal quantile approximation only at this reporting
    # layer. Bootstrap intervals remain the held-out within-seed uncertainty.
    if not np.isclose(confidence, 0.95):
        raise ValueError("paired_seed_interval currently supports confidence=0.95")
    mean = float(values.mean())
    if values.size < 2:
        return {"n": int(values.size), "mean": mean, "se": 0.0, "lo": mean, "hi": mean}
    se = float(values.std(ddof=1) / np.sqrt(values.size))
    critical = float(student_t.ppf(0.5 + confidence / 2.0, df=values.size - 1))
    return {
        "n": int(values.size),
        "mean": mean,
        "se": se,
        "lo": mean - critical * se,
        "hi": mean + critical * se,
    }


def bootstrap_fixed_gate_fisher(
    *,
    membership: np.ndarray,
    task: np.ndarray,
    nuisance: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    make_weights: Callable[[np.ndarray], np.ndarray],
    config: FisherMetricConfig,
    temperature: float,
    hard_mass: float,
    replicates: int,
    confidence: float,
    seed: int,
) -> dict[str, float | int]:
    """Nonparametric held-out bootstrap conditional on a frozen gate rule.

    The callable reprojects the resampled memberships onto the same cap/ESS
    family. Gate parameters are never refit inside the bootstrap.
    """
    if replicates < 1:
        raise ValueError("replicates must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    arrays = [
        _as_vector(membership, "membership"),
        _as_vector(task, "task"),
        _as_vector(nuisance, "nuisance"),
        _as_vector(logits, "logits"),
        _as_vector(labels, "labels", dtype=np.int64),
    ]
    n = arrays[0].size
    if n < 2 or any(array.size != n for array in arrays):
        raise ValueError("bootstrap arrays must align and contain at least two examples")
    rng = np.random.default_rng(seed)
    point_q = make_weights(arrays[0])
    point_metric = evaluate_class_slice(
        task=arrays[1],
        nuisance=arrays[2],
        logits=arrays[3],
        labels=arrays[4],
        q=point_q,
        class_id=class_id,
        temperature=temperature,
        config=config,
        hard_mass=hard_mass,
    )
    draws = np.empty(replicates, dtype=np.float64)
    for replicate in range(replicates):
        sample = rng.integers(0, n, size=n)
        q = make_weights(arrays[0][sample])
        metric = evaluate_class_slice(
            task=arrays[1][sample],
            nuisance=arrays[2][sample],
            logits=arrays[3][sample],
            labels=arrays[4][sample],
            q=q,
            class_id=class_id,
            temperature=temperature,
            config=config,
            hard_mass=hard_mass,
        )
        draws[replicate] = metric["predictive"]["projected_fisher"]
    alpha = (1.0 - confidence) / 2.0
    return {
        "replicates": int(replicates),
        "confidence": float(confidence),
        "point": float(point_metric["predictive"]["projected_fisher"]),
        "mean": float(draws.mean()),
        "standard_error": float(draws.std(ddof=1)) if replicates > 1 else 0.0,
        "lo": float(np.quantile(draws, alpha)),
        "hi": float(np.quantile(draws, 1.0 - alpha)),
    }
