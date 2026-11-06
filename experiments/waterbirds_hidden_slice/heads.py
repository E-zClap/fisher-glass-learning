"""Frozen-feature binary heads and pseudo-slice repair objectives.

This module deliberately knows nothing about Waterbirds metadata.  Oracle
Group DRO receives an explicit integer ``groups`` array from the post-hoc
stage; all group-blind paths call :func:`train_head` without it.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

try:  # Package import (tests and ``python -m``).
    from .common import canonical_probability_weights
except ImportError:  # Direct stage scripts.
    from common import canonical_probability_weights


@dataclass(frozen=True)
class HeadConfig:
    steps: int = 1200
    learning_rate: float = 1e-2
    weight_decay: float = 1e-4
    gradient_clip: float = 10.0
    log_every: int = 100
    early_stopping_patience: int = 0
    early_stopping_eval_every: int = 25
    early_stopping_min_delta: float = 1e-5
    restore_best_checkpoint: bool = True

    def validate(self) -> None:
        if self.steps < 1:
            raise ValueError("steps must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        if self.gradient_clip < 0:
            raise ValueError("gradient_clip must be non-negative")
        if self.log_every < 1:
            raise ValueError("log_every must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping_eval_every < 1:
            raise ValueError("early_stopping_eval_every must be positive")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta must be non-negative")


@dataclass(frozen=True)
class RepairSpec:
    name: str
    lambda_loss: float = 0.0
    lambda_fisher: float = 0.0

    def validate(self) -> None:
        if self.name not in {"no_op", "loss_repair", "fisher_repair", "combined_repair"}:
            raise ValueError(f"unknown repair name: {self.name}")
        if self.lambda_loss < 0 or self.lambda_fisher < 0:
            raise ValueError("repair coefficients must be non-negative")
        if self.name == "no_op" and not (self.lambda_loss == 0 and self.lambda_fisher == 0):
            raise ValueError("no_op requires both coefficients to be zero")
        if self.name == "loss_repair" and not (self.lambda_loss > 0 and self.lambda_fisher == 0):
            raise ValueError("loss_repair requires lambda_loss > 0 and lambda_fisher == 0")
        if self.name == "fisher_repair" and not (self.lambda_fisher > 0 and self.lambda_loss == 0):
            raise ValueError("fisher_repair requires lambda_fisher > 0 and lambda_loss == 0")
        if self.name == "combined_repair" and not (self.lambda_loss > 0 and self.lambda_fisher > 0):
            raise ValueError("combined_repair requires both coefficients to be positive")


@dataclass(frozen=True)
class FrozenHead:
    weight: np.ndarray
    bias: float

    def __post_init__(self) -> None:
        weight = np.asarray(self.weight, dtype=np.float32)
        if weight.ndim != 1 or weight.size == 0:
            raise ValueError("weight must be a non-empty vector")
        if not np.all(np.isfinite(weight)) or not np.isfinite(self.bias):
            raise ValueError("head parameters must be finite")
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "bias", float(self.bias))

    def logits(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim != 2 or features.shape[1] != self.weight.size:
            raise ValueError("feature dimension does not match the frozen head")
        return (features @ self.weight + self.bias).astype(np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {"weight": self.weight.tolist(), "bias": self.bias}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "FrozenHead":
        return cls(np.asarray(payload["weight"], dtype=np.float32), float(payload["bias"]))


@dataclass(frozen=True)
class TrainingResult:
    head: FrozenHead
    history: list[dict[str, float]]
    final: dict[str, float]
    config: dict[str, Any]


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested but CUDA is unavailable: {name}")
    return device


def seed_torch(seed: int) -> None:
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def fit_feature_normalizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    features = np.asarray(features, dtype=np.float32)
    if features.ndim != 2 or features.shape[0] < 2:
        raise ValueError("features must be a two-dimensional array with at least two rows")
    mean = features.mean(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std = features.std(axis=0, keepdims=True, dtype=np.float64).astype(np.float32)
    std[std < 1e-6] = 1.0
    return mean, std


def apply_feature_normalizer(features: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    features = np.asarray(features, dtype=np.float32)
    mean = np.asarray(mean, dtype=np.float32)
    std = np.asarray(std, dtype=np.float32)
    if mean.shape != (1, features.shape[1]) or std.shape != mean.shape:
        raise ValueError("feature normalizer has an incompatible shape")
    if np.any(std <= 0):
        raise ValueError("feature standard deviations must be positive")
    return ((features - mean) / std).astype(np.float32)


def fit_scalar_normalizer(values: np.ndarray) -> tuple[float, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or values.size < 2:
        raise ValueError("values must be a vector with at least two entries")
    mean = float(values.mean())
    std = float(values.std())
    return mean, std if std >= 1e-6 else 1.0


def apply_scalar_normalizer(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    if not np.isfinite(mean) or not np.isfinite(std) or std <= 0:
        raise ValueError("invalid scalar normalizer")
    return ((np.asarray(values, dtype=np.float64) - mean) / std).astype(np.float64)


def probabilities(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = np.clip(np.asarray(logits, dtype=np.float64) / temperature, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scaled))


def per_example_loss(logits: np.ndarray, labels: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    labels = np.asarray(labels, dtype=np.float64)
    p = probabilities(logits, temperature)
    return -(labels * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - labels) * np.log(np.clip(1.0 - p, 1e-12, 1.0)))


def accuracy(logits: np.ndarray, labels: np.ndarray) -> float:
    return float(((np.asarray(logits) >= 0).astype(np.int64) == np.asarray(labels, dtype=np.int64)).mean())


def _initial_parameters(dimension: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    weight = torch.randn(dimension, generator=generator, dtype=torch.float32) * 0.01
    bias = torch.zeros((), dtype=torch.float32)
    return weight.to(device).requires_grad_(True), bias.to(device).requires_grad_(True)


def _head_parameters(initial: FrozenHead | None, dimension: int, seed: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    if initial is None:
        return _initial_parameters(dimension, seed, device)
    if initial.weight.size != dimension:
        raise ValueError("initial head dimension does not match the features")
    weight = torch.as_tensor(initial.weight, dtype=torch.float32, device=device).clone().requires_grad_(True)
    bias = torch.tensor(initial.bias, dtype=torch.float32, device=device, requires_grad=True)
    return weight, bias


def _task_coordinate(logits: torch.Tensor) -> torch.Tensor:
    """Current full-fit logit normalization used by the existing Waterbirds code."""
    return (logits - logits.mean()) / (logits.std(unbiased=False) + 1e-6)


def _slice_fisher(
    logits: torch.Tensor,
    nuisance: torch.Tensor,
    q: torch.Tensor,
    class_mask: torch.Tensor,
    temperature: float,
    shrinkage: float,
    ridge: float,
) -> torch.Tensor:
    """Appendix-B scalar predictive Fisher for a frozen soft slice.

    ``q`` sums to one on ``class_mask`` and is zero elsewhere.  Importantly,
    ``q * p * (1-p)`` is not renormalized: confidence remains part of the
    predictive information scale.
    """
    task = _task_coordinate(logits)
    p = torch.sigmoid(logits / temperature)
    predictive = p * (1.0 - p)

    def blocks(weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        wr = weights * predictive
        return (
            torch.sum(wr * task.square()),
            torch.sum(wr * task * nuisance),
            torch.sum(wr * nuisance.square()),
        )

    local = blocks(q)
    n_class = int(class_mask.sum().item())
    if n_class < 1:
        raise ValueError("the selected class is absent from the repair split")
    uniform = class_mask.to(logits.dtype) / float(n_class)
    global_blocks = blocks(uniform)
    mixed = tuple((1.0 - shrinkage) * left + shrinkage * right for left, right in zip(local, global_blocks))
    j_tt, j_tn, j_nn = mixed
    raw = j_tt - j_tn.square() / (j_nn + ridge)
    return torch.clamp(raw, min=0.0)


def train_head(
    features: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
    device: str,
    config: HeadConfig,
    initial: FrozenHead | None = None,
    repair: RepairSpec | None = None,
    slice_weights: np.ndarray | None = None,
    nuisance: np.ndarray | None = None,
    selected_class: int | None = None,
    temperature: float = 1.0,
    shrinkage: float = 0.1,
    ridge: float = 1e-3,
    epsilon: float = 1e-8,
    groups: np.ndarray | None = None,
    checkpoint_features: np.ndarray | None = None,
    checkpoint_labels: np.ndarray | None = None,
    normalize_fisher_penalty: bool = True,
    fisher_reference: float | None = None,
) -> TrainingResult:
    """Train ERM, a pseudo-slice repair, or the explicit oracle Group DRO head.

    The three modes are mutually exclusive: ``repair`` activates a frozen
    pseudo-slice objective; ``groups`` activates oracle Group DRO; with neither
    argument the objective is ordinary ERM.  Warm-start fine-tuning can use a
    task-label-only checkpoint split.  Its binary NLL is evaluated without
    temperature refitting, and the best checkpoint (including step zero) is
    restored, so every arm uses the same scale-sensitive overfitting guard.

    When requested, the Fisher reciprocal is multiplied by its value of
    ``F`` at the initial head.  The resulting dimensionless penalty equals one
    at the warm start, which makes a lambda comparable across seeds and gates
    without changing the direction of its gradient.
    """
    config.validate()
    if repair is not None:
        repair.validate()
    if repair is not None and groups is not None:
        raise ValueError("repair and oracle Group DRO cannot be enabled together")
    if repair is not None and repair.name == "no_op":
        raise ValueError("no_op is an artifact candidate and must not be passed to train_head")
    if not (0.0 <= shrinkage < 1.0):
        raise ValueError("shrinkage must lie in [0, 1)")
    if ridge <= 0 or epsilon <= 0 or temperature <= 0:
        raise ValueError("ridge, epsilon, and temperature must be positive")
    if fisher_reference is not None and (not np.isfinite(fisher_reference) or fisher_reference <= 0):
        raise ValueError("fisher_reference must be finite and positive")

    features = np.asarray(features, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.int64)
    if features.ndim != 2 or labels.shape != (features.shape[0],):
        raise ValueError("features/labels have incompatible shapes")
    if not np.isin(labels, [0, 1]).all():
        raise ValueError("binary labels must be encoded as 0/1")
    if not np.all(np.isfinite(features)):
        raise ValueError("features contain non-finite values")

    checkpoint_x_np: np.ndarray | None = None
    checkpoint_y_np: np.ndarray | None = None
    if (checkpoint_features is None) != (checkpoint_labels is None):
        raise ValueError("checkpoint_features and checkpoint_labels must be supplied together")
    if checkpoint_features is not None and checkpoint_labels is not None:
        checkpoint_x_np = np.asarray(checkpoint_features, dtype=np.float32)
        checkpoint_y_np = np.asarray(checkpoint_labels, dtype=np.int64)
        if checkpoint_x_np.ndim != 2 or checkpoint_x_np.shape[1] != features.shape[1]:
            raise ValueError("checkpoint_features have an incompatible shape")
        if checkpoint_y_np.shape != (checkpoint_x_np.shape[0],) or checkpoint_y_np.size == 0:
            raise ValueError("checkpoint_labels have an incompatible shape")
        if not np.isin(checkpoint_y_np, [0, 1]).all():
            raise ValueError("checkpoint labels must be encoded as 0/1")
        if not np.all(np.isfinite(checkpoint_x_np)):
            raise ValueError("checkpoint_features contain non-finite values")
    if config.early_stopping_patience > 0 and checkpoint_x_np is None:
        raise ValueError("early stopping requires a checkpoint split")

    q_np: np.ndarray | None = None
    nuisance_np: np.ndarray | None = None
    class_mask_np: np.ndarray | None = None
    if repair is not None:
        if slice_weights is None or nuisance is None or selected_class not in (0, 1):
            raise ValueError("repair requires slice_weights, nuisance, and selected_class")
        q_np = np.asarray(slice_weights, dtype=np.float64)
        nuisance_np = np.asarray(nuisance, dtype=np.float64)
        if q_np.shape != labels.shape or nuisance_np.shape != labels.shape:
            raise ValueError("repair arrays must align with labels")
        class_mask_np = labels == int(selected_class)
        q_np = canonical_probability_weights(q_np, name="slice_weights")
        if np.any(q_np[~class_mask_np] > 1e-10):
            raise ValueError("slice weights must be class-conditional")

    groups_np: np.ndarray | None = None
    if groups is not None:
        groups_np = np.asarray(groups, dtype=np.int64)
        if groups_np.shape != labels.shape:
            raise ValueError("groups must align with labels")
        unique = np.unique(groups_np)
        if not np.array_equal(unique, np.arange(unique.size)):
            raise ValueError("oracle groups must be contiguous integers starting at zero")
        if unique.size < 2:
            raise ValueError("oracle Group DRO requires at least two groups")

    seed_torch(seed)
    torch_device = resolve_device(device)
    x = torch.as_tensor(features, dtype=torch.float32, device=torch_device)
    y = torch.as_tensor(labels, dtype=torch.float32, device=torch_device)
    q = None if q_np is None else torch.as_tensor(q_np, dtype=torch.float32, device=torch_device).detach()
    b = None if nuisance_np is None else torch.as_tensor(nuisance_np, dtype=torch.float32, device=torch_device).detach()
    class_mask = None if class_mask_np is None else torch.as_tensor(class_mask_np, dtype=torch.bool, device=torch_device)
    group_tensor = None if groups_np is None else torch.as_tensor(groups_np, dtype=torch.long, device=torch_device)
    checkpoint_x = (
        None
        if checkpoint_x_np is None
        else torch.as_tensor(checkpoint_x_np, dtype=torch.float32, device=torch_device)
    )
    checkpoint_y = (
        None
        if checkpoint_y_np is None
        else torch.as_tensor(checkpoint_y_np, dtype=torch.float32, device=torch_device)
    )

    weight, bias = _head_parameters(initial, x.shape[1], seed, torch_device)
    optimizer = torch.optim.AdamW([weight, bias], lr=config.learning_rate, weight_decay=config.weight_decay)
    history: list[dict[str, float]] = []
    resolved_fisher_reference: float | None = None
    if repair is not None and repair.lambda_fisher > 0:
        assert q is not None and b is not None and class_mask is not None
        if fisher_reference is None:
            with torch.no_grad():
                initial_fisher = _slice_fisher(
                    x @ weight + bias,
                    b,
                    q,
                    class_mask,
                    temperature,
                    shrinkage,
                    ridge,
                )
            resolved_fisher_reference = max(float(initial_fisher.detach().cpu()), float(epsilon))
        else:
            resolved_fisher_reference = float(fisher_reference)

    def objective_terms(logits: torch.Tensor) -> tuple[torch.Tensor, ...]:
        losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
        erm = losses.mean()
        objective = erm
        slice_loss = torch.zeros((), dtype=logits.dtype, device=torch_device)
        fisher = torch.zeros((), dtype=logits.dtype, device=torch_device)
        inverse_fisher = torch.zeros((), dtype=logits.dtype, device=torch_device)
        normalized_fisher_penalty = torch.zeros((), dtype=logits.dtype, device=torch_device)

        if group_tensor is not None:
            unique_groups = torch.unique(group_tensor, sorted=True)
            group_losses = torch.stack([losses[group_tensor == gid].mean() for gid in unique_groups])
            objective = group_losses.max()
        elif repair is not None:
            assert q is not None and b is not None and class_mask is not None
            slice_loss = torch.sum(q * losses)
            if repair.lambda_fisher > 0:
                fisher = _slice_fisher(logits, b, q, class_mask, temperature, shrinkage, ridge)
                inverse_fisher = 1.0 / (fisher + epsilon)
                if normalize_fisher_penalty:
                    assert resolved_fisher_reference is not None
                    normalized_fisher_penalty = inverse_fisher * resolved_fisher_reference
                else:
                    normalized_fisher_penalty = inverse_fisher
            objective = (
                erm
                + repair.lambda_loss * slice_loss
                + repair.lambda_fisher * normalized_fisher_penalty
            )
        return (
            objective,
            erm,
            slice_loss,
            fisher,
            inverse_fisher,
            normalized_fisher_penalty,
        )

    def checkpoint_metric() -> float:
        assert checkpoint_x is not None and checkpoint_y is not None
        with torch.no_grad():
            checkpoint_logits = checkpoint_x @ weight + bias
            value = F.binary_cross_entropy_with_logits(checkpoint_logits, checkpoint_y)
        return float(value.detach().cpu())

    checkpoint_enabled = checkpoint_x is not None
    best_checkpoint_metric = float("inf")
    best_checkpoint_step = 0
    best_weight = weight.detach().clone()
    best_bias = bias.detach().clone()
    stale_evaluations = 0
    if checkpoint_enabled:
        best_checkpoint_metric = checkpoint_metric()

    optimization_steps_run = 0
    early_stopped = False
    for step in range(config.steps):
        optimizer.zero_grad(set_to_none=True)
        logits = x @ weight + bias
        (
            objective,
            erm,
            slice_loss,
            fisher,
            inverse_fisher,
            normalized_fisher_penalty,
        ) = objective_terms(logits)

        if not torch.isfinite(objective):
            raise FloatingPointError(f"non-finite training objective at step {step}")
        objective.backward()
        if config.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_([weight, bias], config.gradient_clip)
        optimizer.step()
        optimization_steps_run = step + 1

        row = {
            "step": float(step + 1),
            "objective": float(objective.detach().cpu()),
            "erm_loss": float(erm.detach().cpu()),
            "slice_loss": float(slice_loss.detach().cpu()),
            "slice_fisher": float(fisher.detach().cpu()),
            "inverse_slice_fisher": float(inverse_fisher.detach().cpu()),
            "normalized_fisher_penalty": float(normalized_fisher_penalty.detach().cpu()),
        }
        evaluated_checkpoint = checkpoint_enabled and (
            (step + 1) % config.early_stopping_eval_every == 0 or step + 1 == config.steps
        )
        if evaluated_checkpoint:
            current_metric = checkpoint_metric()
            row["checkpoint_binary_nll"] = current_metric
            if current_metric < best_checkpoint_metric - config.early_stopping_min_delta:
                best_checkpoint_metric = current_metric
                best_checkpoint_step = step + 1
                best_weight = weight.detach().clone()
                best_bias = bias.detach().clone()
                stale_evaluations = 0
            else:
                stale_evaluations += 1

        should_log = (
            step == 0
            or step + 1 == config.steps
            or (step + 1) % config.log_every == 0
            or evaluated_checkpoint
        )
        if should_log:
            history.append(dict(row))

        if (
            evaluated_checkpoint
            and config.early_stopping_patience > 0
            and stale_evaluations >= config.early_stopping_patience
        ):
            early_stopped = True
            break

    if checkpoint_enabled and config.restore_best_checkpoint:
        with torch.no_grad():
            weight.copy_(best_weight)
            bias.copy_(best_bias)
        selected_step = best_checkpoint_step
        selected_checkpoint_metric = best_checkpoint_metric
    else:
        selected_step = optimization_steps_run
        selected_checkpoint_metric = checkpoint_metric() if checkpoint_enabled else None

    with torch.no_grad():
        final_terms = objective_terms(x @ weight + bias)
    final = {
        "step": float(selected_step),
        "optimization_steps_run": float(optimization_steps_run),
        "objective": float(final_terms[0].detach().cpu()),
        "erm_loss": float(final_terms[1].detach().cpu()),
        "slice_loss": float(final_terms[2].detach().cpu()),
        "slice_fisher": float(final_terms[3].detach().cpu()),
        "inverse_slice_fisher": float(final_terms[4].detach().cpu()),
        "normalized_fisher_penalty": float(final_terms[5].detach().cpu()),
        "early_stopped": float(early_stopped),
    }
    if selected_checkpoint_metric is not None:
        final["checkpoint_binary_nll"] = float(selected_checkpoint_metric)

    frozen = FrozenHead(weight.detach().cpu().numpy(), float(bias.detach().cpu()))
    mode = "oracle_group_dro" if groups is not None else repair.name if repair is not None else "erm"
    return TrainingResult(
        head=frozen,
        history=history,
        final=final,
        config={
            "mode": mode,
            "head": asdict(config),
            "repair": None if repair is None else asdict(repair),
            "checkpoint_metric": "binary_nll_at_temperature_1" if checkpoint_enabled else None,
            "fisher_penalty": {
                "normalized_to_initial_fisher": bool(
                    repair is not None and repair.lambda_fisher > 0 and normalize_fisher_penalty
                ),
                "initial_fisher_reference": resolved_fisher_reference,
            },
        },
    )
