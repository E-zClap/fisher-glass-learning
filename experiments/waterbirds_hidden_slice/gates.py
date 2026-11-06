"""Transferable, class-conditional linear gates for hidden-slice discovery.

Fisher and high-loss discovery use the same gate architecture, target-mass
penalty, capped-simplex projection, and effective-sample-size constraint.  Only
the adversarial score differs.  Restarts are represented by a leading tensor
dimension and optimized together.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
from typing import Any, Literal

import torch
from torch import nn

try:  # Package import (tests and ``python -m``).
    from .geometry import (
        CenteredFisherDiagnostic,
        ScalarFisherResult,
        centered_scalar_diagnostic,
        shrunken_scalar_fisher,
    )
except ImportError:  # Direct stage scripts: ``python experiments/.../discover.py``.
    from geometry import (  # type: ignore[no-redef]
        CenteredFisherDiagnostic,
        ScalarFisherResult,
        centered_scalar_diagnostic,
        shrunken_scalar_fisher,
    )


Tensor = torch.Tensor
SelectorObjective = Literal["fisher", "high_loss"]


@dataclass(frozen=True)
class GateConstraints:
    """Transferable constraints expressed relative to each class sample size."""

    target_mass: float = 0.10
    minimum_ess_fraction: float = 0.80
    max_weight_multiplier: float = 2.0
    projection_iterations: int = 64

    def __post_init__(self) -> None:
        if not 0.0 < self.target_mass < 1.0:
            raise ValueError("target_mass must lie in (0, 1) for a sigmoid gate")
        if not 0.0 < self.minimum_ess_fraction <= 1.0:
            raise ValueError("minimum_ess_fraction must lie in (0, 1]")
        if self.max_weight_multiplier < 1.0:
            raise ValueError("max_weight_multiplier must be at least one")
        if self.projection_iterations < 16:
            raise ValueError("projection_iterations must be at least 16")

    def resolve(self, count: int) -> "ResolvedGateConstraints":
        if count <= 0:
            raise ValueError("a class-conditional gate needs at least one row")
        nominal_count = self.target_mass * float(count)
        minimum_ess = min(float(count), max(1.0, self.minimum_ess_fraction * nominal_count))
        # Appendix B caps q_i by c_q / m, where m is the nominal slice size.
        # The separate ESS guard may be a fraction of that nominal size.
        cap = min(1.0, self.max_weight_multiplier / nominal_count)
        if cap * count < 1.0 - 1e-12:
            raise ValueError("the maximum-weight constraint is infeasible")
        return ResolvedGateConstraints(
            count=count,
            nominal_count=nominal_count,
            minimum_ess=minimum_ess,
            cap=cap,
        )


@dataclass(frozen=True)
class ResolvedGateConstraints:
    count: int
    nominal_count: float
    minimum_ess: float
    cap: float


@dataclass(frozen=True)
class MembershipMassDiagnostic:
    observed_mass: Tensor
    target_mass: float
    signed_error: Tensor
    relative_squared_error: Tensor


@dataclass(frozen=True)
class WeightDiagnostics:
    membership: MembershipMassDiagnostic
    nominal_count: float
    minimum_ess: float
    effective_sample_size: Tensor
    cap: float
    maximum_weight: Tensor


@dataclass(frozen=True)
class GateOptimizationConfig:
    """Optimization and geometry settings frozen before discovery."""

    objective: SelectorObjective = "fisher"
    constraints: GateConstraints = field(default_factory=GateConstraints)
    restarts: int = 10
    steps: int = 400
    learning_rate: float = 5e-2
    weight_decay: float = 1e-4
    mass_penalty_strength: float = 10.0
    initialization_scale: float = 2e-2
    eta: float = 0.10
    ridge: float = 1e-3
    eps: float = 1e-8
    use_predictive_fisher: bool = True
    seed: int = 17

    def __post_init__(self) -> None:
        if self.objective not in {"fisher", "high_loss"}:
            raise ValueError("objective must be 'fisher' or 'high_loss'")
        if self.restarts <= 0:
            raise ValueError("restarts must be positive")
        if self.steps < 0:
            raise ValueError("steps must be nonnegative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0.0 or self.mass_penalty_strength < 0.0:
            raise ValueError("penalty strengths must be nonnegative")
        if not 0.0 <= self.eta < 1.0:
            raise ValueError("eta must lie in [0, 1)")
        if self.ridge <= 0.0 or self.eps <= 0.0:
            raise ValueError("gate optimization requires positive ridge and eps")


def membership_mass_diagnostic(membership: Tensor, target_mass: float) -> MembershipMassDiagnostic:
    """Measure whether sigmoid memberships represent the requested slice mass."""

    if membership.ndim < 1:
        raise ValueError("membership must have at least one dimension")
    if not 0.0 < target_mass <= 1.0:
        raise ValueError("target_mass must lie in (0, 1]")
    observed = membership.mean(dim=-1)
    error = observed - float(target_mass)
    relative_squared = (error / float(target_mass)).square()
    return MembershipMassDiagnostic(
        observed_mass=observed,
        target_mass=float(target_mass),
        signed_error=error,
        relative_squared_error=relative_squared,
    )


def membership_target_mass_penalty(membership: Tensor, target_mass: float) -> Tensor:
    """Per-restart relative squared target-mass penalty."""

    return membership_mass_diagnostic(membership, target_mass).relative_squared_error


def effective_sample_size(q: Tensor) -> Tensor:
    """Return ``1 / sum(q**2)`` along the final dimension."""

    if q.ndim < 1:
        raise ValueError("q must have at least one dimension")
    return torch.reciprocal(q.square().sum(dim=-1))


def _projection_tolerance(dtype: torch.dtype) -> float:
    """Numerical tolerance for validating a projected probability vector."""

    return 2e-5 if dtype in {torch.float16, torch.bfloat16, torch.float32} else 1e-10


def _capped_simplex_active_sets(
    values: Tensor,
    cap: float,
    iterations: int,
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Resolve the projection active sets without recording bisection branches.

    Differentiating through a finite bisection is incorrect: the sequence of
    Boolean branch decisions is locally constant even though the dual root is
    not.  Bisection is therefore used only to identify the lower, free, and
    upper coordinates.  ``project_capped_simplex`` subsequently reconstructs
    the dual threshold from the fixed active sets, which gives the exact
    almost-everywhere Jacobian of the Euclidean projection.
    """

    with torch.no_grad():
        detached = values.detach()
        cap_value = torch.as_tensor(float(cap), device=detached.device, dtype=detached.dtype)
        lower_dual = (detached - float(cap)).amin(dim=-1, keepdim=True)
        upper_dual = detached.amax(dim=-1, keepdim=True)
        for _ in range(iterations):
            midpoint = 0.5 * (lower_dual + upper_dual)
            candidate = torch.clamp(detached - midpoint, min=0.0, max=float(cap))
            too_large = candidate.sum(dim=-1, keepdim=True) > 1.0
            lower_dual = torch.where(too_large, midpoint, lower_dual)
            upper_dual = torch.where(too_large, upper_dual, midpoint)

        dual = 0.5 * (lower_dual + upper_dual)
        # Refining the root from its active sets removes the finite-bisection
        # residual.  Repeating also stabilizes coordinates extremely close to
        # zero or the cap; those boundary points are nondifferentiable anyway.
        for _ in range(3):
            shifted = detached - dual
            upper = shifted >= float(cap)
            lower = shifted <= 0.0
            free = ~(upper | lower)
            free_count = free.sum(dim=-1, keepdim=True)
            exact_dual = (
                (detached * free).sum(dim=-1, keepdim=True)
                + cap_value * upper.sum(dim=-1, keepdim=True)
                - 1.0
            ) / free_count.clamp_min(1)
            dual = torch.where(free_count > 0, exact_dual, dual)

        shifted = detached - dual
        upper = shifted >= float(cap)
        lower = shifted <= 0.0
        free = ~(upper | lower)
        return lower, free, upper, dual


def project_capped_simplex(
    values: Tensor,
    cap: float,
    *,
    iterations: int = 64,
    validate: bool = True,
) -> Tensor:
    """Vectorized Euclidean projection onto ``q >= 0, sum(q)=1, q<=cap``.

    The backward map is the exact almost-everywhere projection Jacobian.  If
    ``F`` is the free active set, its nonzero block is
    ``I_F - 1_F 1_F^T / |F|``; lower- and upper-clamped coordinates have zero
    derivative while their active sets remain fixed.
    """

    values = torch.as_tensor(values)
    if not values.is_floating_point():
        values = values.to(dtype=torch.get_default_dtype())
    if values.ndim < 1 or values.shape[-1] == 0:
        raise ValueError("values must have a nonempty final dimension")
    if validate and not bool(torch.isfinite(values).all()):
        raise ValueError("values must be finite")
    n = values.shape[-1]
    if cap <= 0.0 or cap * n < 1.0 - 1e-12:
        raise ValueError("cap is infeasible for a unit-sum nonnegative vector")
    cap = min(float(cap), 1.0)
    cap_value = torch.as_tensor(cap, device=values.device, dtype=values.dtype)
    lower, free, upper, detached_dual = _capped_simplex_active_sets(values, cap, iterations)
    free_count = free.sum(dim=-1, keepdim=True)
    differentiable_dual = (
        (values * free).sum(dim=-1, keepdim=True)
        + cap_value * upper.sum(dim=-1, keepdim=True)
        - 1.0
    ) / free_count.clamp_min(1)
    differentiable = torch.where(
        upper,
        torch.full_like(values, cap),
        torch.where(lower, torch.zeros_like(values), values - differentiable_dual),
    )
    # At a face with no free coordinate the projection is locally constant for
    # a fixed active set.  Choose the zero subgradient at that nondifferentiable
    # case rather than inventing a derivative through bisection.
    no_free = torch.clamp(values.detach() - detached_dual, min=0.0, max=cap)
    q = torch.where(free_count > 0, differentiable, no_free)

    if validate:
        tolerance = _projection_tolerance(q.dtype)
        totals = q.detach().sum(dim=-1)
        if not torch.allclose(totals, torch.ones_like(totals), atol=tolerance, rtol=tolerance):
            raise RuntimeError("capped-simplex projection failed to sum to one")
        if bool(((q.detach() < -tolerance) | (q.detach() > cap + tolerance)).any()):
            raise RuntimeError("capped-simplex projection violated its bounds")
    return q


def enforce_minimum_ess(q: Tensor, minimum_ess: float, *, validate: bool = True) -> Tensor:
    """Enforce ESS exactly by mixing with the uniform distribution.

    For unit-sum ``q`` and ``u=(1/n,...,1/n)``, the path
    ``u + t(q-u)`` has squared norm ``1/n + t**2 (||q||**2-1/n)``.  Solving
    this identity for ``t`` enforces the ESS boundary without iteration.  The
    mixing also preserves nonnegativity and every cap that admits ``u``.
    """

    q = torch.as_tensor(q)
    if not q.is_floating_point():
        q = q.to(dtype=torch.get_default_dtype())
    if q.ndim < 1 or q.shape[-1] == 0:
        raise ValueError("q must have a nonempty final dimension")
    n = q.shape[-1]
    if not 1.0 <= minimum_ess <= float(n):
        raise ValueError(f"minimum_ess must lie in [1, {n}]")
    tolerance = 2e-5 if q.dtype in {torch.float16, torch.bfloat16, torch.float32} else 1e-10
    if validate:
        totals = q.detach().sum(dim=-1)
        if not torch.allclose(totals, torch.ones_like(totals), atol=tolerance, rtol=tolerance):
            raise ValueError("q must sum to one before ESS enforcement")
        if bool((q.detach() < -tolerance).any()):
            raise ValueError("q must be nonnegative")
    if n == 1:
        return torch.ones_like(q)

    uniform = torch.full_like(q, 1.0 / float(n))
    squared_norm = q.square().sum(dim=-1)
    uniform_squared_norm = 1.0 / float(n)
    target_squared_norm = 1.0 / float(minimum_ess)
    denominator = (squared_norm - uniform_squared_norm).clamp_min(torch.finfo(q.dtype).tiny)
    boundary_t = torch.sqrt(
        torch.as_tensor(
            target_squared_norm - uniform_squared_norm,
            device=q.device,
            dtype=q.dtype,
        ).clamp_min(0.0)
        / denominator
    ).clamp(max=1.0)
    needs_mixing = squared_norm > target_squared_norm
    t = torch.where(needs_mixing, boundary_t, torch.ones_like(boundary_t))
    return uniform + t.unsqueeze(-1) * (q - uniform)


def admissible_weights(
    membership: Tensor,
    constraints: GateConstraints,
    *,
    validate: bool = True,
) -> tuple[Tensor, WeightDiagnostics]:
    """Convert memberships into capped, minimum-ESS probability weights."""

    membership = torch.as_tensor(membership)
    if not membership.is_floating_point():
        membership = membership.to(dtype=torch.get_default_dtype())
    if membership.ndim < 1 or membership.shape[-1] == 0:
        raise ValueError("membership must have a nonempty final dimension")
    if validate:
        detached = membership.detach()
        if not bool(torch.isfinite(detached).all()):
            raise ValueError("membership must be finite")
        if bool(((detached < 0.0) | (detached > 1.0)).any()):
            raise ValueError("membership must lie in [0, 1]")
    resolved = constraints.resolve(membership.shape[-1])
    tiny = torch.finfo(membership.dtype).tiny
    total_membership = membership.sum(dim=-1, keepdim=True)
    if validate and bool((total_membership.detach() <= tiny).any()):
        raise ValueError("membership must have positive mass")
    initial = membership / total_membership.clamp_min(tiny)

    # ``initial`` already lies on the simplex.  Returning it directly whenever
    # its cap is feasible is both exact and substantially cheaper than solving
    # a redundant projection.  The row-wise ``where`` preserves this identity
    # gradient when only some vectorized restarts need projection.
    cap_feasible = initial.detach().amax(dim=-1) <= resolved.cap
    if bool(cap_feasible.all()):
        capped = initial
    else:
        projected = project_capped_simplex(
            initial,
            resolved.cap,
            iterations=constraints.projection_iterations,
            validate=validate,
        )
        capped = torch.where(cap_feasible.unsqueeze(-1), initial, projected)
    q = enforce_minimum_ess(capped, resolved.minimum_ess, validate=validate)
    mass = membership_mass_diagnostic(membership, constraints.target_mass)
    diagnostics = WeightDiagnostics(
        membership=mass,
        nominal_count=resolved.nominal_count,
        minimum_ess=resolved.minimum_ess,
        effective_sample_size=effective_sample_size(q),
        cap=resolved.cap,
        maximum_weight=q.amax(dim=-1),
    )
    return q, diagnostics


class _CalibratedSigmoidMembership(torch.autograd.Function):
    """Sigmoid memberships with an exactly calibrated mean and implicit gradient."""

    @staticmethod
    def forward(  # type: ignore[override]
        ctx: Any,
        scores: Tensor,
        target_mass: float,
        initial_bias: Tensor,
        newton_iterations: int,
        bisection_iterations: int,
        force_full_solve: bool,
    ) -> tuple[Tensor, Tensor]:
        target = float(target_mass)
        target_logit = math.log(target / (1.0 - target))
        original_shape = scores.shape[:-1]
        flat_scores = scores.reshape(-1, scores.shape[-1])
        flat_initial_bias = initial_bias.reshape(-1)
        if flat_initial_bias.numel() != flat_scores.shape[0]:
            raise ValueError("initial_bias must have one value per score row")
        # These are guaranteed brackets: at the lower endpoint every logit is
        # at most logit(target), and at the upper endpoint every logit is at
        # least logit(target).
        lower = target_logit - flat_scores.amax(dim=-1)
        upper = target_logit - flat_scores.amin(dim=-1)

        def bisection(
            selected_scores: Tensor,
            selected_lower: Tensor,
            selected_upper: Tensor,
        ) -> Tensor:
            for _ in range(int(bisection_iterations)):
                midpoint = 0.5 * (selected_lower + selected_upper)
                observed = torch.sigmoid(
                    selected_scores + midpoint.unsqueeze(-1)
                ).mean(dim=-1)
                too_small = observed < target
                selected_lower = torch.where(too_small, midpoint, selected_lower)
                selected_upper = torch.where(too_small, selected_upper, midpoint)
            return 0.5 * (selected_lower + selected_upper)

        if bool(force_full_solve):
            flat_bias = bisection(flat_scores, lower, upper)
        else:
            flat_bias = torch.minimum(torch.maximum(flat_initial_bias, lower), upper)
            warm_lower = lower
            warm_upper = upper
            tiny = torch.finfo(flat_scores.dtype).tiny
            tolerance = max(1e-12, 8.0 * torch.finfo(flat_scores.dtype).eps)
            for _ in range(int(newton_iterations)):
                warm_membership = torch.sigmoid(flat_scores + flat_bias.unsqueeze(-1))
                observed = warm_membership.mean(dim=-1)
                slope = (warm_membership * (1.0 - warm_membership)).mean(dim=-1)
                too_small = observed < target
                warm_lower = torch.where(too_small, flat_bias, warm_lower)
                warm_upper = torch.where(too_small, warm_upper, flat_bias)
                proposed = flat_bias + (target - observed) / slope.clamp_min(tiny)
                safeguarded = (
                    torch.isfinite(proposed)
                    & (proposed > warm_lower)
                    & (proposed < warm_upper)
                )
                candidate = torch.where(
                    safeguarded,
                    proposed,
                    0.5 * (warm_lower + warm_upper),
                )
                converged = (observed - target).abs() <= tolerance
                flat_bias = torch.where(converged, flat_bias, candidate)

            warm_membership = torch.sigmoid(flat_scores + flat_bias.unsqueeze(-1))
            unresolved = (
                ~torch.isfinite(flat_bias)
                | ~torch.isfinite(warm_membership).all(dim=-1)
                | ((warm_membership.mean(dim=-1) - target).abs() > tolerance)
            )
            if bool(unresolved.any()):
                repaired = bisection(
                    flat_scores[unresolved],
                    warm_lower[unresolved],
                    warm_upper[unresolved],
                )
                flat_bias = flat_bias.clone()
                flat_bias[unresolved] = repaired

        flat_membership = torch.sigmoid(flat_scores + flat_bias.unsqueeze(-1))
        tolerance = max(1e-12, 8.0 * torch.finfo(flat_scores.dtype).eps)
        mass_error = (flat_membership.mean(dim=-1) - target).abs()
        if bool((~torch.isfinite(flat_membership).all(dim=-1) | (mass_error > tolerance)).any()):
            raise RuntimeError(
                "target-mass calibration failed: "
                f"target={target:.8g}, max_error={float(mass_error.max().item()):.8g}"
            )

        membership = flat_membership.reshape(*original_shape, scores.shape[-1])
        bias = flat_bias.reshape(original_shape)
        derivative = membership * (1.0 - membership)
        ctx.save_for_backward(derivative)
        return membership, bias

    @staticmethod
    def backward(  # type: ignore[override]
        ctx: Any,
        grad_membership: Tensor | None,
        grad_bias: Tensor | None,
    ) -> tuple[Tensor, None, None, None, None, None]:
        (derivative,) = ctx.saved_tensors
        denominator = derivative.sum(dim=-1, keepdim=True).clamp_min(
            torch.finfo(derivative.dtype).tiny
        )
        grad_scores = torch.zeros_like(derivative)
        if grad_membership is not None:
            implicit_mean = (grad_membership * derivative).sum(dim=-1, keepdim=True) / denominator
            grad_scores = derivative * (grad_membership - implicit_mean)
        if grad_bias is not None:
            grad_scores = grad_scores - grad_bias.unsqueeze(-1) * derivative / denominator
        return grad_scores, None, None, None, None, None


def calibrated_sigmoid_membership(
    scores: Tensor,
    target_mass: float,
    *,
    initial_bias: Tensor | None = None,
    newton_iterations: int = 4,
    bisection_iterations: int = 64,
    force_full_solve: bool = False,
) -> tuple[Tensor, Tensor]:
    """Calibrate one bias per leading row so mean sigmoid mass is fixed.

    The returned bias is transferable and can be frozen with the linear gate.
    Its derivative with respect to the scores is obtained by implicit
    differentiation of the mean-membership equality, rather than by
    differentiating through bisection.
    """

    scores = torch.as_tensor(scores)
    if not scores.is_floating_point():
        scores = scores.to(dtype=torch.get_default_dtype())
    if scores.ndim < 1 or scores.shape[-1] == 0:
        raise ValueError("scores must have a nonempty final dimension")
    if not 0.0 < target_mass < 1.0:
        raise ValueError("target_mass must lie in (0, 1)")
    if newton_iterations < 1:
        raise ValueError("newton_iterations must be positive")
    if bisection_iterations < 16:
        raise ValueError("bisection_iterations must be at least 16")
    expected_bias_shape = scores.shape[:-1]
    if initial_bias is None:
        target_logit = math.log(target_mass / (1.0 - target_mass))
        initial_bias = torch.full(
            expected_bias_shape,
            target_logit,
            device=scores.device,
            dtype=scores.dtype,
        )
    else:
        initial_bias = torch.as_tensor(initial_bias, device=scores.device, dtype=scores.dtype)
        if initial_bias.shape != expected_bias_shape:
            raise ValueError(
                f"initial_bias must have shape {tuple(expected_bias_shape)}, "
                f"got {tuple(initial_bias.shape)}"
            )
    return _CalibratedSigmoidMembership.apply(
        scores,
        float(target_mass),
        initial_bias,
        int(newton_iterations),
        int(bisection_iterations),
        bool(force_full_solve),
    )


class LinearGateEnsemble(nn.Module):
    """Class-specific linear sigmoid gates with vectorized restart parameters."""

    def __init__(
        self,
        feature_dim: int,
        restarts: int,
        *,
        target_mass: float,
        seed: int,
        initialization_scale: float = 2e-2,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__()
        if feature_dim <= 0 or restarts <= 0:
            raise ValueError("feature_dim and restarts must be positive")
        self.feature_dim = int(feature_dim)
        self.restarts = int(restarts)
        self.target_mass = float(target_mass)
        self.initialization_scale = float(initialization_scale)
        self.weight = nn.Parameter(torch.empty(restarts, feature_dim, device=device, dtype=dtype))
        self.bias = nn.Parameter(torch.empty(restarts, device=device, dtype=dtype))
        self.fresh_init(seed)

    def fresh_init(self, seed: int) -> None:
        """Replace every restart with an independent deterministic initialization."""

        if not 0.0 < self.target_mass < 1.0:
            raise ValueError("a sigmoid gate requires target_mass strictly between zero and one")
        generator = torch.Generator(device=self.weight.device)
        generator.manual_seed(int(seed))
        initial_weight = torch.randn(
            self.weight.shape,
            generator=generator,
            device=self.weight.device,
            dtype=self.weight.dtype,
        ) * self.initialization_scale
        initial_bias = math.log(self.target_mass / (1.0 - self.target_mass))
        with torch.no_grad():
            self.weight.copy_(initial_weight)
            self.bias.fill_(initial_bias)

    def forward(self, features: Tensor) -> Tensor:
        features = torch.as_tensor(features, device=self.weight.device, dtype=self.weight.dtype)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (n, {self.feature_dim}), got {tuple(features.shape)}"
            )
        return torch.sigmoid(self.weight @ features.transpose(0, 1) + self.bias.unsqueeze(-1))

    def calibrated_membership(
        self,
        features: Tensor,
        *,
        initial_bias: Tensor | None = None,
        newton_iterations: int = 4,
        bisection_iterations: int = 64,
        force_full_solve: bool = False,
    ) -> tuple[Tensor, Tensor]:
        """Return target-mass memberships and their transferable biases."""

        features = torch.as_tensor(features, device=self.weight.device, dtype=self.weight.dtype)
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (n, {self.feature_dim}), got {tuple(features.shape)}"
            )
        scores = self.weight @ features.transpose(0, 1)
        return calibrated_sigmoid_membership(
            scores,
            self.target_mass,
            initial_bias=self.bias.detach() if initial_bias is None else initial_bias,
            newton_iterations=newton_iterations,
            bisection_iterations=bisection_iterations,
            force_full_solve=force_full_solve,
        )


@dataclass(frozen=True)
class FrozenLinearGate:
    """One selected restart, sufficient to transfer a slice rule to new rows."""

    weight: Tensor
    bias: float
    class_value: int
    objective: SelectorObjective
    constraints: GateConstraints

    def __post_init__(self) -> None:
        if self.objective not in {"fisher", "high_loss"}:
            raise ValueError("objective must be 'fisher' or 'high_loss'")
        if self.weight.ndim != 1 or self.weight.numel() == 0:
            raise ValueError("a frozen gate weight must be a nonempty vector")
        if not bool(torch.isfinite(self.weight).all()) or not math.isfinite(self.bias):
            raise ValueError("frozen gate parameters must be finite")

    @property
    def feature_dim(self) -> int:
        return int(self.weight.numel())

    def membership(self, features: Tensor) -> Tensor:
        features = torch.as_tensor(features)
        if not features.is_floating_point():
            features = features.to(dtype=torch.get_default_dtype())
        if features.ndim != 2 or features.shape[1] != self.feature_dim:
            raise ValueError(
                f"features must have shape (n, {self.feature_dim}), got {tuple(features.shape)}"
            )
        weight = self.weight.to(device=features.device, dtype=features.dtype)
        bias = torch.as_tensor(self.bias, device=features.device, dtype=features.dtype)
        return torch.sigmoid(features @ weight + bias)

    def state_dict(self) -> dict[str, Any]:
        return {
            "weight": self.weight.detach().cpu().clone(),
            "bias": float(self.bias),
            "class_value": int(self.class_value),
            "objective": self.objective,
            "constraints": asdict(self.constraints),
        }

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        state = self.state_dict()
        state["schema_version"] = 1
        state["weight"] = state["weight"].tolist()
        return state

    @classmethod
    def from_dict(
        cls,
        payload: dict[str, Any],
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> "FrozenLinearGate":
        if int(payload.get("schema_version", 1)) != 1:
            raise ValueError("unsupported frozen-gate schema")
        return cls(
            weight=torch.as_tensor(payload["weight"], device=device, dtype=dtype),
            bias=float(payload["bias"]),
            class_value=int(payload["class_value"]),
            objective=str(payload["objective"]),  # type: ignore[arg-type]
            constraints=GateConstraints(**payload["constraints"]),
        )


@dataclass(frozen=True)
class ClassDiscoveryResult:
    class_value: int
    gate: FrozenLinearGate
    best_restart: int
    objective_score: float
    restart_scores: Tensor
    discovery_weights: Tensor
    weight_diagnostics: WeightDiagnostics
    projected_fisher: float | None
    inverse_fisher: float | None
    centered_projected_fisher: float | None
    weighted_loss: float | None


@dataclass(frozen=True)
class DiscoveryResult:
    """Per-class witnesses and the class chosen solely by discovery score."""

    objective: SelectorObjective
    selected: ClassDiscoveryResult
    per_class: tuple[ClassDiscoveryResult, ...]
    seed: int

    @property
    def gate(self) -> FrozenLinearGate:
        return self.selected.gate


@dataclass(frozen=True)
class FrozenGateApplication:
    class_value: int
    indices: Tensor
    membership: Tensor
    q: Tensor
    full_q: Tensor
    diagnostics: WeightDiagnostics


@dataclass(frozen=True)
class HeldoutSelectorEvaluation:
    application: FrozenGateApplication
    objective_score: Tensor
    fisher: ScalarFisherResult | None
    centered: CenteredFisherDiagnostic | None
    weighted_loss: Tensor | None


def _feature_matrix(features: Tensor) -> Tensor:
    features = torch.as_tensor(features)
    if not features.is_floating_point():
        features = features.to(dtype=torch.get_default_dtype())
    if features.ndim != 2 or features.shape[0] == 0 or features.shape[1] == 0:
        raise ValueError("features must be a nonempty matrix")
    if not bool(torch.isfinite(features).all()):
        raise ValueError("features must be finite")
    return features


def _row_vector(value: Tensor | None, name: str, count: int, features: Tensor) -> Tensor | None:
    if value is None:
        return None
    value = torch.as_tensor(value, device=features.device)
    if not value.is_floating_point():
        value = value.to(dtype=features.dtype)
    else:
        value = value.to(dtype=features.dtype)
    if value.ndim != 1 or value.numel() != count:
        raise ValueError(f"{name} must have shape ({count},)")
    if not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must be finite")
    return value


def _selector_scores(
    objective: SelectorObjective,
    q: Tensor,
    *,
    task: Tensor | None,
    nuisance: Tensor | None,
    predictive_weight: Tensor | None,
    losses: Tensor | None,
    config: GateOptimizationConfig,
) -> tuple[Tensor, ScalarFisherResult | None, Tensor | None]:
    if objective == "fisher":
        if task is None or nuisance is None:
            raise ValueError("Fisher discovery requires task and nuisance coordinates")
        if config.use_predictive_fisher and predictive_weight is None:
            raise ValueError("predictive Fisher discovery requires predictive_weight")
        fisher = shrunken_scalar_fisher(
            task,
            nuisance,
            q,
            eta=config.eta,
            ridge=config.ridge,
            eps=config.eps,
            predictive_weight=predictive_weight if config.use_predictive_fisher else None,
            validate=False,
        )
        score = -torch.log(fisher.projected + config.eps)
        return score, fisher, None
    if losses is None:
        raise ValueError("high-loss discovery requires per-row losses")
    weighted_loss = (q * losses).sum(dim=-1)
    return weighted_loss, None, weighted_loss


def _discover_one_class(
    features: Tensor,
    class_value: int,
    *,
    task: Tensor | None,
    nuisance: Tensor | None,
    predictive_weight: Tensor | None,
    losses: Tensor | None,
    config: GateOptimizationConfig,
    seed: int,
) -> ClassDiscoveryResult:
    gate = LinearGateEnsemble(
        features.shape[1],
        config.restarts,
        target_mass=config.constraints.target_mass,
        seed=seed,
        initialization_scale=config.initialization_scale,
        device=features.device,
        dtype=features.dtype,
    )
    optimizer = torch.optim.AdamW(
        [gate.weight],
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    calibration_iterations = max(64, config.constraints.projection_iterations)
    mass_tolerance = max(1e-6, 64.0 * torch.finfo(features.dtype).eps)

    # A constant gate is an exact, transferable member of the search family.
    # Keeping it as the initial checkpoint guarantees that optimization cannot
    # return a discovery witness worse than the no-slicing sanity candidate.
    with torch.no_grad():
        constant_membership = torch.full(
            (config.restarts, features.shape[0]),
            config.constraints.target_mass,
            device=features.device,
            dtype=features.dtype,
        )
        constant_q, _ = admissible_weights(constant_membership, config.constraints, validate=False)
        constant_scores, _, _ = _selector_scores(
            config.objective,
            constant_q,
            task=task,
            nuisance=nuisance,
            predictive_weight=predictive_weight,
            losses=losses,
            config=config,
        )
        constant_penalty = membership_target_mass_penalty(
            constant_membership,
            config.constraints.target_mass,
        )
        constant_utility = constant_scores - config.mass_penalty_strength * constant_penalty
        best_utility = constant_utility.detach().clone()
        best_weight = torch.zeros_like(gate.weight)
        constant_bias = math.log(
            config.constraints.target_mass / (1.0 - config.constraints.target_mass)
        )
        best_bias = torch.full_like(gate.bias, constant_bias)
        run_remained_finite = torch.isfinite(constant_utility).all()

    def checkpoint(
        membership: Tensor,
        calibrated_bias: Tensor,
        utility: Tensor,
    ) -> None:
        nonlocal best_utility, best_weight, best_bias, run_remained_finite
        with torch.no_grad():
            observed = membership.mean(dim=-1)
            feasible = (observed - config.constraints.target_mass).abs() <= mass_tolerance
            finite = (
                torch.isfinite(membership).all(dim=-1)
                & torch.isfinite(calibrated_bias)
                & torch.isfinite(utility)
                & torch.isfinite(gate.weight).all(dim=-1)
            )
            run_remained_finite = run_remained_finite & finite.all()
            improved = feasible & finite & (utility.detach() > best_utility)
            best_utility = torch.where(improved, utility.detach(), best_utility)
            best_weight = torch.where(improved.unsqueeze(-1), gate.weight.detach(), best_weight)
            best_bias = torch.where(improved, calibrated_bias.detach(), best_bias)

    full_calibration_period = 128
    for step in range(config.steps):
        membership, calibrated_bias = gate.calibrated_membership(
            features,
            initial_bias=gate.bias.detach(),
            newton_iterations=4,
            bisection_iterations=calibration_iterations,
            force_full_solve=(step == 0 or step % full_calibration_period == 0),
        )
        with torch.no_grad():
            gate.bias.copy_(calibrated_bias)
        q, _ = admissible_weights(membership, config.constraints, validate=False)
        scores, _, _ = _selector_scores(
            config.objective,
            q,
            task=task,
            nuisance=nuisance,
            predictive_weight=predictive_weight,
            losses=losses,
            config=config,
        )
        mass_penalty = membership_target_mass_penalty(
            membership,
            config.constraints.target_mass,
        )
        utility = scores - config.mass_penalty_strength * mass_penalty
        checkpoint(membership, calibrated_bias, utility)
        objective = -utility.mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        with torch.no_grad():
            if gate.weight.grad is None:
                raise RuntimeError("gate optimizer produced no weight gradient")
            run_remained_finite = run_remained_finite & torch.isfinite(gate.weight.grad).all()
        optimizer.step()

    # Include the state produced by the final optimizer update.
    membership, calibrated_bias = gate.calibrated_membership(
        features,
        initial_bias=gate.bias.detach(),
        newton_iterations=4,
        bisection_iterations=calibration_iterations,
        force_full_solve=True,
    )
    q, _ = admissible_weights(membership, config.constraints, validate=False)
    scores, _, _ = _selector_scores(
        config.objective,
        q,
        task=task,
        nuisance=nuisance,
        predictive_weight=predictive_weight,
        losses=losses,
        config=config,
    )
    final_penalty = membership_target_mass_penalty(membership, config.constraints.target_mass)
    checkpoint(
        membership,
        calibrated_bias,
        scores - config.mass_penalty_strength * final_penalty,
    )

    with torch.no_grad():
        if not bool(run_remained_finite):
            raise RuntimeError("gate optimization encountered a non-finite value or gradient")
        gate.weight.copy_(best_weight)
        gate.bias.copy_(best_bias)
        membership = gate(features)
        q, diagnostics = admissible_weights(membership, config.constraints)
        scores, fisher, weighted_loss = _selector_scores(
            config.objective,
            q,
            task=task,
            nuisance=nuisance,
            predictive_weight=predictive_weight,
            losses=losses,
            config=config,
        )
        mass_penalty = membership_target_mass_penalty(membership, config.constraints.target_mass)
        selection_scores = scores - config.mass_penalty_strength * mass_penalty
        observed_mass = diagnostics.membership.observed_mass
        if bool(((observed_mass - config.constraints.target_mass).abs() > mass_tolerance).any()):
            raise RuntimeError(
                "calibrated discovery gate violated target membership mass: "
                f"target={config.constraints.target_mass:.8g}, "
                f"observed={observed_mass.detach().cpu().tolist()}"
            )
        score_tolerance = max(1e-6, 64.0 * torch.finfo(selection_scores.dtype).eps)
        if bool((selection_scores + score_tolerance < constant_utility).any()):
            raise RuntimeError("best gate checkpoint regressed below the constant-gate sanity candidate")
        if not bool(torch.isfinite(selection_scores).all()):
            raise RuntimeError("gate selection scores are non-finite")
        best = int(torch.argmax(selection_scores).item())
        frozen = FrozenLinearGate(
            weight=gate.weight[best].detach().cpu().clone(),
            bias=float(gate.bias[best].item()),
            class_value=int(class_value),
            objective=config.objective,
            constraints=config.constraints,
        )
        report_fisher = fisher
        if report_fisher is None and task is not None and nuisance is not None:
            if not config.use_predictive_fisher or predictive_weight is not None:
                report_fisher = shrunken_scalar_fisher(
                    task,
                    nuisance,
                    q,
                    eta=config.eta,
                    ridge=config.ridge,
                    eps=config.eps,
                    predictive_weight=predictive_weight if config.use_predictive_fisher else None,
                    validate=False,
                )
        report_loss = weighted_loss
        if report_loss is None and losses is not None:
            report_loss = (q * losses).sum(dim=-1)
        centered_value: float | None = None
        if report_fisher is not None and task is not None and nuisance is not None:
            centered = centered_scalar_diagnostic(
                task,
                nuisance,
                q[best],
                eta=config.eta,
                ridge=config.ridge,
                eps=config.eps,
                predictive_weight=predictive_weight if config.use_predictive_fisher else None,
            )
            centered_value = float(centered.fisher.projected.item())
        return ClassDiscoveryResult(
            class_value=int(class_value),
            gate=frozen,
            best_restart=best,
            objective_score=float(selection_scores[best].item()),
            restart_scores=selection_scores.detach().cpu().clone(),
            discovery_weights=q[best].detach().cpu().clone(),
            weight_diagnostics=WeightDiagnostics(
                membership=MembershipMassDiagnostic(
                    observed_mass=diagnostics.membership.observed_mass[best].detach().cpu(),
                    target_mass=diagnostics.membership.target_mass,
                    signed_error=diagnostics.membership.signed_error[best].detach().cpu(),
                    relative_squared_error=diagnostics.membership.relative_squared_error[best].detach().cpu(),
                ),
                nominal_count=diagnostics.nominal_count,
                minimum_ess=diagnostics.minimum_ess,
                effective_sample_size=diagnostics.effective_sample_size[best].detach().cpu(),
                cap=diagnostics.cap,
                maximum_weight=diagnostics.maximum_weight[best].detach().cpu(),
            ),
            projected_fisher=None if report_fisher is None else float(report_fisher.projected[best].item()),
            inverse_fisher=None if report_fisher is None else float(report_fisher.inverse[best].item()),
            centered_projected_fisher=centered_value,
            weighted_loss=None if report_loss is None else float(report_loss[best].item()),
        )


def discover_class_conditional(
    features: Tensor,
    labels: Tensor,
    *,
    config: GateOptimizationConfig,
    task_coordinate: Tensor | None = None,
    nuisance_coordinate: Tensor | None = None,
    predictive_weight: Tensor | None = None,
    losses: Tensor | None = None,
    class_values: tuple[int, ...] | list[int] | None = None,
) -> DiscoveryResult:
    """Discover one witness per class, then select by discovery objective only.

    Every call constructs fresh gate parameters from ``config.seed``.  Calling
    this function again after a model change therefore reinitializes the
    adversary instead of warm-starting from an earlier slice.
    """

    features = _feature_matrix(features)
    labels = torch.as_tensor(labels, device=features.device)
    if labels.ndim != 1 or labels.numel() != features.shape[0]:
        raise ValueError("labels must align with feature rows")
    count = features.shape[0]
    task_coordinate = _row_vector(task_coordinate, "task_coordinate", count, features)
    nuisance_coordinate = _row_vector(nuisance_coordinate, "nuisance_coordinate", count, features)
    predictive_weight = _row_vector(predictive_weight, "predictive_weight", count, features)
    losses = _row_vector(losses, "losses", count, features)
    if predictive_weight is not None and bool((predictive_weight < 0).any()):
        raise ValueError("predictive_weight must be nonnegative")
    if losses is not None and bool((losses < 0).any()):
        raise ValueError("losses must be nonnegative")

    if class_values is None:
        class_values = [int(value.item()) for value in torch.unique(labels, sorted=True)]
    if not class_values:
        raise ValueError("at least one class value is required")
    results: list[ClassDiscoveryResult] = []
    for position, class_value in enumerate(class_values):
        mask = labels == int(class_value)
        if not bool(mask.any()):
            raise ValueError(f"class {class_value} has no discovery rows")
        class_seed = int(config.seed) + 104729 * position
        results.append(
            _discover_one_class(
                features[mask],
                int(class_value),
                task=None if task_coordinate is None else task_coordinate[mask],
                nuisance=None if nuisance_coordinate is None else nuisance_coordinate[mask],
                predictive_weight=None if predictive_weight is None else predictive_weight[mask],
                losses=None if losses is None else losses[mask],
                config=config,
                seed=class_seed,
            )
        )
    selected = max(results, key=lambda result: result.objective_score)
    return DiscoveryResult(
        objective=config.objective,
        selected=selected,
        per_class=tuple(results),
        seed=int(config.seed),
    )


def apply_frozen_gate(
    gate: FrozenLinearGate,
    features: Tensor,
    labels: Tensor,
) -> FrozenGateApplication:
    """Apply a frozen class-specific rule without consulting score outer products."""

    features = _feature_matrix(features)
    labels = torch.as_tensor(labels, device=features.device)
    if labels.ndim != 1 or labels.numel() != features.shape[0]:
        raise ValueError("labels must align with feature rows")
    indices = torch.nonzero(labels == gate.class_value, as_tuple=False).flatten()
    if indices.numel() == 0:
        raise ValueError(f"class {gate.class_value} has no held-out rows")
    membership = gate.membership(features[indices])
    q, diagnostics = admissible_weights(membership, gate.constraints)
    full_q = torch.zeros(features.shape[0], device=q.device, dtype=q.dtype)
    full_q[indices] = q
    return FrozenGateApplication(
        class_value=gate.class_value,
        indices=indices,
        membership=membership,
        q=q,
        full_q=full_q,
        diagnostics=diagnostics,
    )


def evaluate_frozen_selector(
    gate: FrozenLinearGate,
    features: Tensor,
    labels: Tensor,
    *,
    eta: float = 0.1,
    ridge: float = 1e-3,
    eps: float = 1e-8,
    use_predictive_fisher: bool = True,
    task_coordinate: Tensor | None = None,
    nuisance_coordinate: Tensor | None = None,
    predictive_weight: Tensor | None = None,
    losses: Tensor | None = None,
) -> HeldoutSelectorEvaluation:
    """Score a frozen rule on held-out rows using freshly computed outer products."""

    features = _feature_matrix(features)
    count = features.shape[0]
    application = apply_frozen_gate(gate, features, labels)
    task_coordinate = _row_vector(task_coordinate, "task_coordinate", count, features)
    nuisance_coordinate = _row_vector(nuisance_coordinate, "nuisance_coordinate", count, features)
    predictive_weight = _row_vector(predictive_weight, "predictive_weight", count, features)
    losses = _row_vector(losses, "losses", count, features)
    indices = application.indices

    fisher: ScalarFisherResult | None = None
    centered: CenteredFisherDiagnostic | None = None
    if task_coordinate is not None and nuisance_coordinate is not None:
        if use_predictive_fisher and predictive_weight is None:
            raise ValueError("predictive Fisher evaluation requires predictive_weight")
        task = task_coordinate[indices]
        nuisance = nuisance_coordinate[indices]
        weight = predictive_weight[indices] if use_predictive_fisher and predictive_weight is not None else None
        fisher = shrunken_scalar_fisher(
            task,
            nuisance,
            application.q,
            eta=eta,
            ridge=ridge,
            eps=eps,
            predictive_weight=weight,
        )
        centered = centered_scalar_diagnostic(
            task,
            nuisance,
            application.q,
            eta=eta,
            ridge=ridge,
            eps=eps,
            predictive_weight=weight,
        )
    elif gate.objective == "fisher":
        raise ValueError("Fisher evaluation requires task and nuisance coordinates")

    weighted_loss = None if losses is None else (application.q * losses[indices]).sum()
    if gate.objective == "fisher":
        assert fisher is not None
        objective_score = -torch.log(fisher.projected + eps)
    else:
        if weighted_loss is None:
            raise ValueError("high-loss evaluation requires per-row losses")
        objective_score = weighted_loss
    return HeldoutSelectorEvaluation(
        application=application,
        objective_score=objective_score,
        fisher=fisher,
        centered=centered,
        weighted_loss=weighted_loss,
    )
