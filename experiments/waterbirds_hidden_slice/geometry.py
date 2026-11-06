"""Differentiable scalar Fisher--Schur geometry for hidden-slice experiments.

The functions in this module implement the scalar task/nuisance specialization
of the Appendix-B construction.  Slice weights ``q`` are probability weights.
In predictive mode the Fisher coefficient is exactly ``q * r``, where
``r = p(1-p)``; ``r`` is deliberately *not* renormalized.  Unweighted mode is
obtained by omitting ``predictive_weight``.

All functions accept either one slice-weight vector of shape ``(n,)`` or a
vectorized collection of vectors with shape ``(..., n)``.  This permits gate
restarts to be optimized in parallel while keeping the score coordinates fixed.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


Tensor = torch.Tensor


@dataclass(frozen=True)
class ScalarMoments:
    """The three entries of a symmetric scalar task/nuisance Fisher block."""

    j_tt: Tensor
    j_tn: Tensor
    j_nn: Tensor


@dataclass(frozen=True)
class ScalarFisherResult:
    """A ridge Schur result with a single, explicit reciprocal floor."""

    moments: ScalarMoments
    raw_projected: Tensor
    projected: Tensor
    inverse: Tensor
    beta: Tensor


@dataclass(frozen=True)
class CenteredFisherDiagnostic:
    """Shift-robust companion diagnostic computed from centered coordinates."""

    fisher: ScalarFisherResult
    projected_over_raw: Tensor
    correlation: Tensor


def _coordinate_vector(value: Tensor, name: str, *, validate: bool = True) -> Tensor:
    value = torch.as_tensor(value)
    if value.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got shape {tuple(value.shape)}")
    if not value.is_floating_point():
        value = value.to(dtype=torch.get_default_dtype())
    if validate and not bool(torch.isfinite(value).all()):
        raise ValueError(f"{name} must contain only finite values")
    return value


def _probability_weights(q: Tensor, n: int, *, validate: bool) -> Tensor:
    q = torch.as_tensor(q)
    if not q.is_floating_point():
        q = q.to(dtype=torch.get_default_dtype())
    if q.ndim < 1 or q.shape[-1] != n:
        raise ValueError(f"q must have final dimension {n}, got shape {tuple(q.shape)}")
    if not bool(torch.isfinite(q).all()):
        raise ValueError("q must contain only finite values")
    if validate:
        detached = q.detach()
        tolerance = 2e-5 if q.dtype in {torch.float16, torch.bfloat16, torch.float32} else 1e-10
        if bool((detached < -tolerance).any()):
            raise ValueError("q must be nonnegative")
        totals = detached.sum(dim=-1)
        if not torch.allclose(totals, torch.ones_like(totals), atol=tolerance, rtol=tolerance):
            raise ValueError("q must sum to one along its final dimension")
    return q


def scalar_moments(
    task: Tensor,
    nuisance: Tensor,
    q: Tensor,
    *,
    predictive_weight: Tensor | None = None,
    center: bool = False,
    validate: bool = True,
) -> ScalarMoments:
    """Compute scalar task/nuisance Fisher moments.

    Parameters
    ----------
    task, nuisance:
        Fixed operational coordinates with shape ``(n,)``.
    q:
        One or more probability vectors with shape ``(..., n)``.
    predictive_weight:
        Optional nonnegative ``r`` with shape ``(n,)``.  The coefficient of
        each outer product is ``q_i r_i``.  In particular, this function never
        divides by ``sum(q * r)``.
    center:
        If true, subtract coefficient-weighted coordinate means before forming
        the block.  The outer-product scale remains ``sum(q * r)`` in
        predictive mode, so centering does not erase confidence saturation.
    """

    task = _coordinate_vector(task, "task", validate=validate)
    nuisance = _coordinate_vector(nuisance, "nuisance", validate=validate).to(
        device=task.device,
        dtype=task.dtype,
    )
    if nuisance.shape != task.shape:
        raise ValueError("task and nuisance must have the same shape")
    q = _probability_weights(q, task.numel(), validate=validate).to(device=task.device, dtype=task.dtype)

    coefficient = q
    if predictive_weight is not None:
        predictive_weight = _coordinate_vector(
            predictive_weight,
            "predictive_weight",
            validate=validate,
        ).to(
            device=task.device,
            dtype=task.dtype,
        )
        if predictive_weight.shape != task.shape:
            raise ValueError("predictive_weight must have the same shape as task")
        if validate and bool((predictive_weight.detach() < 0).any()):
            raise ValueError("predictive_weight must be nonnegative")
        coefficient = coefficient * predictive_weight

    task_used = task
    nuisance_used = nuisance
    if center:
        total = coefficient.sum(dim=-1)
        tiny = torch.finfo(task.dtype).tiny
        safe_total = total.clamp_min(tiny)
        task_mean = (coefficient * task).sum(dim=-1) / safe_total
        nuisance_mean = (coefficient * nuisance).sum(dim=-1) / safe_total
        task_used = task - task_mean.unsqueeze(-1)
        nuisance_used = nuisance - nuisance_mean.unsqueeze(-1)

    return ScalarMoments(
        j_tt=(coefficient * task_used.square()).sum(dim=-1),
        j_tn=(coefficient * task_used * nuisance_used).sum(dim=-1),
        j_nn=(coefficient * nuisance_used.square()).sum(dim=-1),
    )


def uniform_scalar_moments(
    task: Tensor,
    nuisance: Tensor,
    *,
    predictive_weight: Tensor | None = None,
    center: bool = False,
    validate: bool = True,
) -> ScalarMoments:
    """Compute the global block on the supplied class-specific sample."""

    task = _coordinate_vector(task, "task", validate=validate)
    if task.numel() == 0:
        raise ValueError("the class-specific sample must not be empty")
    q = torch.full_like(task, 1.0 / float(task.numel()))
    return scalar_moments(
        task,
        nuisance,
        q,
        predictive_weight=predictive_weight,
        center=center,
        validate=validate,
    )


def shrink_scalar_moments(
    slice_moments: ScalarMoments,
    global_moments: ScalarMoments,
    eta: float,
) -> ScalarMoments:
    """Shrink the complete Fisher block before taking its Schur complement."""

    if not 0.0 <= eta < 1.0:
        raise ValueError(f"eta must lie in [0, 1), got {eta}")
    keep = 1.0 - float(eta)
    return ScalarMoments(
        j_tt=keep * slice_moments.j_tt + eta * global_moments.j_tt,
        j_tn=keep * slice_moments.j_tn + eta * global_moments.j_tn,
        j_nn=keep * slice_moments.j_nn + eta * global_moments.j_nn,
    )


def ridge_schur(
    moments: ScalarMoments,
    *,
    ridge: float = 1e-3,
    eps: float = 1e-8,
    validate: bool = True,
) -> ScalarFisherResult:
    """Apply the scalar ridge Schur complement and reciprocal map.

    Numerical negatives are clamped to zero.  ``eps`` is then added exactly
    once, in the reciprocal.  The nuisance ridge and reciprocal floor therefore
    retain separate meanings.
    """

    if ridge < 0.0:
        raise ValueError(f"ridge must be nonnegative, got {ridge}")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}")
    denominator = moments.j_nn + float(ridge)
    if validate and bool((denominator.detach() <= 0).any()):
        raise ValueError("the nuisance denominator must be positive")
    beta = moments.j_tn / denominator
    raw_projected = moments.j_tt - moments.j_tn.square() / denominator
    projected = torch.clamp(raw_projected, min=0.0)
    inverse = torch.reciprocal(projected + float(eps))
    return ScalarFisherResult(
        moments=moments,
        raw_projected=raw_projected,
        projected=projected,
        inverse=inverse,
        beta=beta,
    )


def shrunken_scalar_fisher(
    task: Tensor,
    nuisance: Tensor,
    q: Tensor,
    *,
    eta: float = 0.1,
    ridge: float = 1e-3,
    eps: float = 1e-8,
    predictive_weight: Tensor | None = None,
    center: bool = False,
    global_moments: ScalarMoments | None = None,
    validate: bool = True,
) -> ScalarFisherResult:
    """Compute the class-specific Appendix-B shrunken ridge Fisher.

    ``task`` and ``nuisance`` must already be restricted to one task class.
    Unless an explicit class-specific ``global_moments`` block is supplied, the
    global block is the uniform block over those same rows.  This prevents
    shrinkage from importing coordinates from another task class.
    """

    slice_moments = scalar_moments(
        task,
        nuisance,
        q,
        predictive_weight=predictive_weight,
        center=center,
        validate=validate,
    )
    if global_moments is None:
        global_moments = uniform_scalar_moments(
            task,
            nuisance,
            predictive_weight=predictive_weight,
            center=center,
            validate=validate,
        )
    shrunken = shrink_scalar_moments(slice_moments, global_moments, eta)
    return ridge_schur(shrunken, ridge=ridge, eps=eps, validate=validate)


def centered_scalar_diagnostic(
    task: Tensor,
    nuisance: Tensor,
    q: Tensor,
    *,
    eta: float = 0.1,
    ridge: float = 1e-3,
    eps: float = 1e-8,
    predictive_weight: Tensor | None = None,
    global_moments: ScalarMoments | None = None,
    validate: bool = True,
) -> CenteredFisherDiagnostic:
    """Return a centered robustness check without changing the primary score."""

    fisher = shrunken_scalar_fisher(
        task,
        nuisance,
        q,
        eta=eta,
        ridge=ridge,
        eps=eps,
        predictive_weight=predictive_weight,
        center=True,
        global_moments=global_moments,
        validate=validate,
    )
    raw = fisher.moments.j_tt
    raw_safe = raw.clamp_min(torch.finfo(raw.dtype).tiny)
    ratio = torch.where(raw > 0, fisher.projected / raw_safe, torch.zeros_like(raw))
    correlation_denominator = torch.sqrt(
        (fisher.moments.j_tt * fisher.moments.j_nn).clamp_min(torch.finfo(raw.dtype).tiny)
    )
    correlation = torch.where(
        (fisher.moments.j_tt > 0) & (fisher.moments.j_nn > 0),
        fisher.moments.j_tn / correlation_denominator,
        torch.zeros_like(fisher.moments.j_tn),
    ).clamp(min=-1.0, max=1.0)
    return CenteredFisherDiagnostic(
        fisher=fisher,
        projected_over_raw=ratio,
        correlation=correlation,
    )
