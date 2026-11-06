"""Fail-closed audit of ridge versus exact scalar Schur information.

The headline Fisher-Glass experiments use one task coordinate ``z`` and one
nuisance coordinate ``b`` per evaluated environment.  This module computes the
ridge and unregularized Schur complements from the *same* held-out score pairs,
retains the Fisher blocks needed to reproduce the comparison, and constructs a
simultaneous lower confidence audit for the exact information.

The confidence calculation is a Gaussian/Wald simulation for the three
empirical moments ``E[z^2]``, ``E[z b]``, and ``E[b^2]``.  A Bonferroni
correction is applied jointly over environments and the nuisance/ridge/gap
bounds.  It is a first-order, conditional audit of a fixed fitted model and
fixed evaluated environment set; it is not the manuscript's separate
nonasymptotic population-environment theorem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np


DEFAULT_RIDGE_GRID = (0.0, 1e-6, 1e-5, 1e-4, 1e-3, 1e-2, 1e-1)


def discrete_cvar(values: np.ndarray, alpha: float, weights: np.ndarray | None = None) -> float:
    """Exact upper-tail CVaR for a finite discrete distribution.

    Boundary atoms are included fractionally.  This matters when the tail mass
    ``1-alpha`` is not an integer multiple of an environment's probability.
    """

    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        raise ValueError("CVaR requires at least one value")
    if not 0.0 <= alpha < 1.0:
        raise ValueError(f"alpha must lie in [0, 1), got {alpha}")
    if not np.all(np.isfinite(values)):
        raise ValueError("CVaR values must be finite")

    if weights is None:
        probabilities = np.full(values.size, 1.0 / values.size, dtype=np.float64)
    else:
        probabilities = np.asarray(weights, dtype=np.float64).reshape(-1)
        if probabilities.shape != values.shape:
            raise ValueError("CVaR weights must have the same shape as values")
        if np.any(probabilities < 0.0) or not np.all(np.isfinite(probabilities)):
            raise ValueError("CVaR weights must be finite and nonnegative")
        total = float(probabilities.sum())
        if total <= 0.0:
            raise ValueError("CVaR weights must have positive total mass")
        probabilities = probabilities / total

    tail_mass = 1.0 - alpha
    order = np.argsort(values)[::-1]
    remaining = tail_mass
    tail_sum = 0.0
    for index in order:
        take = min(float(probabilities[index]), remaining)
        tail_sum += take * float(values[index])
        remaining -= take
        if remaining <= 10.0 * np.finfo(np.float64).eps:
            break
    if remaining > 1e-10:
        raise RuntimeError("CVaR tail integration did not exhaust the requested mass")
    return tail_sum / tail_mass


def _weighted_moments(
    z: np.ndarray,
    b: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return moment vector, covariance of its mean, and effective sample size."""

    z = np.asarray(z, dtype=np.float64).reshape(-1)
    b = np.asarray(b, dtype=np.float64).reshape(-1)
    if z.shape != b.shape:
        raise ValueError("task and nuisance scores must have the same shape")
    if z.size < 2:
        raise ValueError("each environment needs at least two held-out rows")
    if not np.all(np.isfinite(z)) or not np.all(np.isfinite(b)):
        raise ValueError("task and nuisance scores must be finite")

    if weight is None:
        normalized = np.full(z.size, 1.0 / z.size, dtype=np.float64)
    else:
        normalized = np.asarray(weight, dtype=np.float64).reshape(-1)
        if normalized.shape != z.shape:
            raise ValueError("score weights must have the same shape as scores")
        if np.any(normalized < 0.0) or not np.all(np.isfinite(normalized)):
            raise ValueError("score weights must be finite and nonnegative")
        total = float(normalized.sum())
        if total <= 0.0:
            raise ValueError("score weights must have positive total mass")
        normalized = normalized / total

    observations = np.column_stack((z * z, z * b, b * b))
    moments = normalized @ observations
    centered = observations - moments
    squared_weight_sum = float(np.sum(normalized * normalized))
    correction = 1.0 - squared_weight_sum
    if correction <= np.finfo(np.float64).eps:
        raise ValueError("environment has insufficient effective sample size")
    covariance_of_mean = (centered * (normalized * normalized)[:, None]).T @ centered / correction
    covariance_of_mean = 0.5 * (covariance_of_mean + covariance_of_mean.T)
    effective_n = 1.0 / squared_weight_sum
    return moments, covariance_of_mean, effective_n


def schur_from_moments(
    j_tt: float,
    j_tn: float,
    j_nn: float,
    ridge: float,
    singular_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Compute exact/ridge scalar Schur information with fail-closed checks."""

    values = np.asarray([j_tt, j_tn, j_nn, ridge], dtype=np.float64)
    if not np.all(np.isfinite(values)):
        raise ValueError("Fisher moments and ridge must be finite")
    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative")

    scale = max(abs(j_tt), abs(j_nn), 1.0)
    psd_slack = 100.0 * np.finfo(np.float64).eps * scale * scale
    reasons: list[str] = []
    if j_tt < -psd_slack or j_nn < -psd_slack or j_tn * j_tn > j_tt * j_nn + psd_slack:
        reasons.append("fisher_block_not_psd")

    ridge_denominator = j_nn + ridge
    if ridge_denominator <= singular_tolerance:
        ridge_information: float | None = None
        reasons.append("ridge_denominator_nonpositive")
    else:
        raw_ridge = j_tt - (j_tn * j_tn) / ridge_denominator
        if raw_ridge < -psd_slack:
            reasons.append("negative_ridge_information")
        ridge_information = float(max(raw_ridge, 0.0))

    if j_nn <= singular_tolerance:
        exact_information = None
        gap = None
        reasons.append("nuisance_block_singular")
    else:
        raw_exact = j_tt - (j_tn * j_tn) / j_nn
        if raw_exact < -psd_slack:
            reasons.append("negative_exact_information")
        exact_information = float(max(raw_exact, 0.0))
        if ridge_information is None:
            gap = None
        else:
            gap = float(ridge_information - exact_information)
            closed_form_gap = float(ridge * j_tn * j_tn / (j_nn * (j_nn + ridge)))
            gap_tolerance = 1e-10 * max(1.0, abs(gap), abs(closed_form_gap))
            if gap < -gap_tolerance or abs(gap - closed_form_gap) > gap_tolerance:
                reasons.append("ridge_gap_identity_failed")

    return {
        "j_tt": float(j_tt),
        "j_tn": float(j_tn),
        "j_nn": float(j_nn),
        "ridge_information": ridge_information,
        "exact_information": exact_information,
        "operator_gap": gap,
        "eigenvalue_gap": gap,
        "status": "ok" if not reasons else "uncertifiable",
        "reasons": sorted(set(reasons)),
    }


def scalar_fisher_block(
    z: np.ndarray,
    b: np.ndarray,
    ridge: float,
    weight: np.ndarray | None = None,
    singular_tolerance: float = 1e-12,
) -> tuple[dict[str, Any], np.ndarray]:
    """Estimate a scalar Fisher block and return its moment covariance."""

    moments, covariance_of_mean, effective_n = _weighted_moments(z, b, weight)
    block = schur_from_moments(*moments, ridge, singular_tolerance)
    block["n"] = int(np.asarray(z).size)
    block["effective_n"] = float(effective_n)
    return block, covariance_of_mean


def _draw_information(
    draws: np.ndarray,
    ridge: float,
    singular_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    j_tt = draws[:, 0]
    j_tn = draws[:, 1]
    j_nn = draws[:, 2]
    valid_exact = j_nn > singular_tolerance
    valid_ridge = j_nn + ridge > singular_tolerance

    ridge_information = np.zeros(draws.shape[0], dtype=np.float64)
    ridge_information[valid_ridge] = np.maximum(
        j_tt[valid_ridge] - j_tn[valid_ridge] ** 2 / (j_nn[valid_ridge] + ridge),
        0.0,
    )
    exact_information = np.zeros(draws.shape[0], dtype=np.float64)
    exact_information[valid_exact] = np.maximum(
        j_tt[valid_exact] - j_tn[valid_exact] ** 2 / j_nn[valid_exact],
        0.0,
    )
    gap = np.full(draws.shape[0], np.inf, dtype=np.float64)
    jointly_valid = valid_exact & valid_ridge
    # Use the closed form directly.  The difference of the two clamped Schur
    # values is anti-conservative when a Gaussian moment draw is not PSD.
    gap[jointly_valid] = (
        ridge
        * j_tn[jointly_valid] ** 2
        / (j_nn[jointly_valid] * (j_nn[jointly_valid] + ridge))
    )
    return j_nn, ridge_information, exact_information, gap


def audit_scalar_environments(
    environments: Iterable[dict[str, Any]],
    ridge: float,
    eps: float,
    alpha: float,
    bootstrap_replicates: int = 100_000,
    confidence: float = 0.95,
    seed: int = 20260729,
    ridge_grid: Iterable[float] = DEFAULT_RIDGE_GRID,
    singular_tolerance: float = 1e-12,
) -> dict[str, Any]:
    """Audit ridge/exact information for a fixed evaluated environment set.

    Each environment dictionary must contain ``name``, ``z``, and ``b`` and
    may contain a nonnegative ``weight`` array used inside that environment.
    """

    environment_inputs = list(environments)
    if not environment_inputs:
        raise ValueError("at least one environment is required")
    if eps <= 0.0:
        raise ValueError("eps must be positive")
    if bootstrap_replicates < 0:
        raise ValueError("bootstrap_replicates must be nonnegative")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")

    rng = np.random.default_rng(seed)
    records: list[dict[str, Any]] = []
    covariance_matrices: list[np.ndarray | None] = []
    for environment in environment_inputs:
        name = str(environment["name"])
        try:
            block, covariance_of_mean = scalar_fisher_block(
                environment["z"],
                environment["b"],
                ridge,
                environment.get("weight"),
                singular_tolerance,
            )
        except ValueError as exc:
            if not (
                "at least two held-out rows" in str(exc)
                or "insufficient effective sample size" in str(exc)
            ):
                raise
            z = np.asarray(environment["z"], dtype=np.float64).reshape(-1)
            b = np.asarray(environment["b"], dtype=np.float64).reshape(-1)
            if z.shape != b.shape or z.size == 0:
                raise ValueError("an audited environment cannot be empty") from exc
            if not np.all(np.isfinite(z)) or not np.all(np.isfinite(b)):
                raise ValueError("task and nuisance scores must be finite") from exc
            weight = environment.get("weight")
            if weight is None:
                normalized = np.full(z.size, 1.0 / z.size, dtype=np.float64)
            else:
                normalized = np.asarray(weight, dtype=np.float64).reshape(-1)
                if (
                    normalized.shape != z.shape
                    or np.any(normalized < 0.0)
                    or not np.all(np.isfinite(normalized))
                ):
                    raise
                total = float(normalized.sum())
                if total <= 0.0:
                    raise
                normalized = normalized / total
            observations = np.column_stack((z * z, z * b, b * b))
            moments = normalized @ observations
            block = schur_from_moments(*moments, ridge, singular_tolerance)
            block["n"] = int(z.size)
            block["effective_n"] = float(1.0 / np.sum(normalized * normalized))
            block["status"] = "uncertifiable"
            block["reasons"] = sorted(
                set(block["reasons"] + ["insufficient_effective_sample_size"])
            )
            covariance_of_mean = None
        block["name"] = name
        block["moment_covariance_of_mean"] = (
            covariance_of_mean.tolist() if covariance_of_mean is not None else None
        )
        records.append(block)
        covariance_matrices.append(covariance_of_mean)

    failure_probability = 1.0 - confidence
    bonferroni_tail = failure_probability / (3.0 * len(records))
    if bootstrap_replicates:
        minimum_tail_count = bootstrap_replicates * bonferroni_tail
        if minimum_tail_count < 50.0:
            raise ValueError(
                "bootstrap_replicates is too small for the simultaneous tail: "
                f"need at least {int(np.ceil(50.0 / bonferroni_tail))}"
            )

        for record, covariance_of_mean in zip(records, covariance_matrices):
            if covariance_of_mean is None:
                record.update(
                    {
                        "j_nn_lcb": 0.0,
                        "ridge_information_lcb": 0.0,
                        "exact_information_direct_lcb": 0.0,
                        "gaussian_singular_fraction": None,
                        "operator_gap_ucb": None,
                        "eigenvalue_gap_ucb": None,
                        "eta": float(record["ridge_information"] or 0.0),
                        "conservative_information": 0.0,
                    }
                )
                continue
            mean = np.array([record["j_tt"], record["j_tn"], record["j_nn"]], dtype=np.float64)
            draws = rng.multivariate_normal(
                mean,
                covariance_of_mean,
                size=bootstrap_replicates,
                check_valid="ignore",
            )
            j_nn_draw, ridge_draw, exact_draw, gap_draw = _draw_information(
                draws, ridge, singular_tolerance
            )
            record["j_nn_lcb"] = float(
                min(
                    max(np.quantile(j_nn_draw, bonferroni_tail, method="lower"), 0.0),
                    record["j_nn"],
                )
            )
            record["ridge_information_lcb"] = float(
                min(
                    max(np.quantile(ridge_draw, bonferroni_tail, method="lower"), 0.0),
                    record["ridge_information"] or 0.0,
                )
            )
            record["exact_information_direct_lcb"] = float(
                min(
                    max(np.quantile(exact_draw, bonferroni_tail, method="lower"), 0.0),
                    record["exact_information"] or 0.0,
                )
            )
            finite_gap = gap_draw[np.isfinite(gap_draw)]
            invalid_fraction = float(1.0 - finite_gap.size / gap_draw.size)
            record["gaussian_singular_fraction"] = invalid_fraction
            gap_ucb_raw = float(
                np.quantile(gap_draw, 1.0 - bonferroni_tail, method="higher")
            )
            if record["operator_gap"] is None or not np.isfinite(gap_ucb_raw):
                record["operator_gap_ucb"] = None
                record["eigenvalue_gap_ucb"] = None
                record["eta"] = float(record["ridge_information"] or 0.0)
                record["conservative_information"] = 0.0
                record["status"] = "uncertifiable"
                record["reasons"] = sorted(
                    set(record["reasons"] + ["gaussian_draw_nuisance_singular"])
                )
            else:
                gap_ucb = float(max(gap_ucb_raw, record["operator_gap"] or 0.0))
                eta = float(
                    max((record["ridge_information"] or 0.0) - record["ridge_information_lcb"], 0.0)
                )
                decomposed_lower = max(
                    (record["ridge_information"] or 0.0) - eta - gap_ucb,
                    0.0,
                )
                record["operator_gap_ucb"] = gap_ucb
                record["eigenvalue_gap_ucb"] = gap_ucb
                record["eta"] = eta
                record["conservative_information"] = float(
                    min(
                        decomposed_lower,
                        record["exact_information_direct_lcb"],
                        record["exact_information"] or 0.0,
                    )
                )

            if record["j_nn_lcb"] <= singular_tolerance:
                record["conservative_information"] = 0.0
                record["status"] = "uncertifiable"
                record["reasons"] = sorted(set(record["reasons"] + ["nuisance_lcb_nonpositive"]))
    else:
        for record in records:
            record.update(
                {
                    "j_nn_lcb": None,
                    "ridge_information_lcb": None,
                    "exact_information_direct_lcb": None,
                    "gaussian_singular_fraction": None,
                    "operator_gap_ucb": None,
                    "eigenvalue_gap_ucb": None,
                    "eta": None,
                    "conservative_information": None,
                }
            )

    ridge_available = all(record["ridge_information"] is not None for record in records)
    ridge_values = (
        np.array([record["ridge_information"] for record in records], dtype=np.float64)
        if ridge_available
        else None
    )
    exact_available = all(record["exact_information"] is not None for record in records)
    exact_values = (
        np.array([record["exact_information"] for record in records], dtype=np.float64)
        if exact_available
        else None
    )
    conservative_available = bootstrap_replicates > 0 and all(
        record["conservative_information"] is not None for record in records
    )
    conservative_values = (
        np.array([record["conservative_information"] for record in records], dtype=np.float64)
        if conservative_available
        else None
    )

    ridge_certificate = (
        float(1.0 / discrete_cvar(1.0 / (ridge_values + eps), alpha))
        if ridge_values is not None
        else None
    )
    exact_certificate = (
        float(1.0 / discrete_cvar(1.0 / (exact_values + eps), alpha))
        if exact_values is not None
        else None
    )
    conservative_certificate = (
        float(1.0 / discrete_cvar(1.0 / (conservative_values + eps), alpha))
        if conservative_values is not None
        else None
    )

    reasons = sorted({reason for record in records for reason in record["reasons"]})
    if ridge_certificate is None:
        reasons.append("ridge_certificate_unavailable")
    if exact_certificate is None:
        reasons.append("exact_certificate_unavailable")
    if not bootstrap_replicates:
        reasons.append("sampling_audit_not_run")
    if conservative_certificate is not None and conservative_certificate <= eps * (1.0 + 1e-6):
        reasons.append("conservative_certificate_at_numerical_floor")
    ordering_scale = max(
        1.0,
        abs(ridge_certificate or 0.0),
        abs(exact_certificate or 0.0),
        abs(conservative_certificate or 0.0),
    )
    ordering_tolerance = 1e-10 * ordering_scale
    if (
        ridge_certificate is not None
        and exact_certificate is not None
        and exact_certificate > ridge_certificate + ordering_tolerance
    ):
        reasons.append("exact_ridge_certificate_ordering_failed")
    if (
        exact_certificate is not None
        and conservative_certificate is not None
        and conservative_certificate > exact_certificate + ordering_tolerance
    ):
        reasons.append("lower_exact_certificate_ordering_failed")
    reasons = sorted(set(reasons))
    status = "audited" if not reasons else "uncertifiable"

    gaps_available = all(record["operator_gap"] is not None for record in records)
    gap_ucbs_available = bootstrap_replicates > 0 and all(
        record["operator_gap_ucb"] is not None for record in records
    )
    finite_gaps = [record["operator_gap"] for record in records] if gaps_available else []
    finite_gap_ucbs = (
        [record["operator_gap_ucb"] for record in records] if gap_ucbs_available else []
    )
    j_nn_lcbs_available = bootstrap_replicates > 0 and all(
        record["j_nn_lcb"] is not None for record in records
    )
    aggregate = {
        "min_nuisance_eigenvalue": float(min(record["j_nn"] for record in records)),
        "min_nuisance_eigenvalue_lcb": (
            float(min(record["j_nn_lcb"] for record in records))
            if j_nn_lcbs_available
            else None
        ),
        "max_operator_gap": float(max(finite_gaps)) if finite_gaps else None,
        "max_eigenvalue_gap": float(max(finite_gaps)) if finite_gaps else None,
        "max_operator_gap_ucb": float(max(finite_gap_ucbs)) if finite_gap_ucbs else None,
        "max_eigenvalue_gap_ucb": float(max(finite_gap_ucbs)) if finite_gap_ucbs else None,
        "ridge_certificate": ridge_certificate,
        "exact_certificate": exact_certificate,
        "conservative_lower_certificate": conservative_certificate,
        "certificate_gap": (
            float(ridge_certificate - exact_certificate)
            if ridge_certificate is not None and exact_certificate is not None
            else None
        ),
        "weakest_ridge_environment": (
            records[int(np.argmin(ridge_values))]["name"] if ridge_values is not None else None
        ),
        "weakest_exact_environment": (
            records[int(np.argmin(exact_values))]["name"] if exact_values is not None else None
        ),
        "weakest_conservative_environment": (
            records[int(np.argmin(conservative_values))]["name"]
            if conservative_values is not None
            else None
        ),
        "status": status,
        "reasons": reasons,
    }

    ridge_sweep = []
    for sweep_ridge in sorted({float(value) for value in ridge_grid}):
        if sweep_ridge < 0.0:
            raise ValueError("ridge sweep values must be nonnegative")
        sweep_information = []
        for record in records:
            sweep_block = schur_from_moments(
                record["j_tt"],
                record["j_tn"],
                record["j_nn"],
                sweep_ridge,
                singular_tolerance,
            )
            value = sweep_block["exact_information"] if sweep_ridge == 0.0 else sweep_block["ridge_information"]
            if value is None:
                sweep_information = []
                break
            sweep_information.append(value)
        sweep_certificate = (
            float(
                1.0
                / discrete_cvar(
                    1.0 / (np.asarray(sweep_information, dtype=np.float64) + eps),
                    alpha,
                )
            )
            if sweep_information
            else None
        )
        ridge_sweep.append({"ridge": sweep_ridge, "certificate": sweep_certificate})

    return {
        "schema_version": 2,
        "definition": "scalar held-out feature Schur audit",
        "conditional_on_fitted_model": True,
        "environment_weighting": "uniform",
        "alpha": float(alpha),
        "eps": float(eps),
        "ridge": float(ridge),
        "sampling_audit": {
            "method": "Bonferroni simultaneous Gaussian/Wald simulation of Fisher moments",
            "replicates": int(bootstrap_replicates),
            "confidence": float(confidence),
            "seed": int(seed),
            "bonferroni_tail_probability": float(bonferroni_tail),
            "scope": "fixed fitted model and fixed evaluated environment set",
        },
        "environments": records,
        "aggregate": aggregate,
        "ridge_sweep": ridge_sweep,
    }


def rebuttal_values(audit: dict[str, Any]) -> str:
    """Format one explicitly named audit row for a rebuttal draft."""

    aggregate = audit["aggregate"]
    confidence = 100.0 * float(audit["sampling_audit"]["confidence"])

    def formatted(value: Any) -> str:
        return "unavailable" if value is None else f"{float(value):.6g}"

    return (
        f"minimum nuisance eigenvalue {formatted(aggregate['min_nuisance_eigenvalue'])}; "
        f"maximum ridge-to-exact gap {formatted(aggregate['max_operator_gap'])}; "
        f"ridge certificate {formatted(aggregate['ridge_certificate'])}; "
        f"exact certificate {formatted(aggregate['exact_certificate'])}; "
        f"{confidence:g}% simultaneous lower certificate "
        f"{formatted(aggregate['conservative_lower_certificate'])}; "
        f"status {aggregate['status']}"
    )


def self_test() -> None:
    collapse = schur_from_moments(1.0, 1e-3, 1e-6, 1e-3)
    assert abs(collapse["exact_information"]) < 1e-12
    assert abs(collapse["ridge_information"] - 1e-3 / (1e-6 + 1e-3)) < 1e-12

    orthogonal = schur_from_moments(2.0, 0.0, 3.0, 1e-3)
    assert orthogonal["exact_information"] == orthogonal["ridge_information"] == 2.0
    assert orthogonal["operator_gap"] == 0.0

    non_psd_draw = np.array([[0.0, 10.0, 1.0]], dtype=np.float64)
    _, _, _, non_psd_gap = _draw_information(
        non_psd_draw, ridge=1e-3, singular_tolerance=1e-12
    )
    assert non_psd_gap[0] > 0.09

    rng = np.random.default_rng(7)
    environments = []
    for index, correlation in enumerate((0.1, 0.7, 0.98, 0.4)):
        b = rng.normal(size=1200)
        z = correlation * b + np.sqrt(1.0 - correlation * correlation) * rng.normal(size=b.size)
        environments.append({"name": f"e{index}", "z": z, "b": b})
    audit = audit_scalar_environments(
        environments,
        ridge=1e-3,
        eps=1e-8,
        alpha=0.95,
        bootstrap_replicates=12_000,
        confidence=0.95,
        seed=11,
    )
    aggregate = audit["aggregate"]
    assert aggregate["conservative_lower_certificate"] <= aggregate["exact_certificate"] + 1e-12
    assert aggregate["exact_certificate"] <= aggregate["ridge_certificate"] + 1e-12
    sweep = [row["certificate"] for row in audit["ridge_sweep"] if row["certificate"] is not None]
    assert all(left <= right + 1e-12 for left, right in zip(sweep, sweep[1:]))

    singular = schur_from_moments(1.0, 0.0, 0.0, 1e-3)
    assert singular["status"] == "uncertifiable"
    assert singular["exact_information"] is None


def _load_npz_environments(path: Path) -> list[dict[str, Any]]:
    payload = np.load(path, allow_pickle=False)
    required = {"z", "b", "environment"}
    missing = required.difference(payload.files)
    if missing:
        raise ValueError(f"score bundle is missing arrays: {sorted(missing)}")
    z = np.asarray(payload["z"], dtype=np.float64)
    b = np.asarray(payload["b"], dtype=np.float64)
    environment_ids = np.asarray(payload["environment"])
    weight = np.asarray(payload["weight"], dtype=np.float64) if "weight" in payload.files else None
    unique_ids = np.unique(environment_ids)
    if "environment_names" in payload.files:
        names = [str(value) for value in payload["environment_names"].tolist()]
        if len(names) != len(unique_ids):
            raise ValueError("environment_names must align with sorted unique environment IDs")
    else:
        names = [str(value) for value in unique_ids]

    environments = []
    for environment_id, name in zip(unique_ids, names):
        mask = environment_ids == environment_id
        row = {"name": name, "z": z[mask], "b": b[mask]}
        if weight is not None:
            row["weight"] = weight[mask]
        environments.append(row)
    return environments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, help="NPZ with z, b, and environment arrays")
    parser.add_argument("--output", type=Path, default=Path("build/results/ridge_exact_audit.json"))
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--bootstrap-replicates", type=int, default=100_000)
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260729)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    if args.self_test:
        self_test()
        print("ridge_exact_audit self-test: ok")
        return
    if args.scores is None:
        parser.error("--scores is required unless --self-test is used")

    audit = audit_scalar_environments(
        _load_npz_environments(args.scores),
        ridge=args.ridge,
        eps=args.eps,
        alpha=args.alpha,
        bootstrap_replicates=args.bootstrap_replicates,
        confidence=args.confidence,
        seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    print(rebuttal_values(audit))
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
