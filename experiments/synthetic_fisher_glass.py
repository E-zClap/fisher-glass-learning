"""Synthetic Fisher-glass experiment.

This script validates the separation theorem in a controlled hidden-shift
problem. The task feature is z = s + delta_e r, while s is deliberately treated
as a disallowed nuisance-aligned proxy. Raw task Fisher is bounded away from
zero, but the nuisance-projected feature Fisher is delta_e^2. If
P(delta_e < t) ~ t^gamma, then the inverse projected-Fisher loss has tail
exponent gamma / 2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def sample_delta(rng: np.random.Generator, n_env: int, gamma: float, delta_max: float) -> np.ndarray:
    # P(delta < t) = (t / delta_max)^gamma on [0, delta_max].
    u = np.maximum(rng.random(n_env), np.finfo(float).tiny)
    return delta_max * (u ** (1.0 / gamma))


def cvar(values: np.ndarray, alpha: float) -> float:
    threshold = np.quantile(values, alpha)
    tail = values[values >= threshold]
    return float(tail.mean())


def survival_curve(values: np.ndarray, n_points: int = 240) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(values)
    lo = np.quantile(values, 0.50)
    hi = np.quantile(values, 0.9995)
    thresholds = np.geomspace(max(lo, np.finfo(float).tiny), hi, n_points)
    survival = np.array([(values > t).mean() for t in thresholds])
    mask = survival > 0
    return thresholds[mask], survival[mask]


def tail_exponent(values: np.ndarray, tail_frac: float = 0.05) -> float:
    """Fit P(X>x) ~ x^{-beta} on the largest tail_frac observations."""
    values = np.sort(np.asarray(values))
    n = len(values)
    start = max(0, int((1.0 - tail_frac) * n))
    x = values[start:]
    ranks = np.arange(start, n)
    surv = (n - ranks) / n
    mask = (x > 0) & (surv > 0)
    slope, _ = np.polyfit(np.log(x[mask]), np.log(surv[mask]), 1)
    return float(-slope)


def spurious_classifier_metrics(
    rng: np.random.Generator,
    delta: np.ndarray,
    n_per_env: int,
    logit_scale: float,
    chunk: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate f(x)=logit_scale*s on labels y=1{s+delta*r>0}."""
    losses = np.empty(delta.shape[0], dtype=float)
    accuracies = np.empty(delta.shape[0], dtype=float)
    for start in range(0, delta.shape[0], chunk):
        stop = min(start + chunk, delta.shape[0])
        d = delta[start:stop, None]
        s = rng.normal(size=(stop - start, n_per_env))
        r = rng.normal(size=(stop - start, n_per_env))
        z = s + d * r
        y = (z > 0).astype(float)
        logits = logit_scale * s
        losses[start:stop] = np.mean(np.logaddexp(0.0, logits) - y * logits, axis=1)
        accuracies[start:stop] = np.mean((logits > 0) == (y > 0), axis=1)
    return losses, accuracies


def predictive_projected_fisher(
    rng: np.random.Generator,
    delta_grid: np.ndarray,
    n_per_env: int,
    logit_scale: float,
    temperature: float,
) -> np.ndarray:
    """Monte Carlo predictive Fisher in the (z,s) operational coordinates."""
    projected = np.empty(delta_grid.shape[0], dtype=float)
    for i, d in enumerate(delta_grid):
        s = rng.normal(size=n_per_env)
        r = rng.normal(size=n_per_env)
        z = s + d * r
        p = sigmoid(logit_scale * s / temperature)
        h = p * (1.0 - p)
        j_tt = np.mean(h * z * z)
        j_tn = np.mean(h * z * s)
        j_nn = np.mean(h * s * s)
        projected[i] = max(j_tt - (j_tn * j_tn) / max(j_nn, 1e-12), 0.0)
    return projected


def summarize_main(delta: np.ndarray, losses: np.ndarray, accuracies: np.ndarray, alpha: float, eps: float) -> dict:
    raw = 1.0 + delta**2
    projected = delta**2
    inverse_loss = 1.0 / (projected + eps)
    cvar_inverse = cvar(inverse_loss, alpha)
    return {
        "n_env": int(delta.size),
        "alpha": alpha,
        "raw_fisher_mean": float(raw.mean()),
        "raw_fisher_min": float(raw.min()),
        "projected_fisher_mean": float(projected.mean()),
        "projected_fisher_min": float(projected.min()),
        "projected_fisher_q01": float(np.quantile(projected, 0.01)),
        "projected_fisher_q05": float(np.quantile(projected, 0.05)),
        "inverse_loss_cvar": cvar_inverse,
        "fisher_glass_certificate": float(1.0 / cvar_inverse),
        "tail_exponent_beta_hat": tail_exponent(inverse_loss),
        "accuracy_mean": float(accuracies.mean()),
        "accuracy_worst_5pct": float(np.quantile(accuracies, 0.05)),
        "loss_mean": float(losses.mean()),
        "loss_cvar": cvar(losses, alpha),
    }


def reserve_metrics(delta: np.ndarray, alpha: float, eps: float, reserve_r: float, anchor_delta: float) -> dict:
    projected = delta**2
    anchor_fisher = anchor_delta**2
    mixed = (1.0 - reserve_r) * projected + reserve_r * anchor_fisher
    inverse_loss = 1.0 / (mixed + eps)
    return {
        "reserve_r": reserve_r,
        "anchor_delta": anchor_delta,
        "anchor_fisher": anchor_fisher,
        "mixed_fisher_min": float(mixed.min()),
        "inverse_loss_cvar": cvar(inverse_loss, alpha),
        "fisher_glass_certificate": float(1.0 / cvar(inverse_loss, alpha)),
        "theory_lower_bound": float(reserve_r * anchor_fisher),
    }


def portfolio_metrics(
    rng: np.random.Generator,
    n_env: int,
    gamma: float,
    delta_max: float,
    alpha: float,
    eps: float,
    max_k: int,
) -> list[dict]:
    rows = []
    for k in range(1, max_k + 1):
        deltas = np.stack([sample_delta(rng, n_env, gamma, delta_max) for _ in range(k)], axis=1)
        # Equal positive weights. Scaling changes constants, not tail exponents.
        projected = np.mean(deltas**2, axis=1)
        inverse_loss = 1.0 / (projected + eps)
        rows.append(
            {
                "K": k,
                "inverse_loss_cvar": cvar(inverse_loss, alpha),
                "fisher_glass_certificate": float(1.0 / cvar(inverse_loss, alpha)),
                "tail_exponent_beta_hat": tail_exponent(inverse_loss),
            }
        )
    return rows


def make_figures(
    fig_dir: Path,
    rng: np.random.Generator,
    main_delta: np.ndarray,
    main_summary: dict,
    reserve_summary: dict,
    portfolio_summary: list[dict],
    gamma_values: list[float],
    delta_max: float,
    n_env: int,
    alpha: float,
    eps: float,
) -> dict:
    fig_dir.mkdir(parents=True, exist_ok=True)
    raw = 1.0 + main_delta**2
    projected = main_delta**2
    inverse_loss = 1.0 / (projected + eps)

    fig1, axes = plt.subplots(2, 2, figsize=(7.2, 5.2), constrained_layout=True)
    order = np.argsort(projected)
    axes[0, 0].plot(raw[order], label="raw task Fisher", lw=1.4)
    axes[0, 0].plot(projected[order], label="projected Fisher", lw=1.4)
    axes[0, 0].set_yscale("log")
    axes[0, 0].set_title("A. average raw information is benign")
    axes[0, 0].set_xlabel("environment, sorted by projected Fisher")
    axes[0, 0].set_ylabel("Fisher")
    axes[0, 0].legend(fontsize=7)

    qs = np.linspace(0.001, 0.5, 250)
    axes[0, 1].plot(qs, np.quantile(projected, qs), color="tab:red")
    axes[0, 1].set_yscale("log")
    axes[0, 1].set_title("B. projected Fisher tail")
    axes[0, 1].set_xlabel("lower quantile")
    axes[0, 1].set_ylabel(r"$F_{T|N,e}$")

    x, surv = survival_curve(inverse_loss)
    axes[1, 0].loglog(x, surv, color="tab:purple")
    axes[1, 0].set_title(fr"C. inverse tail, $\hat\beta={main_summary['tail_exponent_beta_hat']:.2f}$")
    axes[1, 0].set_xlabel(r"$L=1/F_{T|N,e}$")
    axes[1, 0].set_ylabel(r"$P(L>x)$")

    labels = ["base", "reserve"] + [f"K={row['K']}" for row in portfolio_summary]
    certs = [
        main_summary["fisher_glass_certificate"],
        reserve_summary["fisher_glass_certificate"],
    ] + [row["fisher_glass_certificate"] for row in portfolio_summary]
    axes[1, 1].bar(labels, certs, color=["0.35", "tab:green", "tab:blue", "tab:orange", "tab:red"])
    axes[1, 1].set_yscale("log")
    axes[1, 1].set_title("D. repairs lift the certificate")
    axes[1, 1].set_ylabel(r"$I_{0.95}^{FG}$")
    axes[1, 1].tick_params(axis="x", labelrotation=20)

    fig1_path = fig_dir / "fisherglass_synthetic_diagnostics.pdf"
    fig1.savefig(fig1_path)
    plt.close(fig1)

    fig2, ax = plt.subplots(figsize=(5.6, 3.8), constrained_layout=True)
    tail_rows = []
    for gamma in gamma_values:
        delta = sample_delta(rng, n_env, gamma, delta_max)
        inv = 1.0 / (delta**2 + eps)
        beta_hat = tail_exponent(inv)
        x, surv = survival_curve(inv)
        ax.loglog(x, surv, label=fr"$\gamma={gamma:g}$, $\hat\beta={beta_hat:.2f}$")
        tail_rows.append(
            {
                "gamma": gamma,
                "theory_beta": gamma / 2.0,
                "tail_exponent_beta_hat": beta_hat,
                "fisher_glass_certificate": float(1.0 / cvar(inv, alpha)),
            }
        )
    ax.set_xlabel(r"$L=1/F_{T|N,e}$")
    ax.set_ylabel(r"$P(L>x)$")
    ax.set_title("Tail boundary: inverse-Fisher exponent")
    ax.legend(fontsize=7)
    fig2_path = fig_dir / "fisherglass_tail_boundary.pdf"
    fig2.savefig(fig2_path)
    plt.close(fig2)

    return {"diagnostics": str(fig1_path), "tail_boundary": str(fig2_path), "tail_rows": tail_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--n-env", type=int, default=50000)
    parser.add_argument("--n-per-env", type=int, default=256)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--delta-max", type=float, default=0.35)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-12)
    parser.add_argument("--logit-scale", type=float, default=4.0)
    parser.add_argument("--reserve-r", type=float, default=0.05)
    parser.add_argument("--anchor-delta", type=float, default=0.5)
    parser.add_argument("--out-dir", type=Path, default=Path("build/results"))
    parser.add_argument("--fig-dir", type=Path, default=Path("tex/figures"))
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    args.fig_dir.mkdir(parents=True, exist_ok=True)

    delta = sample_delta(rng, args.n_env, args.gamma, args.delta_max)
    losses, accuracies = spurious_classifier_metrics(
        rng,
        delta,
        n_per_env=args.n_per_env,
        logit_scale=args.logit_scale,
    )
    main_summary = summarize_main(delta, losses, accuracies, args.alpha, args.eps)
    reserve_summary = reserve_metrics(delta, args.alpha, args.eps, args.reserve_r, args.anchor_delta)
    portfolio_summary = portfolio_metrics(
        rng,
        args.n_env,
        args.gamma,
        args.delta_max,
        args.alpha,
        args.eps,
        max_k=3,
    )

    # Calibration/saturation check on a smaller grid.
    grid = np.linspace(0.02, args.delta_max, 20)
    pred_t1 = predictive_projected_fisher(rng, grid, 12000, args.logit_scale, temperature=1.0)
    pred_t4 = predictive_projected_fisher(rng, grid, 12000, args.logit_scale, temperature=4.0)
    feature = grid**2
    saturation = {
        "temperature_1_median_predictive_to_feature_ratio": float(np.median(pred_t1 / feature)),
        "temperature_4_median_predictive_to_feature_ratio": float(np.median(pred_t4 / feature)),
        "note": "Predictive Fisher is scaled down by logistic confidence; feature Fisher preserves the theorem geometry.",
    }

    figure_info = make_figures(
        args.fig_dir,
        rng,
        delta,
        main_summary,
        reserve_summary,
        portfolio_summary,
        gamma_values=[1.0, 2.0, 4.0],
        delta_max=args.delta_max,
        n_env=args.n_env,
        alpha=args.alpha,
        eps=args.eps,
    )

    results = {
        "parameters": vars(args) | {"out_dir": str(args.out_dir), "fig_dir": str(args.fig_dir)},
        "main_gamma": args.gamma,
        "main": main_summary,
        "reserve": reserve_summary,
        "portfolio": portfolio_summary,
        "tail_boundary": figure_info["tail_rows"],
        "saturation_check": saturation,
        "figures": {k: v for k, v in figure_info.items() if k != "tail_rows"},
    }
    out_path = args.out_dir / "synthetic_fisher_glass_results.json"
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
