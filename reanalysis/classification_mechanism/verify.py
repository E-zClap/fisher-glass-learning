"""Numerical checks for the analytic boundary-region classification example.

Requires NumPy. These checks support, but do not replace, the analytic proof.
Run from any directory: python path/to/verify.py
"""
from pathlib import Path
import json
import math
import numpy as np

HERE = Path(__file__).resolve().parent
NODES, WEIGHTS = np.polynomial.legendre.leggauss(64)


def integral(fn, a, b):
    if a == b:
        return 0.0
    x = (a + b) / 2 + (b - a) * NODES / 2
    return float((b - a) / 2 * np.dot(WEIGHTS, fn(x)))


def eta(x, theta, mu, delta):
    return .5 + x / 4 + theta + mu * (np.abs(x) > delta)


def main():
    deltas = [1e-4, 1e-3, .01, .05, .1, .25, .5]
    max_info_error = 0.0
    max_separation_error = 0.0
    max_kl_ratio = 0.0
    max_risk_error = 0.0
    max_reciprocal_error = 0.0
    pair_checks = risk_checks = 0
    raw = integral(lambda x: 8 / (4 - x*x), -1, 1)
    assert math.isclose(raw, 4*math.log(3), rel_tol=1e-12)
    for delta in deltas:
        F = 4 * math.log1p(2*delta/(2-delta))
        numeric_F = integral(lambda x: 8/(4-x*x), -delta, delta)
        max_info_error = max(max_info_error, abs(F-numeric_F))
        assert math.isclose(F, numeric_F, rel_tol=1e-12)
        assert 4*delta*(1-1e-12) <= F <= 64*delta/15*(1+1e-12)
        for fraction in [.001, .1, .5, 1]:
            t = delta*fraction/8
            separation = 2*integral(lambda x: (2*t-x/2)/2, 0, 4*t)
            max_separation_error = max(max_separation_error, abs(separation-4*t*t))
            assert math.isclose(separation, 4*t*t, rel_tol=1e-12)
            def kl_integrand(x):
                q = .5+x/4-t
                d = 2*t
                return ((q+d)*np.log1p(d/q)+(1-q-d)*np.log1p(-d/(1-q)))/2
            kl = integral(kl_integrand, -delta, delta)
            assert kl >= -1e-27
            assert kl <= 5*F*t*t*(1+1e-7)+1e-27
            max_kl_ratio = max(max_kl_ratio, kl/(F*t*t))
            pair_checks += 1
        for theta_fraction in [-1, -.5, 0, .5, 1]:
            theta = theta_fraction*delta/8
            for mu_fraction in [-1, 0, 1]:
                mu = mu_fraction*delta/8
                points = np.array([-1, -delta-1e-9, -delta, -delta/2,
                                   -4*theta, 0, delta/2, delta, delta+1e-9, 1])
                probs = eta(points, theta, mu, delta)
                assert np.all(probs >= 1/8-1e-12) and np.all(probs <= 7/8+1e-12)
                away_from_ties = np.abs(probs-.5) > 1e-12
                assert np.array_equal((probs[away_from_ties] >= .5),
                                      (points[away_from_ties] >= -4*theta))
                for estimate_fraction in [-1, -.5, 0, .5, 1]:
                    estimate = estimate_fraction*delta/8
                    a, b = sorted([-4*theta, -4*estimate])
                    risk = integral(lambda x: np.abs(2*eta(x, theta, mu, delta)-1)/2, a, b)
                    exact = 2*(estimate-theta)**2
                    max_risk_error = max(max_risk_error, abs(risk-exact))
                    assert math.isclose(risk, exact, rel_tol=1e-8, abs_tol=1e-18)
                    risk_checks += 1
        for n in [1, 2, 10, 100]:
            numeric = sum(math.comb(n,k)*delta**k*(1-delta)**(n-k)/(k+1)
                          for k in range(n+1))
            exact = -math.expm1((n+1)*math.log1p(-delta))/((n+1)*delta)
            max_reciprocal_error = max(max_reciprocal_error, abs(numeric-exact))
            assert math.isclose(numeric, exact, rel_tol=1e-11)
            # At N=0 the estimator is zero; its conditional MSE is theta^2.
            assert (delta/8)**2 <= 1/4
    result = {
        "status": "passed",
        "delta_values": deltas,
        "pair_checks": pair_checks,
        "risk_identity_checks": risk_checks,
        "max_absolute_information_integral_error": max_info_error,
        "max_absolute_separation_error": max_separation_error,
        "max_observed_kl_over_F_t_squared": max_kl_ratio,
        "proved_kl_constant": 5,
        "max_absolute_risk_identity_error": max_risk_error,
        "max_absolute_binomial_reciprocal_error": max_reciprocal_error,
        "note": "Deterministic quadrature checks; the analytic proof establishes uniform bounds."
    }
    (HERE / "verification.json").write_text(json.dumps(result, indent=2)+"\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
