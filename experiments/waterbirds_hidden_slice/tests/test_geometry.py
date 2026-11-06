from __future__ import annotations

import unittest

import torch

from experiments.waterbirds_hidden_slice.geometry import (
    centered_scalar_diagnostic,
    ridge_schur,
    scalar_moments,
    shrink_scalar_moments,
    shrunken_scalar_fisher,
    uniform_scalar_moments,
)


class ScalarGeometryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dtype = torch.float64

    def test_predictive_factor_is_not_renormalized(self) -> None:
        task = torch.tensor([1.0, 2.0], dtype=self.dtype)
        nuisance = torch.tensor([0.0, 1.0], dtype=self.dtype)
        q = torch.tensor([0.5, 0.5], dtype=self.dtype)
        predictive = torch.tensor([0.1, 0.2], dtype=self.dtype)

        moments = scalar_moments(task, nuisance, q, predictive_weight=predictive)

        self.assertAlmostEqual(moments.j_tt.item(), 0.45)
        self.assertAlmostEqual(moments.j_tn.item(), 0.20)
        self.assertAlmostEqual(moments.j_nn.item(), 0.10)
        # Normalizing q*r would instead make J_TT equal to 3.0.
        self.assertNotAlmostEqual(moments.j_tt.item(), 3.0)

    def test_unweighted_mode_uses_q_directly_and_vectorizes(self) -> None:
        task = torch.tensor([1.0, 2.0], dtype=self.dtype)
        nuisance = torch.tensor([3.0, -1.0], dtype=self.dtype)
        q = torch.tensor([[0.5, 0.5], [0.25, 0.75]], dtype=self.dtype)

        moments = scalar_moments(task, nuisance, q)

        torch.testing.assert_close(moments.j_tt, torch.tensor([2.5, 3.25], dtype=self.dtype))
        torch.testing.assert_close(moments.j_tn, torch.tensor([0.5, -0.75], dtype=self.dtype))
        torch.testing.assert_close(moments.j_nn, torch.tensor([5.0, 3.0], dtype=self.dtype))

    def test_shrinkage_is_applied_to_block_before_schur(self) -> None:
        task = torch.tensor([1.0, 3.0, -2.0], dtype=self.dtype)
        nuisance = torch.tensor([2.0, -1.0, 0.5], dtype=self.dtype)
        q = torch.tensor([0.8, 0.1, 0.1], dtype=self.dtype)
        eta = 0.25
        ridge = 0.1

        local = scalar_moments(task, nuisance, q)
        global_block = uniform_scalar_moments(task, nuisance)
        expected_block = shrink_scalar_moments(local, global_block, eta)
        result = shrunken_scalar_fisher(task, nuisance, q, eta=eta, ridge=ridge)

        torch.testing.assert_close(result.moments.j_tt, expected_block.j_tt)
        torch.testing.assert_close(result.moments.j_tn, expected_block.j_tn)
        torch.testing.assert_close(result.moments.j_nn, expected_block.j_nn)

        incorrect_post_schur_mix = (
            (1.0 - eta) * ridge_schur(local, ridge=ridge).projected
            + eta * ridge_schur(global_block, ridge=ridge).projected
        )
        self.assertFalse(torch.isclose(result.projected, incorrect_post_schur_mix).item())

    def test_zero_clamp_then_epsilon_once(self) -> None:
        task = torch.tensor([1.0, -1.0], dtype=self.dtype)
        nuisance = task.clone()
        q = torch.tensor([0.5, 0.5], dtype=self.dtype)
        moments = scalar_moments(task, nuisance, q)

        result = ridge_schur(moments, ridge=0.0, eps=1e-8)

        self.assertEqual(result.projected.item(), 0.0)
        self.assertAlmostEqual(result.inverse.item(), 1e8)

    def test_centered_diagnostic_is_shift_invariant(self) -> None:
        task = torch.tensor([-2.0, 0.5, 3.0], dtype=self.dtype)
        nuisance = torch.tensor([1.0, -1.0, 2.0], dtype=self.dtype)
        q = torch.tensor([0.2, 0.3, 0.5], dtype=self.dtype)

        first = centered_scalar_diagnostic(task, nuisance, q, eta=0.0, ridge=0.1)
        shifted = centered_scalar_diagnostic(task + 20.0, nuisance - 7.0, q, eta=0.0, ridge=0.1)

        torch.testing.assert_close(first.fisher.projected, shifted.fisher.projected)
        torch.testing.assert_close(first.projected_over_raw, shifted.projected_over_raw)
        torch.testing.assert_close(first.correlation, shifted.correlation)


if __name__ == "__main__":
    unittest.main()

