from __future__ import annotations

import unittest

import torch

from experiments.waterbirds_hidden_slice.gates import (
    FrozenLinearGate,
    GateConstraints,
    GateOptimizationConfig,
    LinearGateEnsemble,
    admissible_weights,
    apply_frozen_gate,
    calibrated_sigmoid_membership,
    discover_class_conditional,
    effective_sample_size,
    enforce_minimum_ess,
    membership_mass_diagnostic,
    membership_target_mass_penalty,
    project_capped_simplex,
)


class GateConstraintTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dtype = torch.float64

    def test_capped_simplex_projection_is_vectorized(self) -> None:
        values = torch.tensor(
            [[0.90, 0.05, 0.03, 0.02], [0.10, 0.20, 0.30, 0.40]],
            dtype=self.dtype,
        )
        q = project_capped_simplex(values, cap=0.30, iterations=80)

        torch.testing.assert_close(q.sum(dim=-1), torch.ones(2, dtype=self.dtype))
        self.assertTrue(bool((q >= 0.0).all()))
        self.assertTrue(bool((q <= 0.30 + 1e-12).all()))

    def test_capped_simplex_projection_has_correct_interior_backward(self) -> None:
        values = torch.tensor([0.10, 0.20, 0.30, 0.40], dtype=self.dtype, requires_grad=True)
        coefficient = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=self.dtype)
        q = project_capped_simplex(values, cap=1.0, iterations=80)
        gradient, = torch.autograd.grad((q * coefficient).sum(), values)

        torch.testing.assert_close(q, values)
        torch.testing.assert_close(gradient, coefficient - coefficient.mean())
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda item: project_capped_simplex(item, cap=1.0, iterations=80),
                (values.detach().clone().requires_grad_(),),
            )
        )

    def test_capped_simplex_projection_has_correct_cap_active_backward(self) -> None:
        values = torch.tensor([0.70, 0.10, 0.10, 0.10], dtype=self.dtype, requires_grad=True)
        coefficient = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=self.dtype)
        q = project_capped_simplex(values, cap=0.40, iterations=80)
        gradient, = torch.autograd.grad((q * coefficient).sum(), values)

        torch.testing.assert_close(q, torch.tensor([0.40, 0.20, 0.20, 0.20], dtype=self.dtype))
        torch.testing.assert_close(gradient, torch.tensor([0.0, -1.0, 0.0, 1.0], dtype=self.dtype))
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda item: project_capped_simplex(item, cap=0.40, iterations=80),
                (values.detach().clone().requires_grad_(),),
            )
        )

    def test_uniform_mixing_enforces_exact_ess_boundary(self) -> None:
        q = torch.tensor(
            [[0.70, 0.10, 0.10, 0.10], [0.25, 0.25, 0.25, 0.25]],
            dtype=self.dtype,
        )
        mixed = enforce_minimum_ess(q, minimum_ess=3.0)
        ess = effective_sample_size(mixed)

        self.assertAlmostEqual(ess[0].item(), 3.0, places=10)
        self.assertAlmostEqual(ess[1].item(), 4.0, places=10)
        torch.testing.assert_close(mixed.sum(dim=-1), torch.ones(2, dtype=self.dtype))

    def test_target_mass_has_diagnostic_and_penalty(self) -> None:
        membership = torch.tensor([[0.10, 0.10], [0.20, 0.40]], dtype=self.dtype)
        diagnostic = membership_mass_diagnostic(membership, target_mass=0.10)
        penalty = membership_target_mass_penalty(membership, target_mass=0.10)

        torch.testing.assert_close(diagnostic.observed_mass, torch.tensor([0.10, 0.30], dtype=self.dtype))
        self.assertEqual(penalty[0].item(), 0.0)
        self.assertAlmostEqual(penalty[1].item(), 4.0)

    def test_admissible_weights_use_nominal_mass_cap_and_ess_fraction(self) -> None:
        constraints = GateConstraints(
            target_mass=0.25,
            minimum_ess_fraction=0.80,
            max_weight_multiplier=1.0,
            projection_iterations=80,
        )
        membership = torch.tensor([[0.99, 0.005, 0.003, 0.002, 0.0, 0.0, 0.0, 0.0]], dtype=self.dtype)
        q, diagnostic = admissible_weights(membership, constraints)

        # Nominal m=2, hence cap=1/m=0.5; minimum ESS is 0.8m=1.6.
        self.assertAlmostEqual(diagnostic.cap, 0.5)
        self.assertAlmostEqual(diagnostic.minimum_ess, 1.6)
        self.assertLessEqual(q.max().item(), 0.5 + 1e-12)
        self.assertGreaterEqual(effective_sample_size(q).item(), 1.6 - 1e-10)


class LinearGateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.dtype = torch.float64

    def test_fresh_initialization_is_seeded_and_vectorized(self) -> None:
        first = LinearGateEnsemble(3, 4, target_mass=0.1, seed=7, dtype=self.dtype)
        second = LinearGateEnsemble(3, 4, target_mass=0.1, seed=7, dtype=self.dtype)
        features = torch.ones(5, 3, dtype=self.dtype)

        torch.testing.assert_close(first.weight, second.weight)
        self.assertEqual(first(features).shape, (4, 5))
        previous = first.weight.detach().clone()
        first.fresh_init(8)
        self.assertFalse(torch.equal(previous, first.weight))

    def test_calibrated_membership_is_exact_and_passes_gradcheck(self) -> None:
        scores = torch.tensor(
            [[-1.7, -0.4, 0.2, 1.1, 2.0], [1.5, -1.0, 0.7, -0.2, 0.1]],
            dtype=self.dtype,
            requires_grad=True,
        )
        membership, bias = calibrated_sigmoid_membership(
            scores,
            0.25,
            bisection_iterations=80,
            force_full_solve=True,
        )

        torch.testing.assert_close(
            membership.mean(dim=-1),
            torch.full((2,), 0.25, dtype=self.dtype),
            atol=1e-10,
            rtol=1e-10,
        )
        self.assertTrue(bool(torch.isfinite(bias).all()))
        self.assertTrue(
            torch.autograd.gradcheck(
                lambda item: calibrated_sigmoid_membership(
                    item,
                    0.25,
                    bisection_iterations=80,
                    force_full_solve=True,
                )[0],
                (scores.detach().clone().requires_grad_(),),
            )
        )

    def test_warm_started_mass_solve_matches_full_solve(self) -> None:
        scores = torch.tensor(
            [[-2.2, -0.7, 0.3, 1.4, 2.8], [1.9, -1.4, 0.8, -0.5, 0.2]],
            dtype=self.dtype,
        )
        _, old_bias = calibrated_sigmoid_membership(
            scores,
            0.20,
            bisection_iterations=80,
            force_full_solve=True,
        )
        updated_scores = scores + torch.tensor(
            [[0.02, -0.01, 0.01, -0.02, 0.01], [-0.01, 0.02, -0.02, 0.01, 0.00]],
            dtype=self.dtype,
        )
        warm_membership, warm_bias = calibrated_sigmoid_membership(
            updated_scores,
            0.20,
            initial_bias=old_bias,
            newton_iterations=3,
            bisection_iterations=80,
        )
        full_membership, full_bias = calibrated_sigmoid_membership(
            updated_scores,
            0.20,
            bisection_iterations=80,
            force_full_solve=True,
        )

        torch.testing.assert_close(
            warm_membership.mean(dim=-1),
            torch.full((2,), 0.20, dtype=self.dtype),
            atol=1e-10,
            rtol=1e-10,
        )
        torch.testing.assert_close(warm_bias, full_bias, atol=1e-10, rtol=1e-10)
        torch.testing.assert_close(warm_membership, full_membership, atol=1e-10, rtol=1e-10)

    def test_frozen_gate_round_trip_and_heldout_application(self) -> None:
        constraints = GateConstraints(target_mass=0.5, minimum_ess_fraction=1.0)
        gate = FrozenLinearGate(
            weight=torch.tensor([1.0, -1.0], dtype=self.dtype),
            bias=0.25,
            class_value=1,
            objective="fisher",
            constraints=constraints,
        )
        restored = FrozenLinearGate.from_dict(gate.to_dict(), dtype=self.dtype)
        features = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [2.0, 1.0], [1.0, 2.0]],
            dtype=self.dtype,
        )
        labels = torch.tensor([0, 1, 1, 0])
        application = apply_frozen_gate(restored, features, labels)

        torch.testing.assert_close(gate.membership(features), restored.membership(features))
        self.assertEqual(application.indices.tolist(), [1, 2])
        self.assertAlmostEqual(application.q.sum().item(), 1.0)
        self.assertAlmostEqual(application.full_q.sum().item(), 1.0)
        self.assertEqual(application.full_q[0].item(), 0.0)
        self.assertEqual(application.full_q[3].item(), 0.0)

    def test_fisher_and_loss_discovery_share_constraints_but_select_by_own_score(self) -> None:
        features = torch.zeros(8, 2, dtype=self.dtype)
        labels = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        task = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0], dtype=self.dtype)
        nuisance = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0], dtype=self.dtype)
        predictive = torch.ones(8, dtype=self.dtype)
        losses = torch.tensor([0.1, 0.1, 0.1, 0.1, 2.0, 2.0, 2.0, 2.0], dtype=self.dtype)
        constraints = GateConstraints(target_mass=0.5, minimum_ess_fraction=1.0)

        fisher = discover_class_conditional(
            features,
            labels,
            config=GateOptimizationConfig(
                objective="fisher",
                constraints=constraints,
                restarts=3,
                steps=0,
                eta=0.0,
                ridge=1e-3,
                seed=11,
            ),
            task_coordinate=task,
            nuisance_coordinate=nuisance,
            predictive_weight=predictive,
        )
        high_loss = discover_class_conditional(
            features,
            labels,
            config=GateOptimizationConfig(
                objective="high_loss",
                constraints=constraints,
                restarts=3,
                steps=0,
                seed=11,
            ),
            losses=losses,
        )

        self.assertEqual(fisher.selected.class_value, 0)
        self.assertEqual(high_loss.selected.class_value, 1)
        self.assertEqual(fisher.gate.constraints, high_loss.gate.constraints)
        self.assertGreaterEqual(
            fisher.selected.weight_diagnostics.effective_sample_size.item(),
            fisher.selected.weight_diagnostics.minimum_ess,
        )
        self.assertGreaterEqual(
            high_loss.selected.weight_diagnostics.effective_sample_size.item(),
            high_loss.selected.weight_diagnostics.minimum_ess,
        )

    def test_optimized_gate_keeps_exact_mass_and_constant_sanity_floor(self) -> None:
        coordinate = torch.linspace(-2.0, 2.0, 40, dtype=self.dtype)
        features = torch.stack([coordinate, coordinate.square()], dim=-1)
        labels = torch.tensor([0] * 20 + [1] * 20)
        losses = (coordinate + 2.1).square()
        constraints = GateConstraints(target_mass=0.25, minimum_ess_fraction=0.8)
        result = discover_class_conditional(
            features,
            labels,
            config=GateOptimizationConfig(
                objective="high_loss",
                constraints=constraints,
                restarts=3,
                steps=30,
                learning_rate=0.05,
                seed=29,
            ),
            losses=losses,
        )

        for class_result in result.per_class:
            self.assertAlmostEqual(
                class_result.weight_diagnostics.membership.observed_mass.item(),
                constraints.target_mass,
                places=9,
            )
            mask = labels == class_result.class_value
            constant_score = losses[mask].mean().item()
            self.assertGreaterEqual(class_result.objective_score + 1e-10, constant_score)


if __name__ == "__main__":
    unittest.main()
