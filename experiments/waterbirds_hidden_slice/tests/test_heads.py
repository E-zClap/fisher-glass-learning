from __future__ import annotations

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from experiments.waterbirds_hidden_slice.heads import FrozenHead, HeadConfig, RepairSpec, train_head

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))
from repair import _select_candidate  # noqa: E402


class RepairHeadTests(unittest.TestCase):
    def test_repair_does_not_mutate_frozen_slice_weights(self) -> None:
        features = np.array(
            [[-2.0, 0.2], [-1.0, -0.1], [0.8, 0.1], [1.8, -0.2], [1.2, 0.4], [-1.2, -0.4]],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1, 1, 0], dtype=np.int64)
        nuisance = np.array([-1.0, -0.5, 0.5, 1.0, 0.8, -0.8], dtype=np.float64)
        q = np.array([0.5, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float64)
        before = q.copy()
        result = train_head(
            features,
            labels,
            seed=17,
            device="cpu",
            config=HeadConfig(steps=2, log_every=1),
            repair=RepairSpec("combined_repair", lambda_loss=0.1, lambda_fisher=0.001),
            slice_weights=q,
            nuisance=nuisance,
            selected_class=0,
            temperature=1.0,
        )
        np.testing.assert_array_equal(q, before)
        self.assertTrue(np.all(np.isfinite(result.head.weight)))
        self.assertTrue(np.isfinite(result.head.bias))

    def test_repair_rejects_cross_class_slice_mass(self) -> None:
        features = np.eye(4, dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        with self.assertRaisesRegex(ValueError, "class-conditional"):
            train_head(
                features,
                labels,
                seed=1,
                device="cpu",
                config=HeadConfig(steps=1),
                repair=RepairSpec("loss_repair", lambda_loss=0.1),
                slice_weights=np.full(4, 0.25),
                nuisance=np.zeros(4),
                selected_class=0,
            )

    def test_fisher_penalty_is_dimensionless_at_warm_start(self) -> None:
        features = np.array(
            [[-2.0, 0.1], [-1.0, -0.2], [1.0, 0.2], [2.0, -0.1]],
            dtype=np.float32,
        )
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        nuisance = np.array([-1.0, -0.5, 0.5, 1.0], dtype=np.float64)
        q = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float64)
        initial = FrozenHead(np.array([0.3, -0.1], dtype=np.float32), -0.05)
        result = train_head(
            features,
            labels,
            seed=2,
            device="cpu",
            config=HeadConfig(steps=1, learning_rate=1e-8, weight_decay=0.0, gradient_clip=0.0),
            initial=initial,
            repair=RepairSpec("fisher_repair", lambda_fisher=0.01),
            slice_weights=q,
            nuisance=nuisance,
            selected_class=0,
            temperature=1.0,
        )
        reference = result.config["fisher_penalty"]["initial_fisher_reference"]
        self.assertGreater(reference, 0.0)
        self.assertAlmostEqual(
            result.final["normalized_fisher_penalty"],
            reference * result.final["inverse_slice_fisher"],
            places=5,
        )
        self.assertAlmostEqual(result.final["normalized_fisher_penalty"], 1.0, places=4)

    def test_warm_start_checkpoint_can_restore_step_zero(self) -> None:
        features = np.array([[-2.0], [-1.0], [1.0], [2.0]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1], dtype=np.int64)
        checkpoint_labels = 1 - labels
        initial = FrozenHead(np.zeros(1, dtype=np.float32), 0.0)
        result = train_head(
            features,
            labels,
            seed=3,
            device="cpu",
            config=HeadConfig(
                steps=20,
                learning_rate=0.1,
                weight_decay=0.0,
                log_every=5,
                early_stopping_patience=2,
                early_stopping_eval_every=1,
                early_stopping_min_delta=1e-8,
            ),
            initial=initial,
            checkpoint_features=features,
            checkpoint_labels=checkpoint_labels,
        )
        self.assertEqual(result.final["step"], 0.0)
        self.assertEqual(result.final["early_stopped"], 1.0)
        np.testing.assert_allclose(result.head.weight, initial.weight, atol=0.0, rtol=0.0)
        self.assertEqual(result.head.bias, initial.bias)

    def test_no_op_is_valid_but_cannot_be_trained(self) -> None:
        spec = RepairSpec("no_op")
        spec.validate()
        with self.assertRaisesRegex(ValueError, "must not be passed"):
            train_head(
                np.eye(2, dtype=np.float32),
                np.array([0, 1], dtype=np.int64),
                seed=1,
                device="cpu",
                config=HeadConfig(steps=1),
                repair=spec,
            )

    def test_fisher_candidate_selection_uses_safety_then_metric(self) -> None:
        args = SimpleNamespace(
            accuracy_budget=0.02,
            balanced_accuracy_budget=0.02,
            per_class_accuracy_budget=0.02,
            loss_lambdas=[0.1, 1.0],
            fisher_lambdas=[0.001, 0.01],
            combined_loss_lambdas=[0.1, 1.0],
            combined_fisher_lambdas=[0.001, 0.01],
        )
        base_metrics = {
            "accuracy": 0.8,
            "balanced_accuracy": 0.8,
            "per_class_accuracy": {"0": 0.8, "1": 0.8},
        }
        base_slice = {
            "weighted_calibrated_loss": 1.0,
            "predictive": {"projected_fisher": 0.1},
        }

        def row(index, name, value, fisher, class_one=0.8):
            return {
                "candidate_index": index,
                "spec": {"name": name, "lambda_loss": 0.0, "lambda_fisher": value},
                "validation_metrics": {
                    "accuracy": 0.8,
                    "balanced_accuracy": 0.8,
                    "per_class_accuracy": {"0": 0.8, "1": class_one},
                },
                "validation_original_slice": {
                    "weighted_calibrated_loss": 1.0,
                    "predictive": {"projected_fisher": fisher},
                },
            }

        rows = [
            row(0, "no_op", 0.0, 0.1),
            row(1, "fisher_repair", 0.001, 0.2),
            row(2, "fisher_repair", 0.01, 0.3, class_one=0.7),
        ]
        selected, selection = _select_candidate(
            "fisher_repair", rows, base_metrics, base_slice, args
        )
        self.assertEqual(selected["candidate_index"], 1)
        self.assertTrue(selection["accuracy_budget_satisfied"])
        self.assertFalse(rows[2]["task_label_safety"]["satisfied"])


if __name__ == "__main__":
    unittest.main()
