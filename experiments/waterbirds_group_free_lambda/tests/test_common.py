from __future__ import annotations

import unittest

from experiments.waterbirds_group_free_lambda.experiment_common import (
    ALGORITHM_VERSION,
    FROZEN_ERM_REFERENCE_KEY,
    SCHEMA_VERSION,
    crossfit_certificate_scores,
    implementation_manifest,
    json_sha256,
    lambda_key,
    one_sided_mean_bound,
    paired_inverse_diagnostic,
    risk_certificate_choice,
    validate_lambdas,
)


def _crossfit(values: list[float]) -> dict:
    return {
        "folds": [
            {"heldout": {"predictive": {"inverse_fisher": float(value)}}}
            for value in values
        ]
    }


class LambdaContractTests(unittest.TestCase):
    def test_lambda_keys_are_stable_and_grid_is_sorted(self) -> None:
        self.assertEqual(lambda_key(0.0), "lambda_0")
        self.assertEqual(lambda_key(0.03), "lambda_0p03")
        self.assertEqual(validate_lambdas([0.3, 0.03, 0.0, 0.01, 0.1]), [0.0, 0.01, 0.03, 0.1, 0.3])

    def test_v2_identity_and_frozen_reference_key_are_explicit(self) -> None:
        self.assertEqual(SCHEMA_VERSION, 2)
        self.assertEqual(ALGORITHM_VERSION, "group-free-lambda-selection-2.0.0")
        self.assertEqual(FROZEN_ERM_REFERENCE_KEY, "frozen_erm_reference")

    def test_grid_requires_matched_erm_anchor_and_fixed_reference(self) -> None:
        with self.assertRaisesRegex(ValueError, "matched ERM continuation anchor"):
            validate_lambdas([0.01, 0.03])
        with self.assertRaisesRegex(ValueError, "fixed reference"):
            validate_lambdas([0.0, 0.01])
        with self.assertRaises(ValueError):
            lambda_key(float("nan"))

    def test_primary_rule_uses_one_se_then_score_with_small_lambda_tie_break(self) -> None:
        def candidate(coefficient: float, nll: float, se: float, score: float) -> dict:
            return {
                "lambda_fg": coefficient,
                "validation_metrics": {"nll": nll, "nll_standard_error": se},
                "certificate_scores": {"simultaneous_certificate_lower_score": score},
            }

        rows = [
            candidate(0.0, 0.50, 0.02, 0.10),
            candidate(0.01, 0.51, 0.02, 0.20),
            candidate(0.03, 0.52, 0.02, 0.20),
            candidate(0.10, 0.53, 0.02, 0.90),
        ]
        choice = risk_certificate_choice(rows)
        self.assertEqual([row["lambda_fg"] for row in choice["eligible"]], [0.0, 0.01, 0.03])
        self.assertEqual(choice["selected"]["lambda_fg"], 0.01)


class BoundTests(unittest.TestCase):
    def test_one_sided_bound_and_bonferroni_are_ordered(self) -> None:
        marginal = one_sided_mean_bound([1.0, 2.0, 3.0, 4.0, 5.0], confidence=0.95, side="upper")
        simultaneous = one_sided_mean_bound(
            [1.0, 2.0, 3.0, 4.0, 5.0], confidence=0.95, comparisons=5, side="upper"
        )
        self.assertAlmostEqual(marginal["mean"], 3.0)
        self.assertGreater(simultaneous["upper"], marginal["upper"])

    def test_crossfit_certificate_is_reciprocal_of_mean_inverse(self) -> None:
        result = crossfit_certificate_scores(_crossfit([2.0, 2.0, 2.0, 2.0, 2.0]), confidence=0.95, comparisons=5)
        self.assertAlmostEqual(result["point_certificate"], 0.5)
        self.assertAlmostEqual(result["simultaneous_certificate_lower_score"], 0.5)

    def test_paired_diagnostic_requires_same_folds(self) -> None:
        improved = paired_inverse_diagnostic(
            _crossfit([1.0, 1.1, 0.9, 1.0, 1.0]),
            _crossfit([2.0, 2.1, 1.9, 2.0, 2.0]),
            confidence=0.95,
            comparisons=5,
        )
        self.assertTrue(improved["diagnostic_upper_score_below_zero"])
        self.assertIn("fold_difference_candidate_minus_baseline", improved)
        self.assertNotIn("fold_difference_candidate_minus_no_op", improved)
        with self.assertRaisesRegex(ValueError, "same cross-fit"):
            paired_inverse_diagnostic(
                _crossfit([1.0, 1.0]),
                _crossfit([1.0, 1.0, 1.0]),
                confidence=0.95,
                comparisons=2,
            )


class SourceManifestTests(unittest.TestCase):
    def test_manifest_identity_covers_local_and_imported_science(self) -> None:
        manifest = implementation_manifest()
        self.assertIn("waterbirds_group_free_lambda/select_lambda.py", manifest["files"])
        self.assertIn("waterbirds_group_free_lambda/certify.py", manifest["files"])
        self.assertIn("waterbirds_hidden_slice/gates.py", manifest["files"])
        identity = {
            "algorithm_version": manifest["algorithm_version"],
            "files": manifest["files"],
        }
        self.assertEqual(manifest["combined_sha256"], json_sha256(identity))
        for digest in manifest["files"].values():
            self.assertEqual(len(digest), 64)
            int(digest, 16)


if __name__ == "__main__":
    unittest.main()
