from __future__ import annotations

import unittest

import numpy as np

from experiments.waterbirds_hidden_slice.metrics import (
    FisherMetricConfig,
    evaluate_class_slice,
)


class ProbabilityBoundaryTests(unittest.TestCase):
    def test_float32_projection_residue_is_canonicalized_before_geometry(self) -> None:
        n = 37
        q32 = np.full(n, np.float32(1.0 / n), dtype=np.float32)
        self.assertNotEqual(float(q32.astype(np.float64).sum()), 1.0)

        result = evaluate_class_slice(
            task=np.linspace(-1.0, 1.0, n),
            nuisance=np.linspace(0.75, -0.25, n),
            logits=np.linspace(-2.0, 2.0, n),
            labels=np.zeros(n, dtype=np.int64),
            q=q32,
            class_id=0,
            temperature=1.0,
            config=FisherMetricConfig(),
        )

        self.assertAlmostEqual(result["constraints"]["sum"], 1.0, places=12)
        self.assertTrue(np.isfinite(result["predictive"]["projected_fisher"]))


if __name__ == "__main__":
    unittest.main()
