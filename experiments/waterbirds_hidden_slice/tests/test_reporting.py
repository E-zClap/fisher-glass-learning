from __future__ import annotations

import tempfile
import unittest
import json
from pathlib import Path

import numpy as np

from experiments.waterbirds_hidden_slice.aggregate import (
    _bounded_summary,
    _mean_stability,
    _uniform_slice_metrics,
)
from experiments.waterbirds_hidden_slice.make_report import (
    _difference_claim,
    _discovery_control_verdict,
    _qualified_repair_claim,
    _repair_selection_suffix,
    _repair_safety_claim,
    _read_aggregate as _read_report_aggregate,
)
from experiments.waterbirds_hidden_slice.common import json_sha256
from experiments.waterbirds_hidden_slice.make_figure import (
    _repair_tick_label,
    _require_complete_paper_protocol,
)


def _summary(lo: float, hi: float, mean: float = 0.0) -> dict[str, float | int]:
    return {"n": 5, "mean": mean, "lo": lo, "hi": hi}


class ReportingGuardrailTests(unittest.TestCase):
    def test_incomplete_run_cannot_emit_paper_figure(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "incomplete run"):
            _require_complete_paper_protocol({"paper_protocol_complete": False})

    def test_figure_label_discloses_no_op_count(self) -> None:
        label = _repair_tick_label(
            {
                "repair_selection": {
                    "any_no_op": True,
                    "no_op_seeds": 3,
                    "total_seeds": 5,
                }
            },
            "Fisher",
        )
        self.assertIn("NO-OP 3/5", label)

    def test_report_rejects_legacy_aggregate_without_source_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "aggregate.json"
            payload = {
                "schema_version": 1,
                "stage": "five_seed_reporting_aggregate",
            }
            payload["aggregate_sha256"] = json_sha256(payload)
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaisesRegex(RuntimeError, "no scientific implementation manifest"):
                _read_report_aggregate(path)

    def test_stability_excludes_never_selected_class(self) -> None:
        crossfit = {
            "folds": [{"fold": index} for index in range(5)],
            "aggregate": {
                "selected_class_counts": {"0": 5, "1": 0},
                "stability": {
                    "0": {
                        "pairs": 10,
                        "score_correlation_mean": 0.8,
                        "score_correlation_min": 0.6,
                        "top_overlap_mean": 0.4,
                        "top_overlap_min": 0.2,
                    },
                    "1": {
                        "pairs": 10,
                        "score_correlation_mean": -0.8,
                        "score_correlation_min": -0.9,
                        "top_overlap_mean": 0.05,
                        "top_overlap_min": 0.01,
                    },
                },
            },
        }
        result = _mean_stability(crossfit)
        self.assertIsNotNone(result)
        assert result is not None
        self.assertAlmostEqual(float(result["score_correlation"]), 0.8)
        self.assertAlmostEqual(float(result["top_overlap"]), 0.4)
        self.assertAlmostEqual(float(result["selected_class_agreement"]), 1.0)

    def test_bounded_interval_never_reports_impossible_jaccard(self) -> None:
        result = _bounded_summary([0.0, 0.0, 0.0, 0.0, 1.0], 0.95, lower=0.0, upper=1.0)
        self.assertEqual(result["lo"], 0.0)
        self.assertLessEqual(result["hi"], 1.0)
        self.assertLess(result["unbounded_lo"], 0.0)

    def test_unfavorable_intervals_are_not_described_as_unresolved(self) -> None:
        claim = _difference_claim(_summary(0.1, 0.2), True, "the discovery claim")
        self.assertIn("contradicting", claim)
        verdict = _discovery_control_verdict(
            [_summary(-0.2, -0.1), _summary(0.1, 0.2)]
        )
        self.assertIn("significantly unfavorable", verdict)

    def test_worst_group_damage_forces_tradeoff_language(self) -> None:
        claim = _repair_safety_claim(
            _summary(0.1, 0.2),
            _summary(-0.3, -0.2),
            _summary(-0.4, -0.3),
        )
        self.assertIn("tradeoff", claim)
        self.assertNotIn("safe hidden-slice improvement", claim)

    def test_no_op_selection_blocks_repair_claim_and_is_visible(self) -> None:
        row = {
            "method": "Hidden-Fisher repair",
            "repair_selection": {
                "no_op_seeds": 5,
                "active_repair_seeds": 0,
                "total_seeds": 5,
                "any_no_op": True,
                "all_no_op": True,
            },
        }
        self.assertIn("NO-OP", _repair_selection_suffix(row))
        claim = _qualified_repair_claim(
            _summary(0.1, 0.2),
            row,
            "a repaired hidden-slice improvement",
        )
        self.assertIn("does not establish", claim)
        safety = _repair_safety_claim(
            _summary(0.1, 0.2),
            _summary(0.0, 0.1),
            _summary(0.0, 0.1),
            row["repair_selection"],
            row["repair_selection"],
        )
        self.assertIn("no-op", safety.lower())

    def test_uniform_gate_is_computed_from_archived_score_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            score_path = Path(directory) / "scores.npz"
            np.savez_compressed(
                score_path,
                y=np.asarray([0, 0, 1, 1], dtype=np.int64),
                erm_task=np.asarray([-1.0, 1.0, -0.5, 0.5]),
                b_vl=np.asarray([-0.5, 0.5, 1.0, -1.0]),
                erm_logits=np.asarray([-1.0, -0.2, 0.2, 1.0]),
            )
            context = {
                "score_path": str(score_path),
                "blind_test": {"arms": {"erm": {"fisher_selector": {"class_id": 0}}}},
                "discovery": {
                    "base_model": {"temperature": 1.0},
                    "protocol": {
                        "slice_mass": 0.1,
                        "shrinkage": 0.1,
                        "ridge": 1e-3,
                        "epsilon": 1e-8,
                    },
                },
            }
            result = _uniform_slice_metrics(context)
            self.assertAlmostEqual(result["constraints"]["effective_sample_size"], 2.0)
            self.assertEqual(result["selector_objective"], "constant_uniform")
            self.assertEqual(result["gate_transfer"]["absolute_mass_error"], 0.0)


if __name__ == "__main__":
    unittest.main()
