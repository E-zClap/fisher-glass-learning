from __future__ import annotations

import copy
import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from experiments.waterbirds_group_free_lambda.aggregate import (
    _load_seed,
    _rebuttal,
    _reportable_protocol_deviations,
    _summary,
    _validate_cross_seed_provenance,
    _validate_embedded_cohort,
    main as aggregate_main,
)
from experiments.waterbirds_group_free_lambda.experiment_common import (
    FROZEN_ERM_REFERENCE_KEY,
    SCHEMA_VERSION,
    atomic_savez_compressed,
    atomic_write_json,
    canonical_protocol_hash,
    crossfit_certificate_scores,
    derive_seed,
    file_sha256,
    implementation_manifest,
    json_artifact_sha256,
    json_sha256,
    lambda_key,
    risk_certificate_choice,
    task_metrics,
)
from experiments.waterbirds_group_free_lambda.preflight import check_static_firewall

EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from experiments.waterbirds_group_free_lambda.select_lambda import (
    _protocol as selection_protocol,
    _resume_if_valid,
    _validate_args as validate_selection_args,
    build_parser as selection_parser,
)
from experiments.waterbirds_group_free_lambda.certify import (
    _validate_crossfit_evidence,
    _validate_selection,
    main as certify_main,
)
from experiments.waterbirds_group_free_lambda.posthoc import main as posthoc_main
from experiments.waterbirds_group_free_lambda.posthoc import (
    _validate_blind_json_metrics,
    _verify_blind_score_semantics,
)
from blind_protocol import evaluate_gate, make_model_coordinates
from common import stratified_kfold_indices
from gates import FrozenLinearGate
from heads import FrozenHead
from metrics import FisherMetricConfig, evaluate_class_slice


class SummaryTests(unittest.TestCase):
    def test_single_seed_interval_is_explicitly_degenerate(self) -> None:
        result = _summary([0.5], confidence=0.95, bounds=(0.0, 1.0))
        self.assertEqual(result["n"], 1)
        self.assertEqual(result["low"], 0.5)
        self.assertEqual(result["high"], 0.5)

    def test_display_bounds_do_not_hide_unbounded_interval(self) -> None:
        result = _summary([0.0, 1.0], confidence=0.95, bounds=(0.0, 1.0))
        self.assertEqual(result["low"], 0.0)
        self.assertEqual(result["high"], 1.0)
        self.assertLess(result["unbounded_low"], 0.0)
        self.assertGreater(result["unbounded_high"], 1.0)

    def test_reportable_defaults_match_selector_defaults_exactly(self) -> None:
        args = selection_parser().parse_args([])
        lambdas = validate_selection_args(args)
        protocol = selection_protocol(args, lambdas)
        contexts = [
            {
                "seed": seed,
                "selection": {"protocol": protocol},
                "blind": {
                    "certification_protocol": {
                        "bootstrap_replicates": 2000,
                        "bootstrap_confidence": 0.95,
                        "cohort_seeds": [17, 18, 19, 20, 21],
                        "complete_default_cohort_frozen_before_test": True,
                        "smoke": False,
                    }
                },
                "posthoc": {
                    "posthoc_protocol": {
                        "cohort_seeds": [17, 18, 19, 20, 21],
                        "complete_default_cohort_frozen_before_metadata": True,
                        "smoke": False,
                    }
                },
            }
            for seed in (17, 18, 19, 20, 21)
        ]
        self.assertEqual(
            _reportable_protocol_deviations(
                contexts, [17, 18, 19, 20, 21], smoke=False
            ),
            [],
        )
        changed = copy.deepcopy(contexts)
        changed[0]["selection"]["protocol"]["folds"] = 2
        deviations = _reportable_protocol_deviations(
            changed, [17, 18, 19, 20, 21], smoke=False
        )
        self.assertTrue(any("folds=2" in value for value in deviations))


class OutcomeWordingTests(unittest.TestCase):
    @staticmethod
    def _aggregate(worst: dict, certificate: dict) -> dict:
        displayed = {"mean": 0.5, "low": 0.4, "high": 0.6}
        policy = lambda key, counts: {
            "key": key,
            "lambda_counts": counts,
            "metrics": {
                "test_accuracy": displayed,
                "test_worst_group_accuracy": displayed,
                "test_hidden_certificate": displayed,
            },
        }
        zero = {"mean": 0.0, "low": -0.01, "high": 0.01}
        return {
            "seeds": [17, 18, 19, 20, 21],
            "reportable_five_seed_protocol_complete": True,
            "selection_agreement": {
                "group_free_equals_validation_risk_minimizer": 5,
                "n_seeds": 5,
            },
            "policies": [
                policy("frozen_erm_reference", {"n/a": 5}),
                policy("matched_erm_lambda_0", {"0": 5}),
                policy("fixed_0.03", {"0.03": 5}),
                policy("validation_risk_minimizer", {"0.3": 5}),
                policy("group_free", {"0.3": 5}),
                policy("group_supervised_val_comparator", {"0": 5}),
            ],
            "paired_comparisons": [
                {
                    "candidate_policy": "group_free",
                    "baseline_policy": "matched_erm_lambda_0",
                    "metric": "test_worst_group_accuracy",
                    "difference_candidate_minus_baseline": worst,
                },
                {
                    "candidate_policy": "group_free",
                    "baseline_policy": "matched_erm_lambda_0",
                    "metric": "test_hidden_certificate",
                    "difference_candidate_minus_baseline": certificate,
                },
                {
                    "candidate_policy": "group_free",
                    "baseline_policy": "matched_erm_lambda_0",
                    "metric": "test_accuracy",
                    "difference_candidate_minus_baseline": zero,
                },
                {
                    "candidate_policy": "matched_erm_lambda_0",
                    "baseline_policy": "frozen_erm_reference",
                    "metric": "test_worst_group_accuracy",
                    "difference_candidate_minus_baseline": zero,
                },
                {
                    "candidate_policy": "group_free",
                    "baseline_policy": "frozen_erm_reference",
                    "metric": "test_worst_group_accuracy",
                    "difference_candidate_minus_baseline": worst,
                },
            ],
        }

    def test_negative_result_is_not_rendered_as_success(self) -> None:
        text = _rebuttal(
            self._aggregate(
                {"mean": -0.10, "low": -0.15, "high": -0.05},
                {"mean": 0.01, "low": -0.01, "high": 0.03},
            )
        )
        self.assertIn("worst-group harm", text)
        self.assertIn("hidden-certificate contrast was inconclusive", text.lower())
        self.assertNotIn("provide a concrete way to choose lambda", text)

    def test_favorable_result_remains_labeled_post_hoc(self) -> None:
        text = _rebuttal(
            self._aggregate(
                {"mean": 0.05, "low": 0.01, "high": 0.09},
                {"mean": 0.02, "low": 0.01, "high": 0.03},
            )
        )
        self.assertIn("higher worst-group accuracy", text)
        self.assertIn("not a fresh held-out confirmation", text)


class FirewallTests(unittest.TestCase):
    def test_blind_sources_do_not_load_direct_metadata_apis(self) -> None:
        self.assertEqual(check_static_firewall(), [])

    def test_partial_reveal_cohorts_are_rejected_outside_smoke_mode(self) -> None:
        for main_function, script_name in (
            (certify_main, "certify.py"),
            (posthoc_main, "posthoc.py"),
        ):
            with self.subTest(stage=script_name):
                with mock.patch("sys.argv", [script_name, "--seeds", "17"]):
                    with self.assertRaisesRegex(ValueError, "complete default seed cohort"):
                        main_function()


class SemanticChainTests(unittest.TestCase):
    @staticmethod
    def _blind_fixture() -> tuple[dict, dict, dict[str, np.ndarray]]:
        labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
        logits = np.asarray([-1.0, -0.5, 0.5, 1.0], dtype=np.float64)
        task = logits.copy()
        nuisance = np.asarray([-1.0, 0.5, -0.5, 1.0], dtype=np.float64)
        q_full = np.asarray([0.5, 0.5, 0.0, 0.0], dtype=np.float64)
        config = FisherMetricConfig(shrinkage=0.1, ridge=0.001, epsilon=1e-8)
        hidden = evaluate_class_slice(
            task=task[:2],
            nuisance=nuisance[:2],
            logits=logits[:2],
            labels=labels[:2],
            q=q_full[:2],
            class_id=0,
            temperature=1.0,
            config=config,
            hard_mass=0.5,
        )
        hidden["certificate"] = float(hidden["predictive"]["projected_fisher"])
        gate = {
            "schema_version": 1,
            "weight": [0.0, 0.0],
            "bias": 0.0,
            "class_value": 0,
            "objective": "fisher",
            "constraints": {
                "target_mass": 0.5,
                "minimum_ess_fraction": 0.8,
                "max_weight_multiplier": 2.0,
                "projection_iterations": 64,
            },
        }
        head = {"weight": [1.0, 0.0], "bias": 0.0}
        reference = {
            "head": head,
            "task_mean": 0.0,
            "task_std": 1.0,
            "final_gate": gate,
        }
        candidate = {
            "candidate_index": 0,
            "lambda_fg": 0.0,
            "head": head,
            "task_mean": 0.0,
            "task_std": 1.0,
            "final_gate": gate,
        }
        metrics = task_metrics(logits, labels, 1.0)
        selection = {
            "normalization": {
                "feature_mean": [[0.0, 0.0]],
                "feature_std": [[1.0, 1.0]],
                "nuisance_mean": 0.0,
                "nuisance_std": 1.0,
                "base_temperature": 1.0,
            },
            "protocol": {
                "fisher_metric": {"shrinkage": 0.1, "ridge": 0.001, "epsilon": 1e-8}
            },
            FROZEN_ERM_REFERENCE_KEY: reference,
            "candidates": [candidate],
        }
        blind = {
            FROZEN_ERM_REFERENCE_KEY: {
                "task_metrics": copy.deepcopy(metrics),
                "hidden_certificate": copy.deepcopy(hidden),
            },
            "candidates": [
                {
                    "candidate_index": 0,
                    "lambda_fg": 0.0,
                    "task_metrics": copy.deepcopy(metrics),
                    "hidden_certificate": copy.deepcopy(hidden),
                }
            ],
        }
        arrays = {
            "row_id": np.arange(labels.size, dtype=np.int64),
            "y": labels,
            "nuisance": nuisance,
            "frozen_erm_reference__logits": logits,
            "frozen_erm_reference__task": task,
            "frozen_erm_reference__q": q_full,
            "frozen_erm_reference__membership": np.asarray(
                [0.5, 0.5, np.nan, np.nan], dtype=np.float64
            ),
            "lambda_0__logits": logits,
            "lambda_0__task": task,
            "lambda_0__q": q_full,
            "lambda_0__membership": np.asarray(
                [0.5, 0.5, np.nan, np.nan], dtype=np.float64
            ),
        }
        return selection, blind, arrays

    def test_blind_json_metrics_are_reconciled_with_npz_scores(self) -> None:
        selection, blind, arrays = self._blind_fixture()
        _validate_blind_json_metrics(selection, blind, arrays)
        blind["candidates"][0]["task_metrics"]["accuracy"] = 0.0
        with self.assertRaisesRegex(RuntimeError, "task metric accuracy is inconsistent"):
            _validate_blind_json_metrics(selection, blind, arrays)

    def test_blind_npz_is_bound_to_prepared_test_heads_and_gates(self) -> None:
        selection, _blind, arrays = self._blind_fixture()
        split = SimpleNamespace(
            row_id=np.arange(4, dtype=np.int64),
            y=np.asarray([0, 0, 1, 1], dtype=np.int64),
            h=np.asarray(
                [[-1.0, 0.0], [-0.5, 0.0], [0.5, 0.0], [1.0, 0.0]],
                dtype=np.float32,
            ),
            b_vl=np.asarray([-1.0, 0.5, -0.5, 1.0], dtype=np.float64),
        )
        recomputed = _verify_blind_score_semantics(
            selection, arrays, split, label="synthetic blind test"
        )
        self.assertEqual(set(recomputed), set(arrays))
        for field, message in (
            ("nuisance", "nuisance coordinate"),
            ("lambda_0__logits", "logits do not match"),
            ("lambda_0__membership", "membership does not match"),
            ("lambda_0__q", "weights do not match"),
        ):
            with self.subTest(field=field):
                tampered = {key: np.array(value, copy=True) for key, value in arrays.items()}
                if field == "lambda_0__q":
                    tampered[field][:2] = np.asarray([0.6, 0.4])
                else:
                    tampered[field][0] += 0.1
                with self.assertRaisesRegex(RuntimeError, message):
                    _verify_blind_score_semantics(
                        selection, tampered, split, label="synthetic blind test"
                    )

    def test_crossfit_ranking_evidence_is_replayed_from_frozen_fold_gates(self) -> None:
        labels = np.asarray([0, 0, 0, 0, 1, 1, 1, 1], dtype=np.int64)
        features = np.asarray(
            [
                [-1.0, -0.5],
                [-0.7, 0.3],
                [-0.2, 0.9],
                [0.1, -0.8],
                [0.3, 0.7],
                [0.6, -0.4],
                [0.9, 0.2],
                [1.2, -0.1],
            ],
            dtype=np.float32,
        )
        nuisance = np.linspace(-1.0, 1.0, labels.size, dtype=np.float64)
        head = FrozenHead(np.asarray([0.8, -0.2], dtype=np.float32), 0.1)
        bundle = make_model_coordinates(
            head,
            features,
            labels,
            nuisance,
            task_mean=0.0,
            task_std=1.0,
            temperature=1.0,
        )
        gate_payload = {
            "schema_version": 1,
            "weight": [0.3, -0.1],
            "bias": 0.0,
            "class_value": 0,
            "objective": "fisher",
            "constraints": {
                "target_mass": 0.5,
                "minimum_ess_fraction": 0.8,
                "max_weight_multiplier": 2.0,
                "projection_iterations": 64,
            },
        }
        gate = FrozenLinearGate.from_dict(gate_payload)
        metric_config = FisherMetricConfig(0.1, 0.001, 1e-8)
        partition_seed = 123
        crossfit_gate_seed = 456
        gate_config_seed = 789
        fold_pairs = stratified_kfold_indices(labels, 2, partition_seed)
        crossfit_config = {
            "objective": "fisher",
            "constraints": copy.deepcopy(gate_payload["constraints"]),
            "restarts": 2,
            "steps": 20,
            "learning_rate": 0.03,
            "weight_decay": 0.0001,
            "mass_penalty_strength": 10.0,
            "eta": 0.1,
            "ridge": 0.001,
            "eps": 1e-8,
            "use_predictive_fisher": True,
            "seed": gate_config_seed,
        }
        rows = []
        for fold_index, (discover_idx, heldout_idx) in enumerate(fold_pairs):
            heldout, _arrays = evaluate_gate(
                gate,
                bundle.take(heldout_idx),
                metric_config,
                device="cpu",
            )
            rows.append(
                {
                    "fold": fold_index,
                    "n_discover": int(discover_idx.size),
                    "n_cert": int(heldout_idx.size),
                    "gate": copy.deepcopy(gate_payload),
                    "discovery": {
                        "objective": "fisher",
                        "seed": derive_seed(
                            crossfit_gate_seed, "fisher", "fold", fold_index
                        ),
                        "selected_class": 0,
                    },
                    "heldout": heldout,
                }
            )
        record = {
            "crossfit": {
                "config": crossfit_config,
                "partition_seed": partition_seed,
                "heldout_evaluation_device": "cpu",
                "folds": rows,
            }
        }
        gate_protocol = {
            "steps": 20,
            "learning_rate": 0.03,
            "restarts": 2,
            "weight_decay": 0.0001,
            "mass_penalty_strength": 10.0,
            "slice_mass": 0.5,
            "minimum_ess_fraction": 0.8,
            "maximum_weight_multiplier": 2.0,
            "shrinkage": 0.1,
            "ridge": 0.001,
            "epsilon": 1e-8,
        }
        _validate_crossfit_evidence(
            record,
            bundle,
            fold_pairs,
            metric_config,
            crossfit_gate_seed,
            gate_config_seed,
            gate_protocol,
            label="synthetic candidate",
        )
        tampered = copy.deepcopy(record)
        tampered["crossfit"]["folds"][0]["heldout"]["predictive"][
            "inverse_fisher"
        ] += 0.1
        with self.assertRaisesRegex(RuntimeError, "replayed validation evidence"):
            _validate_crossfit_evidence(
                tampered,
                bundle,
                fold_pairs,
                metric_config,
                crossfit_gate_seed,
                gate_config_seed,
                gate_protocol,
                label="synthetic candidate",
            )


class HashChainTests(unittest.TestCase):
    def test_embedded_cohort_digest_and_seed_order_are_enforced(self) -> None:
        cohort = {
            "seeds": [17, 18],
            "artifacts": [{"seed": 17}, {"seed": 18}],
        }
        cohort["sha256"] = canonical_protocol_hash(cohort)
        seeds, entries, _digest = _validate_embedded_cohort(
            cohort, label="synthetic cohort"
        )
        self.assertEqual(seeds, [17, 18])
        self.assertEqual([entry["seed"] for entry in entries], seeds)

        tampered = copy.deepcopy(cohort)
        tampered["artifacts"][1]["seed"] = 17
        with self.assertRaisesRegex(RuntimeError, "digest is inconsistent"):
            _validate_embedded_cohort(tampered, label="synthetic cohort")

        reordered = {
            "seeds": [17, 18],
            "artifacts": [{"seed": 18}, {"seed": 17}],
        }
        reordered["sha256"] = canonical_protocol_hash(reordered)
        with self.assertRaisesRegex(RuntimeError, "seed ordering is invalid"):
            _validate_embedded_cohort(reordered, label="synthetic cohort")

    def _write_upstream_valid_selection(self, root: Path) -> tuple[dict, Path, dict[str, str]]:
        seed = 17
        score_path = root / "validation_scores.npz"
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        logits = np.asarray([-1.0, 1.0, -0.5, 0.5], dtype=np.float64)
        head = {"weight": [0.0, 0.0], "bias": 0.0}
        warm_start_sha256 = json_sha256(head)
        candidates = []
        score_arrays: dict[str, np.ndarray] = {
            "row_id": np.arange(labels.size, dtype=np.int64),
            "y": labels,
            "nuisance": np.asarray([-1.0, 1.0, -0.5, 0.5], dtype=np.float64),
            "fold_assignment": np.asarray([0, 1, 0, 1], dtype=np.int16),
        }
        gate = {
            "schema_version": 1,
            "weight": [0.0, 0.0],
            "bias": 0.0,
            "class_value": 0,
            "objective": "fisher",
            "constraints": {
                "target_mass": 0.1,
                "minimum_ess_fraction": 0.8,
                "max_weight_multiplier": 2.0,
                "projection_iterations": 64,
            },
        }

        reference_crossfit = {
            "folds": [
                {"heldout": {"predictive": {"inverse_fisher": 3.0}}},
                {"heldout": {"predictive": {"inverse_fisher": 3.0}}},
            ]
        }
        frozen_reference = {
            "reference_key": FROZEN_ERM_REFERENCE_KEY,
            "role": "frozen_pre_continuation_erm_reference",
            "certificate_score_role": "diagnostic comparability only; excluded from selection",
            "warm_start_sha256": warm_start_sha256,
            "head": head,
            "final_gate": gate,
            "task_mean": 0.0,
            "task_std": 1.0,
            "validation_metrics": task_metrics(logits, labels, 1.0),
            "crossfit": reference_crossfit,
            "certificate_scores": crossfit_certificate_scores(
                reference_crossfit, confidence=0.95, comparisons=2
            ),
        }
        score_arrays[f"{FROZEN_ERM_REFERENCE_KEY}__logits"] = logits
        score_arrays[f"{FROZEN_ERM_REFERENCE_KEY}__task"] = logits
        score_arrays[f"{FROZEN_ERM_REFERENCE_KEY}__nll"] = np.logaddexp(
            0.0, -(2.0 * labels - 1.0) * logits
        )
        for index, (coefficient, inverse_fisher) in enumerate(((0.0, 2.0), (0.03, 1.0))):
            crossfit = {
                "folds": [
                    {"heldout": {"predictive": {"inverse_fisher": inverse_fisher}}},
                    {"heldout": {"predictive": {"inverse_fisher": inverse_fisher}}},
                ]
            }
            scores = crossfit_certificate_scores(crossfit, confidence=0.95, comparisons=2)
            candidate = {
                "candidate_index": index,
                "lambda_fg": coefficient,
                "role": "matched_erm_continuation" if coefficient == 0.0 else "fisher_repair",
                "warm_start_sha256": warm_start_sha256,
                "head": head,
                "final_gate": gate,
                "task_mean": 0.0,
                "task_std": 1.0,
                "validation_metrics": task_metrics(logits, labels, 1.0),
                "crossfit": crossfit,
                "certificate_scores": scores,
                "training": {
                    "config": {
                        "mode": "erm" if coefficient == 0.0 else "fisher_repair",
                        "head": {"steps": 10, "learning_rate": 0.01},
                        "repair": (
                            None
                            if coefficient == 0.0
                            else {
                                "name": "fisher_repair",
                                "lambda_loss": 0.0,
                                "lambda_fisher": coefficient,
                            }
                        ),
                    },
                    "final": {"optimization_steps_run": 10.0},
                    "head_training_seed": 123,
                    "warm_start_sha256": warm_start_sha256,
                },
            }
            candidates.append(candidate)
            prefix = lambda_key(coefficient)
            score_arrays[f"{prefix}__logits"] = logits
            score_arrays[f"{prefix}__task"] = logits
            score_arrays[f"{prefix}__nll"] = np.logaddexp(
                0.0, -(2.0 * labels - 1.0) * logits
            )
        atomic_savez_compressed(score_path, **score_arrays)
        choice = risk_certificate_choice(candidates)
        eligible = [float(row["lambda_fg"]) for row in choice["eligible"]]
        selection = {
            "schema_version": SCHEMA_VERSION,
            "stage": "metadata_blind_lambda_selection",
            "seed": seed,
            "metadata_accessed": False,
            "test_data_accessed": False,
            "implementation_manifest": implementation_manifest(),
            "input_hashes": {
                "train_blind": "1" * 64,
                "val_blind": "2" * 64,
                "preparation_manifest": "3" * 64,
            },
            "protocol": {
                "lambdas": [0.0, 0.03],
                "fixed_lambda": 0.03,
                "confidence": 0.95,
                "folds": 2,
                "fisher_metric": {"shrinkage": 0.1, "ridge": 0.001, "epsilon": 1e-8},
            },
            "normalization": {
                "feature_mean": [[0.0, 0.0]],
                "feature_std": [[1.0, 1.0]],
                "nuisance_mean": 0.0,
                "nuisance_std": 1.0,
                "base_temperature": 1.0,
            },
            "base_model": {
                "head": head,
                "head_sha256": warm_start_sha256,
                "task_mean": 0.0,
                "task_std": 1.0,
            },
            "matched_randomness": {
                "train_partition_seed": derive_seed(
                    seed, "group-free-lambda", "train-partition"
                ),
                "candidate_head_training_seed": derive_seed(
                    seed, "group-free-lambda", "fisher-repair-head"
                ),
                "candidate_crossfit_gate_seed": derive_seed(
                    seed, "group-free-lambda", "candidate-crossfit-gates"
                ),
                "validation_crossfit_partition_seed": derive_seed(
                    seed, "group-free-lambda", "validation-crossfit-partition"
                ),
                "candidate_final_gate_seed": derive_seed(
                    seed, "group-free-lambda", "candidate-final-gates"
                ),
            },
            FROZEN_ERM_REFERENCE_KEY: frozen_reference,
            "candidates": candidates,
            "selection": {
                "risk_best_lambda": float(choice["risk_minimizer"]["lambda_fg"]),
                "validation_risk_minimizer_lambda": float(choice["risk_minimizer"]["lambda_fg"]),
                "risk_ceiling": float(choice["risk_ceiling"]),
                "risk_eligible_lambdas": eligible,
                "eligible_lambdas": eligible,
                "selected_lambda": float(choice["selected"]["lambda_fg"]),
                "selected_candidate_index": int(choice["selected"]["candidate_index"]),
            },
            "validation_score_artifact_sha256": file_sha256(score_path),
        }
        selection["frozen_selection_sha256"] = json_artifact_sha256(
            selection, "frozen_selection_sha256"
        )
        return selection, score_path, selection["input_hashes"]

    def test_certifier_accepts_exact_upstream_selection_schema(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, score_path, inputs = self._write_upstream_valid_selection(Path(directory))
            candidates, arrays = _validate_selection(
                selection,
                seed=17,
                current_inputs=inputs,
                validation_score_path=score_path,
            )
            self.assertEqual(len(candidates), 2)
            self.assertEqual(set(arrays), {
                "row_id", "y", "nuisance", "fold_assignment",
                "frozen_erm_reference__logits",
                "frozen_erm_reference__task",
                "frozen_erm_reference__nll",
                "lambda_0__logits", "lambda_0__task", "lambda_0__nll",
                "lambda_0p03__logits", "lambda_0p03__task", "lambda_0p03__nll",
            })

    def test_certifier_rejects_certificate_score_inconsistent_with_folds(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, score_path, inputs = self._write_upstream_valid_selection(Path(directory))
            selection["candidates"][1]["certificate_scores"][
                "simultaneous_certificate_lower_score"
            ] = 99.0
            selection["frozen_selection_sha256"] = json_artifact_sha256(
                selection, "frozen_selection_sha256"
            )
            with self.assertRaisesRegex(RuntimeError, "certificate scores are inconsistent"):
                _validate_selection(
                    selection,
                    seed=17,
                    current_inputs=inputs,
                    validation_score_path=score_path,
                )

    def test_certifier_rejects_unmatched_lambda_zero_training(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, score_path, inputs = self._write_upstream_valid_selection(Path(directory))
            selection["candidates"][0]["training"]["config"]["mode"] = "exact_no_op"
            selection["candidates"][0]["training"]["final"]["optimization_steps_run"] = 0.0
            selection["frozen_selection_sha256"] = json_artifact_sha256(
                selection, "frozen_selection_sha256"
            )
            with self.assertRaisesRegex(RuntimeError, "wrong training mode|no continuation"):
                _validate_selection(
                    selection,
                    seed=17,
                    current_inputs=inputs,
                    validation_score_path=score_path,
                )

    def test_certifier_recomputes_logits_from_prepared_validation_features(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, score_path, inputs = self._write_upstream_valid_selection(Path(directory))
            validation_split = SimpleNamespace(
                row_id=np.arange(4, dtype=np.int64),
                y=np.asarray([0, 1, 0, 1], dtype=np.int64),
                h=np.ones((4, 2), dtype=np.float32),
                b_vl=np.asarray([-1.0, 1.0, -0.5, 0.5], dtype=np.float64),
            )
            with self.assertRaisesRegex(RuntimeError, "logits do not match its frozen head"):
                _validate_selection(
                    selection,
                    seed=17,
                    current_inputs=inputs,
                    validation_score_path=score_path,
                    validation_split=validation_split,
                )

    def test_frozen_reference_cannot_masquerade_as_lambda_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, score_path, inputs = self._write_upstream_valid_selection(Path(directory))
            selection[FROZEN_ERM_REFERENCE_KEY]["lambda_fg"] = 0.0
            selection["frozen_selection_sha256"] = json_artifact_sha256(
                selection, "frozen_selection_sha256"
            )
            with self.assertRaisesRegex(RuntimeError, "must not masquerade"):
                _validate_selection(
                    selection,
                    seed=17,
                    current_inputs=inputs,
                    validation_score_path=score_path,
                )

    def test_frozen_reference_score_is_excluded_from_selection(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            selection, score_path, inputs = self._write_upstream_valid_selection(Path(directory))
            reference = selection[FROZEN_ERM_REFERENCE_KEY]
            reference["crossfit"] = {
                "folds": [
                    {"heldout": {"predictive": {"inverse_fisher": 0.01}}},
                    {"heldout": {"predictive": {"inverse_fisher": 0.01}}},
                ]
            }
            reference["certificate_scores"] = crossfit_certificate_scores(
                reference["crossfit"], confidence=0.95, comparisons=2
            )
            selection["frozen_selection_sha256"] = json_artifact_sha256(
                selection, "frozen_selection_sha256"
            )
            candidates, _arrays = _validate_selection(
                selection,
                seed=17,
                current_inputs=inputs,
                validation_score_path=score_path,
            )
            self.assertEqual(selection["selection"]["selected_lambda"], 0.03)
            self.assertEqual(len(candidates), 2)

    def test_v1_selection_cannot_resume_under_v2(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            selection, score_path, inputs = self._write_upstream_valid_selection(root)
            selection["schema_version"] = 1
            selection["frozen_selection_sha256"] = json_artifact_sha256(
                selection, "frozen_selection_sha256"
            )
            selection_path = atomic_write_json(root / "selection.json", selection)
            with self.assertRaisesRegex(RuntimeError, "unsupported schema"):
                _resume_if_valid(
                    selection_path,
                    score_path,
                    seed=17,
                    protocol_sha256=str(selection.get("protocol_sha256", "missing")),
                    input_hashes=inputs,
                    lambdas=[0.0, 0.03],
                )

    def _write_valid_chain(
        self,
        root: Path,
        *,
        seed: int = 17,
        train_digest: str = "1" * 64,
    ) -> tuple[Path, dict]:
        seed_dir = root / "runs" / f"seed_{seed}"
        seed_dir.mkdir(parents=True)
        validation_scores = seed_dir / "validation_scores.npz"
        blind_scores = seed_dir / "blind_test_scores.npz"
        source = implementation_manifest()
        head = {"weight": [0.0, 0.0], "bias": 0.0}
        warm_start_sha256 = json_sha256(head)
        gate = {
            "schema_version": 1,
            "weight": [0.0, 0.0],
            "bias": 0.0,
            "class_value": 0,
            "objective": "fisher",
            "constraints": {
                "target_mass": 0.5,
                "minimum_ess_fraction": 0.8,
                "max_weight_multiplier": 2.0,
                "projection_iterations": 64,
            },
        }

        validation_metrics = {
            "nll": 0.7,
            "accuracy": 0.5,
            "balanced_accuracy": 0.5,
            "per_class_accuracy": {"0": 0.5, "1": 0.5},
        }
        candidate_specs = [
            {
                "candidate_index": index,
                "lambda_fg": coefficient,
                "role": "matched_erm_continuation" if coefficient == 0.0 else "fisher_repair",
                "warm_start_sha256": warm_start_sha256,
                "final_gate": gate,
                "validation_metrics": validation_metrics,
                "certificate_scores": {"simultaneous_certificate_lower_score": 0.2},
            }
            for index, coefficient in enumerate((0.0, 0.03))
        ]
        frozen_selection_reference = {
            "reference_key": FROZEN_ERM_REFERENCE_KEY,
            "role": "frozen_pre_continuation_erm_reference",
            "warm_start_sha256": warm_start_sha256,
            "head": head,
            "final_gate": gate,
            "validation_metrics": validation_metrics,
        }
        labels = np.asarray([0, 1, 0, 1], dtype=np.int64)
        logits = np.asarray([-1.0, 1.0, -0.5, 0.5], dtype=np.float64)
        nuisance = np.asarray([-1.0, 1.0, -0.5, 0.5], dtype=np.float64)
        validation_arrays: dict[str, np.ndarray] = {
            "row_id": np.arange(labels.size, dtype=np.int64),
            "y": labels,
            "nuisance": nuisance,
            "fold_assignment": np.asarray([0, 0, 1, 1], dtype=np.int16),
        }
        blind_arrays: dict[str, np.ndarray] = {
            "row_id": np.arange(labels.size, dtype=np.int64),
            "y": labels,
            "nuisance": nuisance,
        }
        q = np.asarray([0.5, 0.0, 0.5, 0.0], dtype=np.float64)
        membership = np.asarray([0.5, np.nan, 0.5, np.nan], dtype=np.float64)
        for prefix in (
            FROZEN_ERM_REFERENCE_KEY,
            lambda_key(0.0),
            lambda_key(0.03),
        ):
            validation_arrays[f"{prefix}__logits"] = logits
            validation_arrays[f"{prefix}__task"] = logits
            validation_arrays[f"{prefix}__nll"] = np.logaddexp(
                0.0, -(2.0 * labels - 1.0) * logits
            )
            blind_arrays[f"{prefix}__logits"] = logits
            blind_arrays[f"{prefix}__task"] = logits
            blind_arrays[f"{prefix}__q"] = q
            blind_arrays[f"{prefix}__membership"] = membership
        np.savez_compressed(validation_scores, **validation_arrays)
        np.savez_compressed(blind_scores, **blind_arrays)
        selection = {
            "schema_version": SCHEMA_VERSION,
            "stage": "metadata_blind_lambda_selection",
            "seed": seed,
            "metadata_accessed": False,
            "test_data_accessed": False,
            "implementation_manifest": source,
            "input_hashes": {
                "train_blind": train_digest,
                "val_blind": "2" * 64,
                "preparation_manifest": "3" * 64,
            },
            "protocol": {"lambdas": [0.0, 0.03], "fixed_lambda": 0.03, "confidence": 0.95},
            "base_model": {"head": head, "head_sha256": warm_start_sha256},
            FROZEN_ERM_REFERENCE_KEY: frozen_selection_reference,
            "selection": {
                "validation_risk_minimizer_lambda": 0.0,
                "selected_lambda": 0.0,
                "risk_eligible_lambdas": [0.0, 0.03],
                "eligible_lambdas": [0.0, 0.03],
            },
            "candidates": candidate_specs,
            "validation_score_artifact": str(validation_scores),
            "validation_score_artifact_sha256": file_sha256(validation_scores),
        }
        selection["frozen_selection_sha256"] = json_artifact_sha256(selection, "frozen_selection_sha256")
        selection_path = atomic_write_json(seed_dir / "selection.json", selection)
        selection_cohort = {
            "seeds": [seed],
            "artifacts": [
                {
                    "seed": seed,
                    "selection_artifact_sha256": file_sha256(selection_path),
                    "frozen_selection_sha256": selection["frozen_selection_sha256"],
                    "validation_score_artifact_sha256": file_sha256(validation_scores),
                }
            ],
        }
        selection_cohort["sha256"] = canonical_protocol_hash(selection_cohort)

        task = {"accuracy": 0.5, "balanced_accuracy": 0.5, "nll": 0.7}
        hidden = {"certificate": 0.2, "predictive": {"inverse_fisher": 5.0}}
        blind_candidates = [
            {
                **candidate,
                "source": "lambda_grid",
                "task_metrics": task,
                "hidden_certificate": hidden,
            }
            for candidate in candidate_specs
        ]
        blind_reference = {
            "source": "frozen_reference",
            "task_metrics": task,
            "hidden_certificate": hidden,
        }
        certification_protocol = {
            "bootstrap_replicates": 2000,
            "bootstrap_confidence": 0.95,
            "cohort_seeds": [seed],
            "complete_default_cohort_frozen_before_test": False,
            "selection_cohort": selection_cohort,
            "selection_cohort_sha256": selection_cohort["sha256"],
            "smoke": True,
        }
        blind = {
            "schema_version": SCHEMA_VERSION,
            "stage": "blind_heldout_test_certification",
            "seed": seed,
            "metadata_accessed": False,
            "test_data_accessed_after_freeze": True,
            "implementation_manifest": source,
            "input_hashes": {
                "train_blind": train_digest,
                "val_blind": "2" * 64,
                "preparation_manifest": "3" * 64,
                "test_blind": "4" * 64,
            },
            "test_blind_sha256": "4" * 64,
            "selection_artifact_sha256": file_sha256(selection_path),
            "frozen_selection_sha256": selection["frozen_selection_sha256"],
            "validation_score_artifact_sha256": file_sha256(validation_scores),
            "certification_protocol": certification_protocol,
            "certification_protocol_sha256": canonical_protocol_hash(
                {"seed": seed, "certification_protocol": certification_protocol}
            ),
            "score_artifact": str(blind_scores),
            "score_artifact_sha256": file_sha256(blind_scores),
            FROZEN_ERM_REFERENCE_KEY: blind_reference,
            "candidates": blind_candidates,
        }
        blind["blind_test_sha256"] = json_artifact_sha256(blind, "blind_test_sha256")
        blind_path = atomic_write_json(seed_dir / "blind_test.json", blind)
        blind_cohort = {
            "seeds": [seed],
            "selection_cohort_sha256": selection_cohort["sha256"],
            "artifacts": [
                {
                    "seed": seed,
                    "blind_test_artifact_sha256": file_sha256(blind_path),
                    "blind_test_sha256": blind["blind_test_sha256"],
                    "blind_test_score_artifact_sha256": file_sha256(blind_scores),
                }
            ],
        }
        blind_cohort["sha256"] = canonical_protocol_hash(blind_cohort)

        group = {"worst_group_accuracy": 0.5}
        def policy(key: str, coefficient: float, *, group_labels: bool) -> dict:
            return {
                "key": key,
                "lambda_fg": coefficient,
                "source": "lambda_grid",
                "group_labels_used_for_selection": group_labels,
                "validation_group_metrics": group,
                "test_group_metrics": group,
                "blind_test_candidate": {"task_metrics": task, "hidden_certificate": hidden},
            }
        frozen_posthoc_reference = {
            "source": "frozen_reference",
            "validation_group_metrics": group,
            "test_group_metrics": group,
            "blind_test_candidate": {"task_metrics": task, "hidden_certificate": hidden},
        }
        frozen_policy = {
            "key": "frozen_erm_reference",
            "lambda_fg": None,
            "source": "frozen_reference",
            "group_labels_used_for_selection": False,
            "validation_group_metrics": group,
            "test_group_metrics": group,
            "blind_test_candidate": {"task_metrics": task, "hidden_certificate": hidden},
        }
        posthoc_protocol = {
            "selection_artifact_sha256": file_sha256(selection_path),
            "frozen_selection_sha256": selection["frozen_selection_sha256"],
            "validation_score_artifact_sha256": file_sha256(validation_scores),
            "blind_test_artifact_sha256": file_sha256(blind_path),
            "blind_test_sha256": blind["blind_test_sha256"],
            "blind_test_score_artifact_sha256": file_sha256(blind_scores),
            "sealed_metadata_sha256": "5" * 64,
            "cohort_seeds": [seed],
            "complete_default_cohort_frozen_before_metadata": False,
            "blind_cohort": blind_cohort,
            "blind_cohort_sha256": blind_cohort["sha256"],
            "smoke": True,
        }
        posthoc = {
            "schema_version": SCHEMA_VERSION,
            "stage": "posthoc_group_label_audit",
            "seed": seed,
            "metadata_accessed_after_blind_freeze": True,
            "implementation_manifest": source,
            "selection_artifact_sha256": file_sha256(selection_path),
            "frozen_selection_sha256": selection["frozen_selection_sha256"],
            "blind_test_artifact_sha256": file_sha256(blind_path),
            "blind_test_sha256": blind["blind_test_sha256"],
            "primary_selection_unchanged": True,
            "posthoc_protocol": posthoc_protocol,
            "posthoc_protocol_sha256": canonical_protocol_hash(
                {"seed": seed, "posthoc_protocol": posthoc_protocol}
            ),
            FROZEN_ERM_REFERENCE_KEY: frozen_posthoc_reference,
            "candidates": [
                {
                    "candidate_index": 0,
                    "lambda_fg": 0.0,
                    "source": "lambda_grid",
                    "validation_group_metrics": group,
                    "test_group_metrics": group,
                    "blind_test_candidate": {"task_metrics": task, "hidden_certificate": hidden},
                },
                {
                    "candidate_index": 1,
                    "lambda_fg": 0.03,
                    "source": "lambda_grid",
                    "validation_group_metrics": group,
                    "test_group_metrics": group,
                    "blind_test_candidate": {"task_metrics": task, "hidden_certificate": hidden},
                },
            ],
            "policies": {
                "frozen_erm_reference": frozen_policy,
                "matched_erm_lambda_0": policy(
                    "matched_erm_lambda_0", 0.0, group_labels=False
                ),
                "fixed_0.03": policy("fixed_0.03", 0.03, group_labels=False),
                "validation_risk_minimizer": policy(
                    "validation_risk_minimizer", 0.0, group_labels=False
                ),
                "group_free": policy("group_free", 0.0, group_labels=False),
                "group_supervised_val_comparator": policy(
                    "group_supervised_val_comparator", 0.0, group_labels=True
                ),
            },
        }
        posthoc["posthoc_sha256"] = json_artifact_sha256(posthoc, "posthoc_sha256")
        atomic_write_json(seed_dir / "posthoc.json", posthoc)
        return seed_dir, selection

    def test_loader_accepts_complete_current_chain(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_chain(root)
            loaded = _load_seed(root, 17)
            self.assertEqual(loaded["seed"], 17)
            self.assertEqual(
                loaded["selection_reference"]["role"],
                "frozen_pre_continuation_erm_reference",
            )
            self.assertEqual(
                loaded["selection_candidates"][lambda_key(0.0)]["role"],
                "matched_erm_continuation",
            )
            self.assertEqual(
                loaded["policies"]["frozen_erm_reference"]["source"],
                "frozen_reference",
            )
            self.assertEqual(
                loaded["policies"]["matched_erm_lambda_0"]["source"],
                "lambda_grid",
            )

    def test_loader_rejects_tampered_selection_link(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_dir, _selection = self._write_valid_chain(root)
            blind_path = seed_dir / "blind_test.json"
            blind = copy.deepcopy(json.loads(blind_path.read_text(encoding="utf-8")))
            blind["selection_artifact_sha256"] = "0" * 64
            blind["blind_test_sha256"] = json_artifact_sha256(blind, "blind_test_sha256")
            atomic_write_json(blind_path, blind)
            with self.assertRaisesRegex(RuntimeError, "blind/selection file hash mismatch"):
                _load_seed(root, 17)

    def test_loader_rejects_wrong_group_supervised_tie_break(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed_dir, _selection = self._write_valid_chain(root)
            posthoc_path = seed_dir / "posthoc.json"
            posthoc = json.loads(posthoc_path.read_text(encoding="utf-8"))
            posthoc["policies"]["group_supervised_val_comparator"]["lambda_fg"] = 0.03
            posthoc["posthoc_sha256"] = json_artifact_sha256(posthoc, "posthoc_sha256")
            atomic_write_json(posthoc_path, posthoc)
            with self.assertRaisesRegex(RuntimeError, "tie-break rule"):
                _load_seed(root, 17)

    def test_cross_seed_provenance_rejects_mixed_prepared_bundles(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_chain(root, seed=17, train_digest="1" * 64)
            self._write_valid_chain(root, seed=18, train_digest="6" * 64)
            contexts = [_load_seed(root, 17), _load_seed(root, 18)]
            with self.assertRaisesRegex(RuntimeError, "cross-seed provenance"):
                _validate_cross_seed_provenance(contexts)

    def test_smoke_renderer_marks_every_human_output_nonreportable(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._write_valid_chain(root)
            with contextlib.redirect_stdout(io.StringIO()):
                with mock.patch(
                    "sys.argv",
                    ["aggregate.py", "--out-dir", str(root), "--seeds", "17", "--smoke"],
                ):
                    aggregate_main()
            expected = (
                "aggregate.json",
                "main_table.csv",
                "frontier.csv",
                "waterbirds_group_free_lambda_table.tex",
                "report.md",
                "rebuttal_lambda_fg.txt",
            )
            for name in expected:
                self.assertTrue((root / name).is_file(), name)
            self.assertIn("NON-REPORTABLE", (root / "report.md").read_text(encoding="utf-8"))
            self.assertIn(
                "NON-REPORTABLE", (root / "rebuttal_lambda_fg.txt").read_text(encoding="utf-8")
            )
            report_text = (root / "report.md").read_text(encoding="utf-8")
            rebuttal_text = (root / "rebuttal_lambda_fg.txt").read_text(encoding="utf-8")
            self.assertIn("Matched ERM continuation", report_text)
            self.assertNotIn("No-op (lambda=0)", report_text)
            self.assertNotIn("selected the exact no-op", rebuttal_text)
            self.assertIn(
                "NON-REPORTABLE",
                (root / "waterbirds_group_free_lambda_table.tex").read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
