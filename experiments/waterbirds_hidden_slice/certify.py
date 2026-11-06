"""Apply frozen gates once to the untouched Waterbirds test split.

This stage computes held-out certificates and confidence diagnostics without
opening Waterbirds group/background metadata.  The resulting score archive is
the immutable input to the later post-hoc interpretation stage.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from compat import BooleanOptionalAction
from blind_protocol import make_model_coordinates, overall_metrics
from common import (
    atomic_savez_compressed,
    atomic_write_json,
    canonical_probability_weights,
    derive_seed,
    file_sha256,
    json_artifact_sha256,
    json_sha256,
    load_blind_npz,
    require_current_scientific_implementation,
    require_json_artifact_sha256,
)
from gates import FrozenLinearGate, admissible_weights
from heads import FrozenHead, apply_feature_normalizer
from metrics import FisherMetricConfig, bootstrap_fixed_gate_fisher, evaluate_class_slice


SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_all_models_frozen(
    data_dir: Path,
    out_dir: Path,
    seeds: list[int],
) -> tuple[dict[int, tuple[dict[str, Any], dict[str, Any], Path, Path]], str]:
    """Validate every blind model/gate before the test archive is opened."""
    manifest_path = data_dir / "preparation_manifest.json"
    manifest = _read_json(manifest_path)
    current_inputs = {
        "train_blind": file_sha256(data_dir / "train_blind.npz"),
        "val_blind": file_sha256(data_dir / "val_blind.npz"),
        "preparation_manifest": file_sha256(manifest_path),
    }
    contexts: dict[int, tuple[dict[str, Any], dict[str, Any], Path, Path]] = {}
    for seed in seeds:
        seed_dir = out_dir / "runs" / f"seed_{seed}"
        discovery_path = seed_dir / "discovery.json"
        repair_path = seed_dir / "repair.json"
        if not discovery_path.is_file() or not repair_path.is_file():
            raise FileNotFoundError(f"all discovery and repair stages must finish before test access: seed {seed}")
        discovery = _read_json(discovery_path)
        repair = _read_json(repair_path)
        if discovery.get("seed") != seed or repair.get("seed") != seed:
            raise RuntimeError(f"seed {seed}: frozen artifacts carry the wrong seed")
        if discovery.get("metadata_accessed") is not False or discovery.get("test_data_accessed") is not False:
            raise RuntimeError(f"seed {seed}: discovery violates the blind-stage contract")
        if repair.get("metadata_accessed") is not False or repair.get("test_data_accessed") is not False:
            raise RuntimeError(f"seed {seed}: repair violates the blind-stage contract")
        require_current_scientific_implementation(discovery, label=f"seed {seed} discovery")
        require_json_artifact_sha256(discovery, "frozen_blind_sha256", label=f"seed {seed} discovery")
        require_json_artifact_sha256(repair, "frozen_repair_sha256", label=f"seed {seed} repair")
        if discovery.get("input_hashes") != current_inputs or repair.get("input_hashes") != current_inputs:
            raise RuntimeError(f"seed {seed}: frozen artifacts do not match current prepared blind inputs")
        if repair.get("discovery_artifact_sha256") != file_sha256(discovery_path):
            raise RuntimeError(f"seed {seed}: repair/discovery file hash mismatch")
        if repair.get("discovery_frozen_blind_sha256") != discovery["frozen_blind_sha256"]:
            raise RuntimeError(f"seed {seed}: repair/discovery frozen hash mismatch")
        contexts[seed] = (discovery, repair, discovery_path, repair_path)
    # Only after every requested model and gate has passed the freeze audit may
    # even the opaque test-file hash be read.
    test_path = data_dir / "test_blind.npz"
    test_sha = file_sha256(test_path)
    expected_test_sha = manifest.get("artifacts", {}).get("test_blind", {}).get("sha256")
    if test_sha != expected_test_sha:
        raise RuntimeError("test_blind.npz does not match the preparation manifest")
    return contexts, test_sha


def _make_weights(membership: np.ndarray, gate: FrozenLinearGate) -> np.ndarray:
    q, _ = admissible_weights(torch.as_tensor(membership, dtype=torch.float64), gate.constraints)
    return canonical_probability_weights(q.detach().cpu().numpy(), name="certification gate weights")


def _evaluate_gate(
    gate: FrozenLinearGate,
    bundle,
    metric_config: FisherMetricConfig,
    *,
    bootstrap_replicates: int,
    bootstrap_confidence: float,
    bootstrap_seed: int,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    labels_t = torch.as_tensor(bundle.labels, dtype=torch.long)
    features_t = torch.as_tensor(bundle.features, dtype=torch.float32)
    mask = bundle.labels == gate.class_value
    membership = gate.membership(features_t[labels_t == gate.class_value]).detach().cpu().numpy().astype(np.float64)
    q = _make_weights(membership, gate)
    indices = np.flatnonzero(mask)
    metrics = evaluate_class_slice(
        task=bundle.task[indices],
        nuisance=bundle.nuisance[indices],
        logits=bundle.logits[indices],
        labels=bundle.labels[indices],
        q=q,
        class_id=gate.class_value,
        temperature=bundle.temperature,
        config=metric_config,
        hard_mass=gate.constraints.target_mass,
    )
    metrics["selector_objective"] = gate.objective
    realized_mass = float(membership.mean())
    metrics["gate_transfer"] = {
        "target_membership_mass": float(gate.constraints.target_mass),
        "heldout_mean_membership": realized_mass,
        "absolute_mass_error": abs(realized_mass - gate.constraints.target_mass),
        "membership_min": float(membership.min()),
        "membership_max": float(membership.max()),
    }
    metrics["bootstrap_predictive_fisher"] = bootstrap_fixed_gate_fisher(
        membership=membership,
        task=bundle.task[indices],
        nuisance=bundle.nuisance[indices],
        logits=bundle.logits[indices],
        labels=bundle.labels[indices],
        class_id=gate.class_value,
        make_weights=lambda values: _make_weights(values, gate),
        config=metric_config,
        temperature=bundle.temperature,
        hard_mass=gate.constraints.target_mass,
        replicates=bootstrap_replicates,
        confidence=bootstrap_confidence,
        seed=bootstrap_seed,
    )
    full_q = np.zeros(bundle.labels.size, dtype=np.float64)
    full_membership = np.full(bundle.labels.size, np.nan, dtype=np.float64)
    full_q[indices] = q
    full_membership[indices] = membership
    return metrics, {
        "indices": indices.astype(np.int64),
        "membership": membership,
        "q": q,
        "full_q": full_q,
        "full_membership": full_membership,
    }


def _compare_selectors(
    left: dict[str, np.ndarray],
    right: dict[str, np.ndarray],
    *,
    left_class: int,
    right_class: int,
    left_mass: float,
    right_mass: float,
) -> dict[str, Any]:
    """Test-set distinctness of two independently frozen slice rules."""
    left_membership = np.asarray(left["full_membership"], dtype=np.float64)
    right_membership = np.asarray(right["full_membership"], dtype=np.float64)
    left_positions = np.flatnonzero(np.isfinite(left_membership))
    right_positions = np.flatnonzero(np.isfinite(right_membership))
    left_k = max(1, int(np.ceil(left_mass * left_positions.size)))
    right_k = max(1, int(np.ceil(right_mass * right_positions.size)))
    left_top = set(
        left_positions[np.argpartition(left_membership[left_positions], -left_k)[-left_k:]].tolist()
    )
    right_top = set(
        right_positions[np.argpartition(right_membership[right_positions], -right_k)[-right_k:]].tolist()
    )
    union = left_top | right_top
    same_class = left_class == right_class
    correlation = None
    if same_class:
        shared = left_positions
        a = left_membership[shared]
        b = right_membership[shared]
        if a.size >= 2 and np.std(a) > 1e-12 and np.std(b) > 1e-12:
            correlation = float(np.corrcoef(a, b)[0, 1])
    return {
        "left": "hidden_fisher",
        "right": "high_loss",
        "same_selected_class": bool(same_class),
        "left_class": int(left_class),
        "right_class": int(right_class),
        "score_correlation_within_shared_class": correlation,
        "top_slice_jaccard": float(len(left_top & right_top) / max(len(union), 1)),
        "top_slice_intersection": int(len(left_top & right_top)),
        "left_top_size": int(len(left_top)),
        "right_top_size": int(len(right_top)),
    }


def _random_slice_baseline(
    bundle,
    *,
    class_id: int,
    mass: float,
    metric_config: FisherMetricConfig,
    draws: int,
    seed: int,
) -> dict[str, Any]:
    indices = np.flatnonzero(bundle.labels == class_id)
    n = int(indices.size)
    k = max(1, int(np.ceil(mass * n)))
    rng = np.random.default_rng(seed)
    rows = []
    for draw in range(draws):
        selected = rng.choice(n, size=k, replace=False)
        q = np.zeros(n, dtype=np.float64)
        q[selected] = 1.0 / float(k)
        rows.append(
            evaluate_class_slice(
                task=bundle.task[indices],
                nuisance=bundle.nuisance[indices],
                logits=bundle.logits[indices],
                labels=bundle.labels[indices],
                q=q,
                class_id=class_id,
                temperature=bundle.temperature,
                config=metric_config,
                hard_mass=mass,
            )
        )

    def summary(path: tuple[str, ...]) -> dict[str, float]:
        values = np.asarray([
            row[path[0]][path[1]] if len(path) == 2 else row[path[0]]
            for row in rows
        ], dtype=np.float64)
        return {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
            "p2_5": float(np.quantile(values, 0.025)),
            "p50": float(np.quantile(values, 0.5)),
            "p97_5": float(np.quantile(values, 0.975)),
        }

    return {
        "draws": draws,
        "class_id": int(class_id),
        "slice_mass": float(mass),
        "hard_slice_size": k,
        "predictive_fisher": summary(("predictive", "projected_fisher")),
        "inverse_fisher": summary(("predictive", "inverse_fisher")),
        "unweighted_fisher": summary(("unweighted", "projected_fisher")),
        "weighted_loss": summary(("weighted_loss",)),
        "weighted_accuracy": summary(("weighted_accuracy",)),
    }


def _arm_bundle(test, test_h, test_b, payload: dict[str, Any]):
    head = FrozenHead.from_dict(payload["head"])
    return make_model_coordinates(
        head,
        test_h,
        test.y,
        test_b,
        task_mean=float(payload["task_mean"]),
        task_std=float(payload["task_std"]),
        temperature=float(payload["temperature"]),
    )


def _run_seed(
    args: argparse.Namespace,
    test,
    seed: int,
    frozen: tuple[dict[str, Any], dict[str, Any], Path, Path],
    test_sha: str,
) -> dict[str, Any]:
    out_dir = Path(args.out_dir)
    seed_dir = out_dir / "runs" / f"seed_{seed}"
    discovery, repair, discovery_path, repair_path = frozen

    output_path = seed_dir / "blind_test_results.json"
    score_path = seed_dir / "blind_test_scores.npz"
    certification_protocol = {
        "discovery_frozen_blind_sha256": discovery["frozen_blind_sha256"],
        "repair_frozen_sha256": repair["frozen_repair_sha256"],
        "discovery_artifact_sha256": file_sha256(discovery_path),
        "repair_artifact_sha256": file_sha256(repair_path),
        "test_blind_sha256": test_sha,
        "repair_evaluations": [
            "original frozen Fisher gate",
            "freshly reinitialized Fisher gate",
        ],
        "bootstrap_replicates": args.bootstrap_replicates,
        "bootstrap_confidence": args.bootstrap_confidence,
        "random_slices": args.random_slices,
    }
    certification_sha = json_sha256({"seed": seed, "certification": certification_protocol})
    if args.resume and output_path.is_file():
        existing = _read_json(output_path)
        if existing.get("certification_protocol_sha256") != certification_sha:
            raise RuntimeError(f"refusing to overwrite mismatched blind test result for seed {seed}")
        if existing.get("seed") != seed or existing.get("stage") != "blind_heldout_test_certification":
            raise RuntimeError(f"malformed blind test result for seed {seed}")
        if existing.get("metadata_accessed") is not False:
            raise RuntimeError(f"blind test result for seed {seed} violates the metadata contract")
        require_json_artifact_sha256(existing, "blind_test_sha256", label=f"seed {seed} blind certification")
        if score_path.is_file() and file_sha256(score_path) == existing.get("score_artifact_sha256"):
            print(f"seed {seed}: blind test certification already complete", flush=True)
            return existing
        raise RuntimeError(f"seed {seed}: blind score artifact is missing or changed")

    normalization = discovery["normalization"]
    test_h = apply_feature_normalizer(
        test.h,
        np.asarray(normalization["feature_mean"], dtype=np.float32),
        np.asarray(normalization["feature_std"], dtype=np.float32),
    )
    test_b = (np.asarray(test.b_vl, dtype=np.float64) - normalization["vl_nuisance_mean"]) / normalization["vl_nuisance_std"]
    random_b = (test_h @ np.asarray(normalization["random_axis"], dtype=np.float32) - normalization["random_nuisance_mean"]) / normalization["random_nuisance_std"]
    protocol = discovery["protocol"]
    metric_config = FisherMetricConfig(protocol["shrinkage"], protocol["ridge"], protocol["epsilon"])

    base_payload = {
        "head": discovery["base_model"]["head"],
        "temperature": discovery["base_model"]["temperature"],
        "task_mean": normalization["task_mean"],
        "task_std": normalization["task_std"],
    }
    base_bundle = _arm_bundle(test, test_h, test_b, base_payload)
    base_random_bundle = _arm_bundle(test, test_h, random_b, base_payload)
    fisher_gate = FrozenLinearGate.from_dict(discovery["frozen_selectors"]["primary_fisher"]["gate"])
    loss_gate = FrozenLinearGate.from_dict(discovery["frozen_selectors"]["high_loss"]["gate"])
    random_nuisance_gate = FrozenLinearGate.from_dict(discovery["frozen_selectors"]["random_nuisance"]["gate"])

    base_fisher, base_fisher_arrays = _evaluate_gate(
        fisher_gate,
        base_bundle,
        metric_config,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=derive_seed(seed, "test-bootstrap", "base-fisher"),
    )
    base_loss, base_loss_arrays = _evaluate_gate(
        loss_gate,
        base_bundle,
        metric_config,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=derive_seed(seed, "test-bootstrap", "base-loss"),
    )
    random_nuisance, random_nuisance_arrays = _evaluate_gate(
        random_nuisance_gate,
        base_random_bundle,
        metric_config,
        bootstrap_replicates=args.bootstrap_replicates,
        bootstrap_confidence=args.bootstrap_confidence,
        bootstrap_seed=derive_seed(seed, "test-bootstrap", "random-nuisance"),
    )
    random_slices = _random_slice_baseline(
        base_bundle,
        class_id=fisher_gate.class_value,
        mass=fisher_gate.constraints.target_mass,
        metric_config=metric_config,
        draws=args.random_slices,
        seed=derive_seed(seed, "random-slice-baseline"),
    )

    arms: dict[str, Any] = {
        "erm": {
            "overall": overall_metrics(base_bundle),
            "fisher_selector": base_fisher,
            "high_loss_selector": base_loss,
            "random_nuisance_selector": random_nuisance,
            "random_slice_baseline": random_slices,
            "fisher_vs_high_loss": _compare_selectors(
                base_fisher_arrays,
                base_loss_arrays,
                left_class=fisher_gate.class_value,
                right_class=loss_gate.class_value,
                left_mass=fisher_gate.constraints.target_mass,
                right_mass=loss_gate.constraints.target_mass,
            ),
        }
    }
    score_arrays: dict[str, np.ndarray] = {
        "row_id": test.row_id,
        "y": test.y,
        "b_vl": test_b.astype(np.float64),
        "erm_logits": base_bundle.logits,
        "erm_task": base_bundle.task,
        "erm_fisher_q": base_fisher_arrays["full_q"],
        "erm_fisher_membership": base_fisher_arrays["full_membership"],
        "erm_loss_q": base_loss_arrays["full_q"],
        "erm_loss_membership": base_loss_arrays["full_membership"],
        "erm_random_nuisance_q": random_nuisance_arrays["full_q"],
        "erm_random_nuisance_membership": random_nuisance_arrays["full_membership"],
    }

    for arm, payload in repair["selected_arms"].items():
        bundle = _arm_bundle(test, test_h, test_b, payload)
        frozen_original, frozen_original_arrays = _evaluate_gate(
            fisher_gate,
            bundle,
            metric_config,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=derive_seed(seed, "test-bootstrap", arm, "original-gate"),
        )
        gate = FrozenLinearGate.from_dict(payload["fresh_adversary"]["gate"])
        hidden, hidden_arrays = _evaluate_gate(
            gate,
            bundle,
            metric_config,
            bootstrap_replicates=args.bootstrap_replicates,
            bootstrap_confidence=args.bootstrap_confidence,
            bootstrap_seed=derive_seed(seed, "test-bootstrap", arm),
        )
        arms[arm] = {
            "spec": payload["spec"],
            "accuracy_budget_satisfied": payload["selection"]["accuracy_budget_satisfied"],
            "overall": overall_metrics(bundle),
            "frozen_original_fisher_selector": frozen_original,
            "rediscovered_fisher_selector": hidden,
        }
        score_arrays[f"{arm}_logits"] = bundle.logits
        score_arrays[f"{arm}_task"] = bundle.task
        score_arrays[f"{arm}_original_fisher_q"] = frozen_original_arrays["full_q"]
        score_arrays[f"{arm}_original_fisher_membership"] = frozen_original_arrays["full_membership"]
        score_arrays[f"{arm}_fisher_q"] = hidden_arrays["full_q"]
        score_arrays[f"{arm}_fisher_membership"] = hidden_arrays["full_membership"]

    atomic_savez_compressed(score_path, **score_arrays)
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "blind_heldout_test_certification",
        "seed": seed,
        "metadata_accessed": False,
        "test_data_accessed_after_freeze": True,
        "certification_protocol": certification_protocol,
        "certification_protocol_sha256": certification_sha,
        "test_blind_sha256": test_sha,
        "discovery_artifact_sha256": file_sha256(discovery_path),
        "repair_artifact_sha256": file_sha256(repair_path),
        "score_artifact": str(score_path),
        "score_artifact_sha256": file_sha256(score_path),
        "arms": arms,
        "freeze_statement": (
            "All reported test gates and model parameters were frozen before test_blind.npz was opened. "
            "No group/background metadata was opened in this stage."
        ),
    }
    result["blind_test_sha256"] = json_artifact_sha256(result, "blind_test_sha256")
    atomic_write_json(output_path, result)
    print(f"seed {seed}: wrote blind held-out certification {output_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="build/results/waterbirds_hidden_slice_v2/prepared")
    parser.add_argument("--out-dir", default="build/results/waterbirds_hidden_slice_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--bootstrap-replicates", type=int, default=2000)
    parser.add_argument("--bootstrap-confidence", type=float, default=0.95)
    parser.add_argument("--random-slices", type=int, default=200)
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.bootstrap_replicates < 1 or args.random_slices < 1:
        raise ValueError("bootstrap_replicates and random_slices must be positive")
    if not 0.0 < args.bootstrap_confidence < 1.0:
        raise ValueError("bootstrap_confidence must lie in (0, 1)")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    data_dir = Path(args.data_dir)
    contexts, test_sha = _require_all_models_frozen(data_dir, Path(args.out_dir), args.seeds)
    test = load_blind_npz(data_dir / "test_blind.npz")
    for seed in args.seeds:
        _run_seed(args, test, seed, contexts[seed], test_sha)


if __name__ == "__main__":
    main()
