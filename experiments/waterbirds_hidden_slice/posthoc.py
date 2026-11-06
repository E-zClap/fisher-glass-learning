"""Reveal Waterbirds metadata after freezing blind results and add oracle rows.

The script first verifies that discovery, repair, and blind test certification
are complete for *every* requested seed.  Only then does it open
``sealed_metadata.npz``.  Metadata is used for interpretation, predefined-group
metrics, an explicitly labeled background-probe sensitivity analysis, and the
explicitly labeled oracle Group DRO comparator; it never changes a blind row.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from compat import BooleanOptionalAction
from blind_protocol import (
    crossfit_selector,
    discover_final,
    make_gate_config,
    make_model_coordinates,
    overall_metrics,
)
from certify import _evaluate_gate
from common import (
    atomic_savez_compressed,
    atomic_write_json,
    canonical_probability_weights,
    configure_torch_determinism,
    derive_seed,
    file_sha256,
    fit_temperature,
    json_artifact_sha256,
    json_sha256,
    load_blind_npz,
    require_current_scientific_implementation,
    require_json_artifact_sha256,
)
from gates import FrozenLinearGate
from heads import (
    FrozenHead,
    HeadConfig,
    apply_feature_normalizer,
    fit_scalar_normalizer,
    resolve_device,
    train_head,
)
from metrics import FisherMetricConfig, evaluate_class_slice


SCHEMA_VERSION = 1
GROUP_NAMES = {
    0: "landbird/land",
    1: "landbird/water",
    2: "waterbird/land",
    3: "waterbird/water",
}
SEALED_KEYS = {"row_id", "background", "group", "img_filename", "split"}


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _require_blind_freeze(
    out_dir: Path,
    data_dir: Path,
    preparation_path: Path,
    seeds: list[int],
) -> dict[int, tuple[Path, Path, Path]]:
    """Validate the complete blind hash chain before sealed metadata access."""
    current_inputs = {
        "train_blind": file_sha256(data_dir / "train_blind.npz"),
        "val_blind": file_sha256(data_dir / "val_blind.npz"),
        "preparation_manifest": file_sha256(preparation_path),
    }
    current_test_sha = file_sha256(data_dir / "test_blind.npz")
    paths = {}
    for seed in seeds:
        seed_dir = out_dir / "runs" / f"seed_{seed}"
        discovery = seed_dir / "discovery.json"
        repair = seed_dir / "repair.json"
        blind_test = seed_dir / "blind_test_results.json"
        if not all(path.is_file() for path in (discovery, repair, blind_test)):
            raise FileNotFoundError(f"all blind stages must finish before metadata reveal for seed {seed}")
        discovery_payload = _read_json(discovery)
        repair_payload = _read_json(repair)
        blind = _read_json(blind_test)
        if any(payload.get("seed") != seed for payload in (discovery_payload, repair_payload, blind)):
            raise RuntimeError(f"seed {seed}: blind artifact seed mismatch")
        if discovery_payload.get("metadata_accessed") is not False or discovery_payload.get("test_data_accessed") is not False:
            raise RuntimeError(f"seed {seed}: discovery violates the blind contract")
        if repair_payload.get("metadata_accessed") is not False or repair_payload.get("test_data_accessed") is not False:
            raise RuntimeError(f"seed {seed}: repair violates the blind contract")
        if blind.get("metadata_accessed") is not False or blind.get("test_data_accessed_after_freeze") is not True:
            raise RuntimeError(f"seed {seed}: blind certification violates the metadata contract")
        require_current_scientific_implementation(
            discovery_payload,
            label=f"seed {seed} discovery",
        )
        require_json_artifact_sha256(
            discovery_payload,
            "frozen_blind_sha256",
            label=f"seed {seed} discovery",
        )
        require_json_artifact_sha256(
            repair_payload,
            "frozen_repair_sha256",
            label=f"seed {seed} repair",
        )
        require_json_artifact_sha256(blind, "blind_test_sha256", label=f"seed {seed} blind certification")
        if discovery_payload.get("input_hashes") != current_inputs or repair_payload.get("input_hashes") != current_inputs:
            raise RuntimeError(f"seed {seed}: blind artifacts refer to stale prepared inputs")
        if repair_payload.get("discovery_artifact_sha256") != file_sha256(discovery):
            raise RuntimeError(f"seed {seed}: repair/discovery hash mismatch")
        if blind.get("discovery_artifact_sha256") != file_sha256(discovery):
            raise RuntimeError(f"seed {seed}: certification/discovery hash mismatch")
        if blind.get("repair_artifact_sha256") != file_sha256(repair):
            raise RuntimeError(f"seed {seed}: certification/repair hash mismatch")
        if blind.get("test_blind_sha256") != current_test_sha:
            raise RuntimeError(f"seed {seed}: certification/test hash mismatch")
        score_path = seed_dir / "blind_test_scores.npz"
        if not score_path.is_file() or file_sha256(score_path) != blind["score_artifact_sha256"]:
            raise RuntimeError(f"seed {seed} blind score artifact hash mismatch")
        paths[seed] = (discovery, repair, blind_test)
    return paths


def _load_sealed(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if file_sha256(path) != expected_sha256:
        raise RuntimeError("sealed metadata hash does not match the preparation manifest")
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != SEALED_KEYS:
            raise ValueError(f"unexpected sealed metadata schema: {archive.files}")
        result = {key: np.asarray(archive[key]) for key in archive.files}
    row_id = result["row_id"].astype(np.int64)
    if np.unique(row_id).size != row_id.size:
        raise ValueError("sealed metadata row ids are not unique")
    return result


def _join_metadata(split, sealed: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    order = {int(row): index for index, row in enumerate(sealed["row_id"])}
    try:
        positions = np.asarray([order[int(row)] for row in split.row_id], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"blind row id missing from sealed metadata: {exc}") from exc
    return {key: values[positions] for key, values in sealed.items() if key != "row_id"}


def _group_metrics(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    predictions = (np.asarray(logits) >= 0).astype(np.int64)
    rows = []
    for gid in range(4):
        mask = groups == gid
        if not mask.any():
            raise ValueError(f"metadata group {gid} is absent")
        rows.append(
            {
                "group_id": gid,
                "group": GROUP_NAMES[gid],
                "n": int(mask.sum()),
                "accuracy": float((predictions[mask] == labels[mask]).mean()),
            }
        )
    return {
        "worst_group_accuracy": min(row["accuracy"] for row in rows),
        "worst_group": min(rows, key=lambda row: row["accuracy"]),
        "groups": rows,
    }


def _composition(
    *,
    q: np.ndarray,
    membership: np.ndarray,
    groups: np.ndarray,
    labels: np.ndarray,
    class_id: int,
    mass: float,
) -> dict[str, Any]:
    q = canonical_probability_weights(q, name="post-hoc soft slice weights")
    membership = np.asarray(membership, dtype=np.float64)
    if q.shape != groups.shape or membership.shape != groups.shape:
        raise ValueError("composition arrays do not align")
    class_mask = labels == class_id
    finite = np.isfinite(membership)
    if not np.array_equal(finite, class_mask):
        raise ValueError("membership support does not match the frozen class")
    if np.any(q[~class_mask] > 0.0):
        raise ValueError("soft slice weights have support outside the frozen class")
    class_positions = np.flatnonzero(class_mask)
    k = max(1, int(np.ceil(mass * class_positions.size)))
    local_top = np.argpartition(membership[class_positions], -k)[-k:]
    hard_positions = class_positions[local_top]
    baseline = {gid: float(np.mean(groups[class_mask] == gid)) for gid in range(4)}
    weighted = {gid: float(q[groups == gid].sum()) for gid in range(4)}
    hard = {gid: float(np.mean(groups[hard_positions] == gid)) for gid in range(4)}
    enrichment = {
        gid: (weighted[gid] / baseline[gid] if baseline[gid] > 0 else None)
        for gid in range(4)
    }
    dominant_gid = max(weighted, key=weighted.get)
    return {
        "class_id": int(class_id),
        "slice_mass": float(mass),
        "hard_slice_size": k,
        "weighted_group_composition": {GROUP_NAMES[gid]: weighted[gid] for gid in range(4)},
        "hard_top_group_composition": {GROUP_NAMES[gid]: hard[gid] for gid in range(4)},
        "class_baseline_group_composition": {GROUP_NAMES[gid]: baseline[gid] for gid in range(4)},
        "weighted_enrichment_over_class": {GROUP_NAMES[gid]: enrichment[gid] for gid in range(4)},
        "dominant_metadata_group": GROUP_NAMES[dominant_gid],
        "dominant_group_purity": weighted[dominant_gid],
        "metadata_role": "post-hoc interpretation only",
    }


def _predefined_group_fisher(
    *,
    task: np.ndarray,
    nuisance: np.ndarray,
    logits: np.ndarray,
    labels: np.ndarray,
    groups: np.ndarray,
    temperature: float,
    config: FisherMetricConfig,
) -> dict[str, Any]:
    rows = []
    for gid in range(4):
        group_mask = groups == gid
        if not group_mask.any() or np.unique(labels[group_mask]).size != 1:
            raise ValueError(f"metadata group {gid} is empty or not class-conditional")
        class_id = int(labels[group_mask][0])
        class_mask = labels == class_id
        local_group = groups[class_mask] == gid
        q = np.zeros(int(class_mask.sum()), dtype=np.float64)
        q[local_group] = 1.0 / float(local_group.sum())
        metrics = evaluate_class_slice(
            task=task[class_mask],
            nuisance=nuisance[class_mask],
            logits=logits[class_mask],
            labels=labels[class_mask],
            q=q,
            class_id=class_id,
            temperature=temperature,
            config=config,
            hard_mass=float(local_group.mean()),
        )
        rows.append(
            {
                "group_id": gid,
                "group": GROUP_NAMES[gid],
                "class_reference_n": int(class_mask.sum()),
                "group_n": int(local_group.sum()),
                "shrinkage_reference": "full task class, matching hidden-slice certification",
                **metrics,
            }
        )
    return {
        "rows": rows,
        "lowest_fisher_group": min(rows, key=lambda row: row["predictive"]["projected_fisher"]),
        "highest_loss_group": max(rows, key=lambda row: row["weighted_loss"]),
    }


def _scores(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _run_seed(
    args: argparse.Namespace,
    seed: int,
    device: str,
    train,
    val,
    test,
    train_meta: dict[str, np.ndarray],
    val_meta: dict[str, np.ndarray],
    test_meta: dict[str, np.ndarray],
    paths: tuple[Path, Path, Path],
) -> dict[str, Any]:
    discovery_path, repair_path, blind_test_path = paths
    discovery = _read_json(discovery_path)
    repair = _read_json(repair_path)
    blind_test = _read_json(blind_test_path)
    seed_dir = Path(args.out_dir) / "runs" / f"seed_{seed}"
    output_path = seed_dir / "posthoc.json"
    oracle_score_path = seed_dir / "posthoc_oracle_scores.npz"
    posthoc_protocol = {
        "blind_test_sha256": blind_test["blind_test_sha256"],
        "discovery_artifact_sha256": file_sha256(discovery_path),
        "repair_artifact_sha256": file_sha256(repair_path),
        "blind_test_artifact_sha256": file_sha256(blind_test_path),
        "oracle_head_steps": args.head_steps,
        "oracle_head_lr": args.head_lr,
        "oracle_head_weight_decay": args.head_weight_decay,
        "oracle_group_dro_hyperparameters_fixed": True,
        "oracle_nuisance_probe": "balanced logistic probe fit on metadata background in blind fit partition",
        "oracle_score_schema": [
            "row_id",
            "img_filename",
            "group",
            "split",
            "oracle_group_dro_logits",
            "oracle_group_dro_task",
            "oracle_group_dro_q",
            "oracle_group_dro_membership",
            "oracle_probe_q",
            "oracle_probe_membership",
        ],
        "deterministic_torch_algorithms": bool(args.deterministic),
    }
    posthoc_sha = json_sha256({"seed": seed, "posthoc": posthoc_protocol})
    if args.resume and output_path.is_file():
        existing = _read_json(output_path)
        if existing.get("posthoc_protocol_sha256") != posthoc_sha:
            raise RuntimeError(f"refusing to resume seed {seed}: post-hoc protocol changed")
        if existing.get("seed") != seed or existing.get("stage") != "posthoc_metadata_interpretation_and_oracles":
            raise RuntimeError(f"refusing to resume malformed post-hoc artifact for seed {seed}")
        require_json_artifact_sha256(existing, "posthoc_sha256", label=f"seed {seed} post-hoc")
        if not oracle_score_path.is_file() or file_sha256(oracle_score_path) != existing.get("oracle_score_artifact_sha256"):
            raise RuntimeError(f"seed {seed}: post-hoc oracle score artifact is missing or changed")
        print(f"seed {seed}: post-hoc interpretation already complete", flush=True)
        return existing

    scores = _scores(seed_dir / "blind_test_scores.npz")
    if not np.array_equal(scores["row_id"], test.row_id) or not np.array_equal(scores["y"], test.y):
        raise RuntimeError("blind test score rows do not match the test artifact")
    groups_test = test_meta["group"].astype(np.int64)
    groups_train = train_meta["group"].astype(np.int64)
    background_train = train_meta["background"].astype(np.int64)
    background_val = val_meta["background"].astype(np.int64)
    background_test = test_meta["background"].astype(np.int64)
    protocol = discovery["protocol"]
    metric_config = FisherMetricConfig(protocol["shrinkage"], protocol["ridge"], protocol["epsilon"])

    selectors = {
        "erm_fisher": ("erm_fisher_q", "erm_fisher_membership", blind_test["arms"]["erm"]["fisher_selector"]),
        "erm_high_loss": ("erm_loss_q", "erm_loss_membership", blind_test["arms"]["erm"]["high_loss_selector"]),
        "erm_random_nuisance": (
            "erm_random_nuisance_q",
            "erm_random_nuisance_membership",
            blind_test["arms"]["erm"]["random_nuisance_selector"],
        ),
    }
    for arm, arm_result in blind_test["arms"].items():
        if arm == "erm":
            continue
        selectors[arm] = (
            f"{arm}_fisher_q",
            f"{arm}_fisher_membership",
            arm_result["rediscovered_fisher_selector"],
        )
    compositions = {}
    for name, (q_key, membership_key, selector_result) in selectors.items():
        compositions[name] = _composition(
            q=scores[q_key],
            membership=scores[membership_key],
            groups=groups_test,
            labels=test.y,
            class_id=int(selector_result["class_id"]),
            mass=float(selector_result["hard_top_mass"]),
        )

    model_group_metrics = {"erm": _group_metrics(scores["erm_logits"], test.y, groups_test)}
    for arm in repair["selected_arms"]:
        model_group_metrics[arm] = _group_metrics(scores[f"{arm}_logits"], test.y, groups_test)
    predefined = _predefined_group_fisher(
        task=scores["erm_task"],
        nuisance=scores["b_vl"],
        logits=scores["erm_logits"],
        labels=test.y,
        groups=groups_test,
        temperature=float(discovery["base_model"]["temperature"]),
        config=metric_config,
    )

    normalization = discovery["normalization"]
    feature_mean = np.asarray(normalization["feature_mean"], dtype=np.float32)
    feature_std = np.asarray(normalization["feature_std"], dtype=np.float32)
    train_h = apply_feature_normalizer(train.h, feature_mean, feature_std)
    val_h = apply_feature_normalizer(val.h, feature_mean, feature_std)
    test_h = apply_feature_normalizer(test.h, feature_mean, feature_std)
    fit_idx = np.asarray(discovery["partition"]["fit_positions"], dtype=np.int64)
    calibration_idx = np.asarray(discovery["partition"]["calibration_positions"], dtype=np.int64)
    base_head = FrozenHead.from_dict(discovery["base_model"]["head"])

    head_config = HeadConfig(
        steps=args.head_steps,
        learning_rate=args.head_lr,
        weight_decay=args.head_weight_decay,
        log_every=max(1, args.head_steps // 12),
    )
    gdro = train_head(
        train_h[fit_idx],
        train.y[fit_idx],
        seed=derive_seed(seed, "oracle-group-dro-head"),
        device=device,
        config=head_config,
        groups=groups_train[fit_idx],
    )
    gdro_temperature = fit_temperature(gdro.head.logits(train_h[calibration_idx]), train.y[calibration_idx])
    gdro_task_mean, gdro_task_std = fit_scalar_normalizer(gdro.head.logits(train_h[fit_idx]))
    vl_mean = float(normalization["vl_nuisance_mean"])
    vl_std = float(normalization["vl_nuisance_std"])
    train_b_vl = (np.asarray(train.b_vl, dtype=np.float64) - vl_mean) / vl_std
    val_b_vl = (np.asarray(val.b_vl, dtype=np.float64) - vl_mean) / vl_std
    test_b_vl = (np.asarray(test.b_vl, dtype=np.float64) - vl_mean) / vl_std
    gdro_val_bundle = make_model_coordinates(
        gdro.head, val_h, val.y, val_b_vl,
        task_mean=gdro_task_mean, task_std=gdro_task_std, temperature=gdro_temperature,
    )
    gdro_test_bundle = make_model_coordinates(
        gdro.head, test_h, test.y, test_b_vl,
        task_mean=gdro_task_mean, task_std=gdro_task_std, temperature=gdro_temperature,
    )
    gate_config = make_gate_config(
        objective="fisher",
        slice_mass=protocol["slice_mass"],
        minimum_ess_fraction=protocol["minimum_ess_fraction"],
        max_weight_multiplier=protocol["maximum_weight_multiplier"],
        restarts=protocol["gate"]["restarts"],
        steps=protocol["gate"]["steps"],
        learning_rate=protocol["gate"]["learning_rate"],
        shrinkage=protocol["shrinkage"],
        ridge=protocol["ridge"],
        epsilon=protocol["epsilon"],
        seed=derive_seed(seed, "oracle-group-dro-adversary"),
        weight_decay=protocol["gate"]["weight_decay"],
        mass_penalty_strength=protocol["gate"]["mass_penalty_strength"],
    )
    gdro_crossfit = crossfit_selector(
        gdro_val_bundle,
        config=gate_config,
        folds=protocol["crossfit_folds"],
        reference_features=train_h[calibration_idx],
        reference_labels=train.y[calibration_idx],
        device=device,
        seed=derive_seed(seed, "oracle-group-dro-crossfit"),
        partition_seed=derive_seed(seed, "validation-crossfit-partition"),
    )
    gdro_gate, gdro_discovery = discover_final(
        gdro_val_bundle,
        config=gate_config,
        device=device,
        seed=derive_seed(seed, "oracle-group-dro-final"),
    )
    bootstrap_replicates = int(blind_test["certification_protocol"]["bootstrap_replicates"])
    bootstrap_confidence = float(blind_test["certification_protocol"]["bootstrap_confidence"])
    gdro_hidden, gdro_arrays = _evaluate_gate(
        gdro_gate,
        gdro_test_bundle,
        metric_config,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=derive_seed(seed, "oracle-group-dro-bootstrap"),
    )
    gdro_group_metrics = _group_metrics(gdro_test_bundle.logits, test.y, groups_test)

    probe = LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=derive_seed(seed, "oracle-probe") % (2**32 - 1),
    )
    probe.fit(train_h[fit_idx], background_train[fit_idx])
    probe_fit_raw = probe.decision_function(train_h[fit_idx])
    probe_mean, probe_std = fit_scalar_normalizer(probe_fit_raw)
    probe_val = (probe.decision_function(val_h) - probe_mean) / probe_std
    probe_test = (probe.decision_function(test_h) - probe_mean) / probe_std
    oracle_val_bundle = make_model_coordinates(
        base_head, val_h, val.y, probe_val,
        task_mean=float(normalization["task_mean"]),
        task_std=float(normalization["task_std"]),
        temperature=float(discovery["base_model"]["temperature"]),
    )
    oracle_test_bundle = make_model_coordinates(
        base_head, test_h, test.y, probe_test,
        task_mean=float(normalization["task_mean"]),
        task_std=float(normalization["task_std"]),
        temperature=float(discovery["base_model"]["temperature"]),
    )
    oracle_probe_config = make_gate_config(
        objective="fisher",
        slice_mass=protocol["slice_mass"],
        minimum_ess_fraction=protocol["minimum_ess_fraction"],
        max_weight_multiplier=protocol["maximum_weight_multiplier"],
        restarts=protocol["gate"]["restarts"],
        steps=protocol["gate"]["steps"],
        learning_rate=protocol["gate"]["learning_rate"],
        shrinkage=protocol["shrinkage"],
        ridge=protocol["ridge"],
        epsilon=protocol["epsilon"],
        seed=derive_seed(seed, "oracle-probe-adversary"),
        weight_decay=protocol["gate"]["weight_decay"],
        mass_penalty_strength=protocol["gate"]["mass_penalty_strength"],
    )
    oracle_probe_crossfit = crossfit_selector(
        oracle_val_bundle,
        config=oracle_probe_config,
        folds=protocol["crossfit_folds"],
        reference_features=train_h[calibration_idx],
        reference_labels=train.y[calibration_idx],
        device=device,
        seed=derive_seed(seed, "oracle-probe-crossfit"),
        partition_seed=derive_seed(seed, "validation-crossfit-partition"),
    )
    oracle_probe_gate, oracle_probe_discovery = discover_final(
        oracle_val_bundle,
        config=oracle_probe_config,
        device=device,
        seed=derive_seed(seed, "oracle-probe-final"),
    )
    oracle_probe_hidden, oracle_probe_arrays = _evaluate_gate(
        oracle_probe_gate,
        oracle_test_bundle,
        metric_config,
        bootstrap_replicates=bootstrap_replicates,
        bootstrap_confidence=bootstrap_confidence,
        bootstrap_seed=derive_seed(seed, "oracle-probe-bootstrap"),
    )
    oracle_probe_composition = _composition(
        q=oracle_probe_arrays["full_q"],
        membership=oracle_probe_arrays["full_membership"],
        groups=groups_test,
        labels=test.y,
        class_id=oracle_probe_gate.class_value,
        mass=oracle_probe_gate.constraints.target_mass,
    )
    gdro_composition = _composition(
        q=gdro_arrays["full_q"],
        membership=gdro_arrays["full_membership"],
        groups=groups_test,
        labels=test.y,
        class_id=gdro_gate.class_value,
        mass=gdro_gate.constraints.target_mass,
    )

    atomic_savez_compressed(
        oracle_score_path,
        row_id=test.row_id,
        img_filename=test_meta["img_filename"],
        group=groups_test,
        split=test_meta["split"],
        oracle_group_dro_logits=gdro_test_bundle.logits,
        oracle_group_dro_task=gdro_test_bundle.task,
        oracle_group_dro_q=gdro_arrays["full_q"],
        oracle_group_dro_membership=gdro_arrays["full_membership"],
        oracle_probe_q=oracle_probe_arrays["full_q"],
        oracle_probe_membership=oracle_probe_arrays["full_membership"],
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "stage": "posthoc_metadata_interpretation_and_oracles",
        "seed": seed,
        "metadata_revealed_after_blind_freeze": True,
        "blind_test_sha256": blind_test["blind_test_sha256"],
        "posthoc_protocol": posthoc_protocol,
        "posthoc_protocol_sha256": posthoc_sha,
        "compositions": compositions,
        "model_group_metrics": model_group_metrics,
        "predefined_group_reference": predefined,
        "oracle_group_dro": {
            "group_labels_used": True,
            "hyperparameter_selection": "none; fixed predeclared optimization settings",
            "head": gdro.head.to_dict(),
            "temperature": gdro_temperature,
            "overall": overall_metrics(gdro_test_bundle),
            "group_metrics": gdro_group_metrics,
            "rediscovered_hidden_slice": gdro_hidden,
            "crossfit": gdro_crossfit,
            "full_validation_discovery": gdro_discovery,
            "composition": gdro_composition,
        },
        "oracle_nuisance_sensitivity": {
            "group_labels_used_for_nuisance_probe": True,
            "role": "oracle-nuisance sensitivity analysis; not the primary unknown-group result",
            "probe_fit_accuracy": float(probe.score(train_h[fit_idx], background_train[fit_idx])),
            "probe_validation_accuracy": float(((probe.decision_function(val_h) >= 0).astype(int) == background_val).mean()),
            "probe_test_accuracy_posthoc": float(((probe.decision_function(test_h) >= 0).astype(int) == background_test).mean()),
            "probe_coef": probe.coef_.astype(np.float32),
            "probe_intercept": probe.intercept_.astype(np.float32),
            "probe_score_mean": probe_mean,
            "probe_score_std": probe_std,
            "hidden_slice": oracle_probe_hidden,
            "crossfit": oracle_probe_crossfit,
            "full_validation_discovery": oracle_probe_discovery,
            "composition": oracle_probe_composition,
        },
        "oracle_score_artifact": str(oracle_score_path),
        "oracle_score_artifact_sha256": file_sha256(oracle_score_path),
        "scope_statement": (
            "The primary unknown object is the subpopulation conditional on a fixed audited nuisance coordinate. "
            "Waterbirds background/group labels appear only in this post-hoc artifact and in explicitly labeled oracle rows."
        ),
    }
    result["posthoc_sha256"] = json_artifact_sha256(result, "posthoc_sha256")
    atomic_write_json(output_path, result)
    print(f"seed {seed}: wrote post-hoc interpretations and oracle controls {output_path}", flush=True)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="build/results/waterbirds_hidden_slice_v2/prepared")
    parser.add_argument("--out-dir", default="build/results/waterbirds_hidden_slice_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--head-steps", type=int, default=1200)
    parser.add_argument("--head-lr", type=float, default=1e-2)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--deterministic", action=BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_torch_determinism(args.deterministic)
    out_dir = Path(args.out_dir)
    data_dir = Path(args.data_dir)
    preparation_path = data_dir / "preparation_manifest.json"
    preparation = _read_json(preparation_path)
    paths = _require_blind_freeze(out_dir, data_dir, preparation_path, args.seeds)
    sealed = _load_sealed(
        data_dir / "sealed_metadata.npz",
        preparation["artifacts"]["sealed_metadata"]["sha256"],
    )
    train = load_blind_npz(data_dir / "train_blind.npz")
    val = load_blind_npz(data_dir / "val_blind.npz")
    test = load_blind_npz(data_dir / "test_blind.npz")
    train_meta = _join_metadata(train, sealed)
    val_meta = _join_metadata(val, sealed)
    test_meta = _join_metadata(test, sealed)
    device = str(resolve_device(args.device))
    for seed in args.seeds:
        _run_seed(
            args,
            seed,
            device,
            train,
            val,
            test,
            train_meta,
            val_meta,
            test_meta,
            paths[seed],
        )


if __name__ == "__main__":
    main()
