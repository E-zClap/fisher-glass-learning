"""Fit ERM and discover cross-fitted hidden Waterbirds slices without metadata.

Only ``train_blind.npz`` and ``val_blind.npz`` are opened.  The official test
archive and sealed Waterbirds metadata are intentionally outside this stage.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from compat import BooleanOptionalAction
from blind_protocol import (
    crossfit_selector,
    discover_final,
    make_gate_config,
    make_model_coordinates,
    normalize_nuisance,
    overall_metrics,
    random_nuisance_axis,
)
from common import (
    WATERBIRDS_FEATURE_MODEL_ID,
    array_sha256,
    atomic_write_json,
    configure_torch_determinism,
    derive_seed,
    file_sha256,
    fit_temperature,
    json_artifact_sha256,
    json_sha256,
    load_blind_npz,
    require_json_artifact_sha256,
    scientific_implementation_manifest,
    stratified_partition_indices,
)
from heads import (
    HeadConfig,
    apply_feature_normalizer,
    fit_feature_normalizer,
    fit_scalar_normalizer,
    resolve_device,
    train_head,
)


SCHEMA_VERSION = 1


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _validate_inputs(data_dir: Path) -> tuple[Any, Any, dict[str, Any]]:
    manifest_path = data_dir / "preparation_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"missing preparation manifest: {manifest_path}")
    manifest = _read_json(manifest_path)
    configuration = {
        key: manifest[key]
        for key in ("blindness_contract", "dataset", "representation", "nuisance")
    }
    if manifest.get("configuration_sha256") != json_sha256(configuration):
        raise RuntimeError("preparation manifest configuration hash mismatch")
    representation = manifest.get("representation", {})
    if (
        representation.get("model") != WATERBIRDS_FEATURE_MODEL_ID
        or representation.get("cache_attests_model_identity") is not True
    ):
        raise RuntimeError(
            "prepared features lack the required attested MobileNet extractor identity; "
            "run corrected preparation in the fresh v2 output directory"
        )
    train_path = data_dir / "train_blind.npz"
    val_path = data_dir / "val_blind.npz"
    for name, path in (("train_blind", train_path), ("val_blind", val_path)):
        expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
        if expected is None or file_sha256(path) != expected:
            raise RuntimeError(f"prepared input hash mismatch for {path}")
    train = load_blind_npz(train_path)
    val = load_blind_npz(val_path)
    if train.feature_dim != val.feature_dim:
        raise ValueError("train and validation feature dimensions differ")
    if np.intersect1d(train.row_id, val.row_id).size:
        raise ValueError("train and validation row ids overlap")
    return train, val, manifest


def _protocol_payload(args: argparse.Namespace, preparation_manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "implementation": scientific_implementation_manifest(Path(__file__).resolve().parent),
        "preparation_configuration_sha256": preparation_manifest["configuration_sha256"],
        "blind_input_sha256": {
            name: preparation_manifest["artifacts"][name]["sha256"]
            for name in ("train_blind", "val_blind")
        },
        "calibration_fraction": args.calibration_fraction,
        "head": {
            "steps": args.head_steps,
            "learning_rate": args.head_lr,
            "weight_decay": args.head_weight_decay,
        },
        "crossfit_folds": args.folds,
        "crossfit_partition": "one shared stratified validation-fold assignment per experiment seed",
        "gate": {
            "steps": args.gate_steps,
            "learning_rate": args.gate_lr,
            "restarts": args.gate_restarts,
            "weight_decay": args.gate_weight_decay,
            "mass_penalty_strength": args.gate_mass_penalty,
        },
        "slice_mass": args.slice_mass,
        "mass_sensitivity": sorted(set(args.mass_sensitivity)),
        "shrinkage": args.shrinkage,
        "shrinkage_sensitivity": sorted(set(args.shrinkage_sensitivity)),
        "minimum_ess_fraction": args.ess_fraction,
        "maximum_weight_multiplier": args.cap_multiplier,
        "ridge": args.ridge,
        "epsilon": args.epsilon,
        "primary_nuisance": "fixed CLIP water-background minus land-background text direction",
        "negative_control": "one frozen random axis in standardized representation space per seed",
        "selectors": ["predictive_fisher", "high_loss"],
        "metadata_used": False,
        "deterministic_torch_algorithms": bool(args.deterministic),
    }


def _gate_config(args: argparse.Namespace, *, objective: str, mass: float, eta: float, seed: int):
    return make_gate_config(
        objective=objective,
        slice_mass=mass,
        minimum_ess_fraction=args.ess_fraction,
        max_weight_multiplier=args.cap_multiplier,
        restarts=args.gate_restarts,
        steps=args.gate_steps,
        learning_rate=args.gate_lr,
        shrinkage=eta,
        ridge=args.ridge,
        epsilon=args.epsilon,
        seed=seed,
        weight_decay=args.gate_weight_decay,
        mass_penalty_strength=args.gate_mass_penalty,
    )


def _run_seed(
    args: argparse.Namespace,
    train,
    val,
    preparation_manifest: dict[str, Any],
    protocol: dict[str, Any],
    seed: int,
    device: str,
) -> dict[str, Any]:
    seed_dir = Path(args.out_dir) / "runs" / f"seed_{seed}"
    output_path = seed_dir / "discovery.json"
    protocol_hash = json_sha256({"protocol": protocol, "seed": seed})
    if args.resume and output_path.is_file():
        existing = _read_json(output_path)
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError(f"refusing to resume seed {seed}: discovery protocol changed")
        if existing.get("seed") != seed or existing.get("stage") != "metadata_blind_discovery":
            raise RuntimeError(f"refusing to resume malformed discovery artifact for seed {seed}")
        if existing.get("metadata_accessed") is not False or existing.get("test_data_accessed") is not False:
            raise RuntimeError(f"refusing to resume non-blind discovery artifact for seed {seed}")
        require_json_artifact_sha256(existing, "frozen_blind_sha256", label=f"seed {seed} discovery")
        current_hashes = {
            "train_blind": file_sha256(Path(args.data_dir) / "train_blind.npz"),
            "val_blind": file_sha256(Path(args.data_dir) / "val_blind.npz"),
            "preparation_manifest": file_sha256(Path(args.data_dir) / "preparation_manifest.json"),
        }
        if existing.get("input_hashes") != current_hashes:
            raise RuntimeError(f"refusing to resume seed {seed}: prepared blind inputs changed")
        print(f"seed {seed}: discovery already complete", flush=True)
        return existing

    partition = stratified_partition_indices(
        train.y,
        {"fit": 1.0 - args.calibration_fraction, "calibration": args.calibration_fraction},
        seed=derive_seed(seed, "fit-calibration"),
    )
    fit_idx = partition["fit"]
    calibration_idx = partition["calibration"]

    feature_mean, feature_std = fit_feature_normalizer(train.h[fit_idx])
    train_h = apply_feature_normalizer(train.h, feature_mean, feature_std)
    val_h = apply_feature_normalizer(val.h, feature_mean, feature_std)
    (train_b, val_b), (nuisance_mean, nuisance_std) = normalize_nuisance(train.b_vl[fit_idx], val.b_vl)
    # normalize_nuisance returns fit first; reconstruct all train rows with the
    # same fit-only constants rather than fitting on calibration rows.
    train_b = ((np.asarray(train.b_vl, dtype=np.float64) - nuisance_mean) / nuisance_std).astype(np.float64)

    head_config = HeadConfig(
        steps=args.head_steps,
        learning_rate=args.head_lr,
        weight_decay=args.head_weight_decay,
        log_every=max(1, args.head_steps // 12),
    )
    trained = train_head(
        train_h[fit_idx],
        train.y[fit_idx],
        seed=derive_seed(seed, "erm-head"),
        device=device,
        config=head_config,
    )
    calibration_logits = trained.head.logits(train_h[calibration_idx])
    temperature = fit_temperature(calibration_logits, train.y[calibration_idx])
    task_mean, task_std = fit_scalar_normalizer(trained.head.logits(train_h[fit_idx]))

    fit_bundle = make_model_coordinates(
        trained.head,
        train_h[fit_idx],
        train.y[fit_idx],
        train_b[fit_idx],
        task_mean=task_mean,
        task_std=task_std,
        temperature=temperature,
    )
    calibration_bundle = make_model_coordinates(
        trained.head,
        train_h[calibration_idx],
        train.y[calibration_idx],
        train_b[calibration_idx],
        task_mean=task_mean,
        task_std=task_std,
        temperature=temperature,
    )
    val_bundle = make_model_coordinates(
        trained.head,
        val_h,
        val.y,
        val_b,
        task_mean=task_mean,
        task_std=task_std,
        temperature=temperature,
    )

    random_axis = random_nuisance_axis(train_h.shape[1], derive_seed(seed, "random-nuisance-axis"))
    random_fit_raw = train_h[fit_idx] @ random_axis
    random_train_raw = train_h @ random_axis
    random_val_raw = val_h @ random_axis
    random_mean, random_std = fit_scalar_normalizer(random_fit_raw)
    random_train = (random_train_raw - random_mean) / random_std
    random_val = (random_val_raw - random_mean) / random_std
    random_val_bundle = replace(val_bundle, nuisance=random_val.astype(np.float64))

    main_seed = derive_seed(seed, "primary-gate")
    fisher_config = _gate_config(
        args,
        objective="fisher",
        mass=args.slice_mass,
        eta=args.shrinkage,
        seed=main_seed,
    )
    loss_config = _gate_config(
        args,
        objective="high_loss",
        mass=args.slice_mass,
        eta=args.shrinkage,
        seed=derive_seed(seed, "loss-gate"),
    )
    random_config = _gate_config(
        args,
        objective="fisher",
        mass=args.slice_mass,
        eta=args.shrinkage,
        seed=derive_seed(seed, "random-nuisance-gate"),
    )
    cv_partition_seed = derive_seed(seed, "validation-crossfit-partition")

    primary_fisher_cv = crossfit_selector(
        val_bundle,
        config=fisher_config,
        folds=args.folds,
        reference_features=calibration_bundle.features,
        reference_labels=calibration_bundle.labels,
        device=device,
        seed=derive_seed(seed, "fisher-crossfit"),
        partition_seed=cv_partition_seed,
    )
    primary_loss_cv = crossfit_selector(
        val_bundle,
        config=loss_config,
        folds=args.folds,
        reference_features=calibration_bundle.features,
        reference_labels=calibration_bundle.labels,
        device=device,
        seed=derive_seed(seed, "loss-crossfit"),
        partition_seed=cv_partition_seed,
    )
    random_nuisance_cv = crossfit_selector(
        random_val_bundle,
        config=random_config,
        folds=args.folds,
        reference_features=calibration_bundle.features,
        reference_labels=calibration_bundle.labels,
        device=device,
        seed=derive_seed(seed, "random-nuisance-crossfit"),
        partition_seed=cv_partition_seed,
    )

    mass_sensitivity: dict[str, Any] = {}
    for mass in sorted(set(args.mass_sensitivity)):
        fisher_mass = _gate_config(
            args,
            objective="fisher",
            mass=mass,
            eta=args.shrinkage,
            seed=derive_seed(seed, "mass", mass, "fisher"),
        )
        loss_mass = _gate_config(
            args,
            objective="high_loss",
            mass=mass,
            eta=args.shrinkage,
            seed=derive_seed(seed, "mass", mass, "loss"),
        )
        mass_sensitivity[f"{mass:.6g}"] = {
            "fisher": primary_fisher_cv if np.isclose(mass, args.slice_mass) else crossfit_selector(
                val_bundle,
                config=fisher_mass,
                folds=args.folds,
                reference_features=calibration_bundle.features,
                reference_labels=calibration_bundle.labels,
                device=device,
                seed=derive_seed(seed, "mass-crossfit", mass, "fisher"),
                partition_seed=cv_partition_seed,
            ),
            "high_loss": primary_loss_cv if np.isclose(mass, args.slice_mass) else crossfit_selector(
                val_bundle,
                config=loss_mass,
                folds=args.folds,
                reference_features=calibration_bundle.features,
                reference_labels=calibration_bundle.labels,
                device=device,
                seed=derive_seed(seed, "mass-crossfit", mass, "loss"),
                partition_seed=cv_partition_seed,
            ),
        }

    shrinkage_sensitivity: dict[str, Any] = {}
    for eta in sorted(set(args.shrinkage_sensitivity)):
        eta_config = _gate_config(
            args,
            objective="fisher",
            mass=args.slice_mass,
            eta=eta,
            seed=derive_seed(seed, "shrinkage", eta),
        )
        shrinkage_sensitivity[f"{eta:.6g}"] = (
            primary_fisher_cv
            if np.isclose(eta, args.shrinkage)
            else crossfit_selector(
                val_bundle,
                config=eta_config,
                folds=args.folds,
                reference_features=calibration_bundle.features,
                reference_labels=calibration_bundle.labels,
                device=device,
                seed=derive_seed(seed, "shrinkage-crossfit", eta),
                partition_seed=cv_partition_seed,
            )
        )

    fisher_gate, fisher_final = discover_final(
        val_bundle,
        config=fisher_config,
        device=device,
        seed=derive_seed(seed, "fisher-final"),
    )
    loss_gate, loss_final = discover_final(
        val_bundle,
        config=loss_config,
        device=device,
        seed=derive_seed(seed, "loss-final"),
    )
    random_gate, random_final = discover_final(
        random_val_bundle,
        config=random_config,
        device=device,
        seed=derive_seed(seed, "random-nuisance-final"),
    )

    artifact = {
        "schema_version": SCHEMA_VERSION,
        "stage": "metadata_blind_discovery",
        "seed": seed,
        "protocol_sha256": protocol_hash,
        "protocol": protocol,
        "metadata_accessed": False,
        "test_data_accessed": False,
        "input_hashes": {
            "train_blind": file_sha256(Path(args.data_dir) / "train_blind.npz"),
            "val_blind": file_sha256(Path(args.data_dir) / "val_blind.npz"),
            "preparation_manifest": file_sha256(Path(args.data_dir) / "preparation_manifest.json"),
        },
        "partition": {
            "fit_positions": fit_idx,
            "calibration_positions": calibration_idx,
            "fit_row_id_sha256": array_sha256(train.row_id[fit_idx]),
            "calibration_row_id_sha256": array_sha256(train.row_id[calibration_idx]),
            "fit_label_counts": {str(c): int(np.sum(train.y[fit_idx] == c)) for c in (0, 1)},
            "calibration_label_counts": {str(c): int(np.sum(train.y[calibration_idx] == c)) for c in (0, 1)},
        },
        "normalization": {
            "feature_mean": feature_mean,
            "feature_std": feature_std,
            "vl_nuisance_mean": nuisance_mean,
            "vl_nuisance_std": nuisance_std,
            "task_mean": task_mean,
            "task_std": task_std,
            "random_axis": random_axis,
            "random_nuisance_mean": random_mean,
            "random_nuisance_std": random_std,
        },
        "base_model": {
            "head": trained.head.to_dict(),
            "temperature": temperature,
            "training": {"config": trained.config, "history": trained.history, "final": trained.final},
            "fit_metrics": overall_metrics(fit_bundle),
            "calibration_metrics": overall_metrics(calibration_bundle),
            "validation_metrics": overall_metrics(val_bundle),
        },
        "crossfit": {
            "primary_fisher": primary_fisher_cv,
            "high_loss": primary_loss_cv,
            "random_nuisance": random_nuisance_cv,
            "mass_sensitivity": mass_sensitivity,
            "shrinkage_sensitivity": shrinkage_sensitivity,
        },
        "frozen_selectors": {
            "primary_fisher": {"gate": fisher_gate.to_dict(), "discovery": fisher_final, "nuisance": "vl_text"},
            "high_loss": {"gate": loss_gate.to_dict(), "discovery": loss_final, "nuisance": "vl_text"},
            "random_nuisance": {"gate": random_gate.to_dict(), "discovery": random_final, "nuisance": "random_axis"},
        },
        "freeze_statement": (
            "Classifier, calibration temperature, protocol, hyperparameters, cross-fitted diagnostics, and final "
            "transferable gates were frozen without opening test_blind.npz or sealed_metadata.npz."
        ),
    }
    artifact["frozen_blind_sha256"] = json_artifact_sha256(artifact, "frozen_blind_sha256")
    atomic_write_json(output_path, artifact)
    print(
        f"seed {seed}: froze Fisher class {fisher_gate.class_value}, loss class {loss_gate.class_value}; "
        f"artifact={output_path}",
        flush=True,
    )
    return artifact


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", default="build/results/waterbirds_hidden_slice_v2/prepared")
    parser.add_argument("--out-dir", default="build/results/waterbirds_hidden_slice_v2")
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--device", default="auto")
    parser.add_argument("--calibration-fraction", type=float, default=0.15)
    parser.add_argument("--head-steps", type=int, default=1200)
    parser.add_argument("--head-lr", type=float, default=1e-2)
    parser.add_argument("--head-weight-decay", type=float, default=1e-4)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--gate-steps", type=int, default=1000)
    parser.add_argument("--gate-lr", type=float, default=3e-2)
    parser.add_argument("--gate-restarts", type=int, default=10)
    parser.add_argument("--gate-weight-decay", type=float, default=1e-4)
    parser.add_argument("--gate-mass-penalty", type=float, default=10.0)
    parser.add_argument("--slice-mass", type=float, default=0.10)
    parser.add_argument("--mass-sensitivity", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--shrinkage", type=float, default=0.10)
    parser.add_argument("--shrinkage-sensitivity", type=float, nargs="+", default=[0.05, 0.10, 0.20])
    parser.add_argument("--ess-fraction", type=float, default=0.80)
    parser.add_argument("--cap-multiplier", type=float, default=2.0)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--deterministic", action=BooleanOptionalAction, default=True)
    parser.add_argument("--resume", action=BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_torch_determinism(args.deterministic)
    if not 0.0 < args.calibration_fraction < 0.5:
        raise ValueError("calibration_fraction must lie in (0, 0.5)")
    if len(set(args.seeds)) != len(args.seeds):
        raise ValueError("seeds must be unique")
    train, val, preparation_manifest = _validate_inputs(Path(args.data_dir))
    protocol = _protocol_payload(args, preparation_manifest)
    protocol_hash = json_sha256(protocol)
    protocol_path = Path(args.out_dir) / "blind_protocol.json"
    if protocol_path.is_file():
        existing = _read_json(protocol_path)
        if existing.get("protocol_sha256") != json_sha256(existing.get("protocol")):
            raise RuntimeError("existing blind protocol content hash mismatch")
        if existing.get("protocol_sha256") != protocol_hash:
            raise RuntimeError("refusing to overwrite an output directory with a different blind protocol")
    else:
        atomic_write_json(
            protocol_path,
            {
                "schema_version": SCHEMA_VERSION,
                "protocol_sha256": protocol_hash,
                "protocol": protocol,
                "metadata_accessed": False,
            },
        )
    device = str(resolve_device(args.device))
    for seed in args.seeds:
        _run_seed(args, train, val, preparation_manifest, protocol, seed, device)


if __name__ == "__main__":
    main()
