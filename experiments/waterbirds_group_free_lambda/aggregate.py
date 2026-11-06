"""Validate, aggregate, and render the group-free lambda-selection audit.

This is a reporting-only stage.  It refuses incomplete or stale per-seed hash
chains and never fits a model or selector.  Group metrics have already been
added by the explicitly post-hoc stage after the blind artifacts were frozen.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import t as student_t

try:
    from .experiment_common import (
        DEFAULT_SEEDS,
        FROZEN_ERM_REFERENCE_KEY,
        SCHEMA_VERSION,
        atomic_write_json,
        canonical_protocol_hash,
        file_sha256,
        implementation_manifest,
        json_artifact_sha256,
        lambda_key,
        read_json,
        require_current_implementation,
        require_json_artifact_sha256,
    )
except ImportError:
    from experiment_common import (
        DEFAULT_SEEDS,
        FROZEN_ERM_REFERENCE_KEY,
        SCHEMA_VERSION,
        atomic_write_json,
        canonical_protocol_hash,
        file_sha256,
        implementation_manifest,
        json_artifact_sha256,
        lambda_key,
        read_json,
        require_current_implementation,
        require_json_artifact_sha256,
    )


SELECTION_STAGE = "metadata_blind_lambda_selection"
BLIND_STAGE = "blind_heldout_test_certification"
POSTHOC_STAGE = "posthoc_group_label_audit"
POLICY_ORDER = (
    "frozen_erm_reference",
    "matched_erm_lambda_0",
    "fixed_0.03",
    "validation_risk_minimizer",
    "group_free",
    "group_supervised_val_comparator",
)
GRID_POLICY_ORDER = tuple(
    key for key in POLICY_ORDER if key != "frozen_erm_reference"
)
POLICY_LABELS = {
    "frozen_erm_reference": "Frozen ERM reference",
    "matched_erm_lambda_0": "Matched ERM continuation (lambda=0)",
    "fixed_0.03": "Fixed lambda=0.03 (hidden-slice scale)",
    "validation_risk_minimizer": "Risk-only selector",
    "group_free": "Group-free selector",
    "group_supervised_val_comparator": "Group-supervised validation comparator",
}
POLICY_SUPERVISION = {
    "frozen_erm_reference": "predeclared frozen task-label ERM reference",
    "matched_erm_lambda_0": "predeclared matched task-label continuation",
    "fixed_0.03": "predeclared; task labels + fixed CLIP nuisance score",
    "validation_risk_minimizer": "task labels only",
    "group_free": "task labels + fixed CLIP water-vs-land nuisance score; no group annotations",
    "group_supervised_val_comparator": "validation group labels (post-hoc comparator)",
}
PAIRED_CONTRASTS = (
    ("group_free", "matched_erm_lambda_0", "primary_regularization_effect"),
    ("matched_erm_lambda_0", "frozen_erm_reference", "secondary_continuation_effect"),
    ("group_free", "frozen_erm_reference", "secondary_total_effect"),
)
METRICS = (
    "test_accuracy",
    "test_balanced_accuracy",
    "test_nll",
    "test_hidden_certificate",
    "test_inverse_fisher",
    "test_worst_group_accuracy",
    "validation_worst_group_accuracy",
)
BOUNDED_METRICS = {
    "test_accuracy": (0.0, 1.0),
    "test_balanced_accuracy": (0.0, 1.0),
    "test_worst_group_accuracy": (0.0, 1.0),
    "validation_worst_group_accuracy": (0.0, 1.0),
}


def _summary(
    values: Iterable[float],
    confidence: float = 0.95,
    *,
    bounds: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Two-sided Student-t summary across experiment seeds."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("summary values must be a non-empty finite vector")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    mean = float(array.mean())
    standard_deviation = float(array.std(ddof=1)) if array.size > 1 else 0.0
    standard_error = float(standard_deviation / np.sqrt(array.size))
    critical = (
        float(student_t.ppf(0.5 + confidence / 2.0, df=array.size - 1))
        if array.size > 1
        else 0.0
    )
    half_width = float(critical * standard_error)
    unbounded_low = mean - half_width
    unbounded_high = mean + half_width
    low, high = unbounded_low, unbounded_high
    if bounds is not None:
        lower, upper = bounds
        if not lower < upper:
            raise ValueError("bounds must be ordered")
        low = max(float(lower), low)
        high = min(float(upper), high)
    return {
        "n": int(array.size),
        "mean": mean,
        "standard_deviation": standard_deviation,
        "standard_error": standard_error,
        "confidence": float(confidence),
        "critical_value": critical,
        "half_width": half_width,
        "low": float(low),
        "high": float(high),
        "unbounded_low": float(unbounded_low),
        "unbounded_high": float(unbounded_high),
        "bounds": list(bounds) if bounds is not None else None,
        "values": array.tolist(),
        "interval": "two-sided Student-t interval across experiment seeds",
        "interval_scope": (
            "descriptive variation across training/partition seeds on the same dataset; not a population or "
            "dataset-sampling confidence interval"
        ),
    }


def _atomic_write_text(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def _require_stage(
    artifact: Mapping[str, Any],
    *,
    path: Path,
    seed: int,
    stage: str,
    digest_field: str,
) -> None:
    if artifact.get("stage") != stage or int(artifact.get("seed", -1)) != seed:
        raise RuntimeError(f"seed {seed}: malformed {path.name}")
    if int(artifact.get("schema_version", -1)) != SCHEMA_VERSION:
        raise RuntimeError(f"seed {seed}: unsupported schema in {path.name}")
    require_current_implementation(
        artifact.get("implementation_manifest"), label=f"seed {seed} {path.name}"
    )
    require_json_artifact_sha256(artifact, digest_field, label=f"seed {seed} {path.name}")


def _candidate_map(rows: Iterable[Mapping[str, Any]], *, label: str) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        coefficient = float(row["lambda_fg"])
        key = lambda_key(coefficient)
        if key in result:
            raise RuntimeError(f"{label} repeats lambda={coefficient:g}")
        result[key] = row
    if not result:
        raise RuntimeError(f"{label} contains no candidates")
    return result


def _frozen_reference(
    artifact: Mapping[str, Any], *, label: str, propagated: bool
) -> Mapping[str, Any]:
    """Return the schema-v2 frozen ERM root without treating it as lambda=0."""

    record = artifact.get(FROZEN_ERM_REFERENCE_KEY)
    if not isinstance(record, Mapping):
        raise RuntimeError(f"{label} lacks the frozen ERM reference root")
    if propagated:
        if record.get("source") != "frozen_reference":
            raise RuntimeError(f"{label} frozen ERM root has an invalid source")
    elif record.get("reference_key") != FROZEN_ERM_REFERENCE_KEY:
        raise RuntimeError(f"{label} frozen ERM root has an invalid reference key")
    if record.get("lambda_fg") is not None:
        raise RuntimeError(f"{label} frozen ERM reference must not carry a lambda value")
    return record


def _policy_map(posthoc: Mapping[str, Any], *, seed: int) -> dict[str, Mapping[str, Any]]:
    policies = posthoc.get("policies")
    if isinstance(policies, Mapping):
        result = {str(key): value for key, value in policies.items()}
    elif isinstance(policies, list):
        result = {str(row["key"]): row for row in policies}
    else:
        raise RuntimeError(f"seed {seed}: posthoc policies are absent")
    missing = set(POLICY_ORDER).difference(result)
    extra = set(result).difference(POLICY_ORDER)
    if missing or extra:
        raise RuntimeError(f"seed {seed}: policy mismatch; missing={sorted(missing)}, extra={sorted(extra)}")
    return result


def _resolve_recorded_path(value: Any, *, fallback: Path) -> Path:
    path = Path(str(value)) if value else fallback
    if path.is_file():
        return path
    if fallback.is_file():
        return fallback
    return path


def _validate_score_archive_schema(
    path: Path,
    *,
    prefixes: list[str],
    records: Mapping[str, Mapping[str, Any]],
    test: bool,
) -> None:
    """Require the exact aligned score schema before aggregating a bundle."""

    expected = {"row_id", "y", "nuisance"}
    if test:
        for prefix in prefixes:
            expected.update(
                {
                    f"{prefix}__logits",
                    f"{prefix}__task",
                    f"{prefix}__q",
                    f"{prefix}__membership",
                }
            )
    else:
        expected.add("fold_assignment")
        for prefix in prefixes:
            expected.update(
                {f"{prefix}__logits", f"{prefix}__task", f"{prefix}__nll"}
            )
    with np.load(path, allow_pickle=False) as archive:
        actual = set(archive.files)
        if actual != expected:
            raise RuntimeError(
                f"{path} has the wrong schema; missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}"
            )
        arrays = {name: np.asarray(archive[name]) for name in archive.files}
    n = arrays["row_id"].size
    if n == 0 or any(array.ndim != 1 or array.size != n for array in arrays.values()):
        raise RuntimeError(f"{path} must contain non-empty aligned one-dimensional arrays")
    row_id = np.asarray(arrays["row_id"])
    labels = np.asarray(arrays["y"])
    if row_id.dtype.kind not in "iu" or np.unique(row_id).size != n:
        raise RuntimeError(f"{path} row ids must be unique integers")
    if labels.dtype.kind not in "iu" or not np.all(np.isin(labels, (0, 1))):
        raise RuntimeError(f"{path} labels must be binary integers")
    if not np.all(np.isfinite(np.asarray(arrays["nuisance"], dtype=np.float64))):
        raise RuntimeError(f"{path} nuisance coordinate contains non-finite values")
    if not test:
        assignment = np.asarray(arrays["fold_assignment"])
        if assignment.dtype.kind not in "iu" or np.any(assignment < 0):
            raise RuntimeError(f"{path} fold assignment must contain nonnegative integers")
    for prefix in prefixes:
        for suffix in ("logits", "task"):
            if not np.all(
                np.isfinite(np.asarray(arrays[f"{prefix}__{suffix}"], dtype=np.float64))
            ):
                raise RuntimeError(f"{path} {prefix} {suffix} contains non-finite values")
        if not test:
            nll = np.asarray(arrays[f"{prefix}__nll"], dtype=np.float64)
            if np.any(nll < 0.0) or not np.all(np.isfinite(nll)):
                raise RuntimeError(f"{path} {prefix} NLL is invalid")
            continue
        record = records[prefix]
        gate = record.get("final_gate")
        if not isinstance(gate, Mapping) or int(gate.get("class_value", -1)) not in (0, 1):
            raise RuntimeError(f"{path} {prefix} lacks a valid frozen gate")
        class_mask = labels.astype(np.int64) == int(gate["class_value"])
        membership = np.asarray(arrays[f"{prefix}__membership"], dtype=np.float64)
        q = np.asarray(arrays[f"{prefix}__q"], dtype=np.float64)
        if (
            not np.all(np.isfinite(membership[class_mask]))
            or not np.all(np.isnan(membership[~class_mask]))
            or np.any(membership[class_mask] < 0.0)
            or np.any(membership[class_mask] > 1.0)
        ):
            raise RuntimeError(f"{path} {prefix} membership violates its class schema")
        if (
            not np.all(np.isfinite(q))
            or np.any(q < -1e-12)
            or not np.allclose(q[~class_mask], 0.0, rtol=0.0, atol=1e-12)
            or not np.isclose(q.sum(), 1.0, rtol=0.0, atol=1e-8)
        ):
            raise RuntimeError(f"{path} {prefix} weights violate their class schema")


def _validate_embedded_cohort(
    cohort: Any,
    *,
    label: str,
) -> tuple[list[int], list[Mapping[str, Any]], str]:
    if not isinstance(cohort, Mapping):
        raise RuntimeError(f"{label} is absent")
    payload = dict(cohort)
    digest = payload.pop("sha256", None)
    if not isinstance(digest, str) or digest != canonical_protocol_hash(payload):
        raise RuntimeError(f"{label} digest is inconsistent")
    raw_seeds = cohort.get("seeds")
    entries = cohort.get("artifacts")
    if (
        not isinstance(raw_seeds, list)
        or not isinstance(entries, list)
        or any(not isinstance(entry, Mapping) for entry in entries)
    ):
        raise RuntimeError(f"{label} schema is invalid")
    seeds = [int(value) for value in raw_seeds]
    if len(seeds) != len(set(seeds)) or [int(entry.get("seed", -1)) for entry in entries] != seeds:
        raise RuntimeError(f"{label} seed ordering is invalid")
    return seeds, entries, digest


def _load_seed(out_dir: Path, seed: int) -> dict[str, Any]:
    seed_dir = out_dir / "runs" / f"seed_{seed}"
    selection_path = seed_dir / "selection.json"
    blind_path = seed_dir / "blind_test.json"
    posthoc_path = seed_dir / "posthoc.json"
    for path in (selection_path, blind_path, posthoc_path):
        if not path.is_file():
            raise FileNotFoundError(path)
    selection = read_json(selection_path)
    blind = read_json(blind_path)
    posthoc = read_json(posthoc_path)
    _require_stage(
        selection,
        path=selection_path,
        seed=seed,
        stage=SELECTION_STAGE,
        digest_field="frozen_selection_sha256",
    )
    _require_stage(
        blind,
        path=blind_path,
        seed=seed,
        stage=BLIND_STAGE,
        digest_field="blind_test_sha256",
    )
    _require_stage(
        posthoc,
        path=posthoc_path,
        seed=seed,
        stage=POSTHOC_STAGE,
        digest_field="posthoc_sha256",
    )

    if selection.get("metadata_accessed") is not False or selection.get("test_data_accessed") is not False:
        raise RuntimeError(f"seed {seed}: selection did not certify metadata/test blindness")
    if blind.get("metadata_accessed") is not False or blind.get("test_data_accessed_after_freeze") is not True:
        raise RuntimeError(f"seed {seed}: blind-test reveal ordering is invalid")
    if posthoc.get("metadata_accessed_after_blind_freeze") is not True:
        raise RuntimeError(f"seed {seed}: post-hoc metadata reveal ordering is invalid")
    if posthoc.get("primary_selection_unchanged") is not True:
        raise RuntimeError(f"seed {seed}: post-hoc stage does not attest an unchanged primary choice")

    selection_file_sha = file_sha256(selection_path)
    blind_file_sha = file_sha256(blind_path)
    if blind.get("selection_artifact_sha256") != selection_file_sha:
        raise RuntimeError(f"seed {seed}: blind/selection file hash mismatch")
    if blind.get("frozen_selection_sha256") != selection.get("frozen_selection_sha256"):
        raise RuntimeError(f"seed {seed}: blind/selection frozen digest mismatch")
    selection_inputs = selection.get("input_hashes")
    blind_inputs = blind.get("input_hashes")
    if not isinstance(selection_inputs, Mapping) or not isinstance(blind_inputs, Mapping):
        raise RuntimeError(f"seed {seed}: blind input provenance is absent")
    for name, digest in selection_inputs.items():
        if blind_inputs.get(name) != digest:
            raise RuntimeError(f"seed {seed}: blind input {name} changed after selection")
    if blind_inputs.get("test_blind") != blind.get("test_blind_sha256"):
        raise RuntimeError(f"seed {seed}: blind test input digest is inconsistent")
    if posthoc.get("selection_artifact_sha256") != selection_file_sha:
        raise RuntimeError(f"seed {seed}: posthoc/selection file hash mismatch")
    if posthoc.get("frozen_selection_sha256") != selection.get("frozen_selection_sha256"):
        raise RuntimeError(f"seed {seed}: posthoc/selection frozen digest mismatch")
    if posthoc.get("blind_test_artifact_sha256") != blind_file_sha:
        raise RuntimeError(f"seed {seed}: posthoc/blind file hash mismatch")
    if posthoc.get("blind_test_sha256") != blind.get("blind_test_sha256"):
        raise RuntimeError(f"seed {seed}: posthoc/blind content digest mismatch")

    selection_candidates = _candidate_map(
        selection.get("candidates", []), label=f"seed {seed} selection"
    )
    blind_candidates = _candidate_map(
        blind.get("candidates", []), label=f"seed {seed} blind test"
    )
    posthoc_candidates = _candidate_map(
        posthoc.get("candidates", []), label=f"seed {seed} posthoc"
    )
    if set(selection_candidates) != set(blind_candidates) or set(
        blind_candidates
    ) != set(posthoc_candidates):
        raise RuntimeError(f"seed {seed}: lambda grid changed across the frozen chain")
    selection_reference = _frozen_reference(
        selection, label=f"seed {seed} selection", propagated=False
    )
    blind_reference = _frozen_reference(
        blind, label=f"seed {seed} blind test", propagated=True
    )
    posthoc_reference = _frozen_reference(
        posthoc, label=f"seed {seed} posthoc", propagated=True
    )

    validation_path = _resolve_recorded_path(
        selection.get("validation_score_artifact"), fallback=seed_dir / "validation_scores.npz"
    )
    blind_score_path = _resolve_recorded_path(
        blind.get("score_artifact"), fallback=seed_dir / "blind_test_scores.npz"
    )
    if file_sha256(validation_path) != selection.get("validation_score_artifact_sha256"):
        raise RuntimeError(f"seed {seed}: validation-score archive hash mismatch")
    if blind.get("validation_score_artifact_sha256") != file_sha256(validation_path):
        raise RuntimeError(f"seed {seed}: blind stage used a different validation-score archive")
    if file_sha256(blind_score_path) != blind.get("score_artifact_sha256"):
        raise RuntimeError(f"seed {seed}: blind-test score archive hash mismatch")
    score_prefixes = [FROZEN_ERM_REFERENCE_KEY, *selection_candidates.keys()]
    score_records = {
        FROZEN_ERM_REFERENCE_KEY: selection_reference,
        **selection_candidates,
    }
    _validate_score_archive_schema(
        validation_path,
        prefixes=score_prefixes,
        records=score_records,
        test=False,
    )
    _validate_score_archive_schema(
        blind_score_path,
        prefixes=score_prefixes,
        records=score_records,
        test=True,
    )
    posthoc_protocol = posthoc.get("posthoc_protocol")
    if not isinstance(posthoc_protocol, Mapping):
        raise RuntimeError(f"seed {seed}: posthoc protocol is absent")
    certification_protocol = blind.get("certification_protocol")
    if not isinstance(certification_protocol, Mapping):
        raise RuntimeError(f"seed {seed}: certification cohort protocol is absent")
    if blind.get("certification_protocol_sha256") != canonical_protocol_hash(
        {"seed": seed, "certification_protocol": certification_protocol}
    ):
        raise RuntimeError(f"seed {seed}: certification protocol digest is inconsistent")
    selection_cohort_seeds, selection_cohort_entries, selection_cohort_digest = (
        _validate_embedded_cohort(
            certification_protocol.get("selection_cohort"),
            label=f"seed {seed} frozen selection cohort",
        )
    )
    if (
        certification_protocol.get("selection_cohort_sha256")
        != selection_cohort_digest
        or certification_protocol.get("cohort_seeds") != selection_cohort_seeds
    ):
        raise RuntimeError(f"seed {seed}: certification selection-cohort link is inconsistent")
    for entry in selection_cohort_entries:
        cohort_seed = int(entry["seed"])
        cohort_dir = out_dir / "runs" / f"seed_{cohort_seed}"
        cohort_selection_path = cohort_dir / "selection.json"
        cohort_validation_path = cohort_dir / "validation_scores.npz"
        if (
            not cohort_selection_path.is_file()
            or not cohort_validation_path.is_file()
            or file_sha256(cohort_selection_path)
            != entry.get("selection_artifact_sha256")
            or file_sha256(cohort_validation_path)
            != entry.get("validation_score_artifact_sha256")
            or read_json(cohort_selection_path).get("frozen_selection_sha256")
            != entry.get("frozen_selection_sha256")
        ):
            raise RuntimeError(
                f"seed {seed}: frozen selection cohort entry {cohort_seed} changed"
            )
    if posthoc.get("posthoc_protocol_sha256") != canonical_protocol_hash(
        {"seed": seed, "posthoc_protocol": posthoc_protocol}
    ):
        raise RuntimeError(f"seed {seed}: post-hoc protocol digest is inconsistent")
    blind_cohort_seeds, blind_cohort_entries, blind_cohort_digest = (
        _validate_embedded_cohort(
            posthoc_protocol.get("blind_cohort"),
            label=f"seed {seed} frozen blind cohort",
        )
    )
    if (
        posthoc_protocol.get("blind_cohort_sha256") != blind_cohort_digest
        or posthoc_protocol.get("cohort_seeds") != blind_cohort_seeds
        or posthoc_protocol["blind_cohort"].get("selection_cohort_sha256")
        != selection_cohort_digest
    ):
        raise RuntimeError(f"seed {seed}: post-hoc blind-cohort link is inconsistent")
    for entry in blind_cohort_entries:
        cohort_seed = int(entry["seed"])
        cohort_dir = out_dir / "runs" / f"seed_{cohort_seed}"
        cohort_blind_path = cohort_dir / "blind_test.json"
        cohort_score_path = cohort_dir / "blind_test_scores.npz"
        if (
            not cohort_blind_path.is_file()
            or not cohort_score_path.is_file()
            or file_sha256(cohort_blind_path)
            != entry.get("blind_test_artifact_sha256")
            or file_sha256(cohort_score_path)
            != entry.get("blind_test_score_artifact_sha256")
            or read_json(cohort_blind_path).get("blind_test_sha256")
            != entry.get("blind_test_sha256")
        ):
            raise RuntimeError(f"seed {seed}: frozen blind cohort entry {cohort_seed} changed")
    if posthoc_protocol.get("cohort_seeds") != certification_protocol.get("cohort_seeds"):
        raise RuntimeError(f"seed {seed}: certification/post-hoc reveal cohorts differ")
    if posthoc_protocol.get("smoke") is not certification_protocol.get("smoke"):
        raise RuntimeError(f"seed {seed}: certification/post-hoc smoke attestations differ")
    expected_complete_cohort = (
        certification_protocol.get("complete_default_cohort_frozen_before_test") is True
    )
    if (
        posthoc_protocol.get("complete_default_cohort_frozen_before_metadata")
        is not expected_complete_cohort
    ):
        raise RuntimeError(f"seed {seed}: certification/post-hoc complete-cohort attestations differ")
    sealed_digest = posthoc_protocol.get("sealed_metadata_sha256")
    if not isinstance(sealed_digest, str) or len(sealed_digest) != 64:
        raise RuntimeError(f"seed {seed}: posthoc protocol lacks an attested metadata seal")
    if posthoc_protocol.get("validation_score_artifact_sha256") != file_sha256(validation_path):
        raise RuntimeError(f"seed {seed}: posthoc used a different validation-score archive")
    if posthoc_protocol.get("blind_test_score_artifact_sha256") != file_sha256(blind_score_path):
        raise RuntimeError(f"seed {seed}: posthoc used a different blind-test score archive")
    expected_protocol_links = {
        "selection_artifact_sha256": selection_file_sha,
        "frozen_selection_sha256": selection["frozen_selection_sha256"],
        "blind_test_artifact_sha256": blind_file_sha,
        "blind_test_sha256": blind["blind_test_sha256"],
    }
    for field, expected in expected_protocol_links.items():
        if posthoc_protocol.get(field) != expected:
            raise RuntimeError(f"seed {seed}: posthoc protocol carries a stale {field}")

    for stage_label, candidate_map in (
        ("blind test", blind_candidates),
        ("posthoc", posthoc_candidates),
    ):
        if any(row.get("source") != "lambda_grid" for row in candidate_map.values()):
            raise RuntimeError(f"seed {seed}: {stage_label} lambda row has an invalid source")

    zero_key = lambda_key(0.0)
    if zero_key not in selection_candidates:
        raise RuntimeError(f"seed {seed}: matched lambda=0 continuation is absent")
    zero_candidate = selection_candidates[zero_key]
    if zero_candidate.get("role") != "matched_erm_continuation":
        raise RuntimeError(f"seed {seed}: lambda=0 is not attested as a matched ERM continuation")
    if selection_reference.get("role") != "frozen_pre_continuation_erm_reference":
        raise RuntimeError(f"seed {seed}: frozen ERM reference role is invalid")
    if selection_reference.get("head") != selection.get("base_model", {}).get("head"):
        raise RuntimeError(f"seed {seed}: frozen ERM reference is not the recorded base-model head")
    if selection_reference.get("warm_start_sha256") != zero_candidate.get("warm_start_sha256"):
        raise RuntimeError(f"seed {seed}: frozen and matched controls do not share a warm start")
    expected_reference_blind = {
        "task_metrics": blind_reference.get("task_metrics"),
        "hidden_certificate": blind_reference.get("hidden_certificate"),
    }
    if posthoc_reference.get("blind_test_candidate") != expected_reference_blind:
        raise RuntimeError(f"seed {seed}: posthoc frozen-reference blind metrics changed")
    policies = _policy_map(posthoc, seed=seed)
    protocol = selection.get("protocol")
    decision = selection.get("selection")
    if not isinstance(protocol, Mapping) or not isinstance(decision, Mapping):
        raise RuntimeError(f"seed {seed}: selection protocol/decision is missing")
    declared_grid = {lambda_key(float(value)) for value in protocol.get("lambdas", [])}
    if declared_grid != set(selection_candidates):
        raise RuntimeError(f"seed {seed}: candidate grid differs from the predeclared protocol")
    if not np.isclose(float(protocol["fixed_lambda"]), 0.03, rtol=0.0, atol=1e-12):
        raise RuntimeError(f"seed {seed}: fixed numerical reference is not lambda=0.03")
    expected_policy_lambdas = {
        "matched_erm_lambda_0": 0.0,
        "fixed_0.03": float(protocol["fixed_lambda"]),
        "validation_risk_minimizer": float(decision["validation_risk_minimizer_lambda"]),
        "group_free": float(decision["selected_lambda"]),
    }
    for key, expected in expected_policy_lambdas.items():
        if not np.isclose(float(policies[key]["lambda_fg"]), expected, rtol=0.0, atol=1e-12):
            raise RuntimeError(f"seed {seed}: posthoc policy {key} changed the frozen decision")
    frozen_policy = policies["frozen_erm_reference"]
    if frozen_policy.get("lambda_fg") is not None:
        raise RuntimeError(f"seed {seed}: frozen ERM policy was incorrectly represented as a lambda candidate")
    comparator_eligible = [float(value) for value in decision.get("eligible_lambdas", [])]
    if not comparator_eligible:
        raise RuntimeError(f"seed {seed}: frozen one-standard-error risk set is empty")
    comparator_lambda = float(policies["group_supervised_val_comparator"]["lambda_fg"])
    if not any(
        np.isclose(comparator_lambda, value, rtol=0.0, atol=1e-12)
        for value in comparator_eligible
    ):
        raise RuntimeError(f"seed {seed}: group-supervised comparator left the frozen risk set")
    comparator_pool = [
        candidate
        for candidate in posthoc_candidates.values()
        if any(
            np.isclose(float(candidate["lambda_fg"]), value, rtol=0.0, atol=1e-12)
            for value in comparator_eligible
        )
    ]
    expected_comparator = min(
        comparator_pool,
        key=lambda candidate: (
            -float(candidate["validation_group_metrics"]["worst_group_accuracy"]),
            float(candidate["lambda_fg"]),
        ),
    )
    if not np.isclose(
        comparator_lambda,
        float(expected_comparator["lambda_fg"]),
        rtol=0.0,
        atol=1e-12,
    ):
        raise RuntimeError(
            f"seed {seed}: group-supervised comparator does not implement its validation tie-break rule"
        )
    for key in POLICY_ORDER:
        expected_group_use = key == "group_supervised_val_comparator"
        if policies[key].get("group_labels_used_for_selection") is not expected_group_use:
            raise RuntimeError(f"seed {seed}: policy {key} has an incorrect supervision declaration")
        expected_source = "frozen_reference" if key == "frozen_erm_reference" else "lambda_grid"
        if policies[key].get("source") != expected_source:
            raise RuntimeError(f"seed {seed}: policy {key} has an incorrect source declaration")
        if key == "frozen_erm_reference":
            if policies[key].get("validation_group_metrics") != posthoc_reference.get(
                "validation_group_metrics"
            ):
                raise RuntimeError(f"seed {seed}: frozen policy validation group metrics changed")
            if policies[key].get("test_group_metrics") != posthoc_reference.get("test_group_metrics"):
                raise RuntimeError(f"seed {seed}: frozen policy test group metrics changed")
            if policies[key].get("blind_test_candidate") != expected_reference_blind:
                raise RuntimeError(f"seed {seed}: frozen policy blind metrics changed")
            continue
        coefficient = float(policies[key]["lambda_fg"])
        _selection_candidate, candidate_blind, candidate_posthoc = _candidate_for_lambda(
            {
                "seed": seed,
                "selection_candidates": selection_candidates,
                "blind_candidates": blind_candidates,
                "posthoc_candidates": posthoc_candidates,
            },
            coefficient,
        )
        if policies[key].get("validation_group_metrics") != candidate_posthoc.get("validation_group_metrics"):
            raise RuntimeError(f"seed {seed}: policy {key} validation group metrics changed")
        if policies[key].get("test_group_metrics") != candidate_posthoc.get("test_group_metrics"):
            raise RuntimeError(f"seed {seed}: policy {key} test group metrics changed")
        expected_blind = {
            "task_metrics": candidate_blind.get("task_metrics"),
            "hidden_certificate": candidate_blind.get("hidden_certificate"),
        }
        if policies[key].get("blind_test_candidate") != expected_blind:
            raise RuntimeError(f"seed {seed}: policy {key} blind metrics changed post hoc")
    return {
        "seed": seed,
        "selection": selection,
        "blind": blind,
        "posthoc": posthoc,
        "selection_candidates": selection_candidates,
        "blind_candidates": blind_candidates,
        "posthoc_candidates": posthoc_candidates,
        "selection_reference": selection_reference,
        "blind_reference": blind_reference,
        "posthoc_reference": posthoc_reference,
        "policies": policies,
        "paths": {
            "selection": str(selection_path),
            "blind_test": str(blind_path),
            "posthoc": str(posthoc_path),
            "validation_scores": str(validation_path),
            "blind_test_scores": str(blind_score_path),
        },
        "sha256": {
            "selection": selection_file_sha,
            "blind_test": blind_file_sha,
            "posthoc": file_sha256(posthoc_path),
            "validation_scores": file_sha256(validation_path),
            "blind_test_scores": file_sha256(blind_score_path),
        },
    }


def _validate_cross_seed_provenance(contexts: list[dict[str, Any]]) -> None:
    """Reject an output directory assembled from different prepared bundles."""

    if not contexts:
        raise ValueError("at least one seed context is required")
    reference = contexts[0]
    expected = {
        "selection_inputs": reference["selection"].get("input_hashes"),
        "blind_inputs": reference["blind"].get("input_hashes"),
        "preparation_configuration": reference["selection"]
        .get("source_preparation_manifest", {})
        .get("configuration_sha256"),
        "sealed_metadata": reference["posthoc"].get("posthoc_protocol", {}).get("sealed_metadata_sha256"),
        "selection_protocol": reference["selection"].get("protocol"),
        "selection_cohort": reference["blind"].get("certification_protocol", {}).get(
            "selection_cohort"
        ),
        "blind_cohort": reference["posthoc"].get("posthoc_protocol", {}).get(
            "blind_cohort"
        ),
    }
    for context in contexts[1:]:
        observed = {
            "selection_inputs": context["selection"].get("input_hashes"),
            "blind_inputs": context["blind"].get("input_hashes"),
            "preparation_configuration": context["selection"]
            .get("source_preparation_manifest", {})
            .get("configuration_sha256"),
            "sealed_metadata": context["posthoc"].get("posthoc_protocol", {}).get(
                "sealed_metadata_sha256"
            ),
            "selection_protocol": context["selection"].get("protocol"),
            "selection_cohort": context["blind"].get("certification_protocol", {}).get(
                "selection_cohort"
            ),
            "blind_cohort": context["posthoc"].get("posthoc_protocol", {}).get(
                "blind_cohort"
            ),
        }
        for field, value in expected.items():
            if observed[field] != value:
                raise RuntimeError(
                    f"seed {context['seed']}: cross-seed provenance differs for {field}"
                )


def _candidate_for_lambda(context: Mapping[str, Any], coefficient: float) -> tuple[Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]]:
    key = lambda_key(float(coefficient))
    try:
        return (
            context["selection_candidates"][key],
            context["blind_candidates"][key],
            context["posthoc_candidates"][key],
        )
    except KeyError as exc:
        raise RuntimeError(f"seed {context['seed']}: policy selected lambda outside grid: {coefficient}") from exc


def _group_worst(record: Mapping[str, Any], split: str) -> float:
    # The post-hoc schema stores explicit validation/test group dictionaries.
    group = record.get(f"{split}_group_metrics")
    if not isinstance(group, Mapping):
        raise RuntimeError(f"posthoc candidate lacks {split}_group_metrics")
    return float(group["worst_group_accuracy"])


def _seed_policy_row(context: Mapping[str, Any], policy_key: str) -> dict[str, Any]:
    policy = context["policies"][policy_key]
    source = str(policy["source"])
    if source == "frozen_reference":
        coefficient = None
        selection_candidate = context["selection_reference"]
        blind_candidate = context["blind_reference"]
        posthoc_candidate = context["posthoc_reference"]
    elif source == "lambda_grid":
        coefficient = float(policy["lambda_fg"])
        selection_candidate, blind_candidate, posthoc_candidate = _candidate_for_lambda(
            context, coefficient
        )
    else:
        raise RuntimeError(f"seed {context['seed']}: unknown policy source {source!r}")
    task = blind_candidate["task_metrics"]
    hidden = blind_candidate["hidden_certificate"]
    policy_test = policy.get("test_group_metrics")
    policy_validation = policy.get("validation_group_metrics")
    test_worst = (
        float(policy_test["worst_group_accuracy"])
        if isinstance(policy_test, Mapping)
        else _group_worst(posthoc_candidate, "test")
    )
    validation_worst = (
        float(policy_validation["worst_group_accuracy"])
        if isinstance(policy_validation, Mapping)
        else _group_worst(posthoc_candidate, "validation")
    )
    return {
        "seed": int(context["seed"]),
        "policy": policy_key,
        "source": source,
        "lambda_fg": coefficient,
        "test_accuracy": float(task["accuracy"]),
        "test_balanced_accuracy": float(task["balanced_accuracy"]),
        "test_nll": float(task["nll"]),
        "test_hidden_certificate": float(hidden["certificate"]),
        "test_inverse_fisher": float(hidden["predictive"]["inverse_fisher"]),
        "test_worst_group_accuracy": test_worst,
        "validation_worst_group_accuracy": validation_worst,
        "validation_nll": float(selection_candidate["validation_metrics"]["nll"]),
    }


def _aggregate_policies(
    contexts: list[dict[str, Any]], confidence: float
) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    per_seed = {
        key: [_seed_policy_row(context, key) for context in contexts] for key in POLICY_ORDER
    }
    rows: list[dict[str, Any]] = []
    for key in POLICY_ORDER:
        seed_rows = per_seed[key]
        lambda_counts = Counter(
            "n/a" if row["lambda_fg"] is None else f"{float(row['lambda_fg']):.12g}"
            for row in seed_rows
        )
        metrics = {
            metric: _summary(
                (float(row[metric]) for row in seed_rows),
                confidence,
                bounds=BOUNDED_METRICS.get(metric),
            )
            for metric in METRICS
        }
        rows.append(
            {
                "key": key,
                "label": POLICY_LABELS[key],
                "source": str(seed_rows[0]["source"]),
                "selection_supervision": POLICY_SUPERVISION[key],
                "lambda_counts": dict(
                    sorted(
                        lambda_counts.items(),
                        key=lambda item: (item[0] == "n/a", float(item[0]) if item[0] != "n/a" else 0.0),
                    )
                ),
                "metrics": metrics,
            }
        )
    return rows, per_seed


def _paired_rows(
    per_seed: Mapping[str, list[dict[str, Any]]], confidence: float
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for candidate_key, baseline_key, role in PAIRED_CONTRASTS:
        candidate = per_seed[candidate_key]
        baseline = per_seed[baseline_key]
        for metric in METRICS:
            differences = [
                float(candidate[index][metric]) - float(baseline[index][metric])
                for index in range(len(baseline))
            ]
            rows.append(
                {
                    "candidate_policy": candidate_key,
                    "baseline_policy": baseline_key,
                    "contrast_role": role,
                    "metric": metric,
                    "favorable_direction": "lower" if metric in {"test_nll", "test_inverse_fisher"} else "higher",
                    "difference_candidate_minus_baseline": _summary(differences, confidence),
                }
            )
    return rows


def _frontier_rows(
    contexts: list[dict[str, Any]], confidence: float
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    grids = [set(context["selection_candidates"]) for context in contexts]
    if any(grid != grids[0] for grid in grids[1:]):
        raise RuntimeError("lambda grid differs across seeds")
    coefficients = sorted(float(contexts[0]["selection_candidates"][key]["lambda_fg"]) for key in grids[0])
    policy_choices = {
        policy: [float(context["policies"][policy]["lambda_fg"]) for context in contexts]
        for policy in GRID_POLICY_ORDER
    }
    aggregate_rows: list[dict[str, Any]] = []
    flat_rows: list[dict[str, Any]] = []
    for coefficient in coefficients:
        per_seed: list[dict[str, Any]] = []
        for context in contexts:
            selection, blind, posthoc = _candidate_for_lambda(context, coefficient)
            task = blind["task_metrics"]
            hidden = blind["hidden_certificate"]
            scores = selection["certificate_scores"]
            seed_row = {
                "seed": int(context["seed"]),
                "lambda_fg": coefficient,
                "validation_nll": float(selection["validation_metrics"]["nll"]),
                "validation_accuracy": float(selection["validation_metrics"]["accuracy"]),
                "validation_certificate_lower_score": float(
                    scores["simultaneous_certificate_lower_score"]
                ),
                "test_accuracy": float(task["accuracy"]),
                "test_balanced_accuracy": float(task["balanced_accuracy"]),
                "test_nll": float(task["nll"]),
                "test_hidden_certificate": float(hidden["certificate"]),
                "test_inverse_fisher": float(hidden["predictive"]["inverse_fisher"]),
                "validation_worst_group_accuracy": _group_worst(posthoc, "validation"),
                "test_worst_group_accuracy": _group_worst(posthoc, "test"),
            }
            per_seed.append(seed_row)
            flat_rows.append(seed_row)
        metric_names = [key for key in per_seed[0] if key not in {"seed", "lambda_fg"}]
        aggregate_rows.append(
            {
                "lambda_fg": coefficient,
                "metrics": {
                    metric: _summary(
                        (row[metric] for row in per_seed),
                        confidence,
                        bounds=(0.0, 1.0) if metric.endswith("accuracy") else None,
                    )
                    for metric in metric_names
                },
                "selected_counts": {
                    policy: int(sum(np.isclose(value, coefficient, rtol=0.0, atol=1e-12) for value in values))
                    for policy, values in policy_choices.items()
                },
                "per_seed": per_seed,
            }
        )
    return aggregate_rows, flat_rows


def _reportable_protocol_deviations(
    contexts: list[dict[str, Any]], seeds: list[int], *, smoke: bool
) -> list[str]:
    """Explain every departure from the frozen five-seed sensitivity protocol."""

    deviations: list[str] = []
    if tuple(seeds) != DEFAULT_SEEDS:
        deviations.append(f"seeds={seeds}; expected {list(DEFAULT_SEEDS)} in this order")
    if smoke:
        deviations.append("launcher/reporting smoke flag is set")

    def check(seed: int, name: str, observed: Any, expected: Any) -> None:
        equal: bool
        if isinstance(expected, float):
            try:
                equal = bool(np.isclose(float(observed), expected, rtol=0.0, atol=1e-12))
            except (TypeError, ValueError):
                equal = False
        elif isinstance(expected, list):
            try:
                array = np.asarray(observed, dtype=np.float64)
                target = np.asarray(expected, dtype=np.float64)
                equal = bool(array.shape == target.shape and np.allclose(array, target, rtol=0.0, atol=1e-12))
            except (TypeError, ValueError):
                equal = False
        else:
            equal = observed == expected
        if not equal:
            deviations.append(f"seed {seed}: {name}={observed!r}; expected {expected!r}")

    for context in contexts:
        seed = int(context["seed"])
        protocol = context["selection"]["protocol"]
        check(seed, "lambdas", protocol.get("lambdas"), [0.0, 0.01, 0.03, 0.1, 0.3])
        check(seed, "fixed_lambda", protocol.get("fixed_lambda"), 0.03)
        check(seed, "confidence", protocol.get("confidence"), 0.95)
        check(seed, "folds", protocol.get("folds"), 5)
        check(
            seed,
            "analysis_status",
            protocol.get("analysis_status"),
            (
                "post-hoc matched-control sensitivity designed after v1 Waterbirds group/test outcomes "
                "were inspected; not an independent held-out confirmation"
            ),
        )
        fractions = protocol.get("training_partition_fractions", {})
        check(seed, "training.fit_fraction", fractions.get("fit"), 0.70)
        check(seed, "training.calibration_fraction", fractions.get("calibration"), 0.15)
        check(seed, "training.gate_discovery_fraction", fractions.get("gate_discovery"), 0.15)
        head = protocol.get("head", {})
        for name, expected in (
            ("steps", 1200),
            ("learning_rate", 0.01),
            ("weight_decay", 0.0001),
            ("gradient_clip", 10.0),
            ("early_stopping_patience", 12),
            ("early_stopping_eval_every", 25),
            ("early_stopping_min_delta", 1e-5),
        ):
            check(seed, f"head.{name}", head.get(name), expected)
        gate = protocol.get("gate", {})
        for name, expected in (
            ("steps", 1000),
            ("learning_rate", 0.03),
            ("restarts", 10),
            ("weight_decay", 0.0001),
            ("mass_penalty_strength", 10.0),
            ("slice_mass", 0.10),
            ("minimum_ess_fraction", 0.80),
            ("maximum_weight_multiplier", 2.0),
            ("shrinkage", 0.10),
            ("ridge", 0.001),
            ("epsilon", 1e-8),
        ):
            check(seed, f"gate.{name}", gate.get(name), expected)
        fisher = protocol.get("fisher_metric", {})
        for name, expected in (("shrinkage", 0.10), ("ridge", 0.001), ("epsilon", 1e-8)):
            check(seed, f"fisher_metric.{name}", fisher.get(name), expected)
        repair = protocol.get("repair", {})
        check(
            seed,
            "repair.matched_training_includes_lambda_zero",
            repair.get("matched_training_includes_lambda_zero"),
            True,
        )
        check(
            seed,
            "repair.frozen_reference_in_candidate_grid",
            repair.get("frozen_reference_in_candidate_grid"),
            False,
        )
        selection_protocol = protocol.get("selection", {})
        check(
            seed,
            "selection.accuracy_diagnostic_tolerance",
            selection_protocol.get("accuracy_diagnostic_tolerance"),
            0.02,
        )
        check(seed, "deterministic_torch_algorithms", protocol.get("deterministic_torch_algorithms"), True)
        certification = context["blind"].get("certification_protocol", {})
        check(seed, "bootstrap_replicates", certification.get("bootstrap_replicates"), 2000)
        check(seed, "bootstrap_confidence", certification.get("bootstrap_confidence"), 0.95)
        check(seed, "certification.cohort_seeds", certification.get("cohort_seeds"), list(DEFAULT_SEEDS))
        check(
            seed,
            "certification.complete_default_cohort_frozen_before_test",
            certification.get("complete_default_cohort_frozen_before_test"),
            True,
        )
        check(seed, "certification.smoke", certification.get("smoke"), False)
        posthoc_protocol = context["posthoc"].get("posthoc_protocol", {})
        check(seed, "posthoc.cohort_seeds", posthoc_protocol.get("cohort_seeds"), list(DEFAULT_SEEDS))
        check(
            seed,
            "posthoc.complete_default_cohort_frozen_before_metadata",
            posthoc_protocol.get("complete_default_cohort_frozen_before_metadata"),
            True,
        )
        check(seed, "posthoc.smoke", posthoc_protocol.get("smoke"), False)
    return deviations


def _fmt(summary: Mapping[str, Any], scale: float = 100.0, digits: int = 1) -> str:
    return f"{scale * float(summary['mean']):.{digits}f} [{scale * float(summary['low']):.{digits}f}, {scale * float(summary['high']):.{digits}f}]"


def _flat_main(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {
            "policy": row["key"],
            "label": row["label"],
            "source": row["source"],
            "selection_supervision": row["selection_supervision"],
            "selected_lambda_counts": json.dumps(row["lambda_counts"], sort_keys=True),
        }
        for metric, summary in row["metrics"].items():
            flat[f"{metric}_mean"] = summary["mean"]
            flat[f"{metric}_ci_low"] = summary["low"]
            flat[f"{metric}_ci_high"] = summary["high"]
        result.append(flat)
    return result


def _flat_frontier(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in rows:
        flat: dict[str, Any] = {"lambda_fg": row["lambda_fg"]}
        for metric, summary in row["metrics"].items():
            flat[f"{metric}_mean"] = summary["mean"]
            flat[f"{metric}_ci_low"] = summary["low"]
            flat[f"{metric}_ci_high"] = summary["high"]
        for policy, count in row["selected_counts"].items():
            flat[f"selected_by_{policy}"] = count
        result.append(flat)
    return result


def _difference(
    aggregate: Mapping[str, Any],
    candidate_policy: str,
    baseline_policy: str,
    metric: str,
) -> Mapping[str, Any]:
    for row in aggregate["paired_comparisons"]:
        if (
            row["candidate_policy"] == candidate_policy
            and row["baseline_policy"] == baseline_policy
            and row["metric"] == metric
        ):
            return row["difference_candidate_minus_baseline"]
    raise KeyError((candidate_policy, baseline_policy, metric))


def _difference_sentence(
    summary: Mapping[str, Any],
    *,
    scale: float = 100.0,
    unit: str = " points",
) -> str:
    low = float(summary["low"])
    high = float(summary["high"])
    if low > 0.0:
        conclusion = "the across-seed interval is entirely above zero"
    elif high < 0.0:
        conclusion = "the across-seed interval is entirely below zero"
    else:
        conclusion = "the across-seed interval includes zero"
    return (
        f"{scale * float(summary['mean']):+.3f}{unit} "
        f"(95% across-seed t interval {scale * low:+.3f} to {scale * high:+.3f}); {conclusion}"
    )


def _higher_is_better_status(summary: Mapping[str, Any]) -> str:
    low = float(summary["low"])
    high = float(summary["high"])
    if low > 0.0:
        return "improved"
    if high < 0.0:
        return "harmed"
    return "inconclusive"


def _rebuttal(aggregate: Mapping[str, Any]) -> str:
    rows = {row["key"]: row for row in aggregate["policies"]}
    n_seeds = len(aggregate["seeds"])
    group_counts = rows["group_free"]["lambda_counts"]
    group_counts_text = ", ".join(
        f"lambda={coefficient}: {count}" for coefficient, count in group_counts.items()
    )
    risk_agreement = int(
        aggregate.get("selection_agreement", {}).get(
            "group_free_equals_validation_risk_minimizer", 0
        )
    )
    zero_count = int(group_counts.get("0", 0))
    worst_difference = _difference(
        aggregate, "group_free", "matched_erm_lambda_0", "test_worst_group_accuracy"
    )
    certificate_difference = _difference(
        aggregate, "group_free", "matched_erm_lambda_0", "test_hidden_certificate"
    )
    accuracy_difference = _difference(
        aggregate, "group_free", "matched_erm_lambda_0", "test_accuracy"
    )
    continuation_worst = _difference(
        aggregate,
        "matched_erm_lambda_0",
        "frozen_erm_reference",
        "test_worst_group_accuracy",
    )
    total_worst = _difference(
        aggregate, "group_free", "frozen_erm_reference", "test_worst_group_accuracy"
    )
    fixed = rows["fixed_0.03"]
    group_free = rows["group_free"]
    comparator = rows["group_supervised_val_comparator"]
    seed_word = "seed" if n_seeds == 1 else "seeds"
    if zero_count == n_seeds:
        action = (
            f"The rule selected the matched zero-penalty continuation, lambda=0, in all {n_seeds} {seed_word}; "
            "it had the highest "
            "adjusted certificate score within the one-standard-error risk set in each run."
        )
    elif zero_count:
        action = (
            f"The rule selected the matched lambda=0 continuation in {zero_count}/{n_seeds} seeds and a "
            f"nonzero coefficient in {n_seeds - zero_count}/{n_seeds}; selected-lambda counts were "
            f"{group_counts_text}."
        )
    else:
        action = (
            f"It selected a nonzero coefficient in all {n_seeds} {seed_word}; selected-lambda counts were "
            f"{group_counts_text}."
        )

    worst_status = _higher_is_better_status(worst_difference)
    certificate_status = _higher_is_better_status(certificate_difference)
    continuation_status = _higher_is_better_status(continuation_worst)
    if worst_status == "harmed":
        outcome = (
            "The post-hoc group audit therefore shows worst-group harm, not evidence that the certificate rule "
            "is a safe proxy for worst-group tuning."
        )
    elif worst_status == "improved":
        outcome = (
            "The post-hoc group audit shows higher worst-group accuracy in this sensitivity analysis, but the "
            "analysis is not a fresh held-out confirmation."
        )
    else:
        outcome = (
            "The post-hoc group audit leaves the worst-group effect inconclusive, so it does not establish that "
            "the certificate rule recovers worst-group tuning."
        )
    if certificate_status == "improved":
        certificate_outcome = "The hidden-certificate contrast was positive."
    elif certificate_status == "harmed":
        certificate_outcome = "The hidden-certificate contrast was negative."
    else:
        certificate_outcome = "The hidden-certificate contrast was inconclusive."
    if continuation_status == "harmed":
        continuation_guidance = (
            "Because the matched continuation also reduced worst-group accuracy relative to the frozen ERM, "
            "retaining the frozen ERM is the more conservative overall choice in this audit."
        )
    elif continuation_status == "improved":
        continuation_guidance = (
            "The matched continuation improved worst-group accuracy relative to the frozen ERM in this audit."
        )
    else:
        continuation_guidance = (
            "The worst-group effect of continuation itself was inconclusive relative to the frozen ERM."
        )

    warning = (
        "NON-REPORTABLE INCOMPLETE PROTOCOL. "
        if not aggregate.get("reportable_five_seed_protocol_complete")
        else ""
    )
    return (
        warning
        + "Table 4 and this deployment question concern two different uses of lambda_FG. In Table 4, "
        "standalone Fisher-Glass obtains 56.3% worst-group accuracy and Group DRO obtains 67.3%. That table is "
        "a controlled known-group comparison: Group DRO directly optimizes the worst labeled group, while the "
        "standalone Fisher-Glass coefficient was selected by certificate among candidates within two points "
        "of the best validation worst-group accuracy. That group-supervised rule selected 0.03 in all five "
        "seeds, but it does not provide a group-free tuning rule. The certificate moves in the complementary "
        "direction "
        "(0.399 for standalone Fisher-Glass versus 0.233 for Group DRO), and the combined row reaches 69.4% "
        "worst-group accuracy with a 0.614 certificate. Thus Table 4 supports complementarity, not replacement.\n\n"
        "To examine the practical question, we added a separate Waterbirds matched-control sensitivity: a single repair stage "
        "against a train-only slice discovered from the ERM head, followed by fresh Appendix-B cross-fitted "
        "slice rediscovery for validation. We predeclared lambda in {0, 0.01, 0.03, 0.1, 0.3}. The selector "
        "first keeps "
        "candidates whose ordinary validation NLL is within one per-example standard error of the minimum. "
        "Within that risk set, it chooses the largest Bonferroni-adjusted cross-fit hidden-certificate lower "
        "score, breaking exact ties toward the smaller coefficient. The two-point accuracy checks and paired "
        "fold comparison are reported as diagnostics and do not change the choice. The nuisance input is a "
        "frozen CLIP water-versus-land text direction: it uses no Waterbirds background or group annotation, "
        "although it does encode that semantic nuisance hypothesis. Group metadata remains sealed throughout "
        "fitting, validation, and blind test certification. Every lambda candidate, including lambda=0, now receives "
        "the same optimizer continuation and checkpoint rule; the pre-continuation ERM is retained separately. "
        "Because this matched control was designed after the v1 Waterbirds group/test outcomes had been inspected, "
        "the v2 run is a post-hoc sensitivity analysis rather than an independent held-out confirmation.\n\n"
        + action
        + f" It made the same choice as ordinary validation-risk minimization in "
        f"{risk_agreement}/{n_seeds} {seed_word}."
        + " Relative to the matched lambda=0 continuation, its paired test average-accuracy change was "
        + _difference_sentence(accuracy_difference)
        + ". Its worst-group change was "
        + _difference_sentence(
            worst_difference,
        )
        + ". Its hidden-certificate change was "
        + _difference_sentence(
            certificate_difference,
            scale=1.0,
            unit="",
        )
        + ". "
        + outcome
        + " "
        + certificate_outcome
        + " The matched lambda=0 continuation itself changed worst-group accuracy relative to the frozen ERM by "
        + _difference_sentence(continuation_worst)
        + "; the total group-free-minus-frozen-ERM change was "
        + _difference_sentence(total_worst)
        + ". "
        + continuation_guidance
        + " Across seeds, the group-free rule obtained test average accuracy "
        + _fmt(group_free["metrics"]["test_accuracy"])
        + "%, worst-group accuracy "
        + _fmt(group_free["metrics"]["test_worst_group_accuracy"])
        + "%, and hidden certificate "
        + _fmt(group_free["metrics"]["test_hidden_certificate"], scale=1.0, digits=3)
        + ". For context, fixed lambda=0.03 on the hidden-slice penalty scale obtained worst-group accuracy "
        + _fmt(fixed["metrics"]["test_worst_group_accuracy"])
        + "%, while the group-supervised validation comparator obtained "
        + _fmt(comparator["metrics"]["test_worst_group_accuracy"])
        + "%. That comparator uses validation group annotations, is not an upper bound, and is not a deployable "
        "selector.\n\n"
        "The fixed 0.03 row is only a numerical reference on this experiment's dimensionless hidden-slice "
        "penalty scale; it is not numerically identical to the known-group Fisher-Glass objective in Table 4. "
        "All candidate heads, lambda decisions, and validation gates were hashed before any test array was "
        "deserialized or evaluated; preflight had only verified the opaque test-artifact hash and manifest "
        "attestation. Group metadata was unsealed only after blind test predictions and certificates were frozen. "
        "The practical recommendation is correspondingly conservative. A different prospective policy could "
        "predeclare a certificate requirement and choose the smallest lambda meeting it under an ordinary-risk "
        "budget, but that target-based policy was not evaluated in this audit, and the certificate should not be "
        "treated as a surrogate for unseen "
        "worst-group accuracy. Without a certificate target or group-labeled audit data, lambda=0 is the conservative "
        "choice within the matched continuation family; the frozen ERM remains the separate reference for the effect "
        "of continuation itself. These results neither replace the Table-4 group-supervised selection nor claim that "
        "standalone Fisher-Glass closes the Group DRO gap. "
        "Across-seed intervals describe training/partition-seed variability on this dataset, not population "
        "sampling uncertainty.\n"
    )


def _lambda_counts_text(row: Mapping[str, Any], *, separator: str = ", ") -> str:
    if row.get("source") == "frozen_reference":
        return "--"
    return separator.join(f"{key}:{value}" for key, value in row["lambda_counts"].items())


def _latex(aggregate: Mapping[str, Any]) -> str:
    lines = [
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Policy & Selected $\lambda_{\mathrm{FG}}$ & Avg. acc. & Worst-group acc. & Hidden cert. \\",
        r"\midrule",
    ]
    for row in aggregate["policies"]:
        counts = _lambda_counts_text(row)
        label = row["label"].replace("lambda", r"$\lambda$")
        lines.append(
            f"{label} & {counts} & "
            f"{_fmt(row['metrics']['test_accuracy'])} & "
            f"{_fmt(row['metrics']['test_worst_group_accuracy'])} & "
            f"{_fmt(row['metrics']['test_hidden_certificate'], scale=1.0, digits=3)} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            "",
            "% Entries are mean [95% Student-t interval] across paired training/partition seeds.",
            "% These describe seed variability on one dataset, not population sampling uncertainty.",
            "% The comparator uses validation group labels post hoc; all other policies avoid group annotations.",
            "% The frozen ERM is outside the lambda grid; lambda=0 is its matched optimizer continuation.",
        ]
    )
    if not aggregate.get("reportable_five_seed_protocol_complete"):
        lines.insert(0, "% NON-REPORTABLE INCOMPLETE OR SMOKE PROTOCOL")
    return "\n".join(lines) + "\n"


def _report(aggregate: Mapping[str, Any]) -> str:
    banner = (
        "> **NON-REPORTABLE INCOMPLETE PROTOCOL. These values must not enter a paper or rebuttal.**\n\n"
        if not aggregate.get("reportable_five_seed_protocol_complete")
        else ""
    )
    deviation_text = (
        "Reportability deviations: " + "; ".join(aggregate.get("reportability_deviations", []))
        if not aggregate.get("reportable_five_seed_protocol_complete")
        else ""
    )
    lines = [
        "# Waterbirds group-free Fisher-Glass lambda selection: schema-v2 matched-control sensitivity",
        "",
        banner.rstrip(),
        "",
        "## Protocol",
        "",
        "Candidate repairs and lambda selection use task labels plus a frozen CLIP water-versus-land nuisance "
        "score, never Waterbirds background/group annotations. The rule retains the one-standard-error "
        "validation-NLL risk set, then maximizes the Bonferroni-adjusted cross-fit hidden-certificate lower "
        "score within it, with ties toward smaller lambda. Accuracy checks and paired fold comparisons are "
        "diagnostics only. Every lambda, including zero, receives the same continuation/checkpoint protocol; "
        "the pre-continuation ERM is a separate frozen reference. Test group labels are unsealed only post hoc.",
        "",
        "This schema-v2 run is a post-hoc matched-control sensitivity designed after the v1 Waterbirds "
        "group/test outcomes were inspected; it is not a fresh held-out confirmation.",
        "",
        f"Seeds: `{aggregate['seeds']}`. Intervals are two-sided 95% Student-t intervals across paired "
        "training/partition seeds on the same dataset; they are not population sampling intervals.",
        deviation_text,
        "",
        "## Policy results",
        "",
        "| Policy | Selected lambda counts | Test avg. acc. | Test worst-group acc. | Hidden certificate |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in aggregate["policies"]:
        counts = (
            "--"
            if row.get("source") == "frozen_reference"
            else ", ".join(f"{key}: {value}" for key, value in row["lambda_counts"].items())
        )
        lines.append(
            f"| {row['label']} | {counts} | {_fmt(row['metrics']['test_accuracy'])}% | "
            f"{_fmt(row['metrics']['test_worst_group_accuracy'])}% | "
            f"{_fmt(row['metrics']['test_hidden_certificate'], scale=1.0, digits=3)} |"
        )
    lines.extend(
        [
            "",
            "The group-supervised comparator uses validation group labels only after the group-free decision "
            "and blind test artifacts have been frozen. It is a diagnostic, not an available practitioner rule "
            "or an upper bound.",
            "",
            "## Paired contrasts",
            "",
        ]
    )
    contrast_labels = {
        ("group_free", "matched_erm_lambda_0"): "Primary: group-free selector minus matched lambda=0 continuation",
        ("matched_erm_lambda_0", "frozen_erm_reference"): "Secondary: matched lambda=0 continuation minus frozen ERM",
        ("group_free", "frozen_erm_reference"): "Secondary: group-free selector minus frozen ERM",
    }
    for candidate, baseline, _role in PAIRED_CONTRASTS:
        lines.append(f"- **{contrast_labels[(candidate, baseline)]}:**")
        for metric in ("test_accuracy", "test_worst_group_accuracy", "test_hidden_certificate"):
            difference = _difference(aggregate, candidate, baseline, metric)
            scale = 1.0 if metric == "test_hidden_certificate" else 100.0
            unit = "" if scale == 1.0 else " points"
            lines.append(
                f"  - `{metric}`: {scale * difference['mean']:+.3f}{unit} "
                f"(95% across-seed t interval {scale * difference['low']:+.3f} to "
                f"{scale * difference['high']:+.3f})."
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            _rebuttal(aggregate).strip(),
            "",
            "## Provenance",
            "",
            "`aggregate.json` records every input path/digest, the current scientific source manifest, "
            "per-seed values, the complete lambda frontier, and all paired differences. Aggregation fails "
            "closed if any selection, blind-test, post-hoc, NPZ, or source digest is stale.",
        ]
    )
    return "\n".join(line for line in lines if line is not None) + "\n"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", default="build/results/waterbirds_group_free_lambda_v2", type=Path)
    parser.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--smoke", action="store_true", help="mark every rendered artifact non-reportable")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if len(args.seeds) != len(set(args.seeds)):
        raise ValueError("seeds must be unique")
    if not np.isclose(args.confidence, 0.95, rtol=0.0, atol=1e-12):
        raise ValueError("report rendering is predeclared for confidence=0.95")
    contexts = [_load_seed(args.out_dir, seed) for seed in args.seeds]
    _validate_cross_seed_provenance(contexts)
    for context in contexts:
        protocol_confidence = float(context["selection"]["protocol"]["confidence"])
        if not np.isclose(protocol_confidence, args.confidence, rtol=0.0, atol=1e-12):
            raise RuntimeError(
                f"seed {context['seed']}: selection/reporting confidence levels differ"
            )
    policies, per_seed = _aggregate_policies(contexts, args.confidence)
    paired = _paired_rows(per_seed, args.confidence)
    frontier, frontier_per_seed = _frontier_rows(contexts, args.confidence)
    protocol_deviations = _reportable_protocol_deviations(
        contexts, list(args.seeds), smoke=bool(args.smoke)
    )
    complete = not protocol_deviations
    aggregate: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "stage": "group_free_lambda_reporting_aggregate",
        "smoke": bool(args.smoke),
        "reportable_five_seed_protocol_complete": bool(complete),
        "reportability_deviations": protocol_deviations,
        "seeds": list(args.seeds),
        "confidence": float(args.confidence),
        "interval_convention": (
            "two-sided Student-t interval across paired training/partition seeds on the same dataset; "
            "descriptive seed variability, not population or dataset-sampling uncertainty"
        ),
        "analysis_status": (
            "post-hoc matched-control sensitivity designed after v1 Waterbirds group/test outcomes were "
            "inspected; not an independent held-out confirmation"
        ),
        "implementation_manifest": implementation_manifest(),
        "metadata_contract": {
            "selection_and_certification_metadata_accessed": False,
            "posthoc_group_labels_opened_only_after_blind_freeze": True,
            "group_supervised_comparator_role": (
                "post-hoc group-supervised validation comparator; neither a deployable selector nor an upper bound"
            ),
        },
        "policies": policies,
        "paired_comparisons": paired,
        "selected_lambda_counts": {row["key"]: row["lambda_counts"] for row in policies},
        "selection_agreement": {
            "group_free_equals_validation_risk_minimizer": int(
                sum(
                    np.isclose(
                        float(context["policies"]["group_free"]["lambda_fg"]),
                        float(context["policies"]["validation_risk_minimizer"]["lambda_fg"]),
                        rtol=0.0,
                        atol=1e-12,
                    )
                    for context in contexts
                )
            ),
            "n_seeds": len(contexts),
        },
        "frontier": frontier,
        "frontier_per_seed": frontier_per_seed,
        "inputs": {
            str(context["seed"]): {"paths": context["paths"], "sha256": context["sha256"]}
            for context in contexts
        },
    }
    aggregate["aggregate_sha256"] = json_artifact_sha256(aggregate, "aggregate_sha256")
    atomic_write_json(args.out_dir / "aggregate.json", aggregate)

    main_rows = _flat_main(policies)
    main_fields = list(main_rows[0])
    _atomic_write_csv(args.out_dir / "main_table.csv", main_rows, main_fields)
    frontier_rows = _flat_frontier(frontier)
    _atomic_write_csv(args.out_dir / "frontier.csv", frontier_rows, list(frontier_rows[0]))
    _atomic_write_text(args.out_dir / "waterbirds_group_free_lambda_table.tex", _latex(aggregate))
    _atomic_write_text(args.out_dir / "report.md", _report(aggregate))
    _atomic_write_text(args.out_dir / "rebuttal_lambda_fg.txt", _rebuttal(aggregate))
    print(f"wrote validated aggregate and reports to {args.out_dir}", flush=True)
    if not complete:
        print(
            "WARNING: output is not the complete reportable five-seed protocol "
            f"(deviations={protocol_deviations})",
            flush=True,
        )


if __name__ == "__main__":
    main()
