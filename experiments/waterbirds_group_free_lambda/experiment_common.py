"""Shared contracts for the group-free Waterbirds lambda-selection audit.

The scientific implementation deliberately reuses the hardened Appendix-B
hidden-slice primitives, while keeping this experiment in a sibling directory
so that existing W3 artifacts are not invalidated.  Blind stages import data
only through ``load_blind_npz``; Waterbirds metadata is opened solely by the
post-hoc stage after every selection and blind test artifact is frozen.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
from scipy.stats import t as student_t


HERE = Path(__file__).resolve().parent
HIDDEN_SLICE_DIR = HERE.parent / "waterbirds_hidden_slice"
if str(HIDDEN_SLICE_DIR) not in sys.path:
    sys.path.insert(0, str(HIDDEN_SLICE_DIR))

# Re-export the hardened utilities used by the stage scripts.  These modules
# contain no Waterbirds metadata loader; the only blind data entry point is the
# exact-schema loader below.
from common import (  # noqa: E402,F401 -- intentional stage-level re-exports
    array_sha256,
    atomic_savez_compressed,
    atomic_write_json,
    canonical_json_bytes,
    canonical_probability_weights,
    configure_torch_determinism,
    derive_seed,
    file_sha256,
    fit_temperature,
    json_artifact_sha256,
    json_sha256,
    load_blind_npz,
    require_json_artifact_sha256,
    stratified_partition_indices,
)

# The stage scripts import these hardened primitives through this module so
# that the sibling-path bootstrap and provenance boundary live in one place.
__all__ = [
    "array_sha256",
    "atomic_savez_compressed",
    "atomic_write_json",
    "canonical_json_bytes",
    "canonical_probability_weights",
    "configure_torch_determinism",
    "derive_seed",
    "file_sha256",
    "fit_temperature",
    "json_artifact_sha256",
    "json_sha256",
    "load_blind_npz",
    "require_json_artifact_sha256",
    "stratified_partition_indices",
]


SCHEMA_VERSION = 2
ALGORITHM_VERSION = "group-free-lambda-selection-2.0.0"
FROZEN_ERM_REFERENCE_KEY = "frozen_erm_reference"
DEFAULT_SEEDS = (17, 18, 19, 20, 21)
DEFAULT_LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3)
SEALED_KEYS = {"row_id", "background", "group", "img_filename", "split"}
GROUP_NAMES = {
    0: "landbird/land",
    1: "landbird/water",
    2: "waterbird/land",
    3: "waterbird/water",
}

DEPENDENCY_FILES = (
    "blind_protocol.py",
    "common.py",
    "compat.py",
    "gates.py",
    "geometry.py",
    "heads.py",
    "metrics.py",
)


def read_json(path: str | Path) -> dict[str, Any]:
    source = Path(path)
    return json.loads(source.read_text(encoding="utf-8"))


def lambda_key(value: float) -> str:
    """Stable, filesystem/NPZ-safe identifier for a nonnegative coefficient."""

    value = float(value)
    if not np.isfinite(value) or value < 0.0:
        raise ValueError("lambda must be finite and nonnegative")
    text = f"{value:.12g}".replace("-", "m").replace(".", "p").replace("+", "")
    return f"lambda_{text}"


def validate_lambdas(values: Iterable[float], *, fixed_lambda: float = 0.03) -> list[float]:
    result = sorted({float(value) for value in values})
    if not result or any(not np.isfinite(value) or value < 0.0 for value in result):
        raise ValueError("the lambda grid must contain finite nonnegative values")
    if result[0] != 0.0:
        raise ValueError("the lambda grid must include the matched ERM continuation anchor lambda=0")
    if not any(np.isclose(value, fixed_lambda, rtol=0.0, atol=1e-12) for value in result):
        raise ValueError(f"the lambda grid must include the fixed reference {fixed_lambda:g}")
    if len({lambda_key(value) for value in result}) != len(result):
        raise ValueError("the lambda grid does not have unique serialized keys")
    return result


def _canonical_source_hash(path: Path) -> str:
    source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
    return hashlib.sha256(source).hexdigest()


def implementation_manifest() -> dict[str, Any]:
    """Fingerprint this experiment and every imported scientific dependency."""

    files: dict[str, str] = {}
    for path in sorted(HERE.iterdir(), key=lambda item: item.name):
        if path.is_file() and path.suffix.lower() in {".py", ".sh"} and not path.name.endswith(".bak"):
            files[f"waterbirds_group_free_lambda/{path.name}"] = _canonical_source_hash(path)
    for name in DEPENDENCY_FILES:
        path = HIDDEN_SLICE_DIR / name
        if not path.is_file():
            raise FileNotFoundError(path)
        files[f"waterbirds_hidden_slice/{name}"] = _canonical_source_hash(path)
    if not files:
        raise RuntimeError("no scientific source files were found")
    identity = {"algorithm_version": ALGORITHM_VERSION, "files": files}
    return {**identity, "combined_sha256": json_sha256(identity)}


def require_current_implementation(recorded: Any, *, label: str) -> dict[str, Any]:
    current = implementation_manifest()
    if not isinstance(recorded, Mapping):
        raise RuntimeError(f"{label} has no scientific implementation manifest")
    if recorded.get("algorithm_version") != current["algorithm_version"]:
        raise RuntimeError(f"{label} uses a different algorithm version")
    if recorded.get("combined_sha256") != current["combined_sha256"]:
        raise RuntimeError(
            f"{label} was generated by different scientific source code; use a fresh output directory"
        )
    return current


def blind_input_hashes(data_dir: str | Path, *, include_test: bool = False) -> dict[str, str]:
    root = Path(data_dir)
    names = ["train_blind.npz", "val_blind.npz", "preparation_manifest.json"]
    if include_test:
        names.append("test_blind.npz")
    return {Path(name).stem: file_sha256(root / name) for name in names}


def task_metrics(logits: np.ndarray, labels: np.ndarray, temperature: float) -> dict[str, Any]:
    """Task-label-only metrics used by the blind validation selector."""

    from heads import per_example_loss  # Imported lazily after path bootstrap.

    values = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    if values.shape != y.shape or values.ndim != 1:
        raise ValueError("logits and labels must be aligned vectors")
    losses = per_example_loss(values, y, temperature)
    prediction = (values >= 0.0).astype(np.int64)
    per_class: dict[str, float] = {}
    for class_id in (0, 1):
        mask = y == class_id
        if not np.any(mask):
            raise ValueError(f"task class {class_id} is absent")
        per_class[str(class_id)] = float(np.mean(prediction[mask] == y[mask]))
    return {
        "n": int(y.size),
        "nll": float(losses.mean()),
        "nll_standard_error": float(losses.std(ddof=1) / np.sqrt(losses.size)) if losses.size > 1 else 0.0,
        "accuracy": float(np.mean(prediction == y)),
        "balanced_accuracy": float(np.mean(list(per_class.values()))),
        "minimum_class_accuracy": float(min(per_class.values())),
        "per_class_accuracy": per_class,
    }


def risk_certificate_choice(candidates: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Apply the frozen one-SE-risk then certificate-score selection rule."""

    rows = list(candidates)
    if not rows:
        raise ValueError("candidate list must not be empty")

    def values(row: Mapping[str, Any]) -> tuple[float, float, float, float]:
        metrics = row.get("validation_metrics")
        scores = row.get("certificate_scores")
        if not isinstance(metrics, Mapping) or not isinstance(scores, Mapping):
            raise ValueError("candidate lacks validation metrics or certificate scores")
        coefficient = float(row["lambda_fg"])
        nll = float(metrics["nll"])
        standard_error = float(metrics["nll_standard_error"])
        score = float(scores["simultaneous_certificate_lower_score"])
        if not np.all(np.isfinite([coefficient, nll, standard_error, score])):
            raise ValueError("selection inputs must be finite")
        if coefficient < 0.0 or standard_error < 0.0 or score < 0.0:
            raise ValueError("lambda, standard error, and certificate score must be nonnegative")
        return coefficient, nll, standard_error, score

    parsed = [(row, values(row)) for row in rows]
    risk_minimizer, risk_values = min(parsed, key=lambda pair: (pair[1][1], pair[1][0]))
    risk_ceiling = float(risk_values[1] + risk_values[2])
    eligible = [row for row, row_values in parsed if row_values[1] <= risk_ceiling + 1e-15]
    selected = max(
        eligible,
        key=lambda row: (
            float(row["certificate_scores"]["simultaneous_certificate_lower_score"]),
            -float(row["lambda_fg"]),
        ),
    )
    return {
        "risk_minimizer": risk_minimizer,
        "risk_ceiling": risk_ceiling,
        "eligible": eligible,
        "selected": selected,
    }


def one_sided_mean_bound(
    values: Iterable[float],
    *,
    confidence: float,
    comparisons: int = 1,
    side: str = "upper",
) -> dict[str, float | int]:
    """One-sided Student-t bound, optionally Bonferroni-simultaneous."""

    array = np.asarray(list(values), dtype=np.float64)
    if array.ndim != 1 or array.size == 0 or not np.all(np.isfinite(array)):
        raise ValueError("values must be a non-empty finite vector")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must lie in (0, 1)")
    if comparisons < 1:
        raise ValueError("comparisons must be positive")
    if side not in {"upper", "lower"}:
        raise ValueError("side must be 'upper' or 'lower'")
    mean = float(array.mean())
    se = float(array.std(ddof=1) / np.sqrt(array.size)) if array.size > 1 else 0.0
    adjusted_tail = (1.0 - confidence) / float(comparisons)
    critical = float(student_t.ppf(1.0 - adjusted_tail, df=array.size - 1)) if array.size > 1 else 0.0
    endpoint = mean + critical * se if side == "upper" else mean - critical * se
    return {
        "n": int(array.size),
        "mean": mean,
        "standard_error": se,
        "confidence": float(confidence),
        "comparisons": int(comparisons),
        "critical_value": critical,
        side: float(endpoint),
    }


def crossfit_certificate_scores(
    crossfit: Mapping[str, Any],
    *,
    confidence: float,
    comparisons: int,
) -> dict[str, Any]:
    """Aggregate held-out fold inverse Fishers into conservative ranking scores.

    The Student-t-shaped endpoints are deliberately called scores rather than
    confidence bounds: cross-fit gate-training sets overlap, so the five fold
    estimates do not satisfy an independent-fold coverage argument.
    """

    folds = crossfit.get("folds")
    if not isinstance(folds, list) or len(folds) < 2:
        raise ValueError("crossfit artifact must contain at least two folds")
    inverse = np.asarray(
        [row["heldout"]["predictive"]["inverse_fisher"] for row in folds],
        dtype=np.float64,
    )
    if np.any(~np.isfinite(inverse)) or np.any(inverse <= 0.0):
        raise ValueError("held-out inverse Fisher values must be finite and positive")
    marginal = one_sided_mean_bound(inverse, confidence=confidence, comparisons=1, side="upper")
    simultaneous = one_sided_mean_bound(
        inverse,
        confidence=confidence,
        comparisons=comparisons,
        side="upper",
    )

    def certificate(bound: Mapping[str, Any]) -> float:
        upper = float(bound["upper"])
        return 0.0 if upper <= 0.0 else float(1.0 / upper)

    return {
        "fold_inverse_fisher": inverse.tolist(),
        "point_certificate": float(1.0 / inverse.mean()),
        "marginal_inverse_upper_score": marginal,
        "marginal_certificate_lower_score": certificate(marginal),
        "simultaneous_inverse_upper_score": simultaneous,
        "simultaneous_certificate_lower_score": certificate(simultaneous),
        "interpretation": (
            "Student-t-shaped aggregation of held-out fold inverse Fisher; the reciprocal is a conservative "
            "cross-fit certificate ranking score. Bonferroni adjustment is applied over the predeclared "
            "lambda grid. Because gate-training folds overlap, this is not asserted to be a formal "
            "independent-fold confidence bound."
        ),
    }


def paired_inverse_diagnostic(
    candidate_crossfit: Mapping[str, Any],
    baseline_crossfit: Mapping[str, Any],
    *,
    confidence: float,
    comparisons: int,
) -> dict[str, Any]:
    """Paired fold ranking diagnostic; negative inverse-Fisher difference is favorable."""

    candidate = np.asarray(
        [row["heldout"]["predictive"]["inverse_fisher"] for row in candidate_crossfit["folds"]],
        dtype=np.float64,
    )
    baseline = np.asarray(
        [row["heldout"]["predictive"]["inverse_fisher"] for row in baseline_crossfit["folds"]],
        dtype=np.float64,
    )
    if candidate.shape != baseline.shape or candidate.size < 2:
        raise ValueError("candidate and baseline must use the same cross-fit fold count")
    difference = candidate - baseline
    marginal = one_sided_mean_bound(difference, confidence=confidence, comparisons=1, side="upper")
    simultaneous = one_sided_mean_bound(
        difference,
        confidence=confidence,
        comparisons=comparisons,
        side="upper",
    )
    return {
        "fold_difference_candidate_minus_baseline": difference.tolist(),
        "marginal_upper_score": marginal,
        "bonferroni_upper_score": simultaneous,
        "diagnostic_upper_score_below_zero": bool(float(simultaneous["upper"]) < 0.0),
        "favorable_direction": "negative",
        "interpretation": (
            "Diagnostic only. Overlapping cross-fit gate-training sets mean the fold aggregation is not a "
            "formal independent-fold significance test."
        ),
    }


def group_metrics(logits: np.ndarray, labels: np.ndarray, groups: np.ndarray) -> dict[str, Any]:
    values = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=np.int64)
    g = np.asarray(groups, dtype=np.int64)
    if values.shape != y.shape or y.shape != g.shape:
        raise ValueError("logits, labels, and groups must align")
    prediction = (values >= 0.0).astype(np.int64)
    rows = []
    for group_id in range(4):
        mask = g == group_id
        if not np.any(mask):
            raise ValueError(f"Waterbirds group {group_id} is absent")
        rows.append(
            {
                "group_id": group_id,
                "group": GROUP_NAMES[group_id],
                "n": int(mask.sum()),
                "accuracy": float(np.mean(prediction[mask] == y[mask])),
            }
        )
    return {
        "worst_group_accuracy": float(min(row["accuracy"] for row in rows)),
        "groups": rows,
    }


def load_sealed_metadata(path: str | Path, *, expected_sha256: str) -> dict[str, np.ndarray]:
    source = Path(path)
    if file_sha256(source) != expected_sha256:
        raise RuntimeError("sealed metadata hash does not match the preparation manifest")
    with np.load(source, allow_pickle=False) as archive:
        if set(archive.files) != SEALED_KEYS:
            raise ValueError(f"unexpected sealed metadata schema: {archive.files}")
        result = {key: np.asarray(archive[key]) for key in archive.files}
    row_id = np.asarray(result["row_id"], dtype=np.int64)
    if np.unique(row_id).size != row_id.size:
        raise ValueError("sealed metadata row ids are not unique")
    return result


def join_sealed(row_id: np.ndarray, sealed: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    lookup = {int(value): index for index, value in enumerate(np.asarray(sealed["row_id"]))}
    try:
        positions = np.asarray([lookup[int(value)] for value in np.asarray(row_id)], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"blind row id is absent from sealed metadata: {exc}") from exc
    return {key: np.asarray(value)[positions] for key, value in sealed.items() if key != "row_id"}


def verify_preparation_artifact(data_dir: str | Path, name: str) -> str:
    """Require an artifact hash recorded by the preparation manifest."""

    root = Path(data_dir)
    manifest = read_json(root / "preparation_manifest.json")
    path = root / name
    actual = file_sha256(path)
    expected = manifest.get("artifacts", {}).get(Path(name).stem, {}).get("sha256")
    if expected is None:
        # The W3 manifest keys normally omit the extension, but accept the
        # explicit filename for forward compatibility.
        expected = manifest.get("artifacts", {}).get(name, {}).get("sha256")
    if actual != expected:
        raise RuntimeError(f"{name} does not match preparation_manifest.json")
    return actual


def canonical_protocol_hash(payload: Mapping[str, Any]) -> str:
    """A named alias used in stage artifacts for readability."""

    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()
