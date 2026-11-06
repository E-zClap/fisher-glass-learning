"""Shared, group-blind utilities for the Waterbirds hidden-slice experiment.

The central contract in this module is intentionally narrow: code used for
discovery, certification, or blind repair loads data through :class:`BlindSplit`
and therefore cannot receive Waterbirds background or group annotations by
accident.  Post-hoc metadata lives in a separately named, separately hashed
artifact and this loader rejects any archive that contains metadata-like keys.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import random
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


BLIND_KEYS = ("row_id", "h", "y", "b_vl")
FORBIDDEN_BLIND_KEY_TOKENS = ("group", "place", "background", "metadata")
HIDDEN_SLICE_ALGORITHM_VERSION = "2.0.0"
WATERBIRDS_FEATURE_MODEL_ID = "torchvision:mobilenet_v3_small:DEFAULT"


def _readonly_contiguous(array: np.ndarray, dtype: np.dtype) -> np.ndarray:
    # Always copy: a frozen dataclass is not actually immutable if a caller
    # retains a writable alias to one of its NumPy buffers.
    out = np.array(array, dtype=dtype, copy=True, order="C")
    out.setflags(write=False)
    return out


def canonical_probability_weights(
    values: np.ndarray,
    *,
    name: str = "q",
    tolerance: float = 2e-5,
    negative_tolerance: float = 1e-12,
) -> np.ndarray:
    """Validate and canonically normalize projected probability weights.

    Gates are evaluated in the model's float32 dtype. Casting their projected
    weights to float64 preserves a harmless float32 summation residue, so every
    NumPy boundary uses the same explicit tolerance and then renormalizes in
    float64. Grossly malformed, non-finite, or materially negative weights are
    still rejected.
    """

    q = np.asarray(values, dtype=np.float64)
    if q.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional")
    if q.size == 0:
        raise ValueError(f"{name} must not be empty")
    if not np.all(np.isfinite(q)):
        raise ValueError(f"{name} must contain only finite values")
    if np.any(q < -float(negative_tolerance)):
        raise ValueError(f"{name} must be nonnegative")
    total = float(q.sum())
    if not np.isclose(total, 1.0, atol=float(tolerance), rtol=0.0):
        raise ValueError(f"{name} must sum to one, got {total:.17g}")
    q = np.maximum(q, 0.0)
    total = float(q.sum())
    if not np.isfinite(total) or total <= 0.0:
        raise ValueError(f"{name} must have positive finite mass")
    return q / total


@dataclass(frozen=True)
class BlindSplit:
    """A validated split containing no Waterbirds group annotations.

    ``row_id`` is the only join key.  It may be joined to the sealed metadata
    *after* all gates, hyperparameters, certificates, and repair models have
    been frozen, but it carries no group information itself.
    """

    row_id: np.ndarray
    h: np.ndarray
    y: np.ndarray
    b_vl: np.ndarray

    def __post_init__(self) -> None:
        row_id_raw = np.asarray(self.row_id)
        h_raw = np.asarray(self.h)
        y_raw = np.asarray(self.y)
        b_raw = np.asarray(self.b_vl)

        if row_id_raw.ndim != 1:
            raise ValueError(f"row_id must be one-dimensional, got {row_id_raw.shape}")
        if row_id_raw.dtype.kind not in "iu":
            raise TypeError(f"row_id must have an integer dtype, got {row_id_raw.dtype}")
        if h_raw.ndim != 2:
            raise ValueError(f"h must be a two-dimensional feature matrix, got {h_raw.shape}")
        if h_raw.dtype.kind not in "fiu":
            raise TypeError(f"h must be numeric, got {h_raw.dtype}")
        if y_raw.ndim != 1 or y_raw.dtype.kind not in "biu":
            raise TypeError(f"y must be a one-dimensional integer array, got {y_raw.shape}/{y_raw.dtype}")
        if b_raw.ndim != 1 or b_raw.dtype.kind not in "fiu":
            raise TypeError(f"b_vl must be a one-dimensional numeric array, got {b_raw.shape}/{b_raw.dtype}")

        n = int(row_id_raw.size)
        if h_raw.shape[0] != n or y_raw.size != n or b_raw.size != n:
            raise ValueError(
                "blind arrays must have matching first dimensions: "
                f"row_id={n}, h={h_raw.shape[0]}, y={y_raw.size}, b_vl={b_raw.size}"
            )
        if n == 0:
            raise ValueError("a BlindSplit cannot be empty")
        if h_raw.shape[1] == 0:
            raise ValueError("h must contain at least one feature")
        if np.unique(row_id_raw).size != n:
            raise ValueError("row_id values must be unique within a blind split")
        if np.any(row_id_raw < 0):
            raise ValueError("row_id values must be nonnegative")
        if not np.all(np.isfinite(h_raw)):
            raise ValueError("h contains NaN or infinite values")
        if not np.all(np.isfinite(b_raw)):
            raise ValueError("b_vl contains NaN or infinite values")
        labels = np.unique(y_raw)
        if not np.all(np.isin(labels, (0, 1))):
            raise ValueError(f"y must be binary, found labels {labels.tolist()}")

        object.__setattr__(self, "row_id", _readonly_contiguous(row_id_raw, np.dtype(np.int64)))
        object.__setattr__(self, "h", _readonly_contiguous(h_raw, np.dtype(np.float32)))
        object.__setattr__(self, "y", _readonly_contiguous(y_raw, np.dtype(np.int64)))
        object.__setattr__(self, "b_vl", _readonly_contiguous(b_raw, np.dtype(np.float32)))

    def __len__(self) -> int:
        return int(self.row_id.size)

    @property
    def feature_dim(self) -> int:
        return int(self.h.shape[1])

    def arrays(self) -> dict[str, np.ndarray]:
        """Return exactly the public blind schema, with no auxiliary fields."""

        return {key: getattr(self, key) for key in BLIND_KEYS}

    def take(self, indices: np.ndarray | Sequence[int]) -> "BlindSplit":
        index = np.asarray(indices)
        if index.ndim != 1 or index.dtype.kind not in "biu":
            raise TypeError("BlindSplit.take expects a one-dimensional integer or boolean index")
        return BlindSplit(**{key: value[index] for key, value in self.arrays().items()})

    def save(self, path: str | Path) -> Path:
        return atomic_savez_compressed(path, **self.arrays())


def _validate_blind_archive_keys(keys: Iterable[str], path: Path) -> None:
    actual = tuple(keys)
    lowered = {key: key.lower() for key in actual}
    forbidden = [
        key
        for key, lower in lowered.items()
        if any(token in lower for token in FORBIDDEN_BLIND_KEY_TOKENS)
    ]
    if forbidden:
        raise ValueError(
            f"refusing to load {path}: metadata-like key(s) in blind archive: {sorted(forbidden)}"
        )
    missing = sorted(set(BLIND_KEYS).difference(actual))
    extra = sorted(set(actual).difference(BLIND_KEYS))
    if missing or extra:
        raise ValueError(
            f"blind archive {path} must contain exactly {list(BLIND_KEYS)}; "
            f"missing={missing}, extra={extra}"
        )


def load_blind_npz(path: str | Path) -> BlindSplit:
    """Load a blind archive, rejecting metadata keys before reading arrays."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)
    with np.load(source, allow_pickle=False) as archive:
        _validate_blind_archive_keys(archive.files, source)
        arrays = {key: np.asarray(archive[key]) for key in BLIND_KEYS}
    return BlindSplit(**arrays)


def _json_default(value: Any) -> Any:
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"not JSON serializable: {type(value)!r}")


def canonical_json_bytes(payload: Any) -> bytes:
    return json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    ).encode("utf-8")


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def json_artifact_sha256(payload: Mapping[str, Any], digest_key: str) -> str:
    """Hash a JSON artifact while excluding its own digest field.

    Stages use this to make accidental mutation, stale resume, and mixed-run
    artifacts fail closed.  It is an integrity checksum, not a signature.
    """
    unhashed = dict(payload)
    unhashed.pop(digest_key, None)
    return json_sha256(unhashed)


def require_json_artifact_sha256(payload: Mapping[str, Any], digest_key: str, *, label: str) -> str:
    expected = payload.get(digest_key)
    if not isinstance(expected, str) or len(expected) != 64:
        raise RuntimeError(f"{label} is missing {digest_key}")
    actual = json_artifact_sha256(payload, digest_key)
    if expected != actual:
        raise RuntimeError(f"{label} content hash mismatch")
    return expected


def file_sha256(path: str | Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(chunk_size):
            digest.update(block)
    return digest.hexdigest()


def scientific_implementation_manifest(directory: str | Path | None = None) -> dict[str, Any]:
    """Return a source fingerprint that invalidates stale resumed artifacts.

    Earlier artifacts keyed resume only by data and command-line settings.  A
    scientific-code change could therefore silently reuse results generated by
    an older optimizer.  The manifest intentionally covers every stage module
    and launcher in this experiment directory, while excluding tests, bytecode,
    and editor/backup files.
    """

    root = Path(__file__).resolve().parent if directory is None else Path(directory).resolve()
    files: dict[str, str] = {}
    for path in sorted(root.iterdir(), key=lambda item: item.name):
        if not path.is_file() or path.suffix.lower() not in {".py", ".sh", ".ps1"}:
            continue
        if path.name.endswith(".bak"):
            continue
        # Git may materialize text with CRLF on Windows and LF on Linux.  The
        # executable source is otherwise identical, so canonicalize line
        # endings before hashing while retaining every other byte exactly.
        source = path.read_bytes().replace(b"\r\n", b"\n").replace(b"\r", b"\n")
        files[path.name] = hashlib.sha256(source).hexdigest()
    if not files:
        raise RuntimeError(f"no hidden-slice implementation files found under {root}")
    identity = {
        "algorithm_version": HIDDEN_SLICE_ALGORITHM_VERSION,
        "files": files,
    }
    return {
        **identity,
        "combined_sha256": json_sha256(identity),
    }


def require_current_implementation_manifest(
    recorded: Any,
    *,
    label: str,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Reject a missing, stale, or unknown experiment-code manifest."""

    if not isinstance(recorded, dict):
        raise RuntimeError(
            f"{label} has no scientific implementation manifest; "
            "legacy hidden-slice artifacts are not scientifically reusable"
        )
    current = scientific_implementation_manifest(directory)
    if recorded.get("algorithm_version") != current["algorithm_version"]:
        raise RuntimeError(
            f"{label} uses hidden-slice algorithm version "
            f"{recorded.get('algorithm_version')!r}, expected {current['algorithm_version']!r}"
        )
    if recorded.get("combined_sha256") != current["combined_sha256"]:
        raise RuntimeError(
            f"{label} was produced by different scientific source code; "
            "run the corrected pipeline in a fresh output directory"
        )
    return current


def require_current_scientific_implementation(
    discovery: dict[str, Any],
    *,
    label: str,
    directory: str | Path | None = None,
) -> dict[str, Any]:
    """Require the current manifest embedded in a discovery artifact.

    Every downstream stage ultimately depends on discovery.  It must validate
    the source manifest itself rather than relying on launcher ordering, which
    makes direct stage calls fail closed on the invalid pre-v2 artifacts.
    """

    return require_current_implementation_manifest(
        discovery.get("protocol", {}).get("implementation"),
        label=label,
        directory=directory,
    )


def array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("ascii"))
    digest.update(canonical_json_bytes(list(value.shape)))
    digest.update(memoryview(value).cast("B"))
    return digest.hexdigest()


def atomic_write_json(path: str | Path, payload: Any) -> Path:
    """Atomically replace a JSON artifact on the destination filesystem."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        payload,
        indent=2,
        sort_keys=True,
        ensure_ascii=False,
        default=_json_default,
        allow_nan=False,
    ) + "\n"
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def write_json(path: str | Path, payload: Any) -> Path:
    """Compatibility name; JSON writes remain atomic."""

    return atomic_write_json(path, payload)


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def atomic_savez_compressed(path: str | Path, **arrays: np.ndarray) -> Path:
    """Atomically write a compressed NumPy archive without pickle objects."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    safe_arrays = {}
    for name, value in arrays.items():
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"refusing to write object array {name!r} to {destination}")
        safe_arrays[name] = array
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".npz",
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        np.savez_compressed(temporary, **safe_arrays)
        with temporary.open("r+b") as handle:
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return destination


def derive_seed(*parts: Any) -> int:
    """Derive a stable, platform-independent 63-bit seed."""

    digest = hashlib.blake2b(canonical_json_bytes(parts), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) & ((1 << 63) - 1)


def set_seed(seed: int, deterministic: bool = False) -> None:
    """Seed Python, NumPy, and Torch (when installed)."""

    if seed < 0:
        raise ValueError("seed must be nonnegative")
    if deterministic:
        # Must be present before the first CUDA BLAS workspace is created.
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed % (2**32))
    try:
        import torch
    except ImportError:
        return
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True)
        if torch.backends.cudnn.is_available():
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


def configure_torch_determinism(enabled: bool = True) -> None:
    """Configure deterministic Torch kernels before any CUDA work starts."""
    if enabled:
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    try:
        import torch
    except ImportError:
        return
    torch.use_deterministic_algorithms(bool(enabled))
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = bool(enabled)


def sigmoid(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    return 1.0 / (1.0 + np.exp(-np.clip(values, -60.0, 60.0)))


def binary_nll(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(y).reshape(-1)
    if values.shape != labels.shape:
        raise ValueError("logits and y must have matching shapes")
    if not np.all(np.isfinite(values)):
        raise ValueError("logits must be finite")
    if not np.all(np.isin(np.unique(labels), (0, 1))):
        raise ValueError("temperature calibration requires binary labels")
    return np.logaddexp(0.0, values) - labels.astype(np.float64) * values


def fit_temperature(
    logits: np.ndarray,
    y: np.ndarray,
    grid: np.ndarray | None = None,
) -> float:
    """Fit a scalar temperature by deterministic held-out NLL grid search."""

    values = np.asarray(logits, dtype=np.float64).reshape(-1)
    labels = np.asarray(y).reshape(-1)
    if grid is None:
        grid = np.exp(np.linspace(math.log(0.05), math.log(20.0), 400))
    candidates = np.asarray(grid, dtype=np.float64).reshape(-1)
    if candidates.size == 0 or np.any(~np.isfinite(candidates)) or np.any(candidates <= 0.0):
        raise ValueError("temperature grid must contain finite positive values")
    losses = np.array([binary_nll(values / value, labels).mean() for value in candidates])
    return float(candidates[int(np.argmin(losses))])


def stratified_partition_indices(
    y: np.ndarray,
    fractions: Mapping[str, float],
    seed: int,
    *,
    require_each_stratum: bool = True,
) -> dict[str, np.ndarray]:
    """Partition positional indices while preserving every observed stratum.

    Counts are allocated independently within each unique value of ``y`` using
    a deterministic largest-remainder rule.  The returned partitions are
    disjoint and cover every row exactly once.
    """

    labels = np.asarray(y).reshape(-1)
    if labels.size == 0:
        raise ValueError("cannot partition an empty label array")
    names = list(fractions)
    if not names:
        raise ValueError("at least one named partition is required")
    proportions = np.asarray([fractions[name] for name in names], dtype=np.float64)
    if np.any(~np.isfinite(proportions)) or np.any(proportions <= 0.0):
        raise ValueError("all partition fractions must be finite and positive")
    if not np.isclose(float(proportions.sum()), 1.0, atol=1e-10, rtol=0.0):
        raise ValueError(f"partition fractions must sum to one, got {proportions.sum():.12g}")

    buckets: dict[str, list[np.ndarray]] = {name: [] for name in names}
    for stratum in np.unique(labels):
        indices = np.flatnonzero(labels == stratum)
        rng = np.random.default_rng(derive_seed(seed, "stratum", str(stratum)))
        rng.shuffle(indices)
        if require_each_stratum and indices.size < len(names):
            raise ValueError(
                f"stratum {stratum!r} has {indices.size} rows, fewer than {len(names)} partitions"
            )

        reserved = np.ones(len(names), dtype=np.int64) if require_each_stratum else np.zeros(len(names), dtype=np.int64)
        remaining = int(indices.size - reserved.sum())
        raw = proportions * remaining
        counts = reserved + np.floor(raw).astype(np.int64)
        leftover = int(indices.size - counts.sum())
        remainder_order = np.argsort(-(raw - np.floor(raw)), kind="stable")
        counts[remainder_order[:leftover]] += 1

        start = 0
        if len(names) != len(counts):
            raise RuntimeError("internal stratified allocation length mismatch")
        for name, count in zip(names, counts):
            stop = start + int(count)
            buckets[name].append(indices[start:stop])
            start = stop
        if start != indices.size:
            raise RuntimeError("internal stratified allocation error")

    output: dict[str, np.ndarray] = {}
    for name in names:
        joined = np.concatenate(buckets[name]).astype(np.int64, copy=False)
        rng = np.random.default_rng(derive_seed(seed, "partition", name))
        rng.shuffle(joined)
        output[name] = joined
    assert_disjoint_complete(output.values(), labels.size)
    return output


def stratified_kfold_indices(y: np.ndarray, n_splits: int, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return deterministic ``(fit, held_out)`` positional indices."""

    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    fractions = {f"fold_{index}": 1.0 / n_splits for index in range(n_splits)}
    partitions = stratified_partition_indices(y, fractions, seed, require_each_stratum=True)
    all_indices = np.arange(np.asarray(y).size, dtype=np.int64)
    folds = []
    for name in fractions:
        held_out = partitions[name]
        fit = np.setdiff1d(all_indices, held_out, assume_unique=False)
        folds.append((fit, held_out))
    return folds


def assert_disjoint_complete(parts: Iterable[np.ndarray], total_size: int) -> None:
    arrays = [np.asarray(part, dtype=np.int64).reshape(-1) for part in parts]
    joined = np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
    if joined.size != total_size:
        raise ValueError(f"partitions contain {joined.size} rows; expected {total_size}")
    if np.any(joined < 0) or np.any(joined >= total_size):
        raise ValueError("partition index is outside the valid range")
    if np.unique(joined).size != total_size:
        raise ValueError("partitions overlap or omit rows")
