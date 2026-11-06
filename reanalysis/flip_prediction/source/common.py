"""Shared utilities for the prospective Waterbirds nuisance-shift experiment.

Everything in this folder is self-contained: it does not import from the other
experiment scripts so that the frozen artifacts here can be re-run independently.
"""

from __future__ import annotations

import csv
import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

GROUP_NAMES = {
    0: "landbird/land",
    1: "landbird/water",
    2: "waterbird/land",
    3: "waterbird/water",
}
BACKGROUND_NAMES = {0: "land", 1: "water"}
SPLIT_NAMES = {0: "train", 1: "val", 2: "test"}


# --------------------------------------------------------------------------- #
# Metadata
# --------------------------------------------------------------------------- #
@dataclass
class Metadata:
    img_id: np.ndarray
    img_filename: np.ndarray
    y: np.ndarray
    split: np.ndarray
    place: np.ndarray
    place_filename: np.ndarray

    def __len__(self) -> int:
        return int(self.y.size)

    @property
    def group(self) -> np.ndarray:
        return group_ids(self.y, self.place)

    @property
    def species(self) -> np.ndarray:
        """CUB class directory, e.g. ``001.Black_footed_Albatross``."""
        return np.array([name.split("/")[0] for name in self.img_filename])

    def rows_for_split(self, split: int) -> np.ndarray:
        return np.flatnonzero(self.split == split)


def load_metadata(path: str | Path = "data/wilds/waterbirds_v1.0/metadata.csv") -> Metadata:
    rows = list(csv.DictReader(Path(path).open(encoding="utf-8")))
    return Metadata(
        img_id=np.array([int(r["img_id"]) for r in rows], dtype=np.int64),
        img_filename=np.array([r["img_filename"] for r in rows]),
        y=np.array([int(r["y"]) for r in rows], dtype=np.int64),
        split=np.array([int(r["split"]) for r in rows], dtype=np.int64),
        place=np.array([int(r["place"]) for r in rows], dtype=np.int64),
        place_filename=np.array([r["place_filename"] for r in rows]),
    )


def group_ids(y: np.ndarray, place: np.ndarray) -> np.ndarray:
    return (2 * np.asarray(y).astype(int) + np.asarray(place).astype(int)).astype(int)


def image_path(root: str | Path, img_filename: str) -> Path:
    return Path(root) / img_filename


def mask_path(root: str | Path, img_filename: str) -> Path:
    return Path(root) / img_filename.replace(".jpg", ".png")


# --------------------------------------------------------------------------- #
# Fisher-Glass geometry
# --------------------------------------------------------------------------- #
def cvar(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    threshold = np.quantile(values, alpha)
    return float(values[values >= threshold].mean())


def fisher_schur(
    z: np.ndarray,
    b: np.ndarray,
    ridge: float,
    weight: np.ndarray | None = None,
) -> dict[str, float]:
    """2x2 Fisher block with the nuisance direction projected out (Schur complement).

    ``z`` is the task coordinate (standardized logit), ``b`` the nuisance
    coordinate (standardized background-probe score).  ``weight`` optionally
    applies the predictive weighting p(1-p); ``None`` gives the unweighted
    feature-Schur score.
    """
    z = np.asarray(z, dtype=float)
    b = np.asarray(b, dtype=float)
    if weight is None:
        weight = np.ones_like(z, dtype=float)
    weight = np.asarray(weight, dtype=float)
    weight = weight / max(float(weight.sum()), 1e-12)
    j_tt = float(np.sum(weight * z * z))
    j_tn = float(np.sum(weight * z * b))
    j_nn = float(np.sum(weight * b * b))
    projected = max(j_tt - (j_tn * j_tn) / (j_nn + ridge), 0.0)
    return {
        "raw": j_tt,
        "projected": projected,
        "projected_over_raw": float(projected / j_tt) if j_tt > 1e-12 else float("nan"),
        "j_tn": j_tn,
        "j_nn": j_nn,
        "beta": float(j_tn / (j_nn + ridge)),
    }


def sigmoid(logits: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logits, dtype=float), -60.0, 60.0)))


def binary_nll(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = sigmoid(logits)
    y = np.asarray(y, dtype=float)
    return -(y * np.log(np.clip(p, 1e-12, 1.0)) + (1.0 - y) * np.log(np.clip(1.0 - p, 1e-12, 1.0)))


def fit_temperature(logits: np.ndarray, y: np.ndarray, grid: np.ndarray | None = None) -> float:
    """Single-parameter temperature scaling, chosen by NLL on the given split."""
    if grid is None:
        grid = np.exp(np.linspace(np.log(0.05), np.log(20.0), 400))
    losses = [float(binary_nll(np.asarray(logits, dtype=float) / t, y).mean()) for t in grid]
    return float(grid[int(np.argmin(losses))])


def standardize(values: np.ndarray, mean: float, std: float) -> np.ndarray:
    std = std if abs(std) > 1e-8 else 1.0
    return ((np.asarray(values, dtype=float) - mean) / std).astype(np.float64)


# --------------------------------------------------------------------------- #
# Resampling
# --------------------------------------------------------------------------- #
def cluster_bootstrap(
    cluster_key: np.ndarray,
    statistic,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict[str, float]:
    """Cluster bootstrap: resample whole clusters (e.g. one source bird) with replacement.

    ``statistic`` maps an array of row indices to a float.  Independent draws of
    the same bird are not independent observations, so they have to move
    together; the species-level variant additionally absorbs the fact that many
    Waterbirds images share a CUB species.

    The resampled row index is built with a vectorised ragged gather rather than
    per-cluster concatenation, which matters because this is called a few
    hundred times per model.
    """
    cluster_key = np.asarray(cluster_key)
    n = int(cluster_key.size)
    point = float(statistic(np.arange(n))) if n else float("nan")
    if n == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan"), "n_clusters": 0}

    _uniq, inverse = np.unique(cluster_key, return_inverse=True)
    n_clusters = int(inverse.max()) + 1
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=n_clusters)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))

    rng = np.random.default_rng(seed)
    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        picked_counts = counts[picks]
        total = int(picked_counts.sum())
        if total == 0:
            draws[i] = np.nan
            continue
        cumulative = np.cumsum(picked_counts)
        segment_offset = np.concatenate(([0], cumulative[:-1]))
        base = np.repeat(starts[picks] - segment_offset, picked_counts)
        draws[i] = statistic(order[base + np.arange(total)])

    finite = draws[np.isfinite(draws)]
    if finite.size == 0:
        return {"point": point, "lo": float("nan"), "hi": float("nan"), "se": float("nan"), "n_clusters": n_clusters}
    return {
        "point": point,
        "lo": float(np.quantile(finite, 0.025)),
        "hi": float(np.quantile(finite, 0.975)),
        "se": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "n_clusters": n_clusters,
    }


def mean_ci(values: Iterable[float]) -> dict[str, float]:
    arr = np.asarray(list(values), dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return {"mean": float("nan"), "ci95": float("nan"), "n": 0}
    if arr.size == 1:
        return {"mean": float(arr[0]), "ci95": 0.0, "n": 1}
    return {
        "mean": float(arr.mean()),
        "ci95": float(1.96 * arr.std(ddof=1) / np.sqrt(arr.size)),
        "n": int(arr.size),
    }


# --------------------------------------------------------------------------- #
# Determinism and provenance
# --------------------------------------------------------------------------- #
def set_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def derive_seed(*parts: Any) -> int:
    """Stable 63-bit seed from arbitrary parts (no dependence on PYTHONHASHSEED)."""
    payload = "|".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.blake2s(payload, digest_size=8).digest(), "little") >> 1


def file_sha256(path: str | Path, chunk: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            block = handle.read(chunk)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def json_sha256(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode("utf-8")).hexdigest()


def write_json(path: str | Path, payload: Any) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")
    return path


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _json_default(obj: Any) -> Any:
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, Path):
        return str(obj)
    raise TypeError(f"not JSON serializable: {type(obj)}")


def log(message: str) -> None:
    import time

    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)
