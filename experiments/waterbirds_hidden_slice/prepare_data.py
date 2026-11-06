"""Prepare the sealed and group-blind Waterbirds artifacts.

This is the only stage allowed to read background metadata from the raw WILDS
dataset.  It writes three discovery-facing archives whose schema is exactly
``row_id,h,y,b_vl`` and a separate ``sealed_metadata.npz`` for post-hoc
interpretation; only ``posthoc.py`` may later open that sealed file.  The
nuisance coordinate is computed without metadata: it is
the cosine score along a fixed CLIP water-background-minus-land-background text
direction.

Model downloads are disabled by default.  Pass ``--allow-model-download``
explicitly if the ImageNet MobileNet or Hugging Face CLIP weights are not
already cached on the execution machine.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import importlib.metadata
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from tqdm import tqdm
from wilds import get_dataset

from compat import BooleanOptionalAction
from common import (
    BLIND_KEYS,
    WATERBIRDS_FEATURE_MODEL_ID,
    BlindSplit,
    array_sha256,
    atomic_savez_compressed,
    atomic_write_json,
    file_sha256,
    json_sha256,
    set_seed,
)


FEATURE_DIM = 576
FEATURE_MODEL_ID = WATERBIRDS_FEATURE_MODEL_ID
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

# Fixed before any Waterbirds score is inspected.  Prompts mention only the
# audited nuisance semantics, never Waterbirds group names or labels.
WATER_BACKGROUND_PROMPTS = (
    "a photo with a water background",
    "a natural scene with ocean water in the background",
    "a natural scene with lake or river water in the background",
    "a photo taken over open water",
    "a watery landscape behind the main subject",
)
LAND_BACKGROUND_PROMPTS = (
    "a photo with a land background",
    "a natural scene with forest in the background",
    "a natural scene with grass or ground in the background",
    "a photo taken over dry land",
    "a terrestrial landscape behind the main subject",
)


def prompt_specification() -> dict[str, object]:
    return {
        "water": list(WATER_BACKGROUND_PROMPTS),
        "land": list(LAND_BACKGROUND_PROMPTS),
        "ensemble": "L2-normalize each prompt embedding, average within concept, L2-normalize each concept mean",
        "direction": "L2-normalized water concept mean minus land concept mean; positive means more water-like",
    }


def prompt_sha256() -> str:
    return json_sha256(prompt_specification())


def as_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"CUDA device requested ({name}) but torch.cuda.is_available() is false")
    return device


def mobilenet_checkpoint_path() -> Path:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    filename = Path(urlparse(weights.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def _autocast(device: torch.device, enabled: bool):
    if not enabled or device.type != "cuda":
        return contextlib.nullcontext()
    dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    return torch.autocast(device_type="cuda", dtype=dtype)


def _loader_kwargs(num_workers: int, pin_memory: bool) -> dict[str, object]:
    if num_workers < 0:
        raise ValueError("num_workers must be nonnegative")
    kwargs: dict[str, object] = {
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    if num_workers:
        kwargs.update({"persistent_workers": True, "prefetch_factor": 2})
    return kwargs


class _MobileNetImageDataset(Dataset):
    def __init__(self, dataset, transform) -> None:
        self.dataset = dataset
        self.transform = transform

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _y, _metadata = self.dataset[index]
        return self.transform(image), int(index)


class _RawImageDataset(Dataset):
    """Image-only view: task/background metadata never enters CLIP scoring."""

    def __init__(self, dataset) -> None:
        self.dataset = dataset

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, index: int):
        image, _y, _metadata = self.dataset[index]
        if not isinstance(image, Image.Image):
            raise TypeError(f"expected a PIL image from WILDS, got {type(image)!r}")
        return image.convert("RGB"), int(index)


def _collate_pil(batch):
    if any(len(item) != 2 for item in batch):
        raise ValueError("each image batch item must contain exactly (image, index)")
    images, indices = zip(*batch)
    return list(images), torch.tensor(indices, dtype=torch.long)


def _validate_complete_indices(indices: np.ndarray, n_rows: int, label: str) -> np.ndarray:
    values = np.asarray(indices)
    if values.ndim != 1 or values.dtype.kind not in "iu":
        raise ValueError(f"{label} indices must be a one-dimensional integer array")
    values = values.astype(np.int64, copy=False)
    if values.size != n_rows:
        raise ValueError(f"{label} contains {values.size} indices; expected {n_rows}")
    if not np.array_equal(np.sort(values), np.arange(n_rows, dtype=np.int64)):
        raise ValueError(f"{label} indices must contain every row 0..{n_rows - 1} exactly once")
    return values


def load_feature_cache(path: str | Path, n_rows: int) -> np.ndarray:
    source = Path(path)
    with np.load(source, allow_pickle=False) as archive:
        required = {"indices", "features", "feature_model"}
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(
                f"feature cache {source} is missing {sorted(missing)}; "
                "legacy caches without an extractor identity are not accepted"
            )
        indices = _validate_complete_indices(archive["indices"], n_rows, "feature cache")
        features = np.asarray(archive["features"])
        cached_model = _scalar_string(archive, "feature_model")
        if cached_model != FEATURE_MODEL_ID:
            raise ValueError(
                f"feature cache model is {cached_model!r}, expected {FEATURE_MODEL_ID!r}"
            )
    if features.ndim != 2 or features.shape != (n_rows, FEATURE_DIM):
        raise ValueError(
            f"feature cache must have shape {(n_rows, FEATURE_DIM)}, got {features.shape}"
        )
    if features.dtype.kind != "f" or not np.all(np.isfinite(features)):
        raise ValueError("feature cache must contain finite floating-point values")
    aligned = np.empty((n_rows, FEATURE_DIM), dtype=np.float32)
    aligned[indices] = features.astype(np.float32, copy=False)
    return aligned


@torch.inference_mode()
def extract_mobilenet_features(
    dataset,
    cache_path: Path,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    amp: bool,
    allow_model_download: bool,
) -> np.ndarray:
    checkpoint = mobilenet_checkpoint_path()
    if not checkpoint.is_file() and not allow_model_download:
        raise FileNotFoundError(
            "the frozen MobileNetV3-small feature cache is absent and its ImageNet checkpoint is not local: "
            f"{checkpoint}. Re-run with --allow-model-download or pre-stage the checkpoint."
        )

    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights).eval().to(device)
    loader = DataLoader(
        _MobileNetImageDataset(dataset, weights.transforms()),
        batch_size=batch_size,
        shuffle=False,
        **_loader_kwargs(num_workers, pin_memory),
    )
    index_parts: list[np.ndarray] = []
    feature_parts: list[np.ndarray] = []
    for images, indices in tqdm(loader, desc="MobileNetV3 frozen features"):
        images = images.to(device, non_blocking=pin_memory)
        with _autocast(device, amp):
            features = model.features(images)
            features = model.avgpool(features)
            features = torch.flatten(features, 1)
        index_parts.append(indices.numpy())
        feature_parts.append(features.float().cpu().numpy())

    indices = _validate_complete_indices(np.concatenate(index_parts), len(dataset), "extracted features")
    features = np.concatenate(feature_parts).astype(np.float32, copy=False)
    if features.shape != (len(dataset), FEATURE_DIM) or not np.all(np.isfinite(features)):
        raise RuntimeError(f"unexpected extracted MobileNet feature matrix {features.shape}")
    atomic_savez_compressed(
        cache_path,
        indices=indices,
        features=features,
        feature_model=np.asarray(FEATURE_MODEL_ID),
    )
    return load_feature_cache(cache_path, len(dataset))


def get_mobilenet_features(
    dataset,
    cache_path: Path,
    *,
    recompute: bool,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    amp: bool,
    allow_model_download: bool,
) -> tuple[np.ndarray, bool]:
    if cache_path.is_file() and not recompute:
        try:
            return load_feature_cache(cache_path, len(dataset)), True
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid frozen-feature cache {cache_path}: {exc}. "
                "Use --recompute-features to replace it."
            ) from exc
    features = extract_mobilenet_features(
        dataset,
        cache_path,
        device,
        batch_size,
        num_workers,
        pin_memory,
        amp,
        allow_model_download,
    )
    return features, False


def _scalar_string(archive, key: str) -> str:
    value = np.asarray(archive[key])
    if value.ndim != 0 or value.dtype.kind not in "US":
        raise ValueError(f"cache field {key!r} must be a scalar string")
    return str(value.item())


def feature_cache_identity_attested(path: str | Path) -> bool:
    with np.load(Path(path), allow_pickle=False) as archive:
        return (
            "feature_model" in archive.files
            and _scalar_string(archive, "feature_model") == FEATURE_MODEL_ID
        )


def load_nuisance_cache(
    path: str | Path,
    n_rows: int,
    clip_model: str,
    clip_revision: str | None,
) -> np.ndarray:
    source = Path(path)
    required = {"indices", "b_vl_raw", "clip_model", "clip_revision", "prompt_sha256"}
    with np.load(source, allow_pickle=False) as archive:
        missing = required.difference(archive.files)
        if missing:
            raise ValueError(f"nuisance cache {source} is missing {sorted(missing)}")
        indices = _validate_complete_indices(archive["indices"], n_rows, "nuisance cache")
        raw = np.asarray(archive["b_vl_raw"])
        cached_model = _scalar_string(archive, "clip_model")
        cached_revision = _scalar_string(archive, "clip_revision")
        cached_prompt_hash = _scalar_string(archive, "prompt_sha256")
    requested_revision = clip_revision or ""
    if cached_model != clip_model or cached_revision != requested_revision:
        raise ValueError(
            "nuisance cache model mismatch: "
            f"cached={cached_model}@{cached_revision or '<default>'}, "
            f"requested={clip_model}@{requested_revision or '<default>'}"
        )
    if cached_prompt_hash != prompt_sha256():
        raise ValueError("nuisance cache was produced with a different prompt ensemble")
    if raw.shape != (n_rows,) or raw.dtype.kind != "f" or not np.all(np.isfinite(raw)):
        raise ValueError(f"nuisance cache b_vl_raw must be a finite vector of shape {(n_rows,)}")
    aligned = np.empty(n_rows, dtype=np.float32)
    aligned[indices] = raw.astype(np.float32, copy=False)
    return aligned


def nuisance_cache_resolved_revision(path: str | Path) -> str | None:
    with np.load(Path(path), allow_pickle=False) as archive:
        if "resolved_revision" not in archive.files:
            return None
        value = _scalar_string(archive, "resolved_revision")
    return value or None


def _normalize_rows(tensor: torch.Tensor) -> torch.Tensor:
    return tensor / tensor.norm(dim=-1, keepdim=True).clamp_min(1e-12)


@torch.inference_mode()
def extract_clip_nuisance(
    dataset,
    cache_path: Path,
    clip_model: str,
    clip_revision: str | None,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    amp: bool,
    allow_model_download: bool,
) -> tuple[np.ndarray, str | None]:
    from transformers import CLIPModel, CLIPProcessor

    pretrained_kwargs: dict[str, object] = {
        "local_files_only": not allow_model_download,
    }
    if clip_revision:
        pretrained_kwargs["revision"] = clip_revision
    try:
        processor = CLIPProcessor.from_pretrained(clip_model, **pretrained_kwargs)
        model = CLIPModel.from_pretrained(clip_model, **pretrained_kwargs).eval().to(device)
    except OSError as exc:
        mode = "download was allowed" if allow_model_download else "downloads are disabled"
        raise RuntimeError(
            f"could not load CLIP model/processor {clip_model!r} ({mode}). "
            "Pre-stage the Hugging Face snapshot or pass --allow-model-download."
        ) from exc

    prompts = list(WATER_BACKGROUND_PROMPTS + LAND_BACKGROUND_PROMPTS)
    text_batch = processor(text=prompts, return_tensors="pt", padding=True)
    text_batch = {key: value.to(device) for key, value in text_batch.items()}
    with _autocast(device, amp):
        text_features = model.get_text_features(**text_batch)
    text_features = _normalize_rows(text_features.float())
    n_water = len(WATER_BACKGROUND_PROMPTS)
    water_mean = _normalize_rows(text_features[:n_water].mean(dim=0, keepdim=True))[0]
    land_mean = _normalize_rows(text_features[n_water:].mean(dim=0, keepdim=True))[0]
    direction = water_mean - land_mean
    direction = direction / direction.norm().clamp_min(1e-12)

    loader = DataLoader(
        _RawImageDataset(dataset),
        batch_size=batch_size,
        shuffle=False,
        collate_fn=_collate_pil,
        **_loader_kwargs(num_workers, pin_memory),
    )
    index_parts: list[np.ndarray] = []
    score_parts: list[np.ndarray] = []
    for images, indices in tqdm(loader, desc="CLIP metadata-free nuisance"):
        image_batch = processor(images=images, return_tensors="pt")
        pixel_values = image_batch["pixel_values"].to(device, non_blocking=pin_memory)
        with _autocast(device, amp):
            image_features = model.get_image_features(pixel_values=pixel_values)
        image_features = _normalize_rows(image_features.float())
        scores = image_features @ direction
        index_parts.append(indices.numpy())
        score_parts.append(scores.cpu().numpy())

    indices = _validate_complete_indices(np.concatenate(index_parts), len(dataset), "CLIP nuisance")
    raw = np.concatenate(score_parts).astype(np.float32, copy=False)
    if raw.shape != (len(dataset),) or not np.all(np.isfinite(raw)):
        raise RuntimeError(f"unexpected CLIP nuisance vector {raw.shape}")
    resolved_revision = getattr(model.config, "_commit_hash", None)
    atomic_savez_compressed(
        cache_path,
        indices=indices,
        b_vl_raw=raw,
        clip_model=np.asarray(clip_model),
        clip_revision=np.asarray(clip_revision or ""),
        resolved_revision=np.asarray(resolved_revision or ""),
        prompt_sha256=np.asarray(prompt_sha256()),
    )
    return load_nuisance_cache(cache_path, len(dataset), clip_model, clip_revision), resolved_revision


def get_clip_nuisance(
    dataset,
    cache_path: Path,
    *,
    recompute: bool,
    clip_model: str,
    clip_revision: str | None,
    device: torch.device,
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
    amp: bool,
    allow_model_download: bool,
) -> tuple[np.ndarray, bool, str | None]:
    if cache_path.is_file() and not recompute:
        try:
            values = load_nuisance_cache(cache_path, len(dataset), clip_model, clip_revision)
            return values, True, nuisance_cache_resolved_revision(cache_path)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid CLIP nuisance cache {cache_path}: {exc}. "
                "Use --recompute-nuisance to replace it."
            ) from exc
    values, resolved_revision = extract_clip_nuisance(
        dataset,
        cache_path,
        clip_model,
        clip_revision,
        device,
        batch_size,
        num_workers,
        pin_memory,
        amp,
        allow_model_download,
    )
    return values, False, resolved_revision


def _background_column(dataset, metadata: np.ndarray) -> tuple[np.ndarray, str]:
    fields = getattr(dataset, "metadata_fields", None)
    if fields is None:
        fields = getattr(dataset, "_metadata_fields", None)
    names = [str(value).lower() for value in fields] if fields is not None else []
    column = None
    field_name = "metadata_column_0"
    for candidate in ("background", "place"):
        if candidate in names:
            column = names.index(candidate)
            field_name = str(fields[column])
            break
    if column is None:
        if metadata.ndim != 2 or metadata.shape[1] == 0:
            raise ValueError("Waterbirds metadata has no background column")
        column = 0
    background = metadata[:, column].astype(np.int64)
    if not np.all(np.isin(np.unique(background), (0, 1))):
        raise ValueError(f"expected binary Waterbirds background metadata, got {np.unique(background)}")
    return background, field_name


def _image_filenames(dataset, root_dir: Path, n_rows: int) -> np.ndarray:
    values = getattr(dataset, "_input_array", None)
    if values is not None and len(values) == n_rows:
        return np.asarray([str(value) for value in values], dtype=np.str_)

    candidates = []
    for attribute in ("data_dir", "_data_dir"):
        value = getattr(dataset, attribute, None)
        if value:
            candidates.append(Path(value) / "metadata.csv")
    candidates.append(root_dir / "waterbirds_v1.0" / "metadata.csv")
    metadata_csv = next((path for path in candidates if path.is_file()), None)
    if metadata_csv is None:
        raise FileNotFoundError("could not locate Waterbirds metadata.csv to recover image filenames")
    with metadata_csv.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != n_rows or "img_filename" not in rows[0]:
        raise ValueError(f"unexpected Waterbirds metadata file {metadata_csv}")
    return np.asarray([row["img_filename"] for row in rows], dtype=np.str_)


def _package_versions() -> dict[str, str]:
    result = {}
    for package in ("numpy", "torch", "torchvision", "transformers", "wilds", "Pillow"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def _artifact_record(path: Path, arrays: dict[str, np.ndarray] | None = None) -> dict[str, object]:
    record: dict[str, object] = {
        "path": str(path),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
    }
    if arrays is not None:
        record["keys"] = list(arrays)
        record["n_rows"] = int(next(iter(arrays.values())).shape[0])
    return record


def _reuse_prepared_bundle(args: argparse.Namespace) -> bool:
    """Reuse a byte-identical prepared bundle on normal launcher resumes."""
    if args.force or args.recompute_features or args.recompute_nuisance:
        return False
    out_dir = Path(args.out_dir)
    manifest_path = out_dir / "preparation_manifest.json"
    expected_paths = {
        "train_blind": out_dir / "train_blind.npz",
        "val_blind": out_dir / "val_blind.npz",
        "test_blind": out_dir / "test_blind.npz",
        "sealed_metadata": out_dir / "sealed_metadata.npz",
    }
    existing = [manifest_path.is_file(), *(path.is_file() for path in expected_paths.values())]
    if not any(existing):
        return False
    if not all(existing):
        raise RuntimeError(
            f"partial prepared bundle under {out_dir}; use --force only after resolving or archiving downstream runs"
        )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    configuration = {
        key: manifest[key]
        for key in ("blindness_contract", "dataset", "representation", "nuisance")
    }
    if manifest.get("configuration_sha256") != json_sha256(configuration):
        raise RuntimeError("existing preparation manifest configuration hash mismatch")
    representation = manifest.get("representation", {})
    if (
        representation.get("model") != FEATURE_MODEL_ID
        or representation.get("cache_attests_model_identity") is not True
    ):
        raise RuntimeError(
            "existing prepared bundle lacks an attested frozen-feature extractor identity; "
            "use the corrected fresh OUT_DIR and attested feature-cache path"
        )
    requested = {
        "root_dir": str(Path(args.root_dir)),
        "feature_cache": str(Path(args.feature_cache)),
        "clip_model": args.clip_model,
        "clip_revision": args.clip_revision,
        "nuisance_cache": str(Path(args.nuisance_cache)),
        "prompt_sha256": prompt_sha256(),
        "deterministic": bool(args.deterministic),
    }
    recorded = {
        "root_dir": manifest.get("dataset", {}).get("root_dir"),
        "feature_cache": manifest.get("representation", {}).get("cache"),
        "clip_model": manifest.get("nuisance", {}).get("clip_model"),
        "clip_revision": manifest.get("nuisance", {}).get("requested_revision"),
        "nuisance_cache": manifest.get("nuisance", {}).get("cache"),
        "prompt_sha256": manifest.get("nuisance", {}).get("prompt_sha256"),
        "deterministic": manifest.get("execution", {}).get("deterministic_torch_algorithms"),
    }
    if recorded != requested:
        raise RuntimeError(
            "existing prepared bundle uses a different scientific preparation configuration; "
            "choose a new OUT_DIR or pass --force intentionally"
        )
    for name, path in expected_paths.items():
        record = manifest.get("artifacts", {}).get(name, {})
        if record.get("path") != str(path) or record.get("sha256") != file_sha256(path):
            raise RuntimeError(f"existing prepared artifact failed its manifest check: {path}")
    print(
        json.dumps(
            {
                "prepared": str(out_dir),
                "manifest": str(manifest_path),
                "reused_exact_bundle": True,
                "reason": "all scientific preparation settings and artifact hashes match",
            },
            indent=2,
        ),
        flush=True,
    )
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root-dir", default="data/wilds", help="WILDS data root")
    parser.add_argument("--out-dir", default="build/results/waterbirds_hidden_slice_v2/prepared")
    parser.add_argument(
        "--feature-cache",
        default="build/features/waterbirds_mobilenetv3_small_attested.npz",
    )
    parser.add_argument(
        "--nuisance-cache",
        default="build/features/waterbirds_clip_vit_base_patch32_nuisance.npz",
    )
    parser.add_argument("--clip-model", default=DEFAULT_CLIP_MODEL)
    parser.add_argument("--clip-revision", default=None, help="optional immutable Hugging Face revision")
    parser.add_argument("--device", default="auto", help="auto | cpu | cuda | cuda:N")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--download", action="store_true", help="allow WILDS to download a missing dataset")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="allow torchvision/Hugging Face model downloads; model loading is local-only otherwise",
    )
    parser.add_argument("--force", action="store_true", help="recompute both expensive model caches")
    parser.add_argument("--recompute-features", action="store_true")
    parser.add_argument("--recompute-nuisance", action="store_true")
    parser.add_argument("--amp", action=BooleanOptionalAction, default=True)
    parser.add_argument("--pin-memory", action=BooleanOptionalAction, default=None)
    parser.add_argument("--deterministic", action=BooleanOptionalAction, default=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    if _reuse_prepared_bundle(args):
        return

    set_seed(args.seed, deterministic=args.deterministic)
    device = resolve_device(args.device)
    pin_memory = device.type == "cuda" if args.pin_memory is None else bool(args.pin_memory)
    out_dir = Path(args.out_dir)
    feature_cache = Path(args.feature_cache)
    nuisance_cache = Path(args.nuisance_cache)

    print(f"loading local Waterbirds data from {args.root_dir}", flush=True)
    dataset = get_dataset(dataset="waterbirds", root_dir=args.root_dir, download=args.download)
    n_rows = len(dataset)
    row_id = np.arange(n_rows, dtype=np.int64)
    y = as_numpy(dataset._y_array).astype(np.int64)
    split = as_numpy(dataset._split_array).astype(np.int64)
    if y.shape != (n_rows,) or split.shape != (n_rows,):
        raise ValueError("WILDS Waterbirds arrays have inconsistent lengths")
    if not np.all(np.isin(np.unique(y), (0, 1))):
        raise ValueError("Waterbirds task labels must be binary")

    features, feature_cache_reused = get_mobilenet_features(
        dataset,
        feature_cache,
        recompute=bool(args.force or args.recompute_features),
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        amp=args.amp,
        allow_model_download=args.allow_model_download,
    )
    b_vl_raw, nuisance_cache_reused, resolved_revision = get_clip_nuisance(
        dataset,
        nuisance_cache,
        recompute=bool(args.force or args.recompute_nuisance),
        clip_model=args.clip_model,
        clip_revision=args.clip_revision,
        device=device,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
        amp=args.amp,
        allow_model_download=args.allow_model_download,
    )

    split_codes = {
        "train": int(dataset.split_dict["train"]),
        "val": int(dataset.split_dict["val"]),
        "test": int(dataset.split_dict["test"]),
    }
    train_mask = split == split_codes["train"]
    if not train_mask.any():
        raise ValueError("official Waterbirds training split is empty")
    nuisance_mean = float(b_vl_raw[train_mask].mean())
    nuisance_std = float(b_vl_raw[train_mask].std())
    if not np.isfinite(nuisance_std) or nuisance_std < 1e-8:
        raise ValueError(f"CLIP nuisance coordinate has degenerate train standard deviation {nuisance_std}")
    b_vl = ((b_vl_raw - nuisance_mean) / nuisance_std).astype(np.float32)

    artifacts: dict[str, dict[str, object]] = {}
    split_row_ids: dict[str, np.ndarray] = {}
    for name, code in split_codes.items():
        indices = np.flatnonzero(split == code)
        split_row_ids[name] = row_id[indices]
        blind = BlindSplit(row_id=row_id[indices], h=features[indices], y=y[indices], b_vl=b_vl[indices])
        path = blind.save(out_dir / f"{name}_blind.npz")
        artifacts[f"{name}_blind"] = _artifact_record(path, blind.arrays())
        artifacts[f"{name}_blind"]["row_id_sha256"] = array_sha256(blind.row_id)
        artifacts[f"{name}_blind"]["label_counts"] = {
            str(label): int(np.sum(blind.y == label)) for label in (0, 1)
        }

    joined = np.concatenate([split_row_ids[name] for name in ("train", "val", "test")])
    if joined.size != n_rows or np.unique(joined).size != n_rows:
        raise RuntimeError("official split artifacts do not form a disjoint complete partition")

    # Metadata is materialized only after all blind numerical artifacts exist.
    # No value below is passed back into feature or nuisance computation.
    metadata = as_numpy(dataset._metadata_array)
    if metadata.shape[0] != n_rows:
        raise ValueError("WILDS Waterbirds metadata has an inconsistent length")
    background, background_field = _background_column(dataset, metadata)
    group = (2 * y + background).astype(np.int64)
    img_filename = _image_filenames(dataset, Path(args.root_dir), n_rows)
    sealed_arrays = {
        "row_id": row_id,
        "background": background,
        "group": group,
        "img_filename": img_filename,
        "split": split,
    }
    sealed_path = atomic_savez_compressed(out_dir / "sealed_metadata.npz", **sealed_arrays)
    artifacts["sealed_metadata"] = _artifact_record(sealed_path, sealed_arrays)
    artifacts["sealed_metadata"]["row_id_sha256"] = array_sha256(row_id)

    checkpoint = mobilenet_checkpoint_path()
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "blindness_contract": {
            "blind_keys_exact": list(BLIND_KEYS),
            "sealed_keys_exact": list(sealed_arrays),
            "statement": (
                "Background/group metadata is absent from blind files and was not used to construct h or b_vl. "
                "sealed_metadata.npz may be opened only after gates, hyperparameters, certificates, and repair models are frozen."
            ),
        },
        "dataset": {
            "name": "Waterbirds via WILDS",
            "root_dir": str(Path(args.root_dir)),
            "n_rows": n_rows,
            "official_split_codes": split_codes,
            "background_field_in_sealed_source": background_field,
            "download_requested": bool(args.download),
        },
        "representation": {
            "model": FEATURE_MODEL_ID,
            "dimension": FEATURE_DIM,
            "cache": str(feature_cache),
            "cache_sha256": file_sha256(feature_cache),
            "cache_reused": feature_cache_reused,
            "cache_attests_model_identity": feature_cache_identity_attested(feature_cache),
            "local_checkpoint": str(checkpoint),
            "local_checkpoint_sha256": file_sha256(checkpoint) if checkpoint.is_file() else None,
        },
        "nuisance": {
            "kind": "metadata-free fixed vision-language text direction",
            "clip_model": args.clip_model,
            "requested_revision": args.clip_revision,
            "resolved_revision_when_extracted": resolved_revision,
            "cache": str(nuisance_cache),
            "cache_sha256": file_sha256(nuisance_cache),
            "cache_reused": nuisance_cache_reused,
            "prompt_specification": prompt_specification(),
            "prompt_sha256": prompt_sha256(),
            "standardization": {
                "fit_split": "official train only",
                "mean": nuisance_mean,
                "std": nuisance_std,
            },
        },
        "execution": {
            "seed": args.seed,
            "device": str(device),
            "cuda_device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "pin_memory": pin_memory,
            "amp": bool(args.amp and device.type == "cuda"),
            "model_download_allowed": bool(args.allow_model_download),
            "deterministic_torch_algorithms": bool(args.deterministic),
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "package_versions": _package_versions(),
        },
        "artifacts": artifacts,
    }
    manifest["configuration_sha256"] = json_sha256(
        {
            "blindness_contract": manifest["blindness_contract"],
            "dataset": manifest["dataset"],
            "representation": manifest["representation"],
            "nuisance": manifest["nuisance"],
        }
    )
    manifest_path = atomic_write_json(out_dir / "preparation_manifest.json", manifest)
    print(
        json.dumps(
            {
                "prepared": str(out_dir),
                "manifest": str(manifest_path),
                "n_rows": n_rows,
                "feature_dim": FEATURE_DIM,
                "device": str(device),
                "feature_cache_reused": feature_cache_reused,
                "nuisance_cache_reused": nuisance_cache_reused,
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
