"""Read-only preflight for the Waterbirds hidden-slice experiment.

This command never trains a model, downloads a file, creates an artifact, or
deserializes/reveals ``sealed_metadata.npz``; it performs only an opaque file
hash check on the seal.  It checks syntax,
dependencies, CUDA visibility, local data/model-cache availability, and any
already-prepared artifact schemas before GPU time is spent.
"""

from __future__ import annotations

import argparse
import ast
import csv
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path

import numpy as np

from common import BLIND_KEYS, WATERBIRDS_FEATURE_MODEL_ID, file_sha256, load_blind_npz


HERE = Path(__file__).resolve().parent
EXPECTED_ROWS = 11_788
EXPECTED_SPLIT_COUNTS = {0: 4_795, 1: 1_199, 2: 5_794}
SEALED_KEYS = ("row_id", "background", "group", "img_filename", "split")
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"
FEATURE_DIM = 576
FEATURE_MODEL_ID = WATERBIRDS_FEATURE_MODEL_ID


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def check_syntax() -> list[str]:
    problems = []
    files = sorted(HERE.rglob("*.py"))
    for path in files:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            problems.append(f"cannot parse {path.name}: {exc}")
    print(f"syntax       parsed {len(files)} Python modules without executing them")
    return problems


def check_dependencies() -> list[str]:
    problems = []
    imports = (
        ("numpy", "numpy"),
        ("torch", "torch"),
        ("torchvision", "torchvision"),
        ("transformers", "transformers"),
        ("wilds", "wilds"),
        ("PIL", "Pillow"),
        ("tqdm", "tqdm"),
        ("sklearn", "scikit-learn"),
        ("scipy", "scipy"),
        ("matplotlib", "matplotlib"),
    )
    for module_name, package_name in imports:
        try:
            importlib.import_module(module_name)
            print(f"{package_name:<12} {_version(package_name)}")
        except Exception as exc:
            problems.append(f"cannot import {module_name}: {type(exc).__name__}: {exc}")
    return problems


def check_cuda(device_name: str) -> list[str]:
    problems = []
    try:
        import torch
    except ImportError:
        return ["cannot inspect CUDA because torch is missing"]
    if torch.cuda.is_available():
        device = torch.device("cuda" if device_name == "auto" else device_name)
        if device.type != "cuda":
            print("cuda         available but CPU was explicitly requested")
            return problems
        try:
            properties = torch.cuda.get_device_properties(device)
            print(
                f"cuda         {properties.name}; {properties.total_memory / 1e9:.1f} GB; "
                f"bf16={torch.cuda.is_bf16_supported()}"
            )
        except Exception as exc:
            problems.append(f"cannot inspect requested CUDA device {device_name}: {exc}")
    elif device_name.startswith("cuda"):
        problems.append(f"CUDA device {device_name!r} requested but CUDA is unavailable")
    else:
        print("cuda         unavailable; preparation will use CPU and be slower")
    return problems


def check_data(root_dir: Path) -> list[str]:
    problems = []
    dataset_dir = root_dir / "waterbirds_v1.0"
    metadata_path = dataset_dir / "metadata.csv"
    if not metadata_path.is_file():
        return [f"missing {metadata_path}; stage Waterbirds or run prepare_data.py --download"]
    try:
        with metadata_path.open(encoding="utf-8", newline="") as handle:
            rows = list(csv.DictReader(handle))
    except Exception as exc:
        return [f"cannot read {metadata_path}: {exc}"]
    required = {"img_id", "img_filename", "y", "split", "place"}
    columns = set(rows[0]) if rows else set()
    if not required.issubset(columns):
        problems.append(f"metadata.csv is missing columns {sorted(required.difference(columns))}")
        return problems
    if len(rows) != EXPECTED_ROWS:
        problems.append(f"metadata.csv has {len(rows)} rows; expected {EXPECTED_ROWS}")
    try:
        split_counts = {
            code: sum(int(row["split"]) == code for row in rows) for code in EXPECTED_SPLIT_COUNTS
        }
    except (KeyError, ValueError) as exc:
        problems.append(f"metadata.csv has an invalid split field: {exc}")
        return problems
    if split_counts != EXPECTED_SPLIT_COUNTS:
        problems.append(f"unexpected train/val/test counts {split_counts}; expected {EXPECTED_SPLIT_COUNTS}")
    sampled = rows[:: max(1, len(rows) // 20)][:20]
    missing_images = [row["img_filename"] for row in sampled if not (dataset_dir / row["img_filename"]).is_file()]
    if missing_images:
        problems.append(f"{len(missing_images)}/20 sampled Waterbirds images are missing (e.g. {missing_images[0]})")
    else:
        print(
            f"data         {len(rows)} metadata rows; train/val/test "
            f"{split_counts[0]}/{split_counts[1]}/{split_counts[2]}; sampled images present"
        )
    return problems


def _huggingface_weight_cached(model_id: str, revision: str | None) -> bool:
    try:
        from huggingface_hub import try_to_load_from_cache
    except ImportError:
        return False
    candidates = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    for filename in candidates:
        try:
            cached = try_to_load_from_cache(model_id, filename, revision=revision)
        except Exception:
            continue
        if isinstance(cached, str) and Path(cached).is_file():
            return True
    return False


def _mobilenet_checkpoint_path() -> Path:
    import torch
    from torchvision.models import MobileNet_V3_Small_Weights
    from urllib.parse import urlparse

    weights = MobileNet_V3_Small_Weights.DEFAULT
    filename = Path(urlparse(weights.url).path).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def _complete_indices(values: np.ndarray, n_rows: int, label: str) -> np.ndarray:
    indices = np.asarray(values)
    if indices.ndim != 1 or indices.dtype.kind not in "iu":
        raise ValueError(f"{label} indices must be a one-dimensional integer array")
    indices = indices.astype(np.int64, copy=False)
    if indices.size != n_rows or not np.array_equal(
        np.sort(indices), np.arange(n_rows, dtype=np.int64)
    ):
        raise ValueError(f"{label} indices are not a complete 0..{n_rows - 1} permutation")
    return indices


def _load_feature_cache(path: Path, n_rows: int) -> np.ndarray:
    with np.load(path, allow_pickle=False) as archive:
        required = {"indices", "features", "feature_model"}
        if not required.issubset(archive.files):
            raise ValueError(
                f"cache is missing {sorted(required.difference(archive.files))}; "
                "legacy caches without an extractor identity are not accepted"
            )
        indices = _complete_indices(archive["indices"], n_rows, "feature cache")
        values = np.asarray(archive["features"])
        cached_model = _scalar_string(archive, "feature_model")
        if cached_model != FEATURE_MODEL_ID:
            raise ValueError(
                f"cache model {cached_model!r} does not match {FEATURE_MODEL_ID!r}"
            )
    if values.shape != (n_rows, FEATURE_DIM) or values.dtype.kind != "f":
        raise ValueError(f"features must have shape {(n_rows, FEATURE_DIM)} and floating dtype")
    if not np.all(np.isfinite(values)):
        raise ValueError("features contain NaN or infinite values")
    aligned = np.empty_like(values, dtype=np.float32)
    aligned[indices] = values.astype(np.float32, copy=False)
    return aligned


def _scalar_string(archive, key: str) -> str:
    value = np.asarray(archive[key])
    if value.ndim != 0 or value.dtype.kind not in "US":
        raise ValueError(f"{key} must be a scalar string")
    return str(value.item())


def _load_nuisance_cache(
    path: Path,
    n_rows: int,
    clip_model: str,
    clip_revision: str | None,
) -> np.ndarray:
    required = {"indices", "b_vl_raw", "clip_model", "clip_revision", "prompt_sha256"}
    with np.load(path, allow_pickle=False) as archive:
        if not required.issubset(archive.files):
            raise ValueError(f"cache is missing {sorted(required.difference(archive.files))}")
        indices = _complete_indices(archive["indices"], n_rows, "nuisance cache")
        values = np.asarray(archive["b_vl_raw"])
        cached_model = _scalar_string(archive, "clip_model")
        cached_revision = _scalar_string(archive, "clip_revision")
        prompt_hash = _scalar_string(archive, "prompt_sha256")
    if cached_model != clip_model or cached_revision != (clip_revision or ""):
        raise ValueError(
            f"cache model {cached_model}@{cached_revision or '<default>'} does not match "
            f"{clip_model}@{clip_revision or '<default>'}"
        )
    if len(prompt_hash) != 64:
        raise ValueError("prompt_sha256 is malformed")
    if values.shape != (n_rows,) or values.dtype.kind != "f" or not np.all(np.isfinite(values)):
        raise ValueError(f"b_vl_raw must be a finite floating vector of shape {(n_rows,)}")
    aligned = np.empty(n_rows, dtype=np.float32)
    aligned[indices] = values.astype(np.float32, copy=False)
    return aligned


def check_model_inputs(
    feature_cache: Path,
    nuisance_cache: Path,
    clip_model: str,
    clip_revision: str | None,
    allow_model_download: bool,
) -> list[str]:
    problems = []
    if feature_cache.is_file():
        try:
            features = _load_feature_cache(feature_cache, EXPECTED_ROWS)
            print(f"features     valid cache {feature_cache} {features.shape}/{features.dtype}")
        except Exception as exc:
            problems.append(f"invalid feature cache {feature_cache}: {exc}")
    else:
        try:
            checkpoint = _mobilenet_checkpoint_path()
        except Exception as exc:
            problems.append(f"cannot locate torchvision MobileNet checkpoint cache: {exc}")
            checkpoint = Path("<unavailable>")
        if checkpoint.is_file():
            print(f"features     cache absent; local MobileNet checkpoint available at {checkpoint}")
        elif allow_model_download:
            print("features     cache/checkpoint absent; prepare step is authorized to download MobileNet weights")
        else:
            problems.append(
                f"feature cache and local MobileNet checkpoint are absent ({feature_cache}, {checkpoint}); "
                "pre-stage one or pass --allow-model-download to prepare_data.py"
            )

    if nuisance_cache.is_file():
        try:
            values = _load_nuisance_cache(
                nuisance_cache,
                EXPECTED_ROWS,
                clip_model,
                clip_revision,
            )
            print(f"nuisance     valid metadata-free CLIP cache {nuisance_cache} {values.shape}/{values.dtype}")
        except Exception as exc:
            problems.append(f"invalid nuisance cache {nuisance_cache}: {exc}")
    elif _huggingface_weight_cached(clip_model, clip_revision):
        try:
            from transformers import CLIPProcessor

            kwargs = {"local_files_only": True}
            if clip_revision:
                kwargs["revision"] = clip_revision
            CLIPProcessor.from_pretrained(clip_model, **kwargs)
            print(f"nuisance     cache absent; local CLIP model and processor found for {clip_model}")
        except Exception as exc:
            problems.append(f"CLIP weights appear cached but its processor is incomplete: {exc}")
    elif allow_model_download:
        print(f"nuisance     cache absent; prepare step is authorized to download {clip_model}")
    else:
        problems.append(
            f"nuisance cache and local CLIP snapshot are absent ({nuisance_cache}, {clip_model}); "
            "pre-stage the model or pass --allow-model-download to prepare_data.py"
        )
    return problems


def check_prepared(out_dir: Path) -> list[str]:
    problems = []
    paths = {
        name: out_dir / f"{name}_blind.npz" for name in ("train", "val", "test")
    }
    paths["sealed"] = out_dir / "sealed_metadata.npz"
    paths["manifest"] = out_dir / "preparation_manifest.json"
    present = {name: path.is_file() for name, path in paths.items()}
    if not any(present.values()):
        print(f"prepared     no artifacts yet under {out_dir} (expected before discovery, not an error)")
        return problems
    if not all(present.values()):
        missing = [name for name, exists in present.items() if not exists]
        return [f"partial prepared artifact set under {out_dir}; missing {missing}"]

    blind = {}
    try:
        # Preserve the official test boundary: preflight validates the two
        # development archives structurally and verifies the opaque test-file
        # hash below, but never opens test arrays before certification.
        blind = {name: load_blind_npz(paths[name]) for name in ("train", "val")}
        dimensions = {split.feature_dim for split in blind.values()}
        if len(dimensions) != 1:
            problems.append(f"blind split feature dimensions disagree: {sorted(dimensions)}")
        if np.intersect1d(blind["train"].row_id, blind["val"].row_id).size:
            problems.append("blind train and validation row IDs overlap")
    except Exception as exc:
        problems.append(f"blind artifact validation failed: {type(exc).__name__}: {exc}")

    try:
        manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
        if manifest.get("blindness_contract", {}).get("blind_keys_exact") != list(BLIND_KEYS):
            problems.append("preparation manifest does not declare the exact blind schema")
        if manifest.get("blindness_contract", {}).get("sealed_keys_exact") != list(SEALED_KEYS):
            problems.append("preparation manifest does not declare the exact sealed schema")
        representation = manifest.get("representation", {})
        if (
            representation.get("model") != FEATURE_MODEL_ID
            or representation.get("cache_attests_model_identity") is not True
        ):
            problems.append("preparation manifest lacks the required attested feature-extractor identity")
        for artifact_name, record in manifest.get("artifacts", {}).items():
            artifact_path = Path(record["path"])
            if not artifact_path.is_file():
                problems.append(f"manifest artifact is missing: {artifact_name} -> {artifact_path}")
            elif file_sha256(artifact_path) != record.get("sha256"):
                problems.append(f"manifest SHA-256 mismatch for {artifact_name}")
    except Exception as exc:
        problems.append(f"manifest validation failed: {type(exc).__name__}: {exc}")

    if not problems and blind:
        counts = {name: len(value) for name, value in blind.items()}
        print(
            f"prepared     development schema {list(BLIND_KEYS)}; rows {counts}; "
            "opaque test/sealed artifact hashes verified without opening their arrays"
        )
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root-dir", default="data/wilds")
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
    parser.add_argument("--clip-revision", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="do not download; merely treat absent local weights as expected because preparation will be authorized",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("---- Waterbirds hidden-slice preflight (read-only) ----")
    problems = check_syntax()
    problems += check_dependencies()
    problems += check_cuda(args.device)
    problems += check_data(Path(args.root_dir))
    problems += check_model_inputs(
        Path(args.feature_cache),
        Path(args.nuisance_cache),
        args.clip_model,
        args.clip_revision,
        args.allow_model_download,
    )
    problems += check_prepared(Path(args.out_dir))
    print("------------------------------------------------------")
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("all preflight checks passed; no training or downloads were performed")


if __name__ == "__main__":
    main()
