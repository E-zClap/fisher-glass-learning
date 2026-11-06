"""Read-only preflight for the group-free Fisher-Glass lambda audit.

The command deliberately does not deserialize ``test_blind.npz`` or
``sealed_metadata.npz``. It checks their digests and manifest attestations
only. Test arrays are first loaded by ``certify.py`` after every selection is
frozen; the metadata seal is first loaded by ``posthoc.py``.
"""

from __future__ import annotations

import argparse
import ast
import importlib
import importlib.metadata
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

try:
    from .experiment_common import (
        HERE,
        HIDDEN_SLICE_DIR,
        blind_input_hashes,
        file_sha256,
        implementation_manifest,
        load_blind_npz,
    )
except ImportError:
    from experiment_common import (
        HERE,
        HIDDEN_SLICE_DIR,
        blind_input_hashes,
        file_sha256,
        implementation_manifest,
        load_blind_npz,
    )


REQUIRED_BLIND_KEYS = {"row_id", "h", "y", "b_vl"}
EXPECTED_SPLIT_ROWS = {"train": 4_795, "val": 1_199, "test": 5_794}
BLIND_STAGES = ("select_lambda.py", "certify.py")


def _version(package: str) -> str:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def check_syntax() -> list[str]:
    problems: list[str] = []
    files = sorted(HERE.glob("*.py"))
    dependencies = [HIDDEN_SLICE_DIR / name for name in (
        "blind_protocol.py", "common.py", "compat.py", "gates.py", "geometry.py", "heads.py", "metrics.py"
    )]
    for path in [*files, *dependencies]:
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            problems.append(f"cannot parse {path}: {exc}")
    print(f"syntax       parsed {len(files) + len(dependencies)} experiment/dependency modules")
    return problems


def check_dependencies() -> list[str]:
    problems: list[str] = []
    packages = (
        ("numpy", "numpy"),
        ("scipy", "scipy"),
        ("torch", "torch"),
        ("sklearn", "scikit-learn"),
    )
    for module, package in packages:
        try:
            importlib.import_module(module)
            print(f"{package:<12} {_version(package)}")
        except Exception as exc:  # pragma: no cover - environment dependent
            problems.append(f"cannot import {module}: {type(exc).__name__}: {exc}")
    return problems


def check_device(device_name: str) -> list[str]:
    problems: list[str] = []
    try:
        import torch
    except ImportError:  # already reported by dependency check
        return problems
    requested_cuda = device_name.startswith("cuda")
    if torch.cuda.is_available():
        device = torch.device("cuda" if device_name == "auto" else device_name)
        if device.type == "cuda":
            try:
                properties = torch.cuda.get_device_properties(device)
                print(
                    f"cuda         {properties.name}; {properties.total_memory / 1e9:.1f} GB; "
                    f"bf16={torch.cuda.is_bf16_supported()}"
                )
            except Exception as exc:  # pragma: no cover - hardware dependent
                problems.append(f"cannot inspect requested device {device_name!r}: {exc}")
        else:
            print("cuda         available; CPU explicitly selected")
    elif requested_cuda:
        problems.append(f"CUDA device {device_name!r} requested but CUDA is unavailable")
    else:
        print("cuda         unavailable; CPU is valid but slower")
    return problems


def _manifest_artifact(manifest: dict[str, Any], filename: str) -> dict[str, Any] | None:
    artifacts = manifest.get("artifacts", {})
    return artifacts.get(Path(filename).stem) or artifacts.get(filename)


def check_prepared(data_dir: Path) -> list[str]:
    problems: list[str] = []
    manifest_path = data_dir / "preparation_manifest.json"
    if not manifest_path.is_file():
        return [
            f"missing {manifest_path}; first run the Waterbirds hidden-slice preparation pipeline"
        ]
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        return [f"cannot read preparation manifest: {type(exc).__name__}: {exc}"]

    observed: dict[str, int] = {}
    for split in ("train", "val", "test"):
        filename = f"{split}_blind.npz"
        path = data_dir / filename
        record = _manifest_artifact(manifest, filename)
        if not path.is_file():
            problems.append(f"missing blind split: {path}")
            continue
        if not isinstance(record, dict) or record.get("sha256") != file_sha256(path):
            problems.append(f"{filename} does not match preparation_manifest.json")
            continue
        if split == "test":
            # Preserve the held-out boundary: rely on the hashed preparation
            # manifest for test schema/row attestations and do not deserialize
            # even a single test array before all selections are frozen.
            if set(record.get("keys", [])) != REQUIRED_BLIND_KEYS:
                problems.append(f"{filename} has an invalid attested key schema")
                continue
            n_rows = int(record.get("n_rows", -1))
            observed[split] = n_rows
            if n_rows != EXPECTED_SPLIT_ROWS[split]:
                problems.append(
                    f"{filename} attests {n_rows} rows; expected {EXPECTED_SPLIT_ROWS[split]}"
                )
            else:
                print("test seal    SHA-256/schema attestation verified without deserializing arrays")
            continue
        try:
            bundle = load_blind_npz(path)
            # ``load_blind_npz`` enforces the exact schema.  Keep an explicit
            # check here so preflight output documents the firewall contract.
            with np.load(path, allow_pickle=False) as archive:
                if set(archive.files) != REQUIRED_BLIND_KEYS:
                    raise ValueError(f"unexpected keys {archive.files}")
            n_rows = int(np.asarray(bundle.row_id).size)
            observed[split] = n_rows
            if n_rows != EXPECTED_SPLIT_ROWS[split]:
                problems.append(
                    f"{filename} has {n_rows} rows; expected {EXPECTED_SPLIT_ROWS[split]}"
                )
        except Exception as exc:
            problems.append(f"invalid {filename}: {type(exc).__name__}: {exc}")

    sealed_path = data_dir / "sealed_metadata.npz"
    sealed_record = _manifest_artifact(manifest, sealed_path.name)
    if not sealed_path.is_file():
        problems.append(f"missing opaque metadata seal: {sealed_path}")
    elif not isinstance(sealed_record, dict) or sealed_record.get("sha256") != file_sha256(sealed_path):
        problems.append("sealed_metadata.npz does not match preparation_manifest.json")
    else:
        print("seal         SHA-256 verified without deserializing metadata")

    try:
        hashes = blind_input_hashes(data_dir, include_test=True)
        print(f"prepared     blind rows {observed}; {len(hashes)} input digests verified")
    except Exception as exc:
        problems.append(f"cannot hash blind inputs: {type(exc).__name__}: {exc}")
    return problems


def _loaded_names(tree: ast.AST) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            names.add(node.id)
        elif isinstance(node, ast.Attribute) and isinstance(node.ctx, ast.Load):
            names.add(node.attr)
    return names


def check_static_firewall() -> list[str]:
    """Reject direct metadata APIs in the two blind stage modules.

    This is defense in depth; the runtime stages additionally use exact-schema
    blind loaders and carry file/source hashes through the frozen chain.
    """

    problems: list[str] = []
    forbidden_loaded_names = {
        "load_sealed_metadata",
        "join_sealed",
        "group_metrics",
        "get_dataset",
        "read_csv",
        "DictReader",
    }
    for filename in BLIND_STAGES:
        path = HERE / filename
        if not path.is_file():
            problems.append(f"missing blind stage source: {path}")
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (OSError, SyntaxError) as exc:
            problems.append(f"cannot inspect firewall for {filename}: {exc}")
            continue
        offending = sorted(_loaded_names(tree).intersection(forbidden_loaded_names))
        if offending:
            problems.append(f"{filename} loads forbidden metadata APIs: {offending}")
    if not problems:
        print("firewall     blind selection/certification sources use no direct metadata APIs")
    return problems


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--data-dir",
        default="build/results/waterbirds_hidden_slice_v2/prepared",
        type=Path,
    )
    parser.add_argument("--device", default="auto")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    print("---- Waterbirds group-free lambda preflight (read-only) ----")
    problems: list[str] = []
    problems.extend(check_syntax())
    problems.extend(check_dependencies())
    problems.extend(check_device(args.device))
    problems.extend(check_prepared(args.data_dir))
    problems.extend(check_static_firewall())
    try:
        source = implementation_manifest()
        print(f"source       {source['combined_sha256']} ({len(source['files'])} files)")
    except Exception as exc:
        problems.append(f"cannot construct source manifest: {type(exc).__name__}: {exc}")
    print("------------------------------------------------------------")
    if problems:
        for problem in problems:
            print(f"ERROR: {problem}", file=sys.stderr)
        raise SystemExit(2)
    print("all preflight checks passed; no artifacts, downloads, or training were performed")


if __name__ == "__main__":
    main()
