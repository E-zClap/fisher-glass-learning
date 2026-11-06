"""Preflight: compile and import every module, check versions and data layout.

Runs in a couple of seconds and fails loudly, so a typo or a missing dependency
surfaces before an hour of GPU time rather than after it.
"""

from __future__ import annotations

import argparse
import importlib
import py_compile
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

MODULES = [
    "common",
    "compose",
    "data",
    "models",
    "mask_stats",
    "select_slice",
    "train_end_to_end",
    "build_paired_images",
    "evaluate_shift",
    "natural_shift_fallback",
    "predictive_validity",
    "make_figure",
    "make_report",
    "fetch_cub_masks",
]

MIN_PYTHON = (3, 9)


def check_python() -> list[str]:
    problems = []
    print(f"python  {sys.version.split()[0]}  ({sys.executable})")
    if sys.version_info < MIN_PYTHON:
        problems.append(f"Python {'.'.join(map(str, MIN_PYTHON))}+ required, found {sys.version.split()[0]}")
    return problems


def check_dependencies() -> list[str]:
    problems = []
    for name, attr in [
        ("numpy", "__version__"),
        ("torch", "__version__"),
        ("torchvision", "__version__"),
        ("sklearn", "__version__"),
        ("scipy", "__version__"),
        ("cv2", "__version__"),
        ("PIL", "__version__"),
        ("matplotlib", "__version__"),
        ("tqdm", "__version__"),
    ]:
        try:
            module = importlib.import_module(name)
            print(f"{name:<12} {getattr(module, attr, '?')}")
        except Exception as exc:
            problems.append(f"cannot import {name}: {exc}")
    try:
        import torch

        if torch.cuda.is_available():
            props = torch.cuda.get_device_properties(0)
            print(f"cuda         {props.name} ({props.total_memory / 1e9:.1f} GB), bf16={torch.cuda.is_bf16_supported()}")
        else:
            print("cuda         NOT AVAILABLE (training will fall back to CPU and be slow)")
    except Exception as exc:  # pragma: no cover
        problems.append(f"torch.cuda probe failed: {exc}")
    return problems


def check_modules() -> list[str]:
    problems = []
    for name in MODULES:
        path = HERE / f"{name}.py"
        try:
            py_compile.compile(str(path), doraise=True)
        except py_compile.PyCompileError as exc:
            problems.append(f"syntax error in {name}.py: {exc}")
            continue
        try:
            importlib.import_module(name)
        except Exception as exc:
            problems.append(f"cannot import {name}.py: {type(exc).__name__}: {exc}")
    if not problems:
        print(f"modules      all {len(MODULES)} compile and import cleanly")
    return problems


def check_data(image_root: Path, mask_root: Path, metadata: Path) -> list[str]:
    problems = []
    if not metadata.exists():
        problems.append(f"missing {metadata} -- run download_data.sh")
        return problems
    from common import load_metadata

    meta = load_metadata(metadata)
    counts = {split: int((meta.split == split).sum()) for split in (0, 1, 2)}
    print(f"metadata     {len(meta)} rows; train/val/test = {counts[0]}/{counts[1]}/{counts[2]}")
    if len(meta) != 11788:
        problems.append(f"expected 11788 metadata rows, found {len(meta)}")

    missing_images = [n for n in meta.img_filename[:200] if not (image_root / str(n)).exists()]
    if missing_images:
        problems.append(f"{len(missing_images)}/200 sampled images missing under {image_root} (e.g. {missing_images[0]})")
    else:
        print(f"images       200/200 sampled files present under {image_root}")

    if not mask_root.exists():
        problems.append(f"missing {mask_root} -- run download_data.sh")
    else:
        missing_masks = [n for n in meta.img_filename[:200] if not (mask_root / str(n).replace(".jpg", ".png")).exists()]
        if missing_masks:
            problems.append(f"{len(missing_masks)}/200 sampled mattes missing under {mask_root}")
        else:
            print(f"mattes       200/200 sampled files present under {mask_root}")
    return problems


def check_compose(image_root: Path, mask_root: Path, metadata: Path, out_dir: Path) -> list[str]:
    """Build one paired variant end to end -- the riskiest CPU path."""
    import numpy as np

    from common import load_metadata
    from compose import DonorBank, intervene, load_source
    from mask_stats import load_or_compute

    try:
        meta = load_metadata(metadata)
        stats = load_or_compute(out_dir / "matte_stats.npz", mask_root, metadata)
        bank = DonorBank.build(meta, image_root, mask_root, stats["foreground_fraction"])
        row = int(np.flatnonzero((meta.split == 2) & (meta.group == 0))[0])
        source = load_source(meta, row, image_root, mask_root)
        same, d_same = intervene(source, bank, "same", 0)
        opposite, d_opposite = intervene(source, bank, "opposite", 0)
    except Exception as exc:
        return [f"compositor failed: {type(exc).__name__}: {exc}"]

    if same.size != source.size or opposite.size != source.size:
        return ["compositor produced an image of the wrong size"]
    print(
        f"compositor   OK on {source.filename} "
        f"(fg={source.foreground_fraction:.3f}, rim={source.rim_fraction:.3f}, "
        f"visible-inpaint same={d_same['visible_inpaint_fraction']:.3f} "
        f"opposite={d_opposite['visible_inpaint_fraction']:.3f}); donors {bank.counts()}"
    )
    return []


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-root", default="data/wilds/waterbirds_v1.0")
    parser.add_argument("--mask-root", default="data/cub_segmentations/segmentations")
    parser.add_argument("--metadata", default="data/wilds/waterbirds_v1.0/metadata.csv")
    parser.add_argument("--out-dir", default="build/prospective_shift")
    parser.add_argument("--skip-compose", action="store_true", help="skip the compositor check (it builds the matte cache)")
    args = parser.parse_args()

    print("---- preflight ----")
    problems = check_python()
    problems += check_dependencies()
    problems += check_modules()
    data_problems = check_data(Path(args.image_root), Path(args.mask_root), Path(args.metadata))
    problems += data_problems
    if not data_problems and not args.skip_compose:
        problems += check_compose(Path(args.image_root), Path(args.mask_root), Path(args.metadata), Path(args.out_dir))

    print("-------------------")
    if problems:
        print(f"{len(problems)} problem(s):")
        for problem in problems:
            print(f"  - {problem}")
        raise SystemExit(1)
    print("all checks passed")


if __name__ == "__main__":
    main()
