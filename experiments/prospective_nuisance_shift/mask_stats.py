"""Cache per-image CUB matte statistics (foreground area, soft-rim area).

The foreground fraction drives donor selection for the background bank: donors
with a small bird leave a small hole to inpaint, so the recovered Places
background is almost entirely real pixels.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import load_metadata, log, mask_path


def compute(mask_root: Path, metadata_csv: Path) -> dict[str, np.ndarray]:
    meta = load_metadata(metadata_csv)
    n = len(meta)
    foreground = np.zeros(n, dtype=np.float32)
    rim = np.zeros(n, dtype=np.float32)
    height = np.zeros(n, dtype=np.int32)
    width = np.zeros(n, dtype=np.int32)
    for i, name in enumerate(tqdm(meta.img_filename, desc="matte stats")):
        seg = np.asarray(Image.open(mask_path(mask_root, str(name))).convert("L"), dtype=np.float32) / 255.0
        foreground[i] = float(np.mean(seg > 0.5))
        rim[i] = float(np.mean((seg > 0.0) & (seg < 1.0)))
        height[i], width[i] = seg.shape
    return {
        "img_id": meta.img_id,
        "foreground_fraction": foreground,
        "rim_fraction": rim,
        "height": height,
        "width": width,
    }


def load_or_compute(cache: str | Path, mask_root: str | Path, metadata_csv: str | Path) -> dict[str, np.ndarray]:
    cache = Path(cache)
    if cache.exists():
        data = np.load(cache)
        return {k: data[k] for k in data.files}
    stats = compute(Path(mask_root), Path(metadata_csv))
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, **stats)
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask-root", default="data/cub_segmentations/segmentations")
    parser.add_argument("--metadata", default="data/wilds/waterbirds_v1.0/metadata.csv")
    parser.add_argument("--cache", default="build/prospective_shift/matte_stats.npz")
    args = parser.parse_args()

    stats = load_or_compute(args.cache, args.mask_root, args.metadata)
    meta = load_metadata(args.metadata)
    fg = stats["foreground_fraction"]
    log(f"cached matte stats for {fg.size} images -> {args.cache}")
    log(f"foreground fraction: min={fg.min():.3f} median={np.median(fg):.3f} max={fg.max():.3f}")
    log(f"soft-rim fraction:   mean={stats['rim_fraction'].mean():.4f}")
    for threshold in (0.05, 0.08, 0.10, 0.15):
        eligible = (meta.split == 0) & (fg <= threshold)
        land = int(np.sum(eligible & (meta.place == 0)))
        water = int(np.sum(eligible & (meta.place == 1)))
        log(f"train donors with fg<={threshold:.2f}: land={land} water={water}")


if __name__ == "__main__":
    main()
