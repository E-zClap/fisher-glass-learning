"""Step 0: fetch the CUB-200-2011 segmentation masks used to build Waterbirds.

Waterbirds images are composites produced by the official ``generate_waterbirds.py``
script in ``kohpangwei/group_DRO``:

    composite = round(cub_image * seg) + round(crop_and_resize(place, size) * (1 - seg))

with ``seg = segmentation_png / 255`` in ``[0, 1]``.  Downloading the CUB
segmentations therefore gives us the *exact* alpha mattes that were used to
generate the dataset we already have on disk, which is what makes a pixel-exact,
label-preserving background intervention possible without downloading Places365
or the 1.1 GB CUB image archive.

Only ``segmentations.tgz`` (39.3 MB) is fetched.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import urllib.request
from pathlib import Path

SEGMENTATIONS_URL = "https://data.caltech.edu/records/w9d68-gec53/files/segmentations.tgz"
EXPECTED_BYTES = 39272883


def download(url: str, dest: Path, chunk: int = 1 << 20) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size == EXPECTED_BYTES:
        print(f"already downloaded: {dest} ({dest.stat().st_size} bytes)")
        return dest
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    digest = hashlib.sha256()
    total = 0
    with urllib.request.urlopen(request, timeout=120) as response, dest.open("wb") as handle:
        length = int(response.headers.get("Content-Length") or 0)
        while True:
            block = response.read(chunk)
            if not block:
                break
            handle.write(block)
            digest.update(block)
            total += len(block)
            if length:
                print(f"\r  {total / 1e6:6.1f} / {length / 1e6:6.1f} MB", end="", flush=True)
    print()
    print(f"downloaded {dest} ({total} bytes, sha256={digest.hexdigest()})")
    return dest


def extract(archive: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = out_dir / "segmentations"
    if marker.exists() and any(marker.iterdir()):
        print(f"already extracted: {marker}")
        return marker
    with tarfile.open(archive, "r:gz") as tar:
        members = tar.getmembers()
        for member in members:
            # Reject absolute paths and parent-directory escapes before extracting.
            target = (out_dir / member.name).resolve()
            if not str(target).startswith(str(out_dir.resolve())):
                raise RuntimeError(f"unsafe path in archive: {member.name}")
        tar.extractall(out_dir)
    print(f"extracted {len(members)} members to {out_dir}")
    return marker


def verify(seg_root: Path, metadata_csv: Path) -> dict:
    import csv

    rows = list(csv.DictReader(metadata_csv.open(encoding="utf-8")))
    missing = []
    for row in rows:
        png = seg_root / row["img_filename"].replace(".jpg", ".png")
        if not png.exists():
            missing.append(row["img_filename"])
    report = {
        "n_waterbirds_images": len(rows),
        "n_missing_masks": len(missing),
        "missing_examples": missing[:10],
    }
    print(json.dumps(report, indent=2))
    if missing:
        raise SystemExit(f"{len(missing)} Waterbirds images have no CUB segmentation mask")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/cub_segmentations")
    parser.add_argument("--metadata", default="data/wilds/waterbirds_v1.0/metadata.csv")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    archive = download(SEGMENTATIONS_URL, out_dir / "segmentations.tgz")
    seg_root = extract(archive, out_dir)
    verify(seg_root, Path(args.metadata))
    print(f"CUB segmentation masks ready under {seg_root}")


if __name__ == "__main__":
    main()
