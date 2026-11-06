"""Stage the original Waterbirds archive for both pipelines (Python standard library).

Downloads from Stanford's maintained download host; the historical CodaLab
endpoint used by WILDS currently returns 404. No images are redistributed here.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import tarfile
import tempfile
import urllib.request
from collections import Counter
from pathlib import Path, PurePosixPath


URL = "https://downloads.cs.stanford.edu/nlp/data/dro/waterbird_complete95_forest2water2.tar.gz"
ARCHIVE_BYTES = 489698023
OFFICIAL_COUNTS = {
    (0, 0): 3498, (0, 1): 184, (0, 2): 56, (0, 3): 1057,
    (1, 0): 467, (1, 1): 466, (1, 2): 133, (1, 3): 133,
    (2, 0): 2255, (2, 1): 2255, (2, 2): 642, (2, 3): 642,
}


def safe_relative(name: str) -> PurePosixPath:
    path = PurePosixPath(name)
    if not name or path.is_absolute() or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError(f"unsafe archive or image path: {name}")
    return path


def validate_dataset(directory: Path, *, expected_counts=OFFICIAL_COUNTS, expected_species=200) -> dict:
    with (directory / "metadata.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    required = {"img_id", "img_filename", "y", "split", "place", "place_filename"}
    if not rows or not required.issubset(rows[0]):
        raise ValueError("metadata.csv is empty or lacks original Waterbirds columns")
    counts = Counter()
    images, identifiers, species = set(), set(), set()
    for row in rows:
        relative = safe_relative(row["img_filename"])
        if len(relative.parts) != 2:
            raise ValueError(f"unexpected species/image layout: {relative}")
        image = directory.joinpath(*relative.parts)
        if not image.is_file() or image.is_symlink():
            raise ValueError(f"missing image or unsupported link: {relative}")
        if row["img_filename"] in images or row["img_id"] in identifiers:
            raise ValueError("duplicate image filename or image ID in metadata")
        images.add(row["img_filename"])
        identifiers.add(row["img_id"])
        species.add(relative.parts[0])
        label, place, split = int(row["y"]), int(row["place"]), int(row["split"])
        if label not in (0, 1) or place not in (0, 1) or split not in (0, 1, 2):
            raise ValueError("invalid binary labels, backgrounds, or official split codes")
        counts[(split, 2 * label + place)] += 1
    if dict(counts) != dict(expected_counts):
        raise ValueError(f"unexpected split/group counts: {dict(counts)}")
    if len(species) != expected_species:
        raise ValueError(f"expected {expected_species} CUB species, found {len(species)}")
    return {
        "n_images": len(rows), "n_species": len(species),
        "split_counts": {str(s): sum(n for (split, _), n in counts.items() if split == s) for s in (0, 1, 2)},
        "metadata_sha256": hashlib.sha256((directory / "metadata.csv").read_bytes()).hexdigest(),
    }


def write_marker(directory: Path) -> None:
    marker = directory / "RELEASE_v1.0.txt"
    if not marker.exists():
        marker.write_text(
            "Original group_DRO Waterbirds waterbird_complete95_forest2water2, manually staged for WILDS v1.0.\n"
            "All metadata-referenced images and official split/group counts were validated by prepare_waterbirds.py.\n"
            f"Dataset source: {URL}\n", encoding="utf-8",
        )


def stage_archive(archive: Path, root: Path, *, expected_counts=OFFICIAL_COUNTS, expected_species=200) -> Path:
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "waterbirds_v1.0"
    if target.exists() or target.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing dataset: {target}")
    with tempfile.TemporaryDirectory(prefix=".waterbirds-stage-", dir=root) as temporary:
        stage = Path(temporary).resolve()
        if stage.parent != root:
            raise ValueError("Staging directory escaped the requested data root")
        with tarfile.open(archive, "r:gz") as source:
            members = source.getmembers()
            names = set()
            for member in members:
                relative = safe_relative(member.name)
                if not (member.isfile() or member.isdir()) or str(relative).casefold() in names:
                    raise ValueError(f"unsafe archive member type or duplicate path: {member.name}")
                names.add(str(relative).casefold())
            for member in members:
                destination = stage.joinpath(*safe_relative(member.name).parts)
                if member.isdir():
                    destination.mkdir(parents=True, exist_ok=True)
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    with source.extractfile(member) as reader, destination.open("xb") as writer:
                        shutil.copyfileobj(reader, writer)
        candidates = list(stage.rglob("metadata.csv"))
        if len(candidates) != 1:
            raise ValueError(f"Expected one Waterbirds metadata.csv, found {len(candidates)}")
        dataset = candidates[0].parent
        validate_dataset(dataset, expected_counts=expected_counts, expected_species=expected_species)
        write_marker(dataset)
        dataset.rename(target)
    return target


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(URL, headers={"User-Agent": "FisherGlass-reproduction/1.0"})
    with urllib.request.urlopen(request, timeout=60) as response, partial.open("wb") as writer:
        shutil.copyfileobj(response, writer, length=4 << 20)
    if partial.stat().st_size != ARCHIVE_BYTES:
        raise ValueError(f"Incomplete or changed download: expected {ARCHIVE_BYTES} bytes; kept {partial}")
    partial.replace(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/wilds"))
    parser.add_argument("--archive", type=Path, help="Use an already downloaded original archive; no network access")
    parser.add_argument("--verify-only", action="store_true", help="Validate existing data without downloading or writing a marker")
    args = parser.parse_args()
    target = args.root / "waterbirds_v1.0"
    if target.is_symlink():
        raise SystemExit("Dataset directory must not be a symbolic link")
    if target.exists():
        report = validate_dataset(target)
        if not args.verify_only:
            write_marker(target)
    elif args.verify_only:
        raise SystemExit(f"Missing dataset: {target}")
    else:
        archive = args.archive or args.root / "_downloads/waterbird_complete95_forest2water2.tar.gz"
        if not archive.is_file():
            if args.archive:
                raise SystemExit(f"Requested local archive is missing: {archive}")
            print(f"Downloading original Waterbirds from {URL}", flush=True)
            download(archive)
        target = stage_archive(archive, args.root)
        report = validate_dataset(target)
        digest = hashlib.sha256()
        with archive.open("rb") as reader:
            for block in iter(lambda: reader.read(4 << 20), b""):
                digest.update(block)
        report["source_archive_sha256"] = digest.hexdigest()
    print(json.dumps(report, indent=2))
    print(f"Waterbirds ready: {target}")


if __name__ == "__main__":
    main()
