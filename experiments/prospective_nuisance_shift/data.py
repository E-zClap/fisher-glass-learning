"""Datasets and loader helpers.

Two datasets are needed: the original Waterbirds composites (indexed by
metadata row) and the generated intervention images (indexed by a manifest
produced in ``build_paired_images.py``).  Both apply the same architecture
transform so originals and interventions are strictly comparable.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from common import group_ids

MANIFEST_FIELDS = [
    "row",
    "source_row",
    "img_id",
    "img_filename",
    "species",
    "y",
    "source_place",
    "group",
    "arm",
    "draw",
    "target_place",
    "path",
    "donor_filename",
    "visible_inpaint_fraction",
    "foreground_fraction",
    "rim_fraction",
]


class WaterbirdsDataset(Dataset):
    """Original Waterbirds composites, selected by metadata row."""

    def __init__(self, image_root: str | Path, filenames, y, place, transform):
        self.image_root = Path(image_root)
        self.filenames = [str(f) for f in filenames]
        self.y = np.asarray(y, dtype=np.int64)
        self.place = np.asarray(place, dtype=np.int64)
        self.groups = group_ids(self.y, self.place)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.filenames)

    def __getitem__(self, i: int):
        image = Image.open(self.image_root / self.filenames[i]).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(int(self.y[i])),
            torch.tensor(int(self.groups[i])),
            torch.tensor(float(self.place[i]), dtype=torch.float32),
            torch.tensor(int(i)),
        )


class ManifestDataset(Dataset):
    """Generated intervention images listed in a manifest."""

    def __init__(self, records: list[dict], transform):
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, i: int):
        record = self.records[i]
        image = Image.open(record["path"]).convert("RGB")
        return (
            self.transform(image),
            torch.tensor(int(record["y"])),
            torch.tensor(int(record["group"])),
            torch.tensor(float(record["target_place"])),
            torch.tensor(int(i)),
        )


def read_manifest(path: str | Path) -> list[dict]:
    rows = []
    with Path(path).open(encoding="utf-8", newline="") as handle:
        for record in csv.DictReader(handle):
            record["row"] = int(record["row"])
            record["source_row"] = int(record["source_row"])
            record["img_id"] = int(record["img_id"])
            record["y"] = int(record["y"])
            record["source_place"] = int(record["source_place"])
            record["group"] = int(record["group"])
            record["draw"] = int(record["draw"])
            record["target_place"] = int(record["target_place"])
            record["visible_inpaint_fraction"] = float(record["visible_inpaint_fraction"])
            record["foreground_fraction"] = float(record["foreground_fraction"])
            record["rim_fraction"] = float(record["rim_fraction"])
            rows.append(record)
    return rows


def write_manifest(path: str | Path, records: list[dict]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        for record in records:
            writer.writerow({key: record[key] for key in MANIFEST_FIELDS})
    return path


def make_loader(dataset: Dataset, batch_size: int, args, shuffle: bool = False, drop_last: bool = False) -> DataLoader:
    kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "drop_last": drop_last,
        "num_workers": args.num_workers,
        "pin_memory": bool(getattr(args, "pin_memory", False)),
    }
    if args.num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = int(getattr(args, "prefetch_factor", 4))
    return DataLoader(dataset, **kwargs)
