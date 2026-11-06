"""Step 3: build the paired, label-preserving background intervention on held-out images.

For every held-out (test-split) image in the requested groups we build two kinds
of variant, keeping the bird foreground, the matte and the label byte-identical:

``same``      the background is replaced by a *different* background of the same
              type (land -> land, water -> water).  This is the artifact control:
              it pays exactly the same recomposition cost as the intervention.
``opposite``  the background is replaced by one of the opposite type
              (land -> water, water -> land).  This is the nuisance intervention.

Backgrounds come from training-split images (disjoint from everything evaluated)
whose own bird is small, recovered by inpainting that small hole.  See
``compose.py`` for the exact inversion of the official Waterbirds composition.

By default all four test groups are generated, so the claim "the certificate
picked the slice that fails" can be checked against the slices it did *not*
pick.  Generation is model-independent: the same images are scored by every
model in step 4.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
from PIL import Image
from tqdm import tqdm

from common import GROUP_NAMES, derive_seed, load_metadata, log, read_json, write_json
from compose import DonorBank, intervene, load_source
from data import write_manifest
from mask_stats import load_or_compute

_STATE: dict = {}


def _init_worker(cfg: dict) -> None:
    meta = load_metadata(cfg["metadata"])
    stats = load_or_compute(cfg["matte_cache"], cfg["mask_root"], cfg["metadata"])
    bank = DonorBank.build(
        meta,
        cfg["image_root"],
        cfg["mask_root"],
        stats["foreground_fraction"],
        donor_split=0,
        max_foreground=cfg["max_donor_foreground"],
        inpaint_radius=cfg["inpaint_radius"],
        dilate=cfg["dilate"],
    )
    _STATE["meta"] = meta
    _STATE["bank"] = bank
    _STATE["cfg"] = cfg


def _generate(source_row: int) -> list[dict]:
    meta = _STATE["meta"]
    bank = _STATE["bank"]
    cfg = _STATE["cfg"]
    source = load_source(
        meta,
        source_row,
        cfg["image_root"],
        cfg["mask_root"],
        inpaint_radius=cfg["inpaint_radius"],
        dilate=cfg["dilate"],
    )
    species = str(meta.img_filename[source_row]).split("/")[0]
    records = []
    for arm in ("same", "opposite"):
        for draw in range(cfg["draws"]):
            image, diagnostics = intervene(
                source,
                bank,
                arm,
                draw,
                max_visible_inpaint=cfg["max_visible_inpaint"],
                donor_tries=cfg["donor_tries"],
            )
            path = Path(cfg["out_root"]) / arm / f"draw{draw}" / source.filename
            path.parent.mkdir(parents=True, exist_ok=True)
            image.save(path, quality=cfg["jpeg_quality"], subsampling=0)
            records.append(
                {
                    "source_row": int(source_row),
                    "img_id": source.img_id,
                    "img_filename": source.filename,
                    "species": species,
                    "y": source.y,
                    "source_place": source.place,
                    "group": int(2 * source.y + source.place),
                    "arm": arm,
                    "draw": int(draw),
                    "target_place": int(diagnostics["target_place"]),
                    "path": str(path),
                    "donor_filename": diagnostics["donor_filename"],
                    "visible_inpaint_fraction": float(diagnostics["visible_inpaint_fraction"]),
                    "foreground_fraction": float(diagnostics["foreground_fraction"]),
                    "rim_fraction": float(diagnostics["rim_fraction"]),
                }
            )
    return records


def select_rows(meta, groups: list[int], max_per_group: int) -> np.ndarray:
    unknown = [gid for gid in groups if gid not in GROUP_NAMES]
    if unknown:
        raise SystemExit(
            f"--groups got {unknown}, but valid group ids are "
            f"{ {gid: name for gid, name in GROUP_NAMES.items()} }"
        )
    chosen = []
    for gid in groups:
        rows = np.flatnonzero((meta.split == 2) & (meta.group == gid))
        if rows.size == 0:
            raise SystemExit(f"group {gid} ({GROUP_NAMES[gid]}) has no held-out images")
        if 0 < max_per_group < rows.size:
            rng = np.random.default_rng(derive_seed("subsample", gid, max_per_group))
            rows = np.sort(rng.choice(rows, size=max_per_group, replace=False))
        chosen.append(rows)
        log(f"group {gid} ({GROUP_NAMES[gid]}): {rows.size} held-out source images")
    return np.concatenate(chosen)


def save_preview(records: list[dict], meta, image_root: str, out_path: Path, gid: int, n: int = 6) -> None:
    by_source: dict[int, dict[str, str]] = {}
    for record in records:
        if record["group"] != gid or record["draw"] != 0:
            continue
        by_source.setdefault(record["source_row"], {})[record["arm"]] = record["path"]
    sources = [row for row, arms in by_source.items() if {"same", "opposite"} <= set(arms)]
    if not sources:
        return
    rng = np.random.default_rng(derive_seed("preview", gid))
    picks = rng.choice(sources, size=min(n, len(sources)), replace=False)
    tile = 224
    grid = Image.new("RGB", (tile * 3, tile * len(picks)), "white")
    for i, source_row in enumerate(picks):
        paths = [Path(image_root) / str(meta.img_filename[source_row]), by_source[source_row]["same"], by_source[source_row]["opposite"]]
        for j, path in enumerate(paths):
            grid.paste(Image.open(path).convert("RGB").resize((tile, tile), Image.LANCZOS), (tile * j, tile * i))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    grid.save(out_path, quality=92)
    log(f"preview (original | same-type | opposite-type): {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-root", default="data/wilds/waterbirds_v1.0")
    parser.add_argument("--mask-root", default="data/cub_segmentations/segmentations")
    parser.add_argument("--metadata", default="data/wilds/waterbirds_v1.0/metadata.csv")
    parser.add_argument("--out-dir", default="build/prospective_shift")
    parser.add_argument("--groups", type=int, nargs="+", default=[0, 1, 2, 3])
    parser.add_argument("--only-selected-group", action="store_true", help="restrict to the frozen slice from select_slice.py")
    parser.add_argument("--draws", type=int, default=2, help="independent background draws per bird per arm")
    parser.add_argument("--max-per-group", type=int, default=0, help="0 = every held-out image in the group")
    parser.add_argument("--max-donor-foreground", type=float, default=0.08)
    parser.add_argument("--max-visible-inpaint", type=float, default=0.03)
    parser.add_argument("--donor-tries", type=int, default=4)
    parser.add_argument("--inpaint-radius", type=int, default=5)
    parser.add_argument("--dilate", type=int, default=2)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--workers", type=int, default=max(1, (mp.cpu_count() or 2) - 2))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    meta = load_metadata(args.metadata)
    groups = list(args.groups)
    if args.only_selected_group:
        frozen = read_json(out_dir / "frozen_selection.json")
        groups = [int(frozen["selected_group"]["gid"])]
        log(f"restricting to frozen selected slice: {frozen['selected_group']['name']}")

    # Warm the matte cache in the parent so workers only read it.
    load_or_compute(out_dir / "matte_stats.npz", args.mask_root, args.metadata)

    cfg = {
        "metadata": args.metadata,
        "image_root": args.image_root,
        "mask_root": args.mask_root,
        "matte_cache": str(out_dir / "matte_stats.npz"),
        "out_root": str(out_dir / "paired"),
        "draws": args.draws,
        "max_donor_foreground": args.max_donor_foreground,
        "max_visible_inpaint": args.max_visible_inpaint,
        "donor_tries": args.donor_tries,
        "inpaint_radius": args.inpaint_radius,
        "dilate": args.dilate,
        "jpeg_quality": args.jpeg_quality,
    }

    rows = select_rows(meta, groups, args.max_per_group)
    total = rows.size * args.draws * 2
    log(f"generating {total} images for {rows.size} source birds using {args.workers} workers")

    records: list[dict] = []
    if args.workers <= 1:
        _init_worker(cfg)
        for source_row in tqdm(rows, desc="compose"):
            records.extend(_generate(int(source_row)))
    else:
        ctx = mp.get_context("spawn" if sys.platform == "win32" else "fork")
        with ctx.Pool(processes=args.workers, initializer=_init_worker, initargs=(cfg,)) as pool:
            for batch in tqdm(pool.imap_unordered(_generate, [int(r) for r in rows], chunksize=8), total=rows.size, desc="compose"):
                records.extend(batch)

    records.sort(key=lambda r: (r["source_row"], r["arm"], r["draw"]))
    for i, record in enumerate(records):
        record["row"] = i
    manifest = write_manifest(out_dir / "paired_manifest.csv", records)

    visible = np.array([r["visible_inpaint_fraction"] for r in records], dtype=float)
    summary = {
        "args": vars(args),
        "n_source_images": int(rows.size),
        "n_generated": len(records),
        "groups": {GROUP_NAMES[g]: int(sum(1 for r in records if r["group"] == g)) for g in groups},
        "visible_inpaint_fraction": {
            "mean": float(visible.mean()),
            "p95": float(np.quantile(visible, 0.95)),
            "max": float(visible.max()),
        },
        "visible_inpaint_by_arm": {
            arm: float(np.mean([r["visible_inpaint_fraction"] for r in records if r["arm"] == arm]))
            for arm in ("same", "opposite")
        },
        "manifest": str(manifest),
    }
    write_json(out_dir / "paired_generation.json", summary)
    for gid in groups:
        save_preview(records, meta, args.image_root, out_dir / "previews" / f"paired_group{gid}.jpg", gid)
    log(
        f"wrote {len(records)} images; visible-inpaint mean={summary['visible_inpaint_fraction']['mean']:.4f} "
        f"(same={summary['visible_inpaint_by_arm']['same']:.4f}, opposite={summary['visible_inpaint_by_arm']['opposite']:.4f})"
    )
    log(f"manifest: {manifest}")


if __name__ == "__main__":
    main()
