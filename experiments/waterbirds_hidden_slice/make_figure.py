"""Create the four-panel hidden-slice discovery and repair figure.

Panels A--C are downstream, post-hoc views of already-frozen selectors.  The
script consumes only a hashed qualitative artifact emitted by ``posthoc.py``;
it never opens the sealed metadata bundle itself.
"""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from common import (
    atomic_write_json,
    file_sha256,
    json_sha256,
    require_current_implementation_manifest,
    require_json_artifact_sha256,
)


GROUP_NAMES = ("landbird/land", "landbird/water", "waterbird/land", "waterbird/water")
GROUP_COLORS = ("#4C78A8", "#72B7B2", "#F58518", "#E45756")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _read_aggregate(path: Path) -> dict[str, Any]:
    aggregate = _read_json(path)
    if aggregate.get("stage") != "five_seed_reporting_aggregate" or aggregate.get("schema_version") != 1:
        raise RuntimeError("unsupported aggregate schema")
    expected = aggregate.get("aggregate_sha256")
    unhashed = dict(aggregate)
    unhashed.pop("aggregate_sha256", None)
    if expected != json_sha256(unhashed):
        raise RuntimeError("aggregate.json content hash mismatch")
    require_current_implementation_manifest(
        aggregate.get("implementation"),
        label="figure aggregate",
    )
    return aggregate


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        return {key: np.asarray(archive[key]) for key in archive.files}


def _load_posthoc_inputs(out_dir: Path, seed: int, aggregate: dict[str, Any]):
    seed_dir = out_dir / "runs" / f"seed_{seed}"
    posthoc_path = seed_dir / "posthoc.json"
    blind_path = seed_dir / "blind_test_results.json"
    score_path = seed_dir / "blind_test_scores.npz"
    posthoc = _read_json(posthoc_path)
    blind = _read_json(blind_path)
    require_json_artifact_sha256(posthoc, "posthoc_sha256", label=f"seed {seed} post-hoc")
    require_json_artifact_sha256(blind, "blind_test_sha256", label=f"seed {seed} blind certification")
    recorded = aggregate.get("inputs", {}).get(str(seed), {}).get("sha256", {})
    if file_sha256(posthoc_path) != recorded.get("posthoc"):
        raise RuntimeError("post-hoc JSON changed after aggregation")
    if file_sha256(blind_path) != recorded.get("blind_test"):
        raise RuntimeError("blind certification JSON changed after aggregation")
    if posthoc.get("metadata_revealed_after_blind_freeze") is not True:
        raise RuntimeError("refusing to use qualitative metadata before the post-hoc boundary")
    if posthoc.get("blind_test_sha256") != blind.get("blind_test_sha256"):
        raise RuntimeError("post-hoc artifact does not match blind certification")
    if file_sha256(score_path) != blind.get("score_artifact_sha256"):
        raise RuntimeError("blind score artifact hash mismatch")

    qualitative_path = Path(posthoc.get("oracle_score_artifact", ""))
    if not qualitative_path.is_file() or file_sha256(qualitative_path) != posthoc.get("oracle_score_artifact_sha256"):
        raise RuntimeError("post-hoc qualitative artifact is missing or changed")
    qualitative = _load_npz(qualitative_path)
    scores = _load_npz(score_path)
    if not np.array_equal(scores["row_id"], scores["row_id"].astype(np.int64)):
        raise RuntimeError("test score row identifiers are not integral")
    return posthoc, blind, scores, qualitative, {
        "posthoc": posthoc_path,
        "blind_test": blind_path,
        "scores": score_path,
        "posthoc_qualitative": qualitative_path,
    }


def _metadata_for_scores(scores: dict[str, np.ndarray], qualitative: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    required = {"row_id", "img_filename", "group", "split"}
    if not required.issubset(qualitative):
        raise RuntimeError(f"post-hoc qualitative artifact is missing {sorted(required.difference(qualitative))}")
    lookup = {int(row): index for index, row in enumerate(qualitative["row_id"])}
    if len(lookup) != len(qualitative["row_id"]):
        raise RuntimeError("post-hoc qualitative row identifiers are not unique")
    try:
        positions = np.asarray([lookup[int(row)] for row in scores["row_id"]], dtype=np.int64)
    except KeyError as exc:
        raise RuntimeError(f"test row missing from sealed metadata: {exc}") from exc
    if not np.all(np.asarray(qualitative["split"])[positions] == 2):
        raise RuntimeError("qualitative panel contains a non-test image")
    return {
        "img_filename": np.asarray(qualitative["img_filename"])[positions],
        "group": np.asarray(qualitative["group"], dtype=np.int64)[positions],
    }


def _top_positions(scores: dict[str, np.ndarray], prefix: str, top_k: int) -> np.ndarray:
    q = np.asarray(scores[f"{prefix}_q"], dtype=np.float64)
    membership = np.asarray(scores[f"{prefix}_membership"], dtype=np.float64)
    if q.shape != membership.shape:
        raise RuntimeError(f"{prefix} q/membership arrays do not align")
    support = np.flatnonzero(np.isfinite(membership))
    if support.size < top_k:
        raise RuntimeError(f"{prefix} has only {support.size} transferable scores, fewer than top-k={top_k}")
    ordering = np.lexsort((-membership[support], -q[support]))
    return support[ordering[:top_k]]


def _image(path: Path) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"))


def _panel_images(
    figure,
    cell,
    *,
    scores: dict[str, np.ndarray],
    metadata: dict[str, np.ndarray],
    image_root: Path,
    top_k: int,
) -> list[dict[str, Any]]:
    subfigure = figure.add_subfigure(cell)
    subfigure.suptitle(
        "A  Frozen selectors: highest-weight held-out images",
        x=0.0,
        ha="left",
        fontweight="bold",
        fontsize=9,
    )
    grid = subfigure.add_gridspec(2, top_k, wspace=0.03, hspace=0.20)
    manifest = []
    for row_index, (prefix, label) in enumerate(
        (("erm_fisher", "CLIP Fisher"), ("erm_loss", "High loss"))
    ):
        positions = _top_positions(scores, prefix, top_k)
        for column, position in enumerate(positions):
            axis = subfigure.add_subplot(grid[row_index, column])
            filename = str(metadata["img_filename"][position])
            group_id = int(metadata["group"][position])
            axis.imshow(_image(image_root / filename))
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(label, fontsize=8, fontweight="bold")
            species_dir = Path(filename).parent.name
            species = species_dir.split(".", 1)[-1].replace("_", " ")
            axis.set_title(f"{species[:15]}\n{GROUP_NAMES[group_id]}", fontsize=5.5, pad=1)
            for spine in axis.spines.values():
                spine.set_linewidth(1.5)
                spine.set_color("#2F4B7C" if row_index == 0 else "#A05195")
            manifest.append(
                {
                    "selector": prefix,
                    "rank": column + 1,
                    "score_position": int(position),
                    "row_id": int(scores["row_id"][position]),
                    "filename": filename,
                    "posthoc_group": GROUP_NAMES[group_id],
                    "q": float(scores[f"{prefix}_q"][position]),
                    "membership": float(scores[f"{prefix}_membership"][position]),
                }
            )
    return manifest


def _row_map(aggregate: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {row["key"]: row for row in aggregate["main_table"]}


def _require_complete_paper_protocol(aggregate: dict[str, Any]) -> None:
    if not aggregate.get("paper_protocol_complete"):
        raise RuntimeError(
            "refusing to create a paper figure from an incomplete run; complete all five predeclared seeds"
        )


def _repair_tick_label(row: dict[str, Any], label: str) -> str:
    selection = row.get("repair_selection")
    if selection and selection.get("any_no_op"):
        return f"{label}\nNO-OP {selection['no_op_seeds']}/{selection['total_seeds']}"
    return label


def _panel_composition(axis, aggregate: dict[str, Any]) -> None:
    compositions = {entry["key"]: entry["composition"] for entry in aggregate["composition"]}
    keys = ("erm_hidden_fisher", "erm_high_loss", "erm_random_nuisance", "oracle_nuisance_probe")
    labels = ("CLIP\nFisher", "High\nloss", "Random\naxis", "Oracle\nprobe*")
    bottom = np.zeros(len(keys), dtype=np.float64)
    for group, color in zip(GROUP_NAMES, GROUP_COLORS):
        values = np.asarray(
            [compositions[key]["weighted_group_composition"][group]["mean"] for key in keys],
            dtype=np.float64,
        )
        axis.bar(np.arange(len(keys)), values, bottom=bottom, color=color, width=0.72, label=group)
        bottom += values
    axis.set_xticks(np.arange(len(keys)), labels)
    axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Mean soft-slice composition")
    axis.set_title("B  Post-hoc metadata composition", loc="left", fontweight="bold")
    axis.legend(fontsize=7, ncol=2, frameon=False, loc="upper center")
    axis.text(0.99, 0.02, "* oracle nuisance sensitivity", transform=axis.transAxes, ha="right", fontsize=7)
    axis.grid(axis="y", alpha=0.18)


def _point(axis, *, x, y, xerr, yerr, label, color, marker, annotate=True) -> None:
    axis.errorbar(
        x,
        y,
        xerr=xerr,
        yerr=yerr,
        fmt=marker,
        color=color,
        markersize=7,
        capsize=2.5,
        linewidth=1.1,
        label=label,
    )
    if annotate:
        axis.annotate(label, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7)


def _panel_loss_fisher(axis, aggregate: dict[str, Any]) -> None:
    rows = _row_map(aggregate)
    selector_specs = (
        ("erm_hidden_fisher", "CLIP Fisher", "#2F4B7C", "o"),
        ("erm_high_loss", "High loss", "#A05195", "s"),
        ("erm_random_slice", "Random slice", "#7A7A7A", "^"),
        ("erm_random_nuisance", "Random axis", "#59A14F", "D"),
    )
    for key, label, color, marker in selector_specs:
        fisher = rows[key]["metrics"]["predictive_fisher"]
        loss = rows[key]["metrics"]["slice_loss"]
        _point(
            axis,
            x=fisher["mean"],
            y=loss["mean"],
            xerr=fisher["half_width"],
            yerr=loss["half_width"],
            label=label,
            color=color,
            marker=marker,
        )
    for index, group in enumerate(aggregate["predefined_groups"]):
        fisher = group["metrics"]["predictive_fisher"]
        loss = group["metrics"]["loss"]
        short = group["group"].replace("bird/", "/")
        _point(
            axis,
            x=fisher["mean"],
            y=loss["mean"],
            xerr=fisher["half_width"],
            yerr=loss["half_width"],
            label=f"Oracle {short}",
            color=GROUP_COLORS[index],
            marker="x",
            annotate=False,
        )
    axis.set_xlabel("Held-out nuisance-projected Fisher  ← lower information")
    axis.set_ylabel("Held-out slice loss")
    axis.set_title("C  Loss and projected Fisher identify different slices", loc="left", fontweight="bold")
    axis.grid(alpha=0.18)
    axis.legend(fontsize=6.7, ncol=2, frameon=False)


def _panel_repairs(axis, aggregate: dict[str, Any]) -> None:
    rows = _row_map(aggregate)
    keys = (
        "erm_hidden_fisher",
        "erm_continuation",
        "loss_repair",
        "fisher_repair",
        "combined_repair",
        "oracle_group_dro",
    )
    base_labels = ("ERM", "ERM+", "Loss", "Fisher", "Combined", "Group DRO*")
    labels = [_repair_tick_label(rows[key], label) for key, label in zip(keys, base_labels)]
    x = np.arange(len(keys), dtype=np.float64)
    fisher = [rows[key]["metrics"]["predictive_fisher"] for key in keys]
    fisher_values = np.asarray([entry["mean"] for entry in fisher], dtype=np.float64)
    fisher_errors = np.asarray([entry["half_width"] for entry in fisher], dtype=np.float64)
    bars = axis.bar(x, fisher_values, width=0.58, color="#4C78A8", alpha=0.84, label="Hidden Fisher")
    axis.errorbar(x, fisher_values, yerr=fisher_errors, fmt="none", ecolor="black", capsize=2, linewidth=0.8)
    bars[-1].set_hatch("//")

    accuracy_axis = axis.twinx()
    for offset, metric, label, color, marker in (
        (-0.08, "overall_accuracy", "Average accuracy", "#59A14F", "o"),
        (0.08, "oracle_worst_group_accuracy", "Oracle worst accuracy", "#F58518", "s"),
    ):
        summaries = [rows[key]["metrics"][metric] for key in keys]
        values = np.asarray([entry["mean"] for entry in summaries], dtype=np.float64)
        errors = np.asarray([entry["half_width"] for entry in summaries], dtype=np.float64)
        accuracy_axis.errorbar(
            x + offset,
            values,
            yerr=errors,
            color=color,
            marker=marker,
            markersize=4.5,
            linewidth=1.1,
            capsize=2,
            label=label,
        )
    axis.set_xticks(x, labels, rotation=12)
    axis.set_ylim(0.0, max(1e-12, 1.12 * float(np.max(fisher_values + fisher_errors))))
    accuracy_axis.set_ylim(0.0, 1.0)
    axis.set_ylabel("Re-discovered projected Fisher ↑", color="#4C78A8")
    accuracy_axis.set_ylabel("Accuracy", color="#59636E")
    axis.set_title("D  Fresh-adversary evaluation after repair", loc="left", fontweight="bold")
    left_handles, left_labels = axis.get_legend_handles_labels()
    right_handles, right_labels = accuracy_axis.get_legend_handles_labels()
    axis.legend(left_handles + right_handles, left_labels + right_labels, fontsize=6.8, frameon=False, ncol=2)
    axis.grid(axis="y", alpha=0.18)
    axis.text(
        0.99,
        0.02,
        "ERM+: matched extra steps; * oracle labeled-group training; NO-OP counts are disclosed per arm",
        transform=axis.transAxes,
        ha="right",
        fontsize=6.8,
    )


def _atomic_save_figure(figure, path: Path, *, dpi: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=path.parent,
            prefix=f".{path.stem}.",
            suffix=path.suffix,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
        figure.savefig(temporary, format=path.suffix.lstrip("."), dpi=dpi, bbox_inches="tight")
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="build/results/waterbirds_hidden_slice_v2/prepared")
    parser.add_argument("--out-dir", default="build/results/waterbirds_hidden_slice_v2")
    parser.add_argument("--image-root", default="data/wilds/waterbirds_v1.0")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--dpi", type=int, default=220)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.top_k < 1 or args.dpi < 72:
        raise ValueError("--top-k must be positive and --dpi must be at least 72")
    out_dir = Path(args.out_dir)
    aggregate_path = out_dir / "aggregate.json"
    aggregate = _read_aggregate(aggregate_path)
    _require_complete_paper_protocol(aggregate)
    if args.seed not in aggregate["seeds"]:
        raise ValueError(f"seed {args.seed} is not present in aggregate.json")
    _posthoc, _blind, scores, qualitative, input_paths = _load_posthoc_inputs(
        out_dir,
        args.seed,
        aggregate,
    )
    metadata = _metadata_for_scores(scores, qualitative)

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    figure = plt.figure(figsize=(15.5, 9.5), constrained_layout=True)
    grid = figure.add_gridspec(2, 2, height_ratios=(1.08, 1.0), width_ratios=(1.12, 1.0))
    top_images = _panel_images(
        figure,
        grid[0, 0],
        scores=scores,
        metadata=metadata,
        image_root=Path(args.image_root),
        top_k=args.top_k,
    )
    _panel_composition(figure.add_subplot(grid[0, 1]), aggregate)
    _panel_loss_fisher(figure.add_subplot(grid[1, 0]), aggregate)
    _panel_repairs(figure.add_subplot(grid[1, 1]), aggregate)
    figure.suptitle(
        "Waterbirds: metadata-blind hidden-slice discovery, certification, and repair",
        fontsize=13,
        fontweight="bold",
    )

    figure_dir = out_dir / "figures"
    pdf_path = _atomic_save_figure(figure, figure_dir / "waterbirds_hidden_slice.pdf", dpi=args.dpi)
    png_path = _atomic_save_figure(figure, figure_dir / "waterbirds_hidden_slice.png", dpi=args.dpi)
    plt.close(figure)

    manifest = {
        "schema_version": 1,
        "stage": "posthoc_four_panel_figure",
        "seed_for_qualitative_panel": args.seed,
        "top_k": args.top_k,
        "metadata_opened_only_after_posthoc_freeze_check": True,
        "paper_protocol_complete": True,
        "repair_selection": {
            row["key"]: row.get("repair_selection")
            for row in aggregate["main_table"]
            if row["key"] in {"loss_repair", "fisher_repair", "combined_repair"}
        },
        "top_images": top_images,
        "inputs": {
            "aggregate": {"path": str(aggregate_path), "sha256": file_sha256(aggregate_path)},
            **{
                name: {"path": str(path), "sha256": file_sha256(path)} for name, path in input_paths.items()
            },
        },
        "outputs": {
            "pdf": {"path": str(pdf_path), "sha256": file_sha256(pdf_path)},
            "png": {"path": str(png_path), "sha256": file_sha256(png_path)},
        },
    }
    manifest["figure_sha256"] = json_sha256(manifest)
    manifest_path = atomic_write_json(figure_dir / "waterbirds_hidden_slice_manifest.json", manifest)
    print(f"wrote {pdf_path}, {png_path}, and {manifest_path}", flush=True)


if __name__ == "__main__":
    main()
