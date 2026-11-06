"""Step 4: score the frozen models on the frozen paired images.

Everything read here was fixed before any shifted image was seen: the
checkpoint, the temperature, the background probe, the slice definition, the
bin edges and the numerical constants all come from ``frozen_selection.json``,
and the run aborts if a checkpoint hash no longer matches.

Reported per model and per slice
--------------------------------
``acc_original``    held-out accuracy on the untouched images
``acc_same``        after replacing the background with another of the same type
``acc_opposite``    after replacing it with one of the opposite type
``delta_nuisance``  ``acc_same - acc_opposite`` -- the compositing-controlled drop
``confident_flip``  P[ wrong after the opposite swap | correct originally and
                    calibrated confidence >= tau ], i.e. restricted to
                    predictions that source accuracy and source loss call safe
``confident_flip_same``  the identical quantity for the same-type swap, which is
                    the artifact floor the intervention has to beat

Uncertainty is a cluster bootstrap: draws of the same bird are resampled
together, at the source-image level and again at the species level.
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

from common import (
    GROUP_NAMES,
    binary_nll,
    cluster_bootstrap,
    file_sha256,
    log,
    mean_ci,
    read_json,
    sigmoid,
    write_json,
)
from data import ManifestDataset, make_loader, read_manifest
from models import (
    amp_dtype,
    autocast_context,
    enable_cuda_speedups,
    get_transform,
    load_checkpoint,
    resolve_device,
)
from select_slice import assign_bin


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def score_manifest(model, records: list[dict], arch: str, device: torch.device, args) -> np.ndarray:
    """Logits for every manifest row, in manifest order."""
    dataset = ManifestDataset(records, get_transform(arch))
    loader = make_loader(dataset, args.batch_size, args)
    dtype = amp_dtype(args.amp, device)
    out = np.zeros(len(records), dtype=np.float64)
    cursor = 0
    for images, _y, _group, _place, _idx in loader:
        images = images.to(device, non_blocking=True)
        if args.channels_last and device.type == "cuda":
            images = images.contiguous(memory_format=torch.channels_last)
        with autocast_context(device, dtype):
            logits, _ = model(images)
        values = logits.float().cpu().numpy()
        out[cursor : cursor + values.size] = values
        cursor += values.size
    if cursor != len(records):
        raise RuntimeError(f"scored {cursor} of {len(records)} manifest rows")
    return out


# --------------------------------------------------------------------------- #
# Per-source aggregation
# --------------------------------------------------------------------------- #
def write_flip_records(
    path: Path,
    records: list[dict],
    manifest_logits: np.ndarray,
    selection: dict,
    original_logit_by_row: dict[int, float],
    z_by_row: dict[int, float],
    b_by_row: dict[int, float],
    temperature: float,
) -> Path:
    """Persist one row per (source image, arm, draw) for the individual-level analysis.

    Everything a per-image failure model could need: the source-side covariates a
    practitioner already has (loss, calibrated confidence, margin), the group, and
    the Fisher-geometry terms. ``beta`` is the validation Schur coefficient
    J_TN/(J_NN+ridge) of the row's group, so ``beta * b`` is the part of the task
    score attributable to the audited nuisance direction and ``z - beta * b`` is
    the transverse residual.
    """
    beta_by_group = {int(row["gid"]): float(row["beta"]) for row in selection["families"]["groups"]["slices"]}
    species_vocab = sorted({str(r["species"]) for r in records})
    species_index = {name: i for i, name in enumerate(species_vocab)}

    keep = [i for i, r in enumerate(records) if r["source_row"] in original_logit_by_row]
    columns: dict[str, list] = {key: [] for key in (
        "source_row", "species_code", "group", "y", "arm_opposite", "draw",
        "correct_after", "original_correct", "nll", "confidence_true", "margin", "z", "b", "beta",
    )}
    for i in keep:
        record = records[i]
        source = record["source_row"]
        y = int(record["y"])
        original = float(original_logit_by_row[source])
        probability = float(sigmoid(np.array([original / temperature]))[0])
        group = int(record["group"])
        columns["source_row"].append(source)
        columns["species_code"].append(species_index[str(record["species"])])
        columns["group"].append(group)
        columns["y"].append(y)
        columns["arm_opposite"].append(1 if record["arm"] == "opposite" else 0)
        columns["draw"].append(int(record["draw"]))
        columns["correct_after"].append(int((manifest_logits[i] >= 0.0) == y))
        columns["original_correct"].append(int((original >= 0.0) == y))
        columns["nll"].append(float(binary_nll(np.array([original]), np.array([y]))[0]))
        columns["confidence_true"].append(probability if y == 1 else 1.0 - probability)
        columns["margin"].append(abs(original))
        columns["z"].append(float(z_by_row.get(source, np.nan)))
        columns["b"].append(float(b_by_row.get(source, np.nan)))
        columns["beta"].append(beta_by_group.get(group, float("nan")))

    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        species_vocab=np.array(species_vocab),
        **{key: np.asarray(values) for key, values in columns.items()},
    )
    return path


def build_per_source(
    records: list[dict],
    manifest_logits: np.ndarray,
    original_logit_by_row: dict[int, float],
    temperature: float,
) -> dict:
    """Collapse manifest rows into per-source-image correctness arrays."""
    if len(records) != len(manifest_logits):
        raise RuntimeError(f"manifest has {len(records)} rows but {len(manifest_logits)} logits")
    by_source: dict[int, dict] = defaultdict(lambda: {"same": [], "opposite": []})
    info: dict[int, dict] = {}
    for record, logit in zip(records, manifest_logits):
        source_row = record["source_row"]
        by_source[source_row][record["arm"]].append(logit)
        info.setdefault(
            source_row,
            {"y": record["y"], "species": record["species"], "group": record["group"], "place": record["source_place"]},
        )

    sources = sorted(s for s in by_source if s in original_logit_by_row)
    y = np.array([info[s]["y"] for s in sources], dtype=int)
    species = np.array([info[s]["species"] for s in sources])
    group = np.array([info[s]["group"] for s in sources], dtype=int)

    original_logits = np.array([original_logit_by_row[s] for s in sources], dtype=float)
    original_correct = ((original_logits >= 0.0).astype(int) == y).astype(float)
    calibrated = sigmoid(original_logits / temperature)
    confidence_true = np.where(y == 1, calibrated, 1.0 - calibrated)

    correct = {}
    for arm in ("same", "opposite"):
        rows = []
        for s in sources:
            draws = np.asarray(by_source[s][arm], dtype=float)
            rows.append(float(np.mean((draws >= 0.0).astype(int) == info[s]["y"])) if draws.size else np.nan)
        correct[arm] = np.asarray(rows, dtype=float)

    return {
        "sources": np.asarray(sources, dtype=int),
        "y": y,
        "species": species,
        "group": group,
        "original_logits": original_logits,
        "original_correct": original_correct,
        "original_nll": binary_nll(original_logits, y),
        "confidence_true": confidence_true,
        "correct_same": correct["same"],
        "correct_opposite": correct["opposite"],
    }


def slice_report(per_source: dict, mask: np.ndarray, args, label: str) -> dict:
    """All shift statistics for one slice, with cluster bootstraps."""
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return {"slice": label, "n": 0}

    original_correct = per_source["original_correct"][idx]
    correct_same = per_source["correct_same"][idx]
    correct_opposite = per_source["correct_opposite"][idx]
    confidence = per_source["confidence_true"][idx]
    species = per_source["species"][idx]

    # Three nested conditioning sets, from most portable to most stringent.
    #
    # A fixed calibrated-probability threshold is not comparable across slices:
    # Waterbirds is 77% landbird, so the model's landbird logits are large while
    # its waterbird logits sit near zero, and one global temperature can leave a
    # slice with essentially no prediction above p >= 0.9. The quantile variant
    # keeps the "confidently correct" framing with a guaranteed denominator.
    correct_mask = original_correct > 0.5
    absolute_mask = correct_mask & (confidence >= args.confidence_threshold)
    if correct_mask.any():
        quantile_cut = float(np.quantile(confidence[correct_mask], 1.0 - args.confident_quantile))
    else:
        quantile_cut = float("inf")
    quantile_mask = correct_mask & (confidence >= quantile_cut)

    def acc(values):
        return lambda rows: float(np.nanmean(values[rows]))

    def delta(rows):
        return float(np.nanmean(correct_same[rows]) - np.nanmean(correct_opposite[rows]))

    def flip_on(values, keep_mask):
        def statistic(rows):
            keep = rows[keep_mask[rows]]
            if keep.size == 0:
                return float("nan")
            return float(np.nanmean(1.0 - values[keep]))

        return statistic

    everything = np.arange(idx.size)
    source_cluster = np.arange(idx.size)
    report = {
        "slice": label,
        "n": int(idx.size),
        "confidence_threshold": args.confidence_threshold,
        "confident_quantile": args.confident_quantile,
        "n_correct": int(correct_mask.sum()),
        "n_confident_correct": int(absolute_mask.sum()),
        "n_confident_correct_quantile": int(quantile_mask.sum()),
        "quantile_confidence_cut": quantile_cut,
        "acc_original": float(np.mean(original_correct)),
        "nll_original": float(np.mean(per_source["original_nll"][idx])),
        "acc_same": float(np.nanmean(correct_same)),
        "acc_opposite": float(np.nanmean(correct_opposite)),
        "delta_nuisance": float(np.nanmean(correct_same) - np.nanmean(correct_opposite)),
        "delta_vs_original": float(np.mean(original_correct) - np.nanmean(correct_opposite)),
        # P[wrong after the opposite swap | correct before] -- always well populated.
        "flip_among_correct": flip_on(correct_opposite, correct_mask)(everything),
        "flip_among_correct_same": flip_on(correct_same, correct_mask)(everything),
        # Same, restricted to the most-confident fraction of correct predictions.
        "confident_flip_quantile": flip_on(correct_opposite, quantile_mask)(everything),
        "confident_flip_quantile_same": flip_on(correct_same, quantile_mask)(everything),
        # Fixed absolute threshold -- read together with n_confident_correct.
        "confident_flip": flip_on(correct_opposite, absolute_mask)(everything),
        "confident_flip_same": flip_on(correct_same, absolute_mask)(everything),
        "bootstrap_source": {
            "acc_same": cluster_bootstrap(source_cluster, acc(correct_same), args.n_boot, args.boot_seed),
            "acc_opposite": cluster_bootstrap(source_cluster, acc(correct_opposite), args.n_boot, args.boot_seed),
            "delta_nuisance": cluster_bootstrap(source_cluster, delta, args.n_boot, args.boot_seed),
            "flip_among_correct": cluster_bootstrap(
                source_cluster, flip_on(correct_opposite, correct_mask), args.n_boot, args.boot_seed
            ),
            "confident_flip_quantile": cluster_bootstrap(
                source_cluster, flip_on(correct_opposite, quantile_mask), args.n_boot, args.boot_seed
            ),
            "confident_flip": cluster_bootstrap(
                source_cluster, flip_on(correct_opposite, absolute_mask), args.n_boot, args.boot_seed
            ),
        },
        "bootstrap_species": {
            "delta_nuisance": cluster_bootstrap(species, delta, args.n_boot, args.boot_seed + 1),
            "flip_among_correct": cluster_bootstrap(
                species, flip_on(correct_opposite, correct_mask), args.n_boot, args.boot_seed + 1
            ),
            "confident_flip_quantile": cluster_bootstrap(
                species, flip_on(correct_opposite, quantile_mask), args.n_boot, args.boot_seed + 1
            ),
        },
    }
    return report


# --------------------------------------------------------------------------- #
# One model
# --------------------------------------------------------------------------- #
def evaluate_run(tag: str, frozen: dict, records: list[dict], args) -> dict:
    run_dir = Path(args.out_dir) / "runs" / tag
    selection = frozen["runs"][tag]
    device = resolve_device(args.device)
    enable_cuda_speedups(args.tf32)

    actual = file_sha256(run_dir / "checkpoint.pt")
    if actual != selection["checkpoint_sha256"]:
        message = f"{tag}: checkpoint changed since selection was frozen ({actual[:12]} != {selection['checkpoint_sha256'][:12]})"
        if not args.allow_drift:
            raise SystemExit(message + " -- rerun select_slice.py or pass --allow-drift")
        log("WARNING " + message)

    diagnostics = np.load(run_dir / "diagnostics.npz")
    test_rows = diagnostics["test_rows"]
    scores = np.load(run_dir / "scores.npz")
    if not (len(test_rows) == len(scores["test_logits"]) == len(diagnostics["b_test"])):
        raise RuntimeError(f"{tag}: test_rows/logits/probe-score lengths disagree")
    original_logit_by_row = {int(r): float(v) for r, v in zip(test_rows, scores["test_logits"])}
    b_test_by_row = {int(r): float(v) for r, v in zip(test_rows, diagnostics["b_test"])}
    z_test_by_row = {int(r): float(v) for r, v in zip(test_rows, diagnostics["z_test"])}
    temperature = float(selection["temperature"])

    model = load_checkpoint(run_dir / "checkpoint.pt", selection["arch"], device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    log(f"{tag}: scoring {len(records)} intervention images on {device}")
    manifest_logits = score_manifest(model, records, selection["arch"], device, args)
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()

    per_source = build_per_source(records, manifest_logits, original_logit_by_row, temperature)
    write_flip_records(
        run_dir / "flip_records.npz",
        records,
        manifest_logits,
        selection,
        original_logit_by_row,
        z_test_by_row,
        b_test_by_row,
        temperature,
    )

    # ---- every group, so the slice the rule picked can be compared to the ones it did not
    by_group = []
    for gid in sorted(set(per_source["group"].tolist())):
        by_group.append(slice_report(per_source, per_source["group"] == gid, args, GROUP_NAMES[gid]))

    # ---- the frozen headline slice
    selected_gid = int(frozen["selected_group"]["gid"])
    headline = slice_report(per_source, per_source["group"] == selected_gid, args, GROUP_NAMES[selected_gid])
    val_slice = next(
        row for row in selection["families"]["groups"]["slices"] if int(row["gid"]) == selected_gid
    )
    headline["validation"] = val_slice
    headline["validation_rule"] = selection["families"]["groups"]["rule"]

    # ---- refined probe-score bins (validation edges, held-out flip rates)
    # Defaults to the selected group. A different group can be requested when the
    # selected one is too small to bin -- the edges and statistics were all frozen
    # before any shifted image was scored, but choosing *which* group to inspect
    # after the fact is a secondary analysis and should be labelled as one.
    statistic = selection["families"]["groups"]["rule"].get("statistic", "projected_fisher")
    bin_gid = selected_gid if args.bin_group is None else int(args.bin_group)
    bins = []
    edges = selection["bin_edges_by_group"].get(str(bin_gid))
    if edges:
        edge_array = np.asarray(edges, dtype=float)
        b_values = np.array([b_test_by_row.get(int(s), np.nan) for s in per_source["sources"]], dtype=float)
        assigned = assign_bin(b_values, edge_array)
        val_bins = {
            int(row["bin_index"]): row
            for row in selection["families"]["bins"]["slices"]
            if int(row["gid"]) == bin_gid
        }
        for bin_index in sorted(val_bins):
            mask = (per_source["group"] == bin_gid) & (assigned == bin_index) & np.isfinite(b_values)
            report = slice_report(per_source, mask, args, f"{GROUP_NAMES[bin_gid]} | probe bin {bin_index + 1}")
            report["validation"] = val_bins[bin_index]
            report["validation_inverse_projected_fisher"] = val_bins[bin_index]["inverse_projected_fisher"]
            report["validation_selection_statistic"] = val_bins[bin_index][statistic]
            report["selection_statistic_name"] = statistic
            bins.append(report)

    result = {
        "tag": tag,
        "arch": selection["arch"],
        "method": selection["method"],
        "fg_lambda": selection["fg_lambda"],
        "seed": selection["seed"],
        "temperature": temperature,
        "frozen_hash": frozen["frozen_hash"],
        "selected_group": frozen["selected_group"],
        "val_fisher_glass_certificate": selection["val_fisher_glass_certificate"],
        "val_overall_accuracy": selection["val_overall_accuracy"],
        "headline": headline,
        "by_group": by_group,
        "bins": bins,
        "bin_group": GROUP_NAMES[bin_gid],
        "bin_group_is_selected": bin_gid == selected_gid,
    }
    write_json(run_dir / "shift_results.json", result)
    log(
        f"{tag}: [{headline['slice']}] orig={headline['acc_original']:.4f} same={headline['acc_same']:.4f} "
        f"opp={headline['acc_opposite']:.4f} delta={headline['delta_nuisance']:.4f} "
        f"confident_flip={headline['confident_flip']:.4f} (control {headline['confident_flip_same']:.4f})"
    )
    return result


# --------------------------------------------------------------------------- #
# Aggregation across seeds
# --------------------------------------------------------------------------- #
def summarize(label: str, method: str, fg_lambda, items: list[dict], primary: bool) -> dict:
    head = [r["headline"] for r in items]
    return {
        "label": label,
        "method": method,
        "fg_lambda": fg_lambda,
        "primary": primary,
        "n_seeds": len(items),
        "seeds": sorted(r["seed"] for r in items),
        "slice": head[0]["slice"],
        "val_accuracy": mean_ci(r["val_overall_accuracy"] for r in items),
        "val_certificate": mean_ci(r["val_fisher_glass_certificate"] for r in items),
        "val_projected_fisher": mean_ci(h["validation"]["projected_fisher"] for h in head),
        "val_projected_over_raw": mean_ci(h["validation"]["projected_over_raw"] for h in head),
        "acc_original": mean_ci(h["acc_original"] for h in head),
        "nll_original": mean_ci(h["nll_original"] for h in head),
        "acc_same": mean_ci(h["acc_same"] for h in head),
        "acc_opposite": mean_ci(h["acc_opposite"] for h in head),
        "delta_nuisance": mean_ci(h["delta_nuisance"] for h in head),
        "flip_among_correct": mean_ci(h["flip_among_correct"] for h in head),
        "flip_among_correct_same": mean_ci(h["flip_among_correct_same"] for h in head),
        "confident_flip_quantile": mean_ci(h["confident_flip_quantile"] for h in head),
        "confident_flip_quantile_same": mean_ci(h["confident_flip_quantile_same"] for h in head),
        "confident_flip": mean_ci(h["confident_flip"] for h in head),
        "confident_flip_same": mean_ci(h["confident_flip_same"] for h in head),
        # Denominators. Temperature scaling can leave almost nothing above a fixed
        # threshold in a slice, which is what a wide CI on confident_flip means --
        # so report them rather than hiding them.
        "n_correct": mean_ci(h["n_correct"] for h in head),
        "n_confident_correct": mean_ci(h["n_confident_correct"] for h in head),
        "n_confident_correct_quantile": mean_ci(h["n_confident_correct_quantile"] for h in head),
    }


def aggregate(results: list[dict], frozen: dict) -> list[dict]:
    """ERM and the validation-chosen Fisher-Glass arm are the two primary rows.

    Splitting the Fisher-Glass arm by lambda mixes seed counts (the rule picks a
    different lambda per seed), so the per-lambda rows are kept only as a
    secondary breakdown and are excluded from the figure.
    """
    summary = []
    erm = [r for r in results if r["method"] == "erm"]
    if erm:
        summary.append(summarize("ERM", "erm", 0.0, erm, primary=True))

    choice = (frozen.get("fg_lambda_choice") or {}).get("per_seed", {})
    chosen_tags = {entry["chosen_tag"] for entry in choice.values()}
    chosen = [r for r in results if r["tag"] in chosen_tags]
    if chosen:
        lambdas = sorted({r["fg_lambda"] for r in chosen})
        # Plain unicode, not mathtext: this label goes into both a markdown table
        # and a matplotlib axis, and "$\lambda$" only renders in one of them.
        label = "ERM + FG (validation-chosen λ)" if len(lambdas) > 1 else f"ERM + FG (λ={lambdas[0]:g})"
        summary.append(summarize(label, "fisher_glass", None, chosen, primary=True))

    for fg_lambda in sorted({r["fg_lambda"] for r in results if r["method"] != "erm"}):
        items = [r for r in results if r["method"] != "erm" and r["fg_lambda"] == fg_lambda]
        summary.append(
            summarize(f"ERM + FG (λ={fg_lambda:g}) — breakdown", "fisher_glass", fg_lambda, items, primary=False)
        )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="build/prospective_shift")
    parser.add_argument("--manifest", default=None)
    parser.add_argument("--tags", nargs="+", default=None, help="default: ERM runs plus the validation-chosen Fisher-Glass runs")
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument(
        "--confident-quantile",
        type=float,
        default=0.5,
        help="fraction of the *correct* predictions in a slice, taken most-confident-first, used for "
        "the quantile flip rate. 0.5 = the more confident half. Unlike the absolute threshold this "
        "cannot end up with an empty denominator.",
    )
    parser.add_argument(
        "--bin-group",
        type=int,
        default=None,
        help="group id whose probe-score bins are reported (default: the frozen selected slice). Use this "
        "when the selected slice has too few validation examples to bin; the result is a secondary "
        "analysis, not part of the frozen selection.",
    )
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--amp", action="store_true", default=True)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--channels-last", action="store_true", default=True)
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--allow-drift", action="store_true", help="continue even if a checkpoint hash changed")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frozen = read_json(out_dir / "frozen_selection.json")
    records = read_manifest(args.manifest or out_dir / "paired_manifest.csv")

    tags = args.tags
    if tags is None:
        tags = [tag for tag, run in frozen["runs"].items() if run["method"] == "erm"]
        choice = frozen.get("fg_lambda_choice") or {}
        tags += [entry["chosen_tag"] for entry in choice.get("per_seed", {}).values()]
    tags = list(dict.fromkeys(tags))
    log(f"evaluating {len(tags)} models on {len(records)} intervention images: {tags}")

    results = [evaluate_run(tag, frozen, records, args) for tag in tags]
    summary = aggregate(results, frozen)
    write_json(out_dir / "shift_summary.json", {"args": vars(args), "summary": summary, "runs": results})

    import csv

    csv_path = out_dir / "shift_summary.csv"
    fields = [
        "label",
        "method",
        "fg_lambda",
        "n_seeds",
        "slice",
        "val_accuracy",
        "val_certificate",
        "val_projected_fisher",
        "acc_original",
        "nll_original",
        "acc_same",
        "acc_opposite",
        "delta_nuisance",
        "confident_flip",
        "confident_flip_same",
    ]
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(fields)
        for row in summary:
            writer.writerow(
                [row[f]["mean"] if isinstance(row[f], dict) else row[f] for f in fields]
            )
    log(f"wrote {out_dir / 'shift_summary.json'} and {csv_path}")


if __name__ == "__main__":
    main()
