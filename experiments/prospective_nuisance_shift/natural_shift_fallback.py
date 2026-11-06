"""Step 5: the unpaired natural-image version of the same test.

No image generation at all.  Using the slice frozen by ``select_slice.py`` (say
``landbird/land``) we compare, on the untouched official test split, that
slice's accuracy against the same-label / opposite-background slice
(``landbird/water``).  Waterbirds' validation and test splits are deliberately
balanced over class x background, so the rare combination has ample data.

This is causally weaker than the paired pixel intervention -- the birds are not
the same birds -- but it uses fully held-out natural images and needs no
compositing at all, so it is a useful independent check that the paired result
is not an artifact of the recomposition pipeline.

Everything is read from artifacts produced earlier; no model is re-run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np

from common import (
    GROUP_NAMES,
    binary_nll,
    cluster_bootstrap,
    load_metadata,
    log,
    mean_ci,
    read_json,
    sigmoid,
    write_json,
)


def slice_metrics(logits: np.ndarray, y: np.ndarray, species: np.ndarray, temperature: float, threshold: float, n_boot: int, seed: int) -> dict:
    preds = (logits >= 0.0).astype(int)
    correct = (preds == y).astype(float)
    calibrated = sigmoid(logits / temperature)
    # Confidence in the *predicted* class. Conditioning on confidence in the true
    # class would be vacuous here: p(y_true) >= 0.9 implies the prediction is
    # correct, so the conditional accuracy would be 1.0 by construction. (In
    # evaluate_shift.py the true-class version is correct, because there the
    # conditioning is on the original image and the outcome is measured after the
    # intervention.)
    confidence_pred = np.maximum(calibrated, 1.0 - calibrated)
    confident = confidence_pred >= threshold

    def accuracy(rows):
        return float(np.mean(correct[rows]))

    def confident_accuracy(rows):
        keep = rows[confident[rows]]
        return float(np.mean(correct[keep])) if keep.size else float("nan")

    return {
        "n": int(y.size),
        "accuracy": float(correct.mean()),
        "nll": float(binary_nll(logits, y).mean()),
        "nll_calibrated": float(binary_nll(logits / temperature, y).mean()),
        "n_confident": int(confident.sum()),
        "confident_accuracy": confident_accuracy(np.arange(y.size)),
        "bootstrap_species": {
            "accuracy": cluster_bootstrap(species, accuracy, n_boot, seed),
            "confident_accuracy": cluster_bootstrap(species, confident_accuracy, n_boot, seed),
        },
    }


def evaluate_run(tag: str, frozen: dict, meta, args) -> dict:
    run_dir = Path(args.out_dir) / "runs" / tag
    selection = frozen["runs"][tag]
    scores = np.load(run_dir / "scores.npz")
    test_rows = scores["test_rows"]
    logits = scores["test_logits"].astype(np.float64)
    y = scores["test_y"].astype(int)
    group = scores["test_group"].astype(int)
    species = meta.species[test_rows]
    temperature = float(selection["temperature"])

    selected_gid = int(frozen["selected_group"]["gid"])
    label = selected_gid // 2
    background = selected_gid % 2
    partner_gid = 2 * label + (1 - background)

    kwargs = dict(temperature=temperature, threshold=args.confidence_threshold, n_boot=args.n_boot, seed=args.boot_seed)
    selected = slice_metrics(logits[group == selected_gid], y[group == selected_gid], species[group == selected_gid], **kwargs)
    partner = slice_metrics(logits[group == partner_gid], y[group == partner_gid], species[group == partner_gid], **kwargs)

    val_slice = next(row for row in selection["families"]["groups"]["slices"] if int(row["gid"]) == selected_gid)
    result = {
        "tag": tag,
        "method": selection["method"],
        "fg_lambda": selection["fg_lambda"],
        "seed": selection["seed"],
        "selected_slice": GROUP_NAMES[selected_gid],
        "partner_slice": GROUP_NAMES[partner_gid],
        "validation": {
            "accuracy": val_slice["accuracy"],
            "nll": val_slice["nll"],
            "projected_fisher": val_slice["projected_fisher"],
            "projected_over_raw": val_slice["projected_over_raw"],
            "overall_accuracy": selection["val_overall_accuracy"],
        },
        "test_selected_slice": selected,
        "test_partner_slice": partner,
        "delta_accuracy": selected["accuracy"] - partner["accuracy"],
        "delta_confident_accuracy": selected["confident_accuracy"] - partner["confident_accuracy"],
    }
    write_json(run_dir / "natural_shift.json", result)
    log(
        f"{tag}: {result['selected_slice']} acc={selected['accuracy']:.4f} vs "
        f"{result['partner_slice']} acc={partner['accuracy']:.4f} "
        f"(delta={result['delta_accuracy']:.4f})"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", default="build/prospective_shift")
    parser.add_argument("--metadata", default="data/wilds/waterbirds_v1.0/metadata.csv")
    parser.add_argument("--tags", nargs="+", default=None)
    parser.add_argument("--confidence-threshold", type=float, default=0.9)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--boot-seed", type=int, default=4321)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    frozen = read_json(out_dir / "frozen_selection.json")
    meta = load_metadata(args.metadata)

    tags = args.tags
    if tags is None:
        tags = [tag for tag, run in frozen["runs"].items() if run["method"] == "erm"]
        choice = frozen.get("fg_lambda_choice") or {}
        tags += [entry["chosen_tag"] for entry in choice.get("per_seed", {}).values()]
    tags = list(dict.fromkeys(tags))

    results = [evaluate_run(tag, frozen, meta, args) for tag in tags]
    summary = []
    for method, fg_lambda in sorted({(r["method"], r["fg_lambda"]) for r in results}):
        items = [r for r in results if r["method"] == method and r["fg_lambda"] == fg_lambda]
        summary.append(
            {
                "method": method,
                "fg_lambda": fg_lambda,
                "n_seeds": len(items),
                "selected_slice": items[0]["selected_slice"],
                "partner_slice": items[0]["partner_slice"],
                "acc_selected": mean_ci(r["test_selected_slice"]["accuracy"] for r in items),
                "acc_partner": mean_ci(r["test_partner_slice"]["accuracy"] for r in items),
                "delta_accuracy": mean_ci(r["delta_accuracy"] for r in items),
                "confident_acc_selected": mean_ci(r["test_selected_slice"]["confident_accuracy"] for r in items),
                "confident_acc_partner": mean_ci(r["test_partner_slice"]["confident_accuracy"] for r in items),
            }
        )
    path = write_json(out_dir / "natural_shift_summary.json", {"args": vars(args), "summary": summary, "runs": results})
    log(f"wrote {path}")


if __name__ == "__main__":
    main()
