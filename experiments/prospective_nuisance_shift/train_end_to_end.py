"""Step 1: end-to-end training on Waterbirds (CUDA-ready).

This replaces the frozen-feature logistic head used elsewhere in the repo: the
whole backbone is fine-tuned, over several seeds, so the prospective experiment
is about an actually-trained model rather than a linear probe.

Methods
-------
``erm``                     plain cross-entropy (headline model)
``fisher_glass``            ERM + lambda * CVaR_alpha(1 / projected Fisher)
``group_dro``               worst-group loss (secondary control)
``group_dro_fisher_glass``  both (secondary control)

The Fisher-Glass *training* penalty conditions on slices and projects a nuisance
coordinate out of the task logit.  That only means anything if the nuisance
actually varies *inside* a slice.  Conditioning on the four (label, background)
groups while using the binary background annotation does not: background is
constant within a group by construction, the Schur complement degenerates to the
within-group logit variance, and the penalty ends up fighting the classification
loss.  The default is therefore ``--fg-nuisance probe``: a linear background
probe trained online on *detached* pooled features, supplying a continuous score
that varies inside every slice.

That probe is fit on the training split with no cross-fitting; the diagnostic in
``select_slice.py`` fits an independent, cross-fitted probe on the *validation*
representation.  Different data and a different estimator, so the certificate is
not simply read off in the coordinates it was optimized in -- but the two are
related, and the honest statement is that they share a geometry rather than
being independent.

Per run we persist the checkpoint, the logits, and the pooled backbone features
for every train/val/test image.  Every later step reads those artifacts and
never retrains.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common import (
    GROUP_NAMES,
    binary_nll,
    cvar,
    fisher_schur,
    load_metadata,
    log,
    set_seed,
    write_json,
)
from data import WaterbirdsDataset, make_loader
from models import (
    FeatureClassifier,
    amp_dtype,
    autocast_context,
    enable_cuda_speedups,
    get_transform,
    make_grad_scaler,
    resolve_device,
)

NUM_GROUPS = 4


# --------------------------------------------------------------------------- #
# Objectives
# --------------------------------------------------------------------------- #
def torch_cvar(values: torch.Tensor, alpha: float) -> torch.Tensor:
    k = max(1, int(np.ceil((1.0 - alpha) * values.numel())))
    return torch.topk(values, k=k, largest=True).values.mean()


def penalty_slices(args: argparse.Namespace, y: torch.Tensor, groups: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Slices the penalty conditions on.

    ``groups``  the four (label, background) groups -- only meaningful when the
                nuisance coordinate varies *within* a group, which is true for
                ``--fg-nuisance probe`` but false for the binary background
                annotation (background is constant inside a group by definition).
    ``labels``  the two classes, so the binary background varies inside a slice.
    ``batch``   one slice per batch.
    """
    if args.fg_slices == "groups":
        return groups, NUM_GROUPS
    if args.fg_slices == "labels":
        return y.long(), 2
    if args.fg_slices == "batch":
        return torch.zeros_like(groups), 1
    raise ValueError(f"unknown --fg-slices: {args.fg_slices}")


def fisher_glass_penalty(
    logits: torch.Tensor,
    nuisance: torch.Tensor,
    slice_ids: torch.Tensor,
    num_slices: int,
    args: argparse.Namespace,
) -> torch.Tensor:
    """CVaR over slices of the inverse nuisance-projected Fisher information.

    Two guards keep the batch estimator usable:

    * a slice whose nuisance coordinate is (near) constant is skipped.  Without
      this the Schur complement degenerates -- ``J_TN^2/J_NN`` becomes either
      ``E[z]^2`` or 0 depending on where the constant sits relative to the batch
      mean -- and the penalty silently turns into "maximise the within-slice
      logit variance", which fights the classification loss.
    * the projected Fisher is floored at ``--fg-clamp`` so a noisy small-sample
      estimate cannot send ``1/projected`` to 1e8 and blow up the step.
    """
    z = (logits - logits.mean()) / (logits.std(unbiased=False) + 1e-6)
    b = (nuisance - nuisance.mean()) / (nuisance.std(unbiased=False) + 1e-6)
    inverses = []
    for sid in range(num_slices):
        mask = slice_ids == sid
        if int(mask.sum()) < args.fg_min_slice:
            continue
        zs, bs = z[mask], b[mask]
        if float(bs.std(unbiased=False)) < args.fg_min_nuisance_std:
            continue
        j_tt = torch.mean(zs * zs)
        j_tn = torch.mean(zs * bs)
        j_nn = torch.mean(bs * bs)
        projected = torch.clamp(j_tt - (j_tn * j_tn) / (j_nn + args.ridge), min=args.fg_clamp)
        inverses.append(1.0 / projected)
    if not inverses:
        return logits.new_tensor(0.0)
    return torch_cvar(torch.stack(inverses), args.fg_alpha)


def objective(
    method: str,
    logits: torch.Tensor,
    y: torch.Tensor,
    nuisance: torch.Tensor,
    groups: torch.Tensor,
    args: argparse.Namespace,
    fg_lambda: float,
) -> tuple[torch.Tensor, float]:
    """Returns (objective, penalty value) -- the penalty is logged for diagnosis."""
    losses = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    if method in {"group_dro", "group_dro_fisher_glass"}:
        present = [losses[groups == gid].mean() for gid in range(NUM_GROUPS) if int((groups == gid).sum()) > 0]
        base = torch.stack(present).max() if present else losses.mean()
    elif method in {"erm", "fisher_glass"}:
        base = losses.mean()
    else:
        raise ValueError(f"unknown method: {method}")
    if method in {"fisher_glass", "group_dro_fisher_glass"} and fg_lambda > 0.0:
        slice_ids, num_slices = penalty_slices(args, y, groups)
        penalty = fisher_glass_penalty(logits, nuisance, slice_ids, num_slices, args)
        return base + fg_lambda * penalty, float(penalty.detach())
    return base, 0.0


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
@torch.inference_mode()
def collect(model, loader, device: torch.device, dtype) -> dict[str, np.ndarray]:
    model.eval()
    logits, features, ys, groups = [], [], [], []
    for images, y, group, _place, _idx in loader:
        images = images.to(device, non_blocking=True)
        with autocast_context(device, dtype):
            logit, pooled = model(images)
        logits.append(logit.float().cpu().numpy())
        features.append(pooled.float().cpu().numpy())
        ys.append(y.numpy())
        groups.append(group.numpy())
    return {
        "logits": np.concatenate(logits).astype(np.float32),
        "features": np.concatenate(features).astype(np.float32),
        "y": np.concatenate(ys).astype(np.int64),
        "group": np.concatenate(groups).astype(np.int64),
    }


def quick_metrics(train_logits: np.ndarray, pack: dict[str, np.ndarray], place: np.ndarray, args) -> dict:
    """Coarse per-group readout using the binary background as nuisance.

    The real diagnostic (learned cross-fitted probe, calibrated variants) lives
    in ``select_slice.py``; this exists only so training logs are interpretable.
    """
    mean = float(train_logits.mean())
    std = float(train_logits.std()) or 1.0
    z = (pack["logits"] - mean) / std
    place = np.asarray(place, dtype=float)
    b = (place - place.mean()) / (place.std() or 1.0)
    preds = (pack["logits"] >= 0.0).astype(int)
    losses = binary_nll(pack["logits"], pack["y"])
    rows = []
    for gid in range(NUM_GROUPS):
        mask = pack["group"] == gid
        if not mask.any():
            continue
        f = fisher_schur(z[mask], b[mask], args.ridge)
        rows.append(
            {
                "group": GROUP_NAMES[gid],
                "n": int(mask.sum()),
                "accuracy": float((preds[mask] == pack["y"][mask]).mean()),
                "loss": float(losses[mask].mean()),
                "projected_fisher": f["projected"],
            }
        )
    inverse = np.array([1.0 / (row["projected_fisher"] + args.eps) for row in rows])
    return {
        "overall_accuracy": float((preds == pack["y"]).mean()),
        "overall_loss": float(losses.mean()),
        "worst_group_accuracy": float(min(row["accuracy"] for row in rows)),
        "fisher_glass_certificate": float(1.0 / cvar(inverse, args.alpha)),
        "groups": rows,
    }


# --------------------------------------------------------------------------- #
# One run
# --------------------------------------------------------------------------- #
def run_tag(method: str, fg_lambda: float, seed: int) -> str:
    return f"{method}_lam{fg_lambda:g}_seed{seed}"


def run_dir(out_dir: str | Path, method: str, fg_lambda: float, seed: int) -> Path:
    return Path(out_dir) / "runs" / run_tag(method, fg_lambda, seed)


def train_one(args: argparse.Namespace, method: str, fg_lambda: float, seed: int) -> dict:
    out = run_dir(args.out_dir, method, fg_lambda, seed)
    marker = out / "run.json"
    if args.resume and marker.exists() and (out / "scores.npz").exists() and (out / "checkpoint.pt").exists():
        log(f"skip (already complete): {out.name}")
        return json.loads(marker.read_text(encoding="utf-8"))

    set_seed(seed)
    device = resolve_device(args.device)
    enable_cuda_speedups(args.tf32)
    if device.type == "cpu":
        torch.set_num_threads(args.threads)
    dtype = amp_dtype(args.amp, device)
    scaler = make_grad_scaler(dtype)

    meta = load_metadata(args.metadata)
    transform = get_transform(args.arch)
    rows = {name: meta.rows_for_split(split) for name, split in (("train", 0), ("val", 1), ("test", 2))}
    datasets = {
        name: WaterbirdsDataset(args.image_root, meta.img_filename[r], meta.y[r], meta.place[r], transform)
        for name, r in rows.items()
    }
    train_loader = make_loader(datasets["train"], args.batch_size, args, shuffle=True, drop_last=True)
    eval_loaders = {name: make_loader(ds, args.eval_batch_size, args) for name, ds in datasets.items()}

    model = FeatureClassifier(arch=args.arch, pretrained=True, freeze_backbone=args.freeze_backbone).to(device)
    if args.channels_last and device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Online background probe supplying a *continuous* nuisance coordinate.  It
    # is fit on detached features (so it never shapes the representation) and its
    # score is detached before entering the penalty (so b is a fixed coordinate).
    # This mirrors the geometry the diagnostic uses, and unlike the binary
    # background annotation it varies inside every slice.
    uses_penalty = method in {"fisher_glass", "group_dro_fisher_glass"} and fg_lambda > 0.0
    needs_probe = uses_penalty and args.fg_nuisance == "probe"
    nuisance_probe = nn.Linear(model.feature_dim, 1).to(device) if needs_probe else None
    probe_optimizer = (
        torch.optim.AdamW(nuisance_probe.parameters(), lr=args.probe_lr) if needs_probe else None
    )
    log(
        f"{out.name}: arch={args.arch} train={len(datasets['train'])} val={len(datasets['val'])} "
        f"test={len(datasets['test'])} epochs={args.epochs} batch={args.batch_size} device={device} "
        f"amp={dtype} channels_last={args.channels_last and device.type == 'cuda'}"
        + (
            f" | penalty: nuisance={args.fg_nuisance} slices={args.fg_slices} lambda={fg_lambda:g} "
            f"clamp={args.fg_clamp:g} warmup={args.fg_warmup_epochs}"
            if uses_penalty
            else ""
        )
    )

    history = []
    start = time.time()
    for epoch in range(args.epochs):
        model.train()
        epoch_start, total, n_batches = time.time(), 0.0, 0
        penalty_total, probe_total = 0.0, 0.0
        warming_up = epoch < args.fg_warmup_epochs
        for images, y, groups, place, _idx in train_loader:
            images = images.to(device, non_blocking=True)
            if args.channels_last and device.type == "cuda":
                images = images.contiguous(memory_format=torch.channels_last)
            y = y.to(device, non_blocking=True)
            groups = groups.to(device, non_blocking=True)
            place = place.to(device, non_blocking=True)
            with autocast_context(device, dtype):
                logits, features = model(images)
            # The Fisher penalty involves ratios; evaluate everything below in fp32.
            logits, features = logits.float(), features.float()

            if needs_probe:
                probe_loss = F.binary_cross_entropy_with_logits(
                    nuisance_probe(features.detach()).view(-1), place
                )
                probe_optimizer.zero_grad(set_to_none=True)
                probe_loss.backward()
                probe_optimizer.step()
                probe_total += float(probe_loss.detach())
                with torch.no_grad():
                    nuisance = nuisance_probe(features).view(-1)
            else:
                nuisance = place

            # During warm-up the probe is still random, so the penalty would be noise.
            step_lambda = 0.0 if warming_up else fg_lambda
            loss, penalty = objective(method, logits, y, nuisance, groups, args, step_lambda)
            optimizer.zero_grad(set_to_none=True)
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()
            total += float(loss.detach())
            penalty_total += penalty
            n_batches += 1
        entry = {"epoch": epoch + 1, "objective": total / max(n_batches, 1), "seconds": time.time() - epoch_start}
        if uses_penalty:
            entry["fg_penalty"] = penalty_total / max(n_batches, 1)
            entry["warmup"] = bool(warming_up)
            if needs_probe:
                entry["probe_loss"] = probe_total / max(n_batches, 1)
        if args.eval_every and ((epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs):
            val_pack = collect(model, eval_loaders["val"], device, dtype)
            preds = (val_pack["logits"] >= 0.0).astype(int)
            entry["val_accuracy"] = float((preds == val_pack["y"]).mean())
            entry["val_worst_group_accuracy"] = float(
                min(
                    (preds[val_pack["group"] == g] == val_pack["y"][val_pack["group"] == g]).mean()
                    for g in range(NUM_GROUPS)
                    if (val_pack["group"] == g).any()
                )
            )
        history.append(entry)
        log(
            f"{out.name}: epoch {entry['epoch']}/{args.epochs} objective={entry['objective']:.4f} "
            + (f"fg={entry['fg_penalty']:.3f}{' (warmup)' if entry['warmup'] else ''} " if "fg_penalty" in entry else "")
            + (f"probe={entry['probe_loss']:.3f} " if "probe_loss" in entry else "")
            + (f"val_acc={entry['val_accuracy']:.4f} val_worst={entry['val_worst_group_accuracy']:.4f} " if "val_accuracy" in entry else "")
            + f"({entry['seconds']:.0f}s)"
        )
    elapsed = time.time() - start

    log(f"{out.name}: scoring train/val/test")
    packs = {name: collect(model, loader, device, dtype) for name, loader in eval_loaders.items()}

    out.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), out / "checkpoint.pt")
    if nuisance_probe is not None:
        torch.save(nuisance_probe.state_dict(), out / "train_nuisance_probe.pt")
    np.savez_compressed(
        out / "scores.npz",
        **{f"{name}_{key}": value for name, pack in packs.items() for key, value in pack.items()},
        **{f"{name}_rows": rows[name] for name in rows},
    )
    record = {
        "tag": out.name,
        "arch": args.arch,
        "method": method,
        "fg_lambda": fg_lambda,
        "seed": seed,
        "epochs": args.epochs,
        "lr": args.lr,
        "batch_size": args.batch_size,
        "freeze_backbone": bool(args.freeze_backbone),
        "penalty": {
            "nuisance": args.fg_nuisance,
            "slices": args.fg_slices,
            "alpha": args.fg_alpha,
            "clamp": args.fg_clamp,
            "min_slice": args.fg_min_slice,
            "min_nuisance_std": args.fg_min_nuisance_std,
            "warmup_epochs": args.fg_warmup_epochs,
            "ridge": args.ridge,
        }
        if method in {"fisher_glass", "group_dro_fisher_glass"}
        else None,
        "elapsed_seconds": elapsed,
        "history": history,
        "train_quick": quick_metrics(packs["train"]["logits"], packs["train"], meta.place[rows["train"]], args),
        "val_quick": quick_metrics(packs["train"]["logits"], packs["val"], meta.place[rows["val"]], args),
        "test_quick": quick_metrics(packs["train"]["logits"], packs["test"], meta.place[rows["test"]], args),
    }
    write_json(marker, record)
    log(
        f"{out.name}: val acc={record['val_quick']['overall_accuracy']:.4f} "
        f"worst={record['val_quick']['worst_group_accuracy']:.4f} | "
        f"test acc={record['test_quick']['overall_accuracy']:.4f} "
        f"worst={record['test_quick']['worst_group_accuracy']:.4f} ({elapsed / 60:.1f} min)"
    )
    return record


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--image-root", default="data/wilds/waterbirds_v1.0")
    parser.add_argument("--metadata", default="data/wilds/waterbirds_v1.0/metadata.csv")
    parser.add_argument("--out-dir", default="build/prospective_shift")
    parser.add_argument("--arch", default="mobilenet_v3_small", choices=["mobilenet_v3_small", "resnet50"])
    parser.add_argument("--methods", nargs="+", default=["erm", "fisher_glass"])
    parser.add_argument("--fg-lambdas", type=float, nargs="+", default=[0.003, 0.01, 0.03])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17, 18, 19, 20, 21])
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=128)
    parser.add_argument("--eval-every", type=int, default=5, help="0 disables per-epoch validation logging")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--freeze-backbone", action="store_true")
    # runtime
    parser.add_argument("--device", default="auto", help="auto | cuda | cuda:0 | cpu")
    parser.add_argument("--threads", type=int, default=16, help="torch CPU threads when running on CPU")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--prefetch-factor", type=int, default=4)
    parser.add_argument("--pin-memory", action="store_true", default=True)
    parser.add_argument("--no-pin-memory", dest="pin_memory", action="store_false")
    parser.add_argument("--amp", action="store_true", default=True, help="bf16/fp16 autocast on CUDA")
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--channels-last", action="store_true", default=True)
    parser.add_argument("--no-channels-last", dest="channels_last", action="store_false")
    parser.add_argument("--tf32", action="store_true", default=True)
    parser.add_argument("--no-tf32", dest="tf32", action="store_false")
    # Fisher-Glass penalty geometry
    parser.add_argument(
        "--fg-nuisance",
        choices=["probe", "background"],
        default="probe",
        help="'probe': online linear background probe on detached features (continuous, varies "
        "inside every slice). 'background': the binary annotation -- degenerate with "
        "--fg-slices groups, since background is constant within a group.",
    )
    parser.add_argument("--fg-slices", choices=["groups", "labels", "batch"], default="groups")
    parser.add_argument("--fg-clamp", type=float, default=0.05, help="floor on the projected Fisher inside the penalty")
    parser.add_argument("--fg-min-slice", type=int, default=6, help="minimum batch members for a slice to contribute")
    parser.add_argument("--fg-min-nuisance-std", type=float, default=0.05, help="skip slices whose nuisance barely varies")
    parser.add_argument("--fg-warmup-epochs", type=int, default=1, help="epochs with the penalty off while the probe calibrates")
    parser.add_argument("--probe-lr", type=float, default=1e-3)
    # geometry constants (kept identical everywhere downstream)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--fg-alpha", type=float, default=0.75)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    records = []
    for method in args.methods:
        lambdas = args.fg_lambdas if method in {"fisher_glass", "group_dro_fisher_glass"} else [0.0]
        for fg_lambda in lambdas:
            for seed in args.seeds:
                records.append(train_one(args, method, fg_lambda, seed))
    index = Path(args.out_dir) / "training_index.json"
    write_json(index, {"args": vars(args), "runs": records})
    log(f"wrote {index} ({len(records)} runs)")


if __name__ == "__main__":
    main()
