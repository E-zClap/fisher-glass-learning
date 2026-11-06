"""End-to-end deep Fisher-Glass benchmark harness.

The lightweight scripts in this repository use frozen features. This script is
the heavier benchmark harness for end-to-end models:

* Waterbirds with a trainable MobileNetV3-small image classifier.
* CivilComments-WILDS with a trainable DistilBERT classifier when the local
  HuggingFace cache contains the model/tokenizer, or when model downloads are
  explicitly allowed.
* CelebA and MultiNLI entries are represented as availability checks, because
  those datasets are not present in the local WILDS root used for this draft.

The harness writes one JSON file containing completed runs and skipped
benchmarks with reasons. It is designed to run on GPU for paper-scale sweeps,
but also supports small CPU smoke tests via --max-train/--max-val/--max-test.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from wilds import get_dataset

from ridge_exact_audit import audit_scalar_environments, schur_from_moments


GROUP_NAMES_WATERBIRDS = {
    0: "landbird/land",
    1: "landbird/water",
    2: "waterbird/land",
    3: "waterbird/water",
}


def as_numpy(values) -> np.ndarray:
    if hasattr(values, "detach"):
        return values.detach().cpu().numpy()
    return np.asarray(values)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def cvar_np(values: np.ndarray, alpha: float) -> float:
    values = np.asarray(values, dtype=float)
    threshold = np.quantile(values, alpha)
    return float(values[values >= threshold].mean())


def torch_cvar(values: torch.Tensor, alpha: float) -> torch.Tensor:
    k = max(1, int(math.ceil((1.0 - alpha) * values.numel())))
    return torch.topk(values, k=k, largest=True).values.mean()


def fisher_schur(z: np.ndarray, b: np.ndarray, ridge: float) -> dict[str, float]:
    j_tt = float(np.mean(z * z))
    j_tn = float(np.mean(z * b))
    j_nn = float(np.mean(b * b))
    audit = schur_from_moments(j_tt, j_tn, j_nn, ridge)
    return {
        "raw": j_tt,
        "projected": audit["ridge_information"],
        "exact_projected": audit["exact_information"],
        "ridge_to_exact_gap": audit["operator_gap"],
        "j_tn": j_tn,
        "j_nn": j_nn,
        "audit_status": audit["status"],
        "audit_reasons": audit["reasons"],
    }


def group_ids(y: np.ndarray, nuisance_binary: np.ndarray) -> np.ndarray:
    return (2 * y.astype(int) + nuisance_binary.astype(int)).astype(int)


def maybe_subset(indices: np.ndarray, max_items: int | None, seed: int) -> np.ndarray:
    if max_items is None or max_items <= 0 or len(indices) <= max_items:
        return indices
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(indices, size=max_items, replace=False))


def split_cap(args: argparse.Namespace, benchmark: str, split_name: str) -> int | None:
    specific = getattr(args, f"{benchmark}_max_{split_name}")
    return specific if specific is not None else getattr(args, f"max_{split_name}")


def loader_kwargs(num_workers: int, pin_memory: bool, persistent_workers: bool, prefetch_factor: int) -> dict[str, Any]:
    kwargs: dict[str, Any] = {"num_workers": num_workers, "pin_memory": pin_memory}
    if num_workers > 0:
        kwargs["persistent_workers"] = persistent_workers
        kwargs["prefetch_factor"] = prefetch_factor
    return kwargs


def use_amp(args: argparse.Namespace, device: torch.device) -> bool:
    return bool(args.amp and device.type == "cuda")


def move_images(images: torch.Tensor, device: torch.device, args: argparse.Namespace) -> torch.Tensor:
    if args.channels_last and device.type == "cuda":
        return images.to(device, non_blocking=args.pin_memory, memory_format=torch.channels_last)
    return images.to(device, non_blocking=args.pin_memory)


def maybe_compile(model: nn.Module, args: argparse.Namespace, label: str) -> nn.Module:
    if not args.compile_model:
        return model
    try:
        log(f"{label}: compiling model with torch.compile")
        return torch.compile(model)
    except Exception as exc:  # pragma: no cover
        log(f"{label}: torch.compile unavailable or failed; continuing uncompiled: {exc}")
        return model


def fisher_glass_penalty(
    logits: torch.Tensor,
    nuisance: torch.Tensor,
    groups: torch.Tensor,
    ridge: float,
    eps: float,
    alpha: float,
    num_groups: int,
) -> torch.Tensor:
    z = (logits - logits.mean()) / (logits.std(unbiased=False) + 1e-6)
    losses = []
    for gid in range(num_groups):
        mask = groups == gid
        if int(mask.sum()) < 2:
            continue
        zg = z[mask]
        bg = nuisance[mask]
        j_tt = torch.mean(zg * zg)
        j_tn = torch.mean(zg * bg)
        j_nn = torch.mean(bg * bg)
        projected = torch.clamp(j_tt - (j_tn * j_tn) / (j_nn + ridge), min=eps)
        losses.append(1.0 / (projected + eps))
    if not losses:
        return logits.new_tensor(0.0)
    return torch_cvar(torch.stack(losses), alpha)


def objective_from_losses(
    method: str,
    logits: torch.Tensor,
    y: torch.Tensor,
    nuisance: torch.Tensor,
    groups: torch.Tensor,
    num_groups: int,
    alpha: float,
    ridge: float,
    eps: float,
    fg_lambda: float,
) -> torch.Tensor:
    losses = F.binary_cross_entropy_with_logits(logits, y.float(), reduction="none")
    group_losses = []
    for gid in range(num_groups):
        mask = groups == gid
        if int(mask.sum()) > 0:
            group_losses.append(losses[mask].mean())
    group_loss = torch.stack(group_losses).max() if group_losses else losses.mean()
    fg_loss = fisher_glass_penalty(logits, nuisance, groups, ridge, eps, alpha, num_groups)
    if method == "erm":
        return losses.mean()
    if method == "group_dro":
        return group_loss
    if method == "loss_cvar":
        return torch_cvar(losses, 0.9)
    if method == "fisher_glass":
        return losses.mean() + fg_lambda * fg_loss
    if method == "group_dro_fisher_glass":
        return group_loss + fg_lambda * fg_loss
    raise ValueError(f"Unknown method: {method}")


@dataclass
class ScorePack:
    logits: np.ndarray
    y: np.ndarray
    groups: np.ndarray
    nuisance: np.ndarray
    losses: np.ndarray


def evaluate_scores(
    train_logits: np.ndarray,
    scores: ScorePack,
    alpha: float,
    ridge: float,
    eps: float,
    group_names: dict[int, str],
    audit_bootstrap_replicates: int = 0,
    audit_confidence: float = 0.95,
    audit_seed: int = 20260729,
) -> dict[str, Any]:
    train_mean = float(train_logits.mean())
    train_std = float(train_logits.std())
    if train_std < 1e-8:
        train_std = 1.0
    z = (scores.logits - train_mean) / train_std
    probs = 1.0 / (1.0 + np.exp(-np.clip(scores.logits, -60.0, 60.0)))
    preds = (probs >= 0.5).astype(int)

    rows = []
    for gid in sorted(group_names):
        mask = scores.groups == gid
        if int(mask.sum()) == 0:
            continue
        f = fisher_schur(z[mask], scores.nuisance[mask], ridge)
        rows.append(
            {
                "group": group_names[gid],
                "n": int(mask.sum()),
                "accuracy": float((preds[mask] == scores.y[mask]).mean()),
                "loss": float(scores.losses[mask].mean()),
                "projected_feature_fisher": f["projected"],
                "exact_projected_feature_fisher": f["exact_projected"],
                "feature_ridge_to_exact_gap": f["ridge_to_exact_gap"],
                "feature_j_tn": f["j_tn"],
                "feature_j_nn": f["j_nn"],
                "feature_audit_status": f["audit_status"],
                "inverse_feature_loss": float(1.0 / (f["projected"] + eps)),
            }
        )
    audit_group_ids = [gid for gid in sorted(group_names) if int((scores.groups == gid).sum()) > 0]
    ridge_exact_audit = audit_scalar_environments(
        [
            {
                "name": group_names[gid],
                "z": z[scores.groups == gid],
                "b": scores.nuisance[scores.groups == gid],
            }
            for gid in audit_group_ids
        ],
        ridge=ridge,
        eps=eps,
        alpha=alpha,
        bootstrap_replicates=audit_bootstrap_replicates,
        confidence=audit_confidence,
        seed=audit_seed,
    )
    inverse = np.array([row["inverse_feature_loss"] for row in rows], dtype=float)
    group_losses = np.array([row["loss"] for row in rows], dtype=float)
    group_acc = np.array([row["accuracy"] for row in rows], dtype=float)
    return {
        "overall_accuracy": float((preds == scores.y).mean()),
        "overall_loss": float(scores.losses.mean()),
        "worst_group_accuracy": float(group_acc.min()),
        "group_loss_cvar": cvar_np(group_losses, alpha),
        "fisher_glass_certificate": float(1.0 / cvar_np(inverse, alpha)),
        "exact_fisher_glass_certificate": ridge_exact_audit["aggregate"]["exact_certificate"],
        "conservative_lower_certificate": ridge_exact_audit["aggregate"]["conservative_lower_certificate"],
        "inverse_loss_p95": float(np.quantile(inverse, 0.95)),
        "weakest_slice": max(rows, key=lambda row: row["inverse_feature_loss"]),
        "worst_accuracy_slice": min(rows, key=lambda row: row["accuracy"]),
        "groups": rows,
        "ridge_exact_audit": ridge_exact_audit,
    }


class WaterbirdsImageDataset(Dataset):
    def __init__(
        self,
        dataset,
        indices: np.ndarray,
        transform,
        y: np.ndarray,
        background: np.ndarray,
        nuisance_mean: float,
        nuisance_std: float,
    ) -> None:
        self.dataset = dataset
        self.indices = [int(i) for i in indices]
        self.transform = transform
        self.y = y
        self.background = background
        self.groups = group_ids(y, background)
        self.nuisance_mean = nuisance_mean
        self.nuisance_std = nuisance_std if nuisance_std > 1e-8 else 1.0

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, item: int):
        idx = self.indices[item]
        image, _, _ = self.dataset[idx]
        y = int(self.y[item])
        group = int(self.groups[item])
        nuisance = (float(self.background[item]) - self.nuisance_mean) / self.nuisance_std
        return self.transform(image), torch.tensor(y), torch.tensor(group), torch.tensor(nuisance, dtype=torch.float32)


def make_mobilenet(binary: bool = True, freeze_backbone: bool = False) -> nn.Module:
    weights = MobileNet_V3_Small_Weights.DEFAULT
    model = mobilenet_v3_small(weights=weights)
    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad_(False)
    out_dim = 1 if binary else 2
    in_features = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_features, out_dim)
    return model


def collect_image_scores(model: nn.Module, loader: DataLoader, device: torch.device, args: argparse.Namespace) -> ScorePack:
    model.eval()
    logits_all, y_all, group_all, nuisance_all, loss_all = [], [], [], [], []
    with torch.inference_mode():
        for images, y, groups, nuisance in loader:
            images = move_images(images, device, args)
            y = y.to(device, non_blocking=args.pin_memory).float()
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp(args, device)):
                logits = model(images).view(-1)
            logits = logits.float()
            losses = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
            logits_all.append(logits.cpu().numpy())
            y_all.append(y.cpu().numpy().astype(int))
            group_all.append(groups.numpy())
            nuisance_all.append(nuisance.numpy())
            loss_all.append(losses.float().cpu().numpy())
    return ScorePack(
        logits=np.concatenate(logits_all),
        y=np.concatenate(y_all),
        groups=np.concatenate(group_all),
        nuisance=np.concatenate(nuisance_all),
        losses=np.concatenate(loss_all),
    )


def train_waterbirds(args: argparse.Namespace, method: str, seed: int) -> dict[str, Any]:
    set_seed(seed)
    device = torch.device(args.device)
    log(f"waterbirds/{method}/seed={seed}: loading dataset from {args.root_dir}")
    dataset = get_dataset(dataset="waterbirds", root_dir=args.root_dir, download=args.download)
    y = as_numpy(dataset._y_array).astype(int)
    metadata = as_numpy(dataset._metadata_array).astype(int)
    split = as_numpy(dataset._split_array).astype(int)
    background = metadata[:, 0].astype(int)

    train_idx = np.flatnonzero(split == dataset.split_dict["train"])
    val_idx = np.flatnonzero(split == dataset.split_dict["val"])
    test_idx = np.flatnonzero(split == dataset.split_dict["test"])
    train_cap = split_cap(args, "waterbirds", "train")
    val_cap = split_cap(args, "waterbirds", "val")
    test_cap = split_cap(args, "waterbirds", "test")
    train_idx = maybe_subset(train_idx, train_cap, seed)
    val_idx = maybe_subset(val_idx, val_cap, seed + 1)
    test_idx = maybe_subset(test_idx, test_cap, seed + 2)

    nuisance_mean = float(background[train_idx].mean())
    nuisance_std = float(background[train_idx].std()) or 1.0
    transform = MobileNet_V3_Small_Weights.DEFAULT.transforms()
    train_ds = WaterbirdsImageDataset(dataset, train_idx, transform, y[train_idx], background[train_idx], nuisance_mean, nuisance_std)
    val_ds = WaterbirdsImageDataset(dataset, val_idx, transform, y[val_idx], background[val_idx], nuisance_mean, nuisance_std)
    test_ds = WaterbirdsImageDataset(dataset, test_idx, transform, y[test_idx], background[test_idx], nuisance_mean, nuisance_std)
    img_loader_kwargs = loader_kwargs(args.num_workers, args.pin_memory, args.persistent_workers, args.prefetch_factor)
    eval_batch_size = args.eval_batch_size or args.batch_size
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **img_loader_kwargs)
    train_eval_loader = DataLoader(train_ds, batch_size=eval_batch_size, shuffle=False, **img_loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=eval_batch_size, shuffle=False, **img_loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=eval_batch_size, shuffle=False, **img_loader_kwargs)

    model = make_mobilenet(freeze_backbone=args.freeze_backbone)
    if args.channels_last and device.type == "cuda":
        model = model.to(device, memory_format=torch.channels_last)
    else:
        model = model.to(device)
    model = maybe_compile(model, args, f"waterbirds/{method}/seed={seed}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    start_time = time.time()
    log(
        f"waterbirds/{method}/seed={seed}: train n={len(train_ds)} val n={len(val_ds)} "
        f"test n={len(test_ds)} epochs={args.epochs} batch={args.batch_size} eval_batch={eval_batch_size} "
        f"device={device} workers={args.num_workers} pin={args.pin_memory} amp={use_amp(args, device)} "
        f"channels_last={args.channels_last and device.type == 'cuda'}"
    )
    for epoch in range(args.epochs):
        epoch_start = time.time()
        batches = 0
        model.train()
        for images, y_batch, groups, nuisance in train_loader:
            images = move_images(images, device, args)
            y_batch = y_batch.to(device, non_blocking=args.pin_memory)
            groups = groups.to(device, non_blocking=args.pin_memory)
            nuisance = nuisance.to(device, non_blocking=args.pin_memory)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp(args, device)):
                logits = model(images).view(-1)
                objective = objective_from_losses(
                    method,
                    logits,
                    y_batch,
                    nuisance,
                    groups,
                    num_groups=4,
                    alpha=args.alpha,
                    ridge=args.ridge,
                    eps=args.eps,
                    fg_lambda=args.fg_lambda,
                )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            batches += 1
        log(
            f"waterbirds/{method}/seed={seed}: finished epoch {epoch + 1}/{args.epochs} "
            f"({batches} batches, {time.time() - epoch_start:.1f}s)"
        )
    elapsed = time.time() - start_time

    log(f"waterbirds/{method}/seed={seed}: evaluating")
    train_scores = collect_image_scores(model, train_eval_loader, device, args)
    val_scores = collect_image_scores(model, val_loader, device, args)
    test_scores = collect_image_scores(model, test_loader, device, args)
    return {
        "benchmark": "waterbirds",
        "backbone": "MobileNetV3-small",
        "method": method,
        "seed": seed,
        "lambda_fg": args.fg_lambda,
        "ridge": args.ridge,
        "eps": args.eps,
        "alpha": args.alpha,
        "audit_schema_version": 2,
        "epochs": args.epochs,
        "freeze_backbone": args.freeze_backbone,
        "max_train": train_cap,
        "max_val": val_cap,
        "max_test": test_cap,
        "elapsed_seconds": elapsed,
        "val": evaluate_scores(
            train_scores.logits, val_scores, args.alpha, args.ridge, args.eps,
            GROUP_NAMES_WATERBIRDS, args.audit_bootstrap_replicates,
            args.audit_confidence, args.audit_seed,
        ),
        "test": evaluate_scores(
            train_scores.logits, test_scores, args.alpha, args.ridge, args.eps,
            GROUP_NAMES_WATERBIRDS, args.audit_bootstrap_replicates,
            args.audit_confidence, args.audit_seed + 1,
        ),
    }


class CivilCommentsTextDataset(Dataset):
    def __init__(
        self,
        texts: list[str],
        y: np.ndarray,
        identity_any: np.ndarray,
        tokenizer,
        max_length: int,
        nuisance_mean: float,
        nuisance_std: float,
    ) -> None:
        self.texts = texts
        self.y = y.astype(int)
        self.identity_any = identity_any.astype(int)
        self.groups = group_ids(self.y, self.identity_any)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.nuisance_mean = nuisance_mean
        self.nuisance_std = nuisance_std if nuisance_std > 1e-8 else 1.0

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, item: int) -> dict[str, torch.Tensor]:
        enc = self.tokenizer(
            self.texts[item] if isinstance(self.texts[item], str) else "",
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": enc["input_ids"].squeeze(0),
            "attention_mask": enc["attention_mask"].squeeze(0),
            "y": torch.tensor(int(self.y[item])),
            "group": torch.tensor(int(self.groups[item])),
            "nuisance": torch.tensor((float(self.identity_any[item]) - self.nuisance_mean) / self.nuisance_std, dtype=torch.float32),
        }


def texts_from_array(values, indices: np.ndarray) -> list[str]:
    out = []
    for idx in indices:
        value = values[int(idx)]
        out.append(value if isinstance(value, str) else "")
    return out


def train_civilcomments_transformer(args: argparse.Namespace, method: str, seed: int) -> dict[str, Any]:
    try:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
    except Exception as exc:  # pragma: no cover
        return {"benchmark": "civilcomments", "method": method, "seed": seed, "skipped": True, "reason": f"transformers unavailable: {exc}"}

    set_seed(seed)
    device = torch.device(args.device)
    log(f"civilcomments/{method}/seed={seed}: loading dataset from {args.root_dir}")
    dataset = get_dataset(dataset="civilcomments", root_dir=args.root_dir, download=args.download)
    y = as_numpy(dataset._y_array).astype(int)
    metadata = as_numpy(dataset._metadata_array).astype(int)
    split = as_numpy(dataset._split_array).astype(int)
    field_to_idx = {name: i for i, name in enumerate(dataset.metadata_fields)}
    identity_any = metadata[:, field_to_idx["identity_any"]].astype(int)

    train_idx = np.flatnonzero(split == dataset.split_dict["train"])
    val_idx = np.flatnonzero(split == dataset.split_dict["val"])
    test_idx = np.flatnonzero(split == dataset.split_dict["test"])
    train_cap = split_cap(args, "civilcomments", "train")
    val_cap = split_cap(args, "civilcomments", "val")
    test_cap = split_cap(args, "civilcomments", "test")
    train_idx = maybe_subset(train_idx, train_cap, seed)
    val_idx = maybe_subset(val_idx, val_cap, seed + 1)
    test_idx = maybe_subset(test_idx, test_cap, seed + 2)
    nuisance_mean = float(identity_any[train_idx].mean())
    nuisance_std = float(identity_any[train_idx].std()) or 1.0

    try:
        log(f"civilcomments/{method}/seed={seed}: loading text model {args.text_model}")
        tokenizer = AutoTokenizer.from_pretrained(
            args.text_model,
            local_files_only=not args.allow_model_download,
        )
        model = AutoModelForSequenceClassification.from_pretrained(
            args.text_model,
            num_labels=1,
            local_files_only=not args.allow_model_download,
        )
    except Exception as exc:
        mode = "online download or local cache" if args.allow_model_download else "local cache"
        return {
            "benchmark": "civilcomments",
            "method": method,
            "seed": seed,
            "skipped": True,
            "reason": f"text model unavailable from {mode}: {exc}",
        }
    model.to(device)
    train_ds = CivilCommentsTextDataset(
        texts_from_array(dataset._text_array, train_idx),
        y[train_idx],
        identity_any[train_idx],
        tokenizer,
        args.max_length,
        nuisance_mean,
        nuisance_std,
    )
    val_ds = CivilCommentsTextDataset(texts_from_array(dataset._text_array, val_idx), y[val_idx], identity_any[val_idx], tokenizer, args.max_length, nuisance_mean, nuisance_std)
    test_ds = CivilCommentsTextDataset(texts_from_array(dataset._text_array, test_idx), y[test_idx], identity_any[test_idx], tokenizer, args.max_length, nuisance_mean, nuisance_std)
    text_workers = args.text_num_workers if args.text_num_workers is not None else args.num_workers
    text_loader_kwargs = loader_kwargs(text_workers, args.pin_memory, args.persistent_workers, args.prefetch_factor)
    text_eval_batch_size = args.text_eval_batch_size or args.text_batch_size
    train_loader = DataLoader(train_ds, batch_size=args.text_batch_size, shuffle=True, **text_loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=text_eval_batch_size, shuffle=False, **text_loader_kwargs)
    test_loader = DataLoader(test_ds, batch_size=text_eval_batch_size, shuffle=False, **text_loader_kwargs)
    train_eval_loader = DataLoader(train_ds, batch_size=text_eval_batch_size, shuffle=False, **text_loader_kwargs)

    model = maybe_compile(model, args, f"civilcomments/{method}/seed={seed}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.text_lr, weight_decay=args.weight_decay)
    start_time = time.time()
    log(
        f"civilcomments/{method}/seed={seed}: train n={len(train_ds)} val n={len(val_ds)} "
        f"test n={len(test_ds)} epochs={args.epochs} batch={args.text_batch_size} eval_batch={text_eval_batch_size} "
        f"device={device} workers={text_workers} pin={args.pin_memory} amp={use_amp(args, device)}"
    )
    for epoch in range(args.epochs):
        epoch_start = time.time()
        batches = 0
        model.train()
        for batch in train_loader:
            input_ids = batch["input_ids"].to(device, non_blocking=args.pin_memory)
            attention_mask = batch["attention_mask"].to(device, non_blocking=args.pin_memory)
            y_batch = batch["y"].to(device, non_blocking=args.pin_memory)
            groups = batch["group"].to(device, non_blocking=args.pin_memory)
            nuisance = batch["nuisance"].to(device, non_blocking=args.pin_memory)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp(args, device)):
                logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.view(-1)
                objective = objective_from_losses(method, logits, y_batch, nuisance, groups, 4, args.alpha, args.ridge, args.eps, args.fg_lambda)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            batches += 1
        log(
            f"civilcomments/{method}/seed={seed}: finished epoch {epoch + 1}/{args.epochs} "
            f"({batches} batches, {time.time() - epoch_start:.1f}s)"
        )
    elapsed = time.time() - start_time

    def collect(loader: DataLoader) -> ScorePack:
        model.eval()
        logits_all, y_all, group_all, nuisance_all, loss_all = [], [], [], [], []
        with torch.inference_mode():
            for batch in loader:
                input_ids = batch["input_ids"].to(device, non_blocking=args.pin_memory)
                attention_mask = batch["attention_mask"].to(device, non_blocking=args.pin_memory)
                y_batch = batch["y"].to(device, non_blocking=args.pin_memory).float()
                with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_amp(args, device)):
                    logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.view(-1)
                logits = logits.float()
                losses = F.binary_cross_entropy_with_logits(logits, y_batch, reduction="none")
                logits_all.append(logits.cpu().numpy())
                y_all.append(y_batch.cpu().numpy().astype(int))
                group_all.append(batch["group"].numpy())
                nuisance_all.append(batch["nuisance"].numpy())
                loss_all.append(losses.float().cpu().numpy())
        return ScorePack(np.concatenate(logits_all), np.concatenate(y_all), np.concatenate(group_all), np.concatenate(nuisance_all), np.concatenate(loss_all))

    group_names = {0: "nonidentity/y=0", 1: "nonidentity/y=1", 2: "identity/y=0", 3: "identity/y=1"}
    log(f"civilcomments/{method}/seed={seed}: evaluating")
    train_scores = collect(train_eval_loader)
    val_scores = collect(val_loader)
    test_scores = collect(test_loader)
    return {
        "benchmark": "civilcomments",
        "backbone": args.text_model,
        "method": method,
        "seed": seed,
        "lambda_fg": args.fg_lambda,
        "ridge": args.ridge,
        "eps": args.eps,
        "alpha": args.alpha,
        "audit_schema_version": 2,
        "epochs": args.epochs,
        "max_train": train_cap,
        "max_val": val_cap,
        "max_test": test_cap,
        "elapsed_seconds": elapsed,
        "val": evaluate_scores(
            train_scores.logits, val_scores, args.alpha, args.ridge, args.eps,
            group_names, args.audit_bootstrap_replicates,
            args.audit_confidence, args.audit_seed,
        ),
        "test": evaluate_scores(
            train_scores.logits, test_scores, args.alpha, args.ridge, args.eps,
            group_names, args.audit_bootstrap_replicates,
            args.audit_confidence, args.audit_seed + 1,
        ),
    }


def summarize_runs(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    complete = [r for r in runs if not r.get("skipped")]
    rows = []
    keys = {(r["benchmark"], r["method"], r.get("lambda_fg")) for r in complete}
    for key in sorted(keys, key=lambda value: tuple(str(part) for part in value)):
        benchmark, method, lambda_fg = key
        items = [
            r
            for r in complete
            if (r["benchmark"], r["method"], r.get("lambda_fg")) == key
        ]

        def available(values: list[float | None]) -> list[float]:
            return [float(value) for value in values if value is not None]

        def mean(values: list[float | None]) -> float | None:
            numeric = available(values)
            return float(np.mean(numeric)) if numeric else None

        def minimum(values: list[float | None]) -> float | None:
            numeric = available(values)
            return min(numeric) if numeric else None

        def maximum(values: list[float | None]) -> float | None:
            numeric = available(values)
            return max(numeric) if numeric else None

        def audit_value(run: dict[str, Any], key: str) -> float | None:
            return (
                run.get("test", {})
                .get("ridge_exact_audit", {})
                .get("aggregate", {})
                .get(key)
            )

        rows.append(
            {
                "benchmark": benchmark,
                "method": method,
                "lambda_fg": lambda_fg,
                "n": len(items),
                "test_overall_accuracy": mean([r["test"]["overall_accuracy"] for r in items]),
                "test_worst_group_accuracy": mean([r["test"]["worst_group_accuracy"] for r in items]),
                "test_loss_cvar": mean([r["test"]["group_loss_cvar"] for r in items]),
                "test_fisher_glass_certificate": mean([r["test"]["fisher_glass_certificate"] for r in items]),
                "test_exact_fisher_glass_certificate": mean(
                    [r["test"].get("exact_fisher_glass_certificate") for r in items]
                ),
                "test_exact_fisher_glass_certificate_min": minimum(
                    [r["test"].get("exact_fisher_glass_certificate") for r in items]
                ),
                "test_conservative_lower_certificate": mean(
                    [r["test"].get("conservative_lower_certificate") for r in items]
                ),
                "test_conservative_lower_certificate_min": minimum(
                    [r["test"].get("conservative_lower_certificate") for r in items]
                ),
                "test_minimum_nuisance_eigenvalue": minimum(
                    [audit_value(r, "min_nuisance_eigenvalue") for r in items]
                ),
                "test_maximum_ridge_to_exact_gap": maximum(
                    [audit_value(r, "max_operator_gap") for r in items]
                ),
                "test_inverse_loss_p95": mean([r["test"]["inverse_loss_p95"] for r in items]),
            }
        )
    return rows


def run_key(record: dict[str, Any]) -> tuple[str, str, int] | None:
    if "benchmark" not in record or "method" not in record or "seed" not in record:
        return None
    return (str(record["benchmark"]), str(record["method"]), int(record["seed"]))


def has_current_audit(record: dict[str, Any]) -> bool:
    """Return whether a completed record contains the current W2 audit schema."""

    if record.get("skipped"):
        return True
    return all(
        record.get(split, {}).get("ridge_exact_audit", {}).get("schema_version") == 2
        for split in ("val", "test")
    )


def write_checkpoint(args: argparse.Namespace, runs: list[dict[str, Any]]) -> None:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    result = {
        "schema_version": 2,
        "note": "Paper-scale use should run this harness on GPU without sample caps when feasible.",
        "device": args.device,
        "parameters": vars(args),
        "summary": summarize_runs(runs),
        "runs": runs,
    }
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(result, indent=2), encoding="utf-8")
    tmp.replace(output)
    log(f"checkpoint wrote {len(runs)} run records to {output}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root-dir", default="data/wilds")
    parser.add_argument("--output", default="build/results/end_to_end_deep_suite_results.json")
    parser.add_argument("--benchmarks", nargs="+", default=["waterbirds", "civilcomments", "celeba", "multinli"])
    parser.add_argument("--methods", nargs="+", default=["erm", "group_dro", "fisher_glass", "group_dro_fisher_glass"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[17])
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--text-batch-size", type=int, default=4)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--text-eval-batch-size", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader workers for image datasets.")
    parser.add_argument("--text-num-workers", type=int, default=None, help="DataLoader workers for text datasets; defaults to --num-workers.")
    parser.add_argument("--pin-memory", action="store_true", help="Pin DataLoader memory for faster host-to-GPU copies.")
    parser.add_argument("--persistent-workers", action="store_true", help="Keep DataLoader workers alive across epochs when workers > 0.")
    parser.add_argument("--prefetch-factor", type=int, default=2, help="DataLoader prefetch factor when workers > 0.")
    parser.add_argument("--amp", action="store_true", help="Use CUDA BF16 autocast for training/evaluation.")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 matmul/cuDNN kernels on CUDA.")
    parser.add_argument("--channels-last", action="store_true", help="Use channels-last memory format for image models on CUDA.")
    parser.add_argument("--compile-model", action="store_true", help="Try torch.compile(model) before training.")
    parser.add_argument("--download", action="store_true", help="Allow WILDS to download missing datasets under --root-dir.")
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow HuggingFace to download the text model/tokenizer for CivilComments.",
    )
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--text-lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--alpha", type=float, default=0.95)
    parser.add_argument("--ridge", type=float, default=1e-3)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--audit-bootstrap-replicates", type=int, default=100000)
    parser.add_argument("--audit-confidence", type=float, default=0.95)
    parser.add_argument("--audit-seed", type=int, default=20260729)
    parser.add_argument("--fg-lambda", type=float, default=0.03)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--max-train", type=int, default=None)
    parser.add_argument("--max-val", type=int, default=None)
    parser.add_argument("--max-test", type=int, default=None)
    parser.add_argument("--waterbirds-max-train", type=int, default=None)
    parser.add_argument("--waterbirds-max-val", type=int, default=None)
    parser.add_argument("--waterbirds-max-test", type=int, default=None)
    parser.add_argument("--civilcomments-max-train", type=int, default=None)
    parser.add_argument("--civilcomments-max-val", type=int, default=None)
    parser.add_argument("--civilcomments-max-test", type=int, default=None)
    parser.add_argument("--resume", action="store_true", help="Load existing --output and skip completed benchmark/method/seed runs.")
    parser.add_argument("--text-model", default="distilbert-base-cased")
    parser.add_argument("--max-length", type=int, default=128)
    args = parser.parse_args()

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        log("CUDA requested but unavailable; falling back to CPU.")
        args.device = "cpu"
    if args.tf32 and args.device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
        log("enabled CUDA TF32 and cuDNN benchmark mode")
    log(f"starting end-to-end suite on device={args.device}")

    output = Path(args.output)
    runs: list[dict[str, Any]] = []
    if args.resume and output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            runs = list(existing.get("runs", []))
            legacy_count = sum(not has_current_audit(record) for record in runs)
            runs = [record for record in runs if has_current_audit(record)]
            log(
                f"resume loaded {len(runs)} current run records from {output}; "
                f"discarded {legacy_count} legacy records that require a W2 rerun"
            )
        except Exception as exc:
            log(f"resume could not read {output}: {exc}; starting a fresh run list")
    completed = {
        key
        for key, record in ((run_key(record), record) for record in runs)
        if key is not None and not record.get("skipped")
    }

    root = Path(args.root_dir)
    for benchmark in args.benchmarks:
        log(f"benchmark={benchmark}: starting")
        if benchmark == "waterbirds":
            if not args.download and not (root / "waterbirds_v1.0").exists():
                runs.append({"benchmark": benchmark, "skipped": True, "reason": "data/wilds/waterbirds_v1.0 not found"})
                write_checkpoint(args, runs)
                continue
            for seed in args.seeds:
                for method in args.methods:
                    key = ("waterbirds", method, seed)
                    if key in completed:
                        log(f"benchmark=waterbirds method={method} seed={seed}: already complete, skipping")
                        continue
                    log(f"benchmark=waterbirds method={method} seed={seed}: starting run")
                    runs.append(train_waterbirds(args, method, seed))
                    if not runs[-1].get("skipped"):
                        completed.add(key)
                        log(
                            f"benchmark=waterbirds method={method} seed={seed}: "
                            f"test worst={runs[-1]['test']['worst_group_accuracy']:.4f} "
                            f"cert={runs[-1]['test']['fisher_glass_certificate']:.4g}"
                        )
                    write_checkpoint(args, runs)
        elif benchmark == "civilcomments":
            if not args.download and not (root / "civilcomments_v1.0").exists():
                runs.append({"benchmark": benchmark, "skipped": True, "reason": "data/wilds/civilcomments_v1.0 not found"})
                write_checkpoint(args, runs)
                continue
            for seed in args.seeds:
                for method in args.methods:
                    key = ("civilcomments", method, seed)
                    if key in completed:
                        log(f"benchmark=civilcomments method={method} seed={seed}: already complete, skipping")
                        continue
                    log(f"benchmark=civilcomments method={method} seed={seed}: starting run")
                    runs.append(train_civilcomments_transformer(args, method, seed))
                    if runs[-1].get("skipped"):
                        log(f"benchmark=civilcomments method={method} seed={seed}: skipped: {runs[-1]['reason']}")
                    else:
                        completed.add(key)
                        log(
                            f"benchmark=civilcomments method={method} seed={seed}: "
                            f"test worst={runs[-1]['test']['worst_group_accuracy']:.4f} "
                            f"cert={runs[-1]['test']['fisher_glass_certificate']:.4g}"
                        )
                    write_checkpoint(args, runs)
        elif benchmark == "celeba":
            runs.append({"benchmark": benchmark, "skipped": True, "reason": "CelebA data not present in local workspace"})
            write_checkpoint(args, runs)
        elif benchmark in {"multinli", "mnli"}:
            runs.append({"benchmark": benchmark, "skipped": True, "reason": "MultiNLI data not present in local workspace"})
            write_checkpoint(args, runs)
        else:
            runs.append({"benchmark": benchmark, "skipped": True, "reason": f"unknown benchmark {benchmark}"})
            write_checkpoint(args, runs)

    write_checkpoint(args, runs)
    log(f"wrote results to {output}")
    print(json.dumps({"summary": summarize_runs(runs), "skipped": [r for r in runs if r.get("skipped")]}, indent=2), flush=True)


if __name__ == "__main__":
    main()
