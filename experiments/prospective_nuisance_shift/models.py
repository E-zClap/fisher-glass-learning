"""Architectures, transforms and checkpoint loading shared by every step.

Both the training step and the shift-evaluation step must build the *identical*
preprocessing pipeline, otherwise the intervention images would be measured
under a different transform than the originals.  Keeping the factory here makes
that impossible to get wrong.
"""

from __future__ import annotations

import contextlib
from pathlib import Path

import torch
import torch.nn as nn
from torchvision.models import (
    MobileNet_V3_Small_Weights,
    ResNet50_Weights,
    mobilenet_v3_small,
    resnet50,
)

ARCHS = ("mobilenet_v3_small", "resnet50")


def get_transform(arch: str):
    """Inference/-training transform bundled with the ImageNet weights."""
    if arch == "mobilenet_v3_small":
        return MobileNet_V3_Small_Weights.DEFAULT.transforms()
    if arch == "resnet50":
        return ResNet50_Weights.IMAGENET1K_V2.transforms()
    raise ValueError(f"unknown arch: {arch} (expected one of {ARCHS})")


class FeatureClassifier(nn.Module):
    """Backbone + binary head that also exposes the pooled representation.

    ``forward`` returns ``(logit, pooled_feature)``.  The pooled feature is what
    the cross-fitted background probe is fit on in ``select_slice.py``.
    """

    def __init__(self, arch: str = "mobilenet_v3_small", pretrained: bool = True, freeze_backbone: bool = False):
        super().__init__()
        self.arch = arch
        if arch == "mobilenet_v3_small":
            weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
            net = mobilenet_v3_small(weights=weights)
            self.trunk = nn.Sequential(net.features, net.avgpool, nn.Flatten(1))
            head = net.classifier
            head[-1] = nn.Linear(head[-1].in_features, 1)
            self.head = head
            self.feature_dim = 576
        elif arch == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
            net = resnet50(weights=weights)
            self.feature_dim = net.fc.in_features
            net.fc = nn.Identity()
            self.trunk = net
            self.head = nn.Linear(self.feature_dim, 1)
        else:
            raise ValueError(f"unknown arch: {arch} (expected one of {ARCHS})")

        if freeze_backbone:
            for param in self.trunk.parameters():
                param.requires_grad_(False)

    def forward(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = self.trunk(images)
        return self.head(pooled).view(-1), pooled


def load_checkpoint(path: str | Path, arch: str, device: torch.device) -> FeatureClassifier:
    model = FeatureClassifier(arch=arch, pretrained=False)
    state = torch.load(Path(path), map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    return model.to(device).eval()


# --------------------------------------------------------------------------- #
# Device / precision helpers
# --------------------------------------------------------------------------- #
def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def amp_dtype(enabled: bool, device: torch.device) -> torch.dtype | None:
    """bfloat16 when the GPU supports it, otherwise float16; None disables autocast."""
    if not enabled or device.type != "cuda":
        return None
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def autocast_context(device: torch.device, dtype: torch.dtype | None):
    """``torch.autocast`` when a dtype was chosen, otherwise a no-op context."""
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=dtype)


def make_grad_scaler(dtype: torch.dtype | None):
    """GradScaler only for fp16; bf16 and fp32 need none. Tolerates older torch."""
    enabled = dtype is torch.float16
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):  # pragma: no cover - torch < 2.4
        return torch.cuda.amp.GradScaler(enabled=enabled)


def enable_cuda_speedups(tf32: bool, benchmark: bool = True) -> None:
    if not torch.cuda.is_available():
        return
    if tf32:
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:  # pragma: no cover - older torch
            pass
    torch.backends.cudnn.benchmark = benchmark
