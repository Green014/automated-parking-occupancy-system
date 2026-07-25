from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small
from torchvision.ops.misc import SqueezeExcitation

from .data import SlotSample, read_image
from .patches import extract_slot_patch

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)
MOBILENET_VARIANTS = (
    "standard",
    "leakyrelu6",
    "cbam",
    "cbam_leakyrelu6",
)


class LeakyReLU6(nn.Module):
    """Leaky ReLU capped at six; alpha is an explicit local adaptation choice."""

    def __init__(self, negative_slope: float = 0.1) -> None:
        super().__init__()
        if negative_slope < 0.0:
            raise ValueError("negative_slope must be non-negative")
        self.negative_slope = negative_slope

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return torch.clamp(
            F.leaky_relu(inputs, negative_slope=self.negative_slope),
            max=6.0,
        )


class CBAM(nn.Module):
    """Channel-then-spatial CBAM with an identity-preserving residual gate.

    Each sigmoid gate is multiplied by two.  Zero-initialising its last
    projection therefore makes the module an exact identity at construction,
    while still allowing attention values in [0, 2].  This is a local
    transfer-learning adaptation; it is not the exact CBAM replacement used
    by the source paper.
    """

    def __init__(
        self,
        channels: int,
        reduction: int = 4,
        spatial_kernel_size: int = 7,
    ) -> None:
        super().__init__()
        if channels <= 0 or reduction <= 0:
            raise ValueError("channels and reduction must be positive")
        if spatial_kernel_size not in {3, 7}:
            raise ValueError("spatial_kernel_size must be 3 or 7")
        hidden = max(1, channels // reduction)
        self.channel_mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, 1, bias=True),
        )
        padding = spatial_kernel_size // 2
        self.spatial = nn.Conv2d(
            2,
            1,
            spatial_kernel_size,
            padding=padding,
            bias=True,
        )
        nn.init.zeros_(self.channel_mlp[2].weight)
        nn.init.zeros_(self.channel_mlp[2].bias)
        nn.init.zeros_(self.spatial.weight)
        nn.init.zeros_(self.spatial.bias)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        average = F.adaptive_avg_pool2d(inputs, 1)
        maximum = F.adaptive_max_pool2d(inputs, 1)
        channel_scale = 2.0 * torch.sigmoid(
            self.channel_mlp(average) + self.channel_mlp(maximum)
        )
        channel_refined = inputs * channel_scale
        spatial_input = torch.cat(
            (
                torch.mean(channel_refined, dim=1, keepdim=True),
                torch.amax(channel_refined, dim=1, keepdim=True),
            ),
            dim=1,
        )
        return channel_refined * (
            2.0 * torch.sigmoid(self.spatial(spatial_input))
        )


class SEWithCBAM(nn.Module):
    """Preserve pretrained SE and add trainable CBAM as a supplement."""

    def __init__(self, se: SqueezeExcitation, channels: int) -> None:
        super().__init__()
        self.se = se
        self.cbam = CBAM(channels)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.cbam(self.se(inputs))


def implementation_label(variant: str) -> str:
    labels = {
        "standard": "adapted_standard_mobilenet_v3_small",
        "leakyrelu6": "paper_inspired_mobilenet_v3_leakyrelu6",
        "cbam": "paper_inspired_mobilenet_v3_cbam",
        "cbam_leakyrelu6": "paper_inspired_mobilenet_v3_cbam_leakyrelu6",
    }
    try:
        return labels[variant]
    except KeyError as error:
        raise ValueError(f"unsupported MobileNet variant: {variant}") from error


def _replace_modules(
    parent: nn.Module,
    predicate: type[nn.Module],
    factory: Any,
) -> int:
    replacements = 0
    for name, child in list(parent.named_children()):
        if isinstance(child, predicate):
            setattr(parent, name, factory(child))
            replacements += 1
        else:
            replacements += _replace_modules(child, predicate, factory)
    return replacements


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.isdigit():
        return torch.device(f"cuda:{requested}")
    return torch.device(requested)


def build_mobilenet_classifier(
    pretrained: bool = True,
    freeze_backbone: bool = True,
    variant: str = "standard",
) -> nn.Module:
    """Build the standard or explicitly paper-inspired MobileNetV3 adaptation."""

    implementation_label(variant)
    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    use_cbam = variant in {"cbam", "cbam_leakyrelu6"}
    use_leakyrelu6 = variant in {"leakyrelu6", "cbam_leakyrelu6"}
    if use_leakyrelu6:
        shallow = nn.Sequential(*list(model.features.children())[:4])
        replaced = _replace_modules(
            shallow,
            nn.ReLU,
            lambda _module: LeakyReLU6(negative_slope=0.1),
        )
        if replaced == 0:
            raise RuntimeError("no shallow MobileNetV3 ReLU modules were found")
    if use_cbam:
        replaced = _replace_modules(
            model.features,
            SqueezeExcitation,
            lambda module: SEWithCBAM(
                module,
                int(module.fc2.out_channels),
            ),
        )
        if replaced == 0:
            raise RuntimeError("no MobileNetV3 SE modules were found")
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, 2)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
        if use_leakyrelu6:
            for feature in list(model.features.children())[:4]:
                for parameter in feature.parameters():
                    parameter.requires_grad = True
        for module in model.features.modules():
            if isinstance(module, CBAM):
                for parameter in module.parameters():
                    parameter.requires_grad = True
    return model


def patch_to_tensor(patch: np.ndarray) -> torch.Tensor:
    if patch.ndim != 3 or patch.shape[2] != 3:
        raise ValueError("patch must be HxWx3 BGR")
    rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    normalized = (rgb - IMAGENET_MEAN) / IMAGENET_STD
    channels_first = np.ascontiguousarray(normalized.transpose(2, 0, 1))
    return torch.from_numpy(channels_first)


class SlotPatchDataset(Dataset[tuple[torch.Tensor, int]]):
    """On-demand OpenCV slot-patch dataset with a small image cache."""

    def __init__(
        self,
        samples: Sequence[SlotSample],
        output_size: tuple[int, int] = (224, 224),
        augment: bool = False,
    ) -> None:
        if not samples:
            raise ValueError("Dataset needs at least one sample")
        self.samples = tuple(samples)
        self.output_size = output_size
        self.augment = augment
        self._image_cache: dict[Path, np.ndarray] = {}

    def __len__(self) -> int:
        return len(self.samples)

    def _image(self, path: Path) -> np.ndarray:
        if path not in self._image_cache:
            self._image_cache[path] = read_image(path)
        return self._image_cache[path]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        sample = self.samples[index]
        patch = extract_slot_patch(
            self._image(sample.image_path),
            sample.points,
            output_size=self.output_size,
        )
        if self.augment and random.random() < 0.5:
            patch = cv2.flip(patch, 1)
        return patch_to_tensor(patch), sample.label


class MobileNetSlotClassifier:
    """Checkpoint-backed slot classifier that returns occupied probability."""

    def __init__(
        self,
        checkpoint_path: str | Path,
        device: str = "auto",
        batch_size: int = 64,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.device = resolve_device(device)
        self.batch_size = batch_size
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        payload = torch.load(
            self.checkpoint_path,
            map_location=self.device,
            weights_only=False,
        )
        self.model = build_mobilenet_classifier(
            pretrained=False,
            freeze_backbone=bool(payload.get("freeze_backbone", True)),
            variant=str(payload.get("variant", "standard")),
        )
        self.model.load_state_dict(payload["model_state"])
        self.model.to(self.device).eval()
        self.patch_size = tuple(payload.get("patch_size", (224, 224)))
        self.checkpoint_metadata = {
            key: payload.get(key)
            for key in (
                "architecture",
                "seed",
                "best_epoch",
                "best_development_macro_f1",
                "freeze_backbone",
                "variant",
            )
        }

    @torch.inference_mode()
    def predict_patches(self, patches: Sequence[np.ndarray]) -> list[float]:
        if not patches:
            return []
        probabilities: list[float] = []
        for start in range(0, len(patches), self.batch_size):
            batch = torch.stack(
                [patch_to_tensor(patch) for patch in patches[start : start + self.batch_size]]
            ).to(self.device)
            logits = self.model(batch)
            occupied = torch.softmax(logits, dim=1)[:, 1]
            probabilities.extend(float(value) for value in occupied.cpu())
        return probabilities

    def metadata(self) -> dict[str, Any]:
        return {
            "backend": "torchvision",
            "implementation_label": implementation_label(
                str(self.checkpoint_metadata.get("variant") or "standard")
            ),
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "patch_size": list(self.patch_size),
            **self.checkpoint_metadata,
        }
