from __future__ import annotations

import random
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from .data import SlotSample, read_image
from .patches import extract_slot_patch

IMAGENET_MEAN = np.asarray((0.485, 0.456, 0.406), dtype=np.float32)
IMAGENET_STD = np.asarray((0.229, 0.224, 0.225), dtype=np.float32)


def resolve_device(requested: str = "auto") -> torch.device:
    if requested == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if requested.isdigit():
        return torch.device(f"cuda:{requested}")
    return torch.device(requested)


def build_mobilenet_classifier(
    pretrained: bool = True,
    freeze_backbone: bool = True,
) -> nn.Module:
    """Build the documented standard-MobileNetV3 adaptation."""

    weights = MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
    model = mobilenet_v3_small(weights=weights)
    input_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(input_features, 2)
    if freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
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
            "implementation_label": "adapted_standard_mobilenet_v3_small",
            "checkpoint": str(self.checkpoint_path),
            "device": str(self.device),
            "patch_size": list(self.patch_size),
            **self.checkpoint_metadata,
        }
