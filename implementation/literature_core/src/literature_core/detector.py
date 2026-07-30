from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .models import Detection

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
ULTRALYTICS_CONFIG_DIR = PACKAGE_ROOT / ".ultralytics"
ULTRALYTICS_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("YOLO_CONFIG_DIR", str(ULTRALYTICS_CONFIG_DIR))


class ObjectDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Return raw object detections for one BGR frame."""

    def metadata(self) -> dict[str, Any]:
        """Return serializable model/runtime provenance."""


def _name(names: dict[int, str] | list[str], class_id: int) -> str:
    return str(names[class_id])


def track_ids_from_output(
    tracks: np.ndarray,
    detection_count: int,
) -> list[int | None]:
    """Map ByteTrack output IDs to raw detections without dropping unmatched boxes."""

    track_ids: list[int | None] = [None] * detection_count
    for track in tracks:
        detection_index = int(track[-1])
        if 0 <= detection_index < detection_count:
            track_ids[detection_index] = int(track[4])
    return track_ids


class YOLOWorldDetector:
    """Lazy Ultralytics YOLO-World adapter with explicit text categories."""

    def __init__(
        self,
        weights: str | Path,
        prompts: Sequence[str] = ("car", "truck", "bus", "motorcycle"),
        confidence: float = 0.025,
        image_size: int = 1280,
        device: str = "auto",
    ) -> None:
        if not prompts or any(not str(prompt).strip() for prompt in prompts):
            raise ValueError("At least one non-empty prompt is required")
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        self.weights = str(weights)
        self.prompts = tuple(str(prompt) for prompt in prompts)
        self.confidence = confidence
        self.image_size = int(image_size)
        self.requested_device = device
        self._device: str | int | None = None
        self._model: Any | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from ultralytics import YOLOWorld

        self._device = (
            0
            if self.requested_device == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.requested_device == "auto"
            else self.requested_device
        )
        self._model = YOLOWorld(self.weights)
        self._model.set_classes(list(self.prompts))

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._ensure_model()
        results = self._model.predict(  # type: ignore[union-attr]
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            device=self._device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        return [
            Detection(
                bbox=tuple(float(value) for value in box),
                confidence=float(score),
                class_id=int(class_id),
                label=_name(result.names, int(class_id)),
            )
            for box, score, class_id in zip(
                boxes,
                confidences,
                class_ids,
                strict=True,
            )
        ]

    def metadata(self) -> dict[str, Any]:
        self._ensure_model()
        import torch
        import ultralytics

        return {
            "backend": "ultralytics_yolo_world",
            "paper": "Cheng et al., CVPR 2024",
            "weights": self.weights,
            "prompts": list(self.prompts),
            "confidence": self.confidence,
            "image_size": self.image_size,
            "requested_device": self.requested_device,
            "resolved_device": self._device,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "license_note": "Ultralytics runtime/weights AGPL-3.0 for this coursework",
            "vacant_inference": "absence_of_sufficient_mapped_occupancy_evidence",
        }


class ClosedSetYOLODetector:
    """Comparable pretrained YOLOv8 vehicle adapter for E0."""

    def __init__(
        self,
        weights: str | Path,
        class_ids: Sequence[int] = (2, 3, 5, 7),
        confidence: float = 0.025,
        image_size: int = 1280,
        device: str = "auto",
        use_tracking: bool = False,
        tracker_config: str | Path = "bytetrack.yaml",
    ) -> None:
        self.weights = str(weights)
        self.class_ids = tuple(int(value) for value in class_ids)
        self.confidence = float(confidence)
        self.image_size = int(image_size)
        self.requested_device = device
        self.use_tracking = use_tracking
        self.tracker_config = str(tracker_config)
        self._device: str | int | None = None
        self._model: Any | None = None
        self._tracker: Any | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from ultralytics import YOLO

        self._device = (
            0
            if self.requested_device == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.requested_device == "auto"
            else self.requested_device
        )
        self._model = YOLO(self.weights)

    def _ensure_tracker(self) -> None:
        if not self.use_tracking or self._tracker is not None:
            return
        from ultralytics.trackers.byte_tracker import BYTETracker
        from ultralytics.utils import IterableSimpleNamespace, YAML
        from ultralytics.utils.checks import check_yaml

        tracker_path = check_yaml(self.tracker_config)
        tracker_args = IterableSimpleNamespace(**YAML.load(tracker_path))
        if tracker_args.tracker_type != "bytetrack":
            raise ValueError("Only ByteTrack is supported for E5")
        self._tracker = BYTETracker(args=tracker_args)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._ensure_model()
        self._ensure_tracker()
        results = self._model.predict(  # type: ignore[union-attr]
            source=frame,
            conf=self.confidence,
            imgsz=self.image_size,
            classes=list(self.class_ids),
            device=self._device,
            verbose=False,
        )
        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return []
        if self.use_tracking:
            assert self._tracker is not None
            raw_boxes = result.boxes.cpu().numpy()
            tracks = self._tracker.update(raw_boxes, frame)
            track_ids = track_ids_from_output(tracks, len(raw_boxes))
        else:
            track_ids = [None] * len(result.boxes)
        boxes = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        return [
            Detection(
                bbox=tuple(float(value) for value in box),
                confidence=float(score),
                class_id=int(class_id),
                label=_name(result.names, int(class_id)),
                track_id=track_id,
            )
            for box, score, class_id, track_id in zip(
                boxes,
                confidences,
                class_ids,
                track_ids,
                strict=True,
            )
        ]

    def metadata(self) -> dict[str, Any]:
        self._ensure_model()
        import ultralytics

        return {
            "backend": "ultralytics_closed_set_yolo",
            "weights": self.weights,
            "class_ids": list(self.class_ids),
            "confidence": self.confidence,
            "image_size": self.image_size,
            "tracking": self.use_tracking,
            "tracker_config": self.tracker_config if self.use_tracking else None,
            "tracking_output_policy": (
                "retain_raw_detections_attach_matched_ids"
                if self.use_tracking
                else None
            ),
            "resolved_device": self._device,
            "ultralytics_version": ultralytics.__version__,
        }
