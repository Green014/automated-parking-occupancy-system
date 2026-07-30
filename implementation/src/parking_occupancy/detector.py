from __future__ import annotations

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .models import Detection

COCO_VEHICLE_CLASS_IDS = (2, 3, 5, 7)  # car, motorcycle, bus, truck
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))


def track_ids_from_output(
    tracks: np.ndarray,
    detection_count: int,
) -> list[int | None]:
    """Map ByteTrack output IDs back to raw detection indices.

    Ultralytics appends the raw detection index as the last column. Keeping
    unmatched raw detections prevents the tracker from becoming an unintended
    second detection-confidence gate.
    """

    track_ids: list[int | None] = [None] * detection_count
    for track in tracks:
        detection_index = int(track[-1])
        if 0 <= detection_index < detection_count:
            track_ids[detection_index] = int(track[4])
    return track_ids


class Detector(Protocol):
    def detect(self, frame: np.ndarray) -> list[Detection]:
        """Detect or track vehicles in one BGR frame."""

    def metadata(self) -> dict[str, Any]:
        """Return serializable backend metadata."""


class UltralyticsDetector:
    """Lazy Ultralytics YOLOv8 detector with optional ByteTrack association."""

    def __init__(
        self,
        weights: str = "yolov8n.pt",
        confidence: float = 0.25,
        image_size: int = 640,
        device: str = "auto",
        vehicle_class_ids: Sequence[int] = COCO_VEHICLE_CLASS_IDS,
        use_tracking: bool = False,
        tracker_config: str = "bytetrack.yaml",
        nms_iou: float | None = None,
        agnostic_nms: bool | None = None,
        max_detections: int | None = None,
        augmentation: bool | None = None,
        rect: bool | None = None,
        half: bool | None = None,
    ) -> None:
        if not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if nms_iou is not None and not 0.0 <= nms_iou <= 1.0:
            raise ValueError("nms_iou must be in [0, 1]")
        if max_detections is not None and max_detections <= 0:
            raise ValueError("max_detections must be positive")
        self.weights = weights
        self.confidence = confidence
        self.image_size = image_size
        self.requested_device = device
        self.vehicle_class_ids = tuple(int(value) for value in vehicle_class_ids)
        self.use_tracking = use_tracking
        self.tracker_config = tracker_config
        self.nms_iou = nms_iou
        self.agnostic_nms = agnostic_nms
        self.max_detections = max_detections
        self.augmentation = augmentation
        self.rect = rect
        self.half = half
        self._model: Any | None = None
        self._resolved_device: str | int | None = None
        self._tracker: Any | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from ultralytics import YOLO

        if self.requested_device == "auto":
            self._resolved_device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self._resolved_device = self.requested_device
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
            raise ValueError(
                "This baseline currently supports only tracker_type=bytetrack"
            )
        self._tracker = BYTETracker(args=tracker_args)

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._ensure_model()
        self._ensure_tracker()
        assert self._model is not None
        kwargs = {
            "source": frame,
            "conf": self.confidence,
            "imgsz": self.image_size,
            "classes": list(self.vehicle_class_ids),
            "device": self._resolved_device,
            "verbose": False,
        }
        optional = {
            "iou": self.nms_iou,
            "agnostic_nms": self.agnostic_nms,
            "max_det": self.max_detections,
            "augment": self.augmentation,
            "rect": self.rect,
            "half": self.half,
        }
        kwargs.update(
            {
                key: value
                for key, value in optional.items()
                if value is not None and not (key == "half" and value is False)
            }
        )
        results = self._model.predict(**kwargs)

        result = results[0]
        if result.boxes is None:
            return []
        track_ids: list[int | None]
        if self.use_tracking:
            assert self._tracker is not None
            raw_boxes = result.boxes.cpu().numpy()
            tracks = self._tracker.update(raw_boxes, frame)
            track_ids = track_ids_from_output(tracks, len(raw_boxes))
        else:
            track_ids = [None] * len(result.boxes)
        if len(result.boxes) == 0:
            return []

        xyxy = result.boxes.xyxy.detach().cpu().numpy()
        confidences = result.boxes.conf.detach().cpu().numpy()
        class_ids = result.boxes.cls.detach().cpu().numpy().astype(int)
        names = result.names
        return [
            Detection(
                bbox=tuple(float(value) for value in box),
                confidence=float(confidence),
                class_id=int(class_id),
                class_name=str(names[int(class_id)]),
                track_id=track_id,
            )
            for box, confidence, class_id, track_id in zip(
                xyxy,
                confidences,
                class_ids,
                track_ids,
                strict=True,
            )
        ]

    def metadata(self) -> dict[str, Any]:
        self._ensure_model()
        import cv2
        import torch
        import ultralytics

        return {
            "backend": "ultralytics",
            "weights": self.weights,
            "confidence": self.confidence,
            "image_size": self.image_size,
            "vehicle_class_ids": list(self.vehicle_class_ids),
            "tracking": self.use_tracking,
            "tracker_config": self.tracker_config if self.use_tracking else None,
            "tracking_output_policy": (
                "retain_raw_detections_attach_matched_ids"
                if self.use_tracking
                else None
            ),
            "nms_iou": self.nms_iou,
            "agnostic_nms": self.agnostic_nms,
            "max_detections": self.max_detections,
            "augmentation": self.augmentation,
            "rect": self.rect,
            "half": self.half,
            "requested_device": self.requested_device,
            "resolved_device": self._resolved_device,
            "torch_version": torch.__version__,
            "torch_cuda_version": torch.version.cuda,
            "cuda_available": torch.cuda.is_available(),
            "gpu_name": (
                torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
            ),
            "ultralytics_version": ultralytics.__version__,
            "opencv_version": cv2.__version__,
        }
