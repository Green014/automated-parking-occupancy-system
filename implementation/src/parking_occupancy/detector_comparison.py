from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .models import Detection


class ComparisonProtocolError(ValueError):
    """Raised when a comparison input conflicts with the frozen protocol."""


class ComparisonNotReady(RuntimeError):
    """Raised before inference when a required model binding is unavailable."""


V1_COMPARISON_PROTOCOL_ID = "D-COMP-NDISPARK-DEV-20260727-01"
V2_COMPARISON_PROTOCOLS = {
    "D-COMP-NDISPARK-DEV-V2-MAXDET300-20260727-01": 300,
    "D-COMP-NDISPARK-DEV-V2-MAXDET1000-20260727-01": 1000,
}


@dataclass(frozen=True)
class DetectorSpec:
    method_id: str
    name: str
    backend: str
    status: str
    weights_name: str | None
    weights_sha256: str | None
    source_class_ids: tuple[int, ...]
    source_class_names: tuple[str, ...]
    prompts: tuple[str, ...]
    project_class_id: int
    project_class_name: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_comparison_protocol(
    config_path: Path,
) -> tuple[dict[str, Any], dict[str, DetectorSpec]]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    protocol_id = payload.get("comparison_protocol_id")
    if protocol_id == V1_COMPARISON_PROTOCOL_ID:
        if payload.get("status") != "frozen_before_new_predictions":
            raise ComparisonProtocolError("Comparison protocol is not frozen")
        expected_max_detections = 300
        require_agnostic_nms = False
    elif protocol_id in V2_COMPARISON_PROTOCOLS:
        if payload.get("status") != (
            "frozen_before_corrected_predictions"
        ):
            raise ComparisonProtocolError(
                "Stage I-v2 comparison protocol is not frozen"
            )
        if payload.get("protocol_generation") != "stage_i_v2":
            raise ComparisonProtocolError(
                "Stage I-v2 protocol_generation must be explicit"
            )
        if payload.get("evaluation_role") != (
            "corrected_development_evaluation"
        ):
            raise ComparisonProtocolError(
                "Stage I-v2 evaluation role must be corrected development"
            )
        expected_max_detections = V2_COMPARISON_PROTOCOLS[str(protocol_id)]
        require_agnostic_nms = True
    else:
        raise ComparisonProtocolError("Unexpected comparison protocol ID")
    if list(payload.get("models", {})) != ["D0", "D1", "D2", "D3"]:
        raise ComparisonProtocolError("Model registry must be D0/D1/D2/D3")
    if payload["models"]["D3"].get("status") != "deferred":
        raise ComparisonProtocolError("D3 must remain deferred")

    common = payload["common_inference"]
    expected_common = {
        "split": "val",
        "imgsz": 640,
        "confidence_floor": 0.001,
        "nms_iou": 0.7,
        "max_detections": expected_max_detections,
        "batch": 1,
        "single_class_evaluation": True,
        "augmentation": False,
    }
    for key, expected in expected_common.items():
        if common.get(key) != expected:
            raise ComparisonProtocolError(
                f"Frozen common setting {key} must be {expected!r}"
            )
    if require_agnostic_nms and common.get("agnostic_nms") is not True:
        raise ComparisonProtocolError(
            "Stage I-v2 must explicitly set agnostic_nms: true"
        )

    specs: dict[str, DetectorSpec] = {}
    for method_id in ("D0", "D1", "D2"):
        item = payload["models"][method_id]
        specs[method_id] = DetectorSpec(
            method_id=method_id,
            name=str(item["name"]),
            backend=str(item["backend"]),
            status=str(item["status"]),
            weights_name=(
                str(item["weights_name"])
                if item.get("weights_name") is not None
                else None
            ),
            weights_sha256=(
                str(item["weights_sha256"])
                if item.get("weights_sha256") is not None
                else None
            ),
            source_class_ids=tuple(
                int(value) for value in item["source_class_ids"]
            ),
            source_class_names=tuple(
                str(value) for value in item["source_class_names"]
            ),
            prompts=tuple(str(value) for value in item.get("prompts", [])),
            project_class_id=int(item["project_class_id"]),
            project_class_name=str(item["project_class_name"]),
        )
        if (
            specs[method_id].project_class_id != 0
            or specs[method_id].project_class_name != "vehicle"
        ):
            raise ComparisonProtocolError(
                f"{method_id} does not map to class 0 vehicle"
            )

    for key in ("preparation_record", "validation_manifest"):
        artifact = payload["data"][key]
        path = _resolve_from_config(config_path, str(artifact["path"]))
        if not path.is_file():
            raise ComparisonProtocolError(f"Missing frozen artifact: {path}")
        if path.stat().st_size != int(artifact["bytes"]):
            raise ComparisonProtocolError(f"{key} byte size mismatch")
        if sha256_file(path) != str(artifact["sha256"]):
            raise ComparisonProtocolError(f"{key} SHA-256 mismatch")
    return payload, specs


def _resolve_dataset_root(data_yaml: Path, payload: dict[str, Any]) -> Path:
    root = Path(str(payload.get("path", data_yaml.parent)))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    return root


def _split_paths(
    data_yaml: Path,
    payload: dict[str, Any],
    split: str,
) -> tuple[Path, Path]:
    root = _resolve_dataset_root(data_yaml, payload)
    image_path = Path(str(payload[split]))
    if not image_path.is_absolute():
        image_path = root / image_path
    try:
        relative = image_path.resolve().relative_to(root.resolve())
    except ValueError as exc:
        raise ComparisonProtocolError(
            f"{split} image path escapes the dataset root"
        ) from exc
    parts = list(relative.parts)
    if not parts or parts[0] != "images":
        raise ComparisonProtocolError(
            f"{split} image path must be below images/"
        )
    parts[0] = "labels"
    label_path = root.joinpath(*parts)
    return image_path.resolve(), label_path.resolve()


def validate_prepared_dataset(
    *,
    data_yaml: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    data_yaml = data_yaml.resolve()
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in data.get("names", {}).items()}
    if names != {0: "vehicle"}:
        raise ComparisonProtocolError(
            f"Dataset names must be {{0: 'vehicle'}}, got {names}"
        )

    summary: dict[str, Any] = {}
    for data_split, expected_prefix in (("train", "train"), ("val", "validation")):
        image_root, label_root = _split_paths(data_yaml, data, data_split)
        image_paths = sorted(
            path
            for pattern in ("*.jpg", "*.jpeg", "*.png")
            for path in image_root.glob(pattern)
        )
        label_paths = sorted(label_root.glob("*.txt"))
        if len(image_paths) != len(label_paths):
            raise ComparisonProtocolError(
                f"{data_split} image/label file counts differ"
            )
        box_count = 0
        classes: set[int] = set()
        for label_path in label_paths:
            for line_number, line in enumerate(
                label_path.read_text(encoding="utf-8").splitlines(),
                start=1,
            ):
                if not line.strip():
                    continue
                values = line.split()
                if len(values) != 5:
                    raise ComparisonProtocolError(
                        f"Invalid YOLO row {label_path}:{line_number}"
                    )
                class_id = int(values[0])
                coordinates = [float(value) for value in values[1:]]
                if class_id != 0 or any(
                    value < 0.0 or value > 1.0 for value in coordinates
                ):
                    raise ComparisonProtocolError(
                        f"Invalid class/coordinates {label_path}:{line_number}"
                    )
                classes.add(class_id)
                box_count += 1

        expected = protocol["data"]["expected"]
        expected_images = int(expected[f"{expected_prefix}_images"])
        expected_boxes = int(expected[f"{expected_prefix}_boxes"])
        if len(image_paths) != expected_images or box_count != expected_boxes:
            raise ComparisonProtocolError(
                f"{data_split} prepared counts differ from protocol"
            )
        summary[data_split] = {
            "images": len(image_paths),
            "boxes": box_count,
            "classes": sorted(classes),
            "image_root": str(image_root),
        }
    return summary


def canonicalize_detections(
    *,
    boxes_xyxy: np.ndarray,
    confidences: np.ndarray,
    source_class_ids: np.ndarray,
    allowed_source_class_ids: tuple[int, ...],
    agnostic_nms_iou: float | None = None,
) -> list[Detection]:
    if not (
        len(boxes_xyxy) == len(confidences) == len(source_class_ids)
    ):
        raise ValueError("Ultralytics detection arrays have different lengths")
    boxes_xyxy = np.asarray(boxes_xyxy)
    confidences = np.asarray(confidences)
    source_class_ids = np.asarray(source_class_ids)
    allowed = set(allowed_source_class_ids)
    allowed_indices = np.asarray(
        [
            index
            for index, source_class_id in enumerate(source_class_ids)
            if int(source_class_id) in allowed
        ],
        dtype=int,
    )
    if agnostic_nms_iou is not None:
        if not 0.0 <= float(agnostic_nms_iou) <= 1.0:
            raise ValueError("agnostic_nms_iou must be in [0, 1]")
        ordered = allowed_indices[
            np.argsort(-confidences[allowed_indices], kind="stable")
        ]
        kept: list[int] = []
        for index in ordered:
            if kept:
                overlap = pairwise_iou_xyxy(
                    boxes_xyxy[np.asarray(kept, dtype=int)],
                    boxes_xyxy[np.asarray([index], dtype=int)],
                )
                if bool(np.any(overlap[:, 0] > float(agnostic_nms_iou))):
                    continue
            kept.append(int(index))
        allowed_indices = np.asarray(kept, dtype=int)

    detections = []
    for index in allowed_indices:
        box = boxes_xyxy[index]
        confidence = confidences[index]
        detections.append(
            Detection(
                bbox=tuple(float(value) for value in box),
                confidence=float(confidence),
                class_id=0,
                class_name="vehicle",
            )
        )
    return detections


def pairwise_iou_xyxy(
    ground_truth_boxes: np.ndarray,
    predicted_boxes: np.ndarray,
) -> np.ndarray:
    """Return an M-by-N IoU matrix for xyxy ground truth and predictions."""

    ground_truth_boxes = np.asarray(ground_truth_boxes, dtype=np.float64)
    predicted_boxes = np.asarray(predicted_boxes, dtype=np.float64)
    if ground_truth_boxes.size == 0:
        ground_truth_boxes = ground_truth_boxes.reshape(0, 4)
    if predicted_boxes.size == 0:
        predicted_boxes = predicted_boxes.reshape(0, 4)
    if (
        ground_truth_boxes.ndim != 2
        or predicted_boxes.ndim != 2
        or ground_truth_boxes.shape[1] != 4
        or predicted_boxes.shape[1] != 4
    ):
        raise ValueError("IoU inputs must have shape (N, 4)")

    intersection_min = np.maximum(
        ground_truth_boxes[:, None, :2],
        predicted_boxes[None, :, :2],
    )
    intersection_max = np.minimum(
        ground_truth_boxes[:, None, 2:],
        predicted_boxes[None, :, 2:],
    )
    intersection_size = np.clip(
        intersection_max - intersection_min,
        a_min=0.0,
        a_max=None,
    )
    intersection = intersection_size[..., 0] * intersection_size[..., 1]
    ground_truth_area = np.prod(
        np.clip(
            ground_truth_boxes[:, 2:] - ground_truth_boxes[:, :2],
            a_min=0.0,
            a_max=None,
        ),
        axis=1,
    )
    predicted_area = np.prod(
        np.clip(
            predicted_boxes[:, 2:] - predicted_boxes[:, :2],
            a_min=0.0,
            a_max=None,
        ),
        axis=1,
    )
    union = (
        ground_truth_area[:, None]
        + predicted_area[None, :]
        - intersection
    )
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0.0,
    )


def match_detection_boxes(
    *,
    ground_truth_boxes: np.ndarray,
    predicted_boxes: np.ndarray,
    iou_thresholds: np.ndarray | None = None,
) -> np.ndarray:
    """Match canonical single-class predictions one-to-one at each IoU."""

    thresholds = (
        np.linspace(0.5, 0.95, 10)
        if iou_thresholds is None
        else np.asarray(iou_thresholds, dtype=np.float64)
    )
    predicted_boxes = np.asarray(predicted_boxes, dtype=np.float64)
    if predicted_boxes.size == 0:
        predicted_boxes = predicted_boxes.reshape(0, 4)
    correct = np.zeros(
        (len(predicted_boxes), len(thresholds)),
        dtype=bool,
    )
    iou = pairwise_iou_xyxy(ground_truth_boxes, predicted_boxes)
    for threshold_index, threshold in enumerate(thresholds):
        matches = np.argwhere(iou >= float(threshold))
        if not len(matches):
            continue
        scores = iou[matches[:, 0], matches[:, 1]]
        matches = np.column_stack((matches, scores))
        if len(matches) > 1:
            matches = matches[np.argsort(matches[:, 2])[::-1]]
            matches = matches[
                np.unique(matches[:, 1], return_index=True)[1]
            ]
            matches = matches[np.argsort(matches[:, 2])[::-1]]
            matches = matches[
                np.unique(matches[:, 0], return_index=True)[1]
            ]
        correct[matches[:, 1].astype(int), threshold_index] = True
    return correct


def _load_ground_truth_boxes(
    *,
    label_path: Path,
    image_width: int,
    image_height: int,
) -> np.ndarray:
    boxes = []
    for line in label_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, x_center, y_center, width, height = (
            float(value) for value in line.split()
        )
        if int(class_id) != 0:
            raise ComparisonProtocolError(
                f"Non-vehicle class in prepared label: {label_path}"
            )
        x_center *= image_width
        width *= image_width
        y_center *= image_height
        height *= image_height
        boxes.append(
            [
                x_center - width / 2.0,
                y_center - height / 2.0,
                x_center + width / 2.0,
                y_center + height / 2.0,
            ]
        )
    return np.asarray(boxes, dtype=np.float32).reshape(-1, 4)


class ComparisonDetectorAdapter:
    """Lazy D0/D1/D2 adapter with a shared inference contract."""

    def __init__(
        self,
        *,
        spec: DetectorSpec,
        weights_path: Path,
        common: dict[str, Any],
        device: str,
    ) -> None:
        self.spec = spec
        self.weights_path = weights_path.resolve()
        self.common = common
        self.requested_device = device
        self._resolved_device: str | int | None = None
        self._model: Any | None = None

    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import torch
        from ultralytics import YOLO, YOLOWorld

        self._resolved_device = (
            0
            if self.requested_device == "auto" and torch.cuda.is_available()
            else "cpu"
            if self.requested_device == "auto"
            else self.requested_device
        )
        if self.spec.backend == "ultralytics_yolo":
            self._model = YOLO(str(self.weights_path))
        elif self.spec.backend == "ultralytics_yolo_world":
            self._model = YOLOWorld(str(self.weights_path))
            self._model.set_classes(list(self.spec.prompts))
        else:
            raise ComparisonProtocolError(
                f"Unsupported backend: {self.spec.backend}"
            )

    @property
    def prediction_class_filter(self) -> list[int] | None:
        if self.spec.backend == "ultralytics_yolo_world":
            return None
        return list(self.spec.source_class_ids)

    def _canonicalize_result(self, result: Any) -> list[Detection]:
        if result.boxes is None or len(result.boxes) == 0:
            return []
        return canonicalize_detections(
            boxes_xyxy=result.boxes.xyxy.detach().cpu().numpy(),
            confidences=result.boxes.conf.detach().cpu().numpy(),
            source_class_ids=(
                result.boxes.cls.detach().cpu().numpy().astype(int)
            ),
            allowed_source_class_ids=self.spec.source_class_ids,
            agnostic_nms_iou=(
                float(self.common["nms_iou"])
                if bool(self.common.get("agnostic_nms", False))
                else None
            ),
        )

    def detect(self, frame: np.ndarray) -> list[Detection]:
        self._ensure_model()
        assert self._model is not None
        result = self._model.predict(
            source=frame,
            conf=float(self.common["confidence_floor"]),
            iou=float(self.common["nms_iou"]),
            imgsz=int(self.common["imgsz"]),
            max_det=int(self.common["max_detections"]),
            agnostic_nms=bool(self.common.get("agnostic_nms", False)),
            classes=self.prediction_class_filter,
            device=self._resolved_device,
            augment=bool(self.common["augmentation"]),
            verbose=False,
        )[0]
        return self._canonicalize_result(result)

    def predict_images(self, image_paths: list[Path]) -> list[Any]:
        self._ensure_model()
        assert self._model is not None
        return list(
            self._model.predict(
                source=[str(path) for path in image_paths],
                conf=float(self.common["confidence_floor"]),
                iou=float(self.common["nms_iou"]),
                imgsz=int(self.common["imgsz"]),
                max_det=int(self.common["max_detections"]),
                agnostic_nms=bool(
                    self.common.get("agnostic_nms", False)
                ),
                batch=int(self.common["batch"]),
                classes=self.prediction_class_filter,
                device=self._resolved_device,
                augment=bool(self.common["augmentation"]),
                rect=bool(self.common["rect"]),
                half=bool(self.common["half"]),
                verbose=False,
            )
        )

    def model_metadata(self) -> dict[str, Any]:
        self._ensure_model()
        import cv2
        import torch
        import ultralytics

        assert self._model is not None
        parameters = sum(
            parameter.numel()
            for parameter in self._model.model.parameters()
        )
        return {
            "method_id": self.spec.method_id,
            "name": self.spec.name,
            "backend": self.spec.backend,
            "weights_name": self.weights_path.name,
            "weights_bytes": self.weights_path.stat().st_size,
            "weights_sha256": sha256_file(self.weights_path),
            "parameter_count": int(parameters),
            "source_class_ids": list(self.spec.source_class_ids),
            "source_class_names": list(self.spec.source_class_names),
            "prompts": list(self.spec.prompts),
            "project_class": {"id": 0, "name": "vehicle"},
            "requested_device": self.requested_device,
            "resolved_device": self._resolved_device,
            "python": platform.python_version(),
            "torch": torch.__version__,
            "torch_cuda": torch.version.cuda,
            "ultralytics": ultralytics.__version__,
            "opencv": cv2.__version__,
        }

    def metadata(self) -> dict[str, Any]:
        """Expose the shared Detector protocol metadata method."""

        return self.model_metadata()

    def release(self) -> None:
        """Release one comparison model before measuring the next model."""

        if self._model is None:
            return
        try:
            self._model.model.to("cpu")
        finally:
            self._model = None
        import gc
        import torch

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def comparison_preflight(
    *,
    config_path: Path,
    data_yaml: Path,
    weight_paths: dict[str, Path | None],
    runtime_weight_hashes: dict[str, str | None] | None = None,
) -> tuple[dict[str, Any], dict[str, DetectorSpec]]:
    protocol, specs = load_comparison_protocol(config_path)
    dataset = validate_prepared_dataset(
        data_yaml=data_yaml,
        protocol=protocol,
    )
    runtime_weight_hashes = runtime_weight_hashes or {}

    model_checks: dict[str, Any] = {}
    all_ready = True
    for method_id, spec in specs.items():
        path = weight_paths.get(method_id)
        expected_hash = spec.weights_sha256 or runtime_weight_hashes.get(
            method_id
        )
        reasons: list[str] = []
        actual_hash = None
        if path is None:
            reasons.append("weights_path_missing")
        elif not path.is_file():
            reasons.append("weights_file_missing")
        else:
            actual_hash = sha256_file(path)
            if expected_hash is None:
                reasons.append("runtime_weights_hash_binding_missing")
            elif actual_hash != expected_hash:
                reasons.append("weights_sha256_mismatch")
        if spec.status == "not_trained" and path is None:
            reasons.append("model_not_trained")
        ready = not reasons
        all_ready = all_ready and ready
        model_checks[method_id] = {
            "name": spec.name,
            "configured_status": spec.status,
            "weights_name": path.name if path is not None else None,
            "expected_sha256": expected_hash,
            "actual_sha256": actual_hash,
            "ready": ready,
            "reasons": reasons,
        }

    report = {
        "schema_version": 1,
        "comparison_protocol_id": protocol["comparison_protocol_id"],
        "task": "single_class_vehicle_box_detection_development_comparison",
        "dataset": dataset,
        "models": model_checks,
        "all_required_models_ready": all_ready,
        "predictions_run": False,
        "execution_gate": (
            "open" if all_ready else "blocked_before_output_creation"
        ),
    }
    return report, specs


def _evaluate_adapter(
    *,
    adapter: ComparisonDetectorAdapter,
    data_yaml: Path,
    method_root: Path,
    protocol: dict[str, Any],
) -> dict[str, Any]:
    """Evaluate canonical vehicle predictions without filtering class-0 truth."""

    ultralytics_config = method_root.parent / "_ultralytics_config"
    ultralytics_config.mkdir(parents=True, exist_ok=True)
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config.resolve())
    import torch
    from ultralytics.utils.metrics import (
        ConfusionMatrix,
        ap_per_class,
        smooth,
    )

    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    image_root, label_root = _split_paths(data_yaml, data, "val")
    image_paths = sorted(
        path
        for pattern in ("*.jpg", "*.jpeg", "*.png")
        for path in image_root.glob(pattern)
    )
    measure_cuda = (
        torch.cuda.is_available()
        and str(adapter.requested_device).lower() not in {"cpu", "mps"}
    )
    if measure_cuda:
        torch.cuda.reset_peak_memory_stats()
    results = adapter.predict_images(image_paths)
    if measure_cuda:
        torch.cuda.synchronize()
        peak_cuda_allocated = int(torch.cuda.max_memory_allocated())
        peak_cuda_reserved = int(torch.cuda.max_memory_reserved())
    else:
        peak_cuda_allocated = None
        peak_cuda_reserved = None
    if len(results) != len(image_paths):
        raise RuntimeError(
            f"{adapter.spec.method_id} returned an unexpected result count"
        )

    method_root.mkdir(parents=True)
    correct_parts: list[np.ndarray] = []
    confidence_parts: list[np.ndarray] = []
    prediction_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    per_image: list[dict[str, Any]] = []
    speed_totals: dict[str, float] = {}
    detection_lines = []
    max_detections = int(protocol["common_inference"]["max_detections"])
    images_reaching_max_det = 0
    for image_index, (image_path, result) in enumerate(
        zip(image_paths, results, strict=True)
    ):
        raw_detection_count = (
            len(result.boxes) if result.boxes is not None else 0
        )
        if raw_detection_count >= max_detections:
            images_reaching_max_det += 1
        detections = adapter._canonicalize_result(result)
        predicted_boxes = np.asarray(
            [detection.bbox for detection in detections],
            dtype=np.float32,
        ).reshape(-1, 4)
        confidences = np.asarray(
            [detection.confidence for detection in detections],
            dtype=np.float32,
        )
        image_height, image_width = (
            int(result.orig_shape[0]),
            int(result.orig_shape[1]),
        )
        ground_truth_boxes = _load_ground_truth_boxes(
            label_path=label_root / f"{image_path.stem}.txt",
            image_width=image_width,
            image_height=image_height,
        )
        correct = match_detection_boxes(
            ground_truth_boxes=ground_truth_boxes,
            predicted_boxes=predicted_boxes,
        )
        correct_parts.append(correct)
        confidence_parts.append(confidences)
        prediction_parts.append(
            np.zeros(len(predicted_boxes), dtype=np.float32)
        )
        target_parts.append(
            np.zeros(len(ground_truth_boxes), dtype=np.float32)
        )
        for key, value in result.speed.items():
            speed_totals[str(key)] = (
                speed_totals.get(str(key), 0.0) + float(value)
            )
        record = {
            "image_index": image_index,
            "image_name": image_path.name,
            "ground_truth_box_count": len(ground_truth_boxes),
            "raw_ultralytics_detection_count": raw_detection_count,
            "detection_count": len(detections),
            "detections": [
                {
                    "bbox_xyxy": list(detection.bbox),
                    "confidence": detection.confidence,
                    "class_id": 0,
                    "class_name": "vehicle",
                }
                for detection in detections
            ],
        }
        per_image.append(
            {
                "ground_truth_boxes": ground_truth_boxes,
                "predicted_boxes": predicted_boxes,
                "confidences": confidences,
            }
        )
        detection_lines.append(json.dumps(record))

    (method_root / "detections.jsonl").write_text(
        "\n".join(detection_lines) + "\n",
        encoding="utf-8",
    )
    correct = np.concatenate(correct_parts, axis=0)
    confidences = np.concatenate(confidence_parts)
    prediction_classes = np.concatenate(prediction_parts)
    target_classes = np.concatenate(target_parts)
    metrics = ap_per_class(
        correct,
        confidences,
        prediction_classes,
        target_classes,
        plot=True,
        save_dir=method_root,
        names={0: "vehicle"},
    )
    (
        _true_positives,
        _false_positives,
        precision,
        recall,
        f1,
        average_precision,
        _class_ids,
        _precision_curve,
        _recall_curve,
        f1_curve,
        confidence_axis,
        _pr_precision,
    ) = metrics
    best_index = int(smooth(f1_curve.mean(0), 0.1).argmax())
    operating_confidence = float(confidence_axis[best_index])

    confusion = ConfusionMatrix(names={0: "vehicle"}, task="detect")
    for item in per_image:
        confusion.process_batch(
            {
                "bboxes": torch.from_numpy(item["predicted_boxes"]),
                "conf": torch.from_numpy(item["confidences"]),
                "cls": torch.zeros(
                    len(item["predicted_boxes"]),
                    dtype=torch.float32,
                ),
            },
            {
                "bboxes": torch.from_numpy(item["ground_truth_boxes"]),
                "cls": torch.zeros(
                    len(item["ground_truth_boxes"]),
                    dtype=torch.float32,
                ),
            },
            conf=operating_confidence,
            iou_thres=0.5,
        )
    confusion.plot(normalize=False, save_dir=str(method_root))
    confusion.plot(normalize=True, save_dir=str(method_root))

    speed = {
        key: value / len(image_paths)
        for key, value in speed_totals.items()
    }
    latency = sum(
        speed.get(key, 0.0)
        for key in ("preprocess", "inference", "postprocess")
    )
    metadata = adapter.model_metadata()
    return {
        "method_id": adapter.spec.method_id,
        "name": adapter.spec.name,
        "task": "single_class_vehicle_box_detection",
        "dataset_role": "consumed_development_validation",
        "images": len(image_paths),
        "ground_truth_boxes": len(target_classes),
        "predictions": len(prediction_classes),
        "max_detections": max_detections,
        "images_reaching_max_det": images_reaching_max_det,
        "precision": float(np.mean(precision)),
        "recall": float(np.mean(recall)),
        "f1": float(np.mean(f1)),
        "operating_confidence_max_f1": operating_confidence,
        "map_50": float(np.mean(average_precision[:, 0])),
        "map_50_95": float(np.mean(average_precision)),
        "speed_ms_per_image": speed,
        "framework_pipeline_latency_ms_per_image": latency,
        "framework_pipeline_fps": 1000.0 / latency if latency > 0 else None,
        "peak_cuda_memory_allocated_bytes": peak_cuda_allocated,
        "peak_cuda_memory_reserved_bytes": peak_cuda_reserved,
        "source_class_filter": list(adapter.spec.source_class_ids),
        "canonical_project_class": {"id": 0, "name": "vehicle"},
        "matching": "one_to_one_descending_iou",
        "iou_thresholds": [
            float(value) for value in np.linspace(0.5, 0.95, 10)
        ],
        "common_inference": protocol["common_inference"],
        "runtime_metadata": metadata,
        "ultralytics_output_dir": method_root.name,
    }


def run_comparison(
    *,
    config_path: Path,
    data_yaml: Path,
    output_root: Path,
    weight_paths: dict[str, Path | None],
    runtime_weight_hashes: dict[str, str | None],
    device: str,
) -> dict[str, Any]:
    preflight, specs = comparison_preflight(
        config_path=config_path,
        data_yaml=data_yaml,
        weight_paths=weight_paths,
        runtime_weight_hashes=runtime_weight_hashes,
    )
    if not preflight["all_required_models_ready"]:
        raise ComparisonNotReady(
            "D0/D1/D2 comparison preflight failed; no inference was run"
        )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite comparison output: {output_root}"
        )

    protocol, _ = load_comparison_protocol(config_path)
    common = protocol["common_inference"]
    adapters = {
        method_id: ComparisonDetectorAdapter(
            spec=spec,
            weights_path=weight_paths[method_id],  # type: ignore[arg-type]
            common=common,
            device=device,
        )
        for method_id, spec in specs.items()
    }
    output_root.mkdir(parents=True)
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )

    reports = {}
    runtime_rows = []
    for method_id, adapter in adapters.items():
        method_root = output_root / method_id
        report = _evaluate_adapter(
            adapter=adapter,
            data_yaml=data_yaml.resolve(),
            method_root=method_root,
            protocol=protocol,
        )
        if int(report["ground_truth_boxes"]) != int(
            protocol["data"]["expected"]["validation_boxes"]
        ):
            raise RuntimeError(
                f"{method_id} loaded an unexpected ground-truth box count"
            )
        metadata = report["runtime_metadata"]
        latency = report["framework_pipeline_latency_ms_per_image"]
        (method_root / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        (method_root / "runtime_metadata.json").write_text(
            json.dumps(metadata, indent=2) + "\n",
            encoding="utf-8",
        )
        reports[method_id] = report
        runtime_rows.append(
            {
                "method_id": method_id,
                "weights_bytes": metadata["weights_bytes"],
                "parameter_count": metadata["parameter_count"],
                "latency_ms_per_image": latency,
                "fps": report["framework_pipeline_fps"],
                "images_reaching_max_det": report[
                    "images_reaching_max_det"
                ],
                "peak_cuda_memory_allocated_bytes": report[
                    "peak_cuda_memory_allocated_bytes"
                ],
                "peak_cuda_memory_reserved_bytes": report[
                    "peak_cuda_memory_reserved_bytes"
                ],
            }
        )
        adapter.release()

    comparison = {
        "schema_version": (
            2
            if protocol.get("protocol_generation") == "stage_i_v2"
            else 1
        ),
        "comparison_protocol_id": protocol["comparison_protocol_id"],
        "protocol_generation": protocol.get(
            "protocol_generation",
            "stage_i_v1",
        ),
        "evaluation_role": protocol.get(
            "evaluation_role",
            "historical_development_evaluation",
        ),
        "dataset_role": "consumed_development_validation",
        "common_inference": protocol["common_inference"],
        "selection_policy": protocol["selection_policy"],
        "models": reports,
        "negative_results_retained": True,
    }
    (output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "model_runtime_table.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(runtime_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(runtime_rows)
    return comparison
