from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import json
import platform
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .models import Detection, ParkingSlot


STAGE_M_PROTOCOL_ID = "STAGE-M-OPEN-SOURCE-TRACKING-ROBUSTNESS-20260728-01"
SUPPORTED_TRACKER_TYPES = {"bytetrack", "tracktrack"}


class StageMProtocolError(ValueError):
    """Raised when a Stage M freeze or data-gate invariant is violated."""


@dataclass(frozen=True, slots=True)
class InferenceSettings:
    """Frozen detector settings shared by OS0 and T0-T3."""

    weights: str
    confidence: float
    nms_iou: float
    image_size: int
    class_ids: tuple[int, ...]
    max_detections: int
    device: str = "auto"
    agnostic_nms: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not 0.0 <= self.nms_iou <= 1.0:
            raise ValueError("nms_iou must be in [0, 1]")
        if self.image_size <= 0:
            raise ValueError("image_size must be positive")
        if self.max_detections <= 0:
            raise ValueError("max_detections must be positive")
        if not self.class_ids:
            raise ValueError("class_ids must not be empty")


@dataclass(frozen=True, slots=True)
class OS0FrameResult:
    """Official ParkingManagement output plus local audit logging."""

    annotated_frame: np.ndarray
    detections: tuple[Detection, ...]
    slot_states: dict[str, bool]
    filled_slots: int
    available_slots: int
    logic_provenance: str = (
        "ultralytics_official_centre_point_in_polygon;"
        "local_per_slot_replay_for_logging_only"
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def load_stage_m_protocol(
    config_path: Path,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    """Load the Stage M freeze and optionally verify every runnable input."""

    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise StageMProtocolError("Unsupported Stage M protocol schema")
    if payload.get("protocol_id") != STAGE_M_PROTOCOL_ID:
        raise StageMProtocolError("Unexpected Stage M protocol ID")
    scope = payload.get("scope", {})
    for key in (
        "stage_l_rerun",
        "stage_l_artifact_modification",
        "parameter_selection_from_stage_k",
        "parameter_selection_from_virat_0502",
        "parameter_selection_after_freeze",
        "formal_execution_without_open_data_gate",
    ):
        if scope.get(key) != "prohibited":
            raise StageMProtocolError(f"Stage M scope.{key} must be prohibited")

    methods = payload.get("methods", {})
    if tuple(methods) != ("OS0-Controlled", "T0", "T1", "T2", "T3"):
        raise StageMProtocolError("Stage M must freeze OS0-Controlled and T0-T3")
    if methods["OS0-Controlled"].get("tracker") != "tracktrack":
        raise StageMProtocolError("OS0-Controlled must use TrackTrack")
    if methods["T2"].get("tracker") != "ByteTrack":
        raise StageMProtocolError("T2 must use ByteTrack")
    if methods["T3"].get("tracker") != "TrackTrack":
        raise StageMProtocolError("T3 must use TrackTrack")

    shared = payload.get("shared_inference", {})
    frozen_values = {
        "confidence": 0.30,
        "nms_iou": 0.70,
        "imgsz": 640,
        "source_class_ids": [0],
        "max_detections": 300,
    }
    for key, expected in frozen_values.items():
        if shared.get(key) != expected:
            raise StageMProtocolError(
                f"Stage M shared_inference.{key} changed from Stage L"
            )
    if payload.get("classifier", {}).get("occupied_threshold") != 0.76:
        raise StageMProtocolError("E1b threshold must remain 0.76")
    if payload.get("mapping", {}).get("minimum_slot_coverage") != 0.40:
        raise StageMProtocolError("B1 coverage must remain 0.40")

    trackers = payload.get("trackers", {})
    for key, tracker_type in (
        ("bytetrack", "bytetrack"),
        ("tracktrack", "tracktrack"),
    ):
        record = trackers.get(key, {})
        tracker_path = _resolve_from_config(
            config_path, str(record.get("config_path", ""))
        )
        loaded = load_tracker_config(tracker_path, expected_type=tracker_type)
        if loaded["tracker_type"] != tracker_type:
            raise StageMProtocolError(f"Invalid {key} tracker freeze")
        if verify_files:
            if tracker_path.stat().st_size != int(record["config_bytes"]):
                raise StageMProtocolError(f"{key} tracker byte count mismatch")
            if sha256_file(tracker_path) != str(record["config_sha256"]):
                raise StageMProtocolError(f"{key} tracker SHA-256 mismatch")

    if verify_files:
        artifact_records = [
            (
                _resolve_from_config(config_path, str(shared["weights_path"])),
                shared["weights_bytes"],
                shared["weights_sha256"],
                "D1 weights",
            ),
            (
                _resolve_from_config(
                    config_path,
                    str(payload["classifier"]["checkpoint_path"]),
                ),
                payload["classifier"]["checkpoint_bytes"],
                payload["classifier"]["checkpoint_sha256"],
                "E1b checkpoint",
            ),
            (
                _resolve_from_config(
                    config_path,
                    str(payload["smoke"]["source_image"]["path"]),
                ),
                payload["smoke"]["source_image"]["bytes"],
                payload["smoke"]["source_image"]["sha256"],
                "smoke source image",
            ),
            (
                _resolve_from_config(
                    config_path,
                    str(payload["smoke"]["regions"]["path"]),
                ),
                payload["smoke"]["regions"]["bytes"],
                payload["smoke"]["regions"]["sha256"],
                "smoke regions",
            ),
        ]
        for section in ("config", "report", "registry"):
            item = payload["stage_l_preservation"][section]
            artifact_records.append(
                (
                    _resolve_from_config(config_path, str(item["path"])),
                    item["bytes"],
                    item["sha256"],
                    f"Stage L {section}",
                )
            )
        for path, expected_bytes, expected_hash, label in artifact_records:
            if not path.is_file():
                raise StageMProtocolError(f"Missing {label}: {path}")
            if path.stat().st_size != int(expected_bytes):
                raise StageMProtocolError(f"{label} byte count mismatch")
            if sha256_file(path) != str(expected_hash):
                raise StageMProtocolError(f"{label} SHA-256 mismatch")
    return payload


def verify_ultralytics_runtime(protocol: Mapping[str, Any]) -> dict[str, Any]:
    """Verify installed Ultralytics source files against the Stage M freeze."""

    import ultralytics
    from ultralytics.solutions import ParkingManagement
    from ultralytics.solutions.solutions import BaseSolution
    from ultralytics.trackers import TRACKTRACK
    from ultralytics.trackers import track as tracker_registration
    from ultralytics.utils.checks import check_yaml

    expected = protocol["runtime"]["source_files"]
    files = {
        "parking_management": Path(inspect.getfile(ParkingManagement)).resolve(),
        "base_solution": Path(inspect.getfile(BaseSolution)).resolve(),
        "tracker_registration": Path(
            inspect.getfile(tracker_registration)
        ).resolve(),
        "tracktrack_implementation": Path(inspect.getfile(TRACKTRACK)).resolve(),
        "upstream_tracktrack_yaml": Path(check_yaml("tracktrack.yaml")).resolve(),
    }
    rows: dict[str, Any] = {}
    for key, path in files.items():
        actual = sha256_file(path)
        frozen = str(expected[key]["sha256"])
        if actual != frozen:
            raise StageMProtocolError(
                f"Ultralytics {key} SHA-256 mismatch ({actual} != {frozen})"
            )
        rows[key] = {"path": str(path), "sha256": actual}
    version = importlib.metadata.version("ultralytics")
    if version != str(protocol["runtime"]["ultralytics"]):
        raise StageMProtocolError(
            f"Ultralytics version mismatch ({version} != "
            f"{protocol['runtime']['ultralytics']})"
        )

    package_versions = {
        "python": platform.python_version(),
        "torch": importlib.metadata.version("torch"),
        "opencv_python": importlib.metadata.version("opencv-python"),
        "numpy": importlib.metadata.version("numpy"),
        "pyyaml": importlib.metadata.version("PyYAML"),
        "lap": importlib.metadata.version("lap"),
        "shapely": importlib.metadata.version("shapely"),
    }
    for key, actual in package_versions.items():
        frozen = str(protocol["runtime"][key])
        if actual != frozen:
            raise StageMProtocolError(
                f"Runtime {key} version mismatch ({actual} != {frozen})"
            )

    license_specs = {
        "ultralytics": (
            "ultralytics",
            "ultralytics-8.4.104.dist-info/licenses/LICENSE",
            protocol["runtime"]["license"],
        ),
        "shapely": (
            "shapely",
            "shapely-2.1.2.dist-info/licenses/LICENSE.txt",
            protocol["runtime"]["shapely_license"],
        ),
    }
    licenses: dict[str, Any] = {}
    for key, (distribution_name, relative_path, frozen) in license_specs.items():
        distribution = importlib.metadata.distribution(distribution_name)
        path = Path(distribution.locate_file(relative_path)).resolve()
        actual_hash = sha256_file(path)
        if actual_hash != str(frozen["local_license_sha256"]):
            raise StageMProtocolError(f"{key} license SHA-256 mismatch")
        if "local_license_bytes" in frozen:
            if path.stat().st_size != int(frozen["local_license_bytes"]):
                raise StageMProtocolError(f"{key} license byte count mismatch")
        licenses[key] = {
            "path": str(path),
            "sha256": actual_hash,
            "spdx": str(frozen["spdx"]),
        }
    return {
        "ultralytics_version": version,
        "ultralytics_package": str(Path(inspect.getfile(ultralytics)).resolve()),
        "package_versions": package_versions,
        "licenses": licenses,
        "source_files": rows,
    }


def load_tracker_config(
    path: Path,
    *,
    expected_type: str | None = None,
) -> dict[str, Any]:
    """Load and validate a frozen Ultralytics tracker configuration."""

    path = path.resolve()
    if not path.is_file():
        raise StageMProtocolError(f"Missing tracker configuration: {path}")
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageMProtocolError("Tracker configuration must be a mapping")
    tracker_type = str(payload.get("tracker_type", "")).lower()
    if tracker_type not in SUPPORTED_TRACKER_TYPES:
        raise StageMProtocolError(
            f"Unsupported tracker_type={tracker_type!r}; "
            f"expected one of {sorted(SUPPORTED_TRACKER_TYPES)}"
        )
    if expected_type is not None and tracker_type != expected_type:
        raise StageMProtocolError(
            f"Expected tracker_type={expected_type}, got {tracker_type}"
        )
    for key in (
        "track_high_thresh",
        "track_low_thresh",
        "new_track_thresh",
        "track_buffer",
        "match_thresh",
    ):
        if key not in payload:
            raise StageMProtocolError(f"Tracker configuration misses {key}")
    return payload


def load_parking_regions(path: Path) -> tuple[ParkingSlot, ...]:
    """Load the JSON format accepted by official ParkingManagement."""

    path = path.resolve()
    if not path.is_file():
        raise StageMProtocolError(f"Missing parking-region JSON: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list) or not payload:
        raise StageMProtocolError("Parking-region JSON must be a non-empty list")

    slots: list[ParkingSlot] = []
    for index, item in enumerate(payload):
        if not isinstance(item, dict) or "points" not in item:
            raise StageMProtocolError(f"Region {index} has no points")
        slot_id = str(item.get("slot_id", item.get("id", f"slot_{index + 1:03d}")))
        slots.append(
            ParkingSlot(
                slot_id=slot_id,
                points=tuple(
                    (float(point[0]), float(point[1]))
                    for point in item["points"]
                ),
            )
        )
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise StageMProtocolError("Parking-region slot IDs must be unique")
    return tuple(slots)


def centre_point_slot_states(
    detections: Sequence[Detection],
    slots: Sequence[ParkingSlot],
) -> dict[str, bool]:
    """Replay the official centre-point rule for per-slot audit logging."""

    states: dict[str, bool] = {}
    for slot in slots:
        polygon = np.asarray(slot.points, dtype=np.int32).reshape((-1, 1, 2))
        states[slot.slot_id] = any(
            cv2.pointPolygonTest(polygon, detection.center, False) >= 0
            for detection in detections
        )
    return states


def _as_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _result_detections(result: Any) -> tuple[Detection, ...]:
    boxes = getattr(result, "boxes", None)
    if boxes is None or len(boxes) == 0:
        return ()

    xyxy = _as_numpy(boxes.xyxy)
    confidences = _as_numpy(boxes.conf)
    class_ids = _as_numpy(boxes.cls).astype(int)
    raw_ids = getattr(boxes, "id", None)
    if raw_ids is None:
        track_ids: list[int | None] = [None] * len(boxes)
    else:
        track_ids = [int(value) for value in _as_numpy(raw_ids)]
    names = getattr(result, "names", {})

    detections = []
    for box, confidence, class_id, track_id in zip(
        xyxy,
        confidences,
        class_ids,
        track_ids,
        strict=True,
    ):
        name = names[int(class_id)] if isinstance(names, (dict, list)) else class_id
        detections.append(
            Detection(
                bbox=tuple(float(value) for value in box),
                confidence=float(confidence),
                class_id=int(class_id),
                class_name=str(name),
                track_id=track_id,
            )
        )
    return tuple(detections)


class UltralyticsSequenceAdapter:
    """Use ``YOLO.track`` without bypassing TrackTrack predictor hooks.

    A model instance is retained only for consecutive frames from one source.
    Switching sources, or declaring static diagnostics, constructs a fresh
    model so tracker state and raw-prediction hooks cannot leak.
    """

    def __init__(
        self,
        settings: InferenceSettings,
        *,
        tracker_config: Path | None,
        model_factory: Callable[[str], Any] | None = None,
    ) -> None:
        self.settings = settings
        self.tracker_config = (
            None if tracker_config is None else tracker_config.resolve()
        )
        self._model_factory = model_factory
        self._model: Any | None = None
        self._source_id: str | None = None
        self._continuous = False
        self._generation = 0
        if self.tracker_config is not None:
            load_tracker_config(self.tracker_config)

    @property
    def generation(self) -> int:
        """Number of clean model/tracker sessions constructed."""

        return self._generation

    def _new_model(self) -> Any:
        if self._model_factory is None:
            from ultralytics import YOLO

            factory: Callable[[str], Any] = YOLO
        else:
            factory = self._model_factory
        self._generation += 1
        return factory(self.settings.weights)

    def begin_source(self, source_id: str, *, continuous: bool) -> None:
        """Start a source and reset all state when continuity is not valid."""

        if not source_id:
            raise ValueError("source_id must not be empty")
        must_reset = (
            self._model is None
            or not continuous
            or self._source_id != source_id
            or self._continuous != continuous
        )
        if must_reset:
            self._model = self._new_model()
        self._source_id = source_id
        self._continuous = continuous

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        if self._model is None or self._source_id is None:
            raise RuntimeError("begin_source must be called before detect")
        kwargs = {
            "source": frame,
            "conf": self.settings.confidence,
            "iou": self.settings.nms_iou,
            "imgsz": self.settings.image_size,
            "classes": list(self.settings.class_ids),
            "max_det": self.settings.max_detections,
            "agnostic_nms": self.settings.agnostic_nms,
            "device": (
                None if self.settings.device == "auto" else self.settings.device
            ),
            "verbose": False,
        }
        if self.tracker_config is None:
            results = self._model.predict(**kwargs)
        else:
            results = self._model.track(
                **kwargs,
                persist=True,
                tracker=str(self.tracker_config),
            )
        if not results:
            return ()
        return _result_detections(results[0])

    def metadata(self) -> dict[str, Any]:
        tracker_type = None
        tracker_sha256 = None
        if self.tracker_config is not None:
            tracker_type = load_tracker_config(self.tracker_config)["tracker_type"]
            tracker_sha256 = sha256_file(self.tracker_config)
        return {
            "backend": "ultralytics_model_track"
            if self.tracker_config is not None
            else "ultralytics_model_predict",
            "tracker_type": tracker_type,
            "tracker_config": (
                None
                if self.tracker_config is None
                else str(self.tracker_config)
            ),
            "tracker_config_sha256": tracker_sha256,
            "persist_policy": (
                "true_only_for_consecutive_frames_from_same_source"
            ),
            "source_switch_reset": "fresh_YOLO_model_instance",
            "generation": self._generation,
            "settings": {
                "weights": self.settings.weights,
                "confidence": self.settings.confidence,
                "nms_iou": self.settings.nms_iou,
                "image_size": self.settings.image_size,
                "class_ids": list(self.settings.class_ids),
                "max_detections": self.settings.max_detections,
                "device": self.settings.device,
                "agnostic_nms": self.settings.agnostic_nms,
            },
        }


class OS0ParkingAdapter:
    """Auditable wrapper around official ``solutions.ParkingManagement``."""

    def __init__(
        self,
        settings: InferenceSettings,
        *,
        region_json: Path,
        tracker_config: Path,
        manager_factory: Callable[..., Any] | None = None,
    ) -> None:
        self.settings = settings
        self.region_json = region_json.resolve()
        self.tracker_config = tracker_config.resolve()
        self.slots = load_parking_regions(self.region_json)
        load_tracker_config(self.tracker_config, expected_type="tracktrack")
        self._manager_factory = manager_factory
        self._manager: Any | None = None
        self._source_id: str | None = None
        self._continuous = False
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def _new_manager(self) -> Any:
        if self._manager_factory is None:
            from ultralytics.solutions import ParkingManagement

            factory: Callable[..., Any] = ParkingManagement
        else:
            factory = self._manager_factory
        self._generation += 1
        return factory(
            model=self.settings.weights,
            json_file=str(self.region_json),
            classes=list(self.settings.class_ids),
            conf=self.settings.confidence,
            iou=self.settings.nms_iou,
            imgsz=self.settings.image_size,
            max_det=self.settings.max_detections,
            device=(
                None if self.settings.device == "auto" else self.settings.device
            ),
            tracker=str(self.tracker_config),
            show=False,
            verbose=False,
        )

    def begin_source(self, source_id: str, *, continuous: bool) -> None:
        if not source_id:
            raise ValueError("source_id must not be empty")
        must_reset = (
            self._manager is None
            or not continuous
            or self._source_id != source_id
            or self._continuous != continuous
        )
        if must_reset:
            self._manager = self._new_manager()
        self._source_id = source_id
        self._continuous = continuous

    def process(self, frame: np.ndarray) -> OS0FrameResult:
        if self._manager is None or self._source_id is None:
            raise RuntimeError("begin_source must be called before process")
        official = self._manager.process(frame.copy())
        boxes = list(getattr(self._manager, "boxes", []))
        class_ids = list(getattr(self._manager, "clss", []))
        confidences = list(getattr(self._manager, "confs", []))
        track_ids = list(getattr(self._manager, "track_ids", []))
        if not confidences:
            confidences = [1.0] * len(boxes)
        if not track_ids:
            track_ids = [None] * len(boxes)

        names = getattr(getattr(self._manager, "model", None), "names", {})
        detections: list[Detection] = []
        for box, class_id, confidence, track_id in zip(
            boxes,
            class_ids,
            confidences,
            track_ids,
            strict=True,
        ):
            coordinates = _as_numpy(box).reshape(-1)
            class_id = int(class_id)
            name = (
                names[class_id]
                if isinstance(names, (dict, list))
                else class_id
            )
            detections.append(
                Detection(
                    bbox=tuple(float(value) for value in coordinates[:4]),
                    confidence=float(confidence),
                    class_id=class_id,
                    class_name=str(name),
                    track_id=None if track_id is None else int(track_id),
                )
            )
        slot_states = centre_point_slot_states(detections, self.slots)
        filled_slots = int(official.filled_slots)
        local_filled = sum(slot_states.values())
        if local_filled != filled_slots:
            raise RuntimeError(
                "Local per-slot centre-point log disagrees with official "
                f"ParkingManagement count ({local_filled} != {filled_slots})"
            )
        return OS0FrameResult(
            annotated_frame=np.asarray(official.plot_im),
            detections=tuple(detections),
            slot_states=slot_states,
            filled_slots=filled_slots,
            available_slots=int(official.available_slots),
        )

    def metadata(self) -> dict[str, Any]:
        from ultralytics.solutions import ParkingManagement

        source_path = Path(inspect.getfile(ParkingManagement)).resolve()
        return {
            "baseline": "OS0-Controlled",
            "official_object": "ultralytics.solutions.ParkingManagement",
            "official_logic": "centre_point_in_polygon",
            "local_adapter_role": "per_slot_logging_metrics_and_exports_only",
            "static_policy": "fresh_manager_per_image",
            "continuous_policy": "persist_only_within_one_video_source",
            "source_switch_reset": "fresh_ParkingManagement_instance",
            "generation": self._generation,
            "ultralytics_version": importlib.metadata.version("ultralytics"),
            "parking_management_source": str(source_path),
            "parking_management_source_sha256": sha256_file(source_path),
            "tracker_config": str(self.tracker_config),
            "tracker_config_sha256": sha256_file(self.tracker_config),
            "region_json": str(self.region_json),
            "region_json_sha256": sha256_file(self.region_json),
        }
