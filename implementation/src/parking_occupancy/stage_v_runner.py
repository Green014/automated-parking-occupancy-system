from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .evaluate import binary_metrics
from .integrated_runner import (
    create_classifier,
    create_detector,
    load_integrated_config,
    resolve_tracker_config,
)
from .integrated_cli import DEFAULT_FINAL_INTEGRATED_CONFIG
from .models import Detection, ParkingSlot
from .slots import SlotMap, slot_map_from_dict
from .stage_v import (
    ClassicPixelBackend,
    ClassicPixelConfig,
    DetectionBackend,
    DetectionCache,
    FrameOccupancyResult,
    FusionBackend,
    OccupancyBackend,
    draw_stage_v_frame,
    validate_frame_result,
)


EXPECTED_D1 = {
    "bytes": 6_255_409,
    "sha256": "0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64",
}
EXPECTED_E1B = {
    "bytes": 8_045_704,
    "sha256": "f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3",
}
EXPECTED_FROZEN_CONFIG = {
    "filename": "p3_stage_r_recommended_default_20260729.yaml",
    "sha256": "198f627689cd93f66ca0f087af6686d3afc697ff51e2aa77ee56124187b981b0",
    "config_id": "P3-STAGE-R-RECOMMENDED-DEFAULT-20260729-01",
}
FROZEN_CRITICAL_PARAMETERS: tuple[tuple[str, Any], ...] = (
    ("config_id", "P3-STAGE-R-RECOMMENDED-DEFAULT-20260729-01"),
    ("detector.confidence", 0.30),
    ("detector.image_size", 640),
    ("detector.nms_iou", 0.70),
    ("detector.agnostic_nms", True),
    ("detector.max_detections", 300),
    ("mapping.minimum_slot_coverage", 0.40),
    ("mapping.one_to_one", True),
    ("classifier.occupied_threshold", 0.76),
    ("classifier.patch_size", [224, 224]),
    ("classifier.perspective_warp", True),
    ("fusion.detector_positive_is_occupied", True),
    ("temporal.default_enabled", False),
    ("tracking.default_backend", "none"),
)
IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}
MODES = ("classic", "detection", "fusion")


class FrozenArtifactError(RuntimeError):
    pass


class FrozenConfigError(RuntimeError):
    pass


def _nested_value(config: Mapping[str, Any], dotted_path: str) -> Any:
    value: Any = config
    for part in dotted_path.split("."):
        if not isinstance(value, Mapping) or part not in value:
            return None
        value = value[part]
    return value


def validate_stage_v_config(
    path: Path,
    *,
    allow_custom_config: bool = False,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load P3 configuration and establish its frozen/custom identity."""

    resolved = path.resolve()
    if not resolved.is_file():
        raise FileNotFoundError(resolved)
    config = load_integrated_config(resolved)
    actual_hash = sha256_file(resolved)
    exact_sha256_match = (
        actual_hash == EXPECTED_FROZEN_CONFIG["sha256"]
    )
    differences = []
    for key, expected in FROZEN_CRITICAL_PARAMETERS:
        actual = _nested_value(config, key)
        if actual != expected:
            differences.append(
                {
                    "parameter": key,
                    "expected": expected,
                    "actual": actual,
                }
            )
    critical_parameters_match = not differences
    frozen = exact_sha256_match and critical_parameters_match
    if not frozen and not allow_custom_config:
        reasons = []
        if not exact_sha256_match:
            reasons.append(
                "configuration SHA-256 mismatch "
                f"(expected {EXPECTED_FROZEN_CONFIG['sha256']}, "
                f"actual {actual_hash})"
            )
        if differences:
            changed = ", ".join(item["parameter"] for item in differences)
            reasons.append(f"changed critical parameters: {changed}")
        raise FrozenConfigError(
            "Stage V Detection/Fusion requires the formal frozen P3 "
            f"configuration; {'; '.join(reasons)}. "
            "Use --allow-custom-config to run an explicitly labelled custom "
            "method."
        )
    identity = {
        "classification": "frozen" if frozen else "custom",
        "method_identity": "frozen_C1_C2" if frozen else "custom",
        "filename": resolved.name,
        "sha256": actual_hash,
        "expected_filename": EXPECTED_FROZEN_CONFIG["filename"],
        "expected_sha256": EXPECTED_FROZEN_CONFIG["sha256"],
        "exact_sha256_match": exact_sha256_match,
        "hash_mismatch": not exact_sha256_match,
        "critical_parameters_match": critical_parameters_match,
        "parameter_values_changed": not critical_parameters_match,
        "frozen_parameters_changed": not frozen,
        "allow_custom_config": bool(allow_custom_config),
        "parameter_differences": differences,
    }
    return config, identity


def stage_v_slot_map_from_dict(payload: Mapping[str, Any]) -> SlotMap:
    """Normalize rectangles and the member polygon schema to canonical slots."""

    width = payload.get("source_width", payload.get("frame_width"))
    height = payload.get("source_height", payload.get("frame_height"))
    if width is None or height is None:
        raise ValueError(
            "Slot map requires source_width/source_height "
            "(or frame_width/frame_height)"
        )
    normalized_slots = []
    for item in payload.get("slots", payload.get("spaces", [])):
        if "points" in item and "rect" in item:
            raise ValueError(
                f"Slot {item.get('id', '')} must define points or rect, not both"
            )
        if "points" in item:
            points = item["points"]
        elif "rect" in item:
            rectangle = item["rect"]
            if isinstance(rectangle, Mapping):
                x = float(rectangle["x"])
                y = float(rectangle["y"])
                rectangle_width = float(rectangle["width"])
                rectangle_height = float(rectangle["height"])
            else:
                if len(rectangle) != 4:
                    raise ValueError("rect must contain x, y, width, height")
                x, y, rectangle_width, rectangle_height = map(float, rectangle)
            if rectangle_width <= 0 or rectangle_height <= 0:
                raise ValueError("Rectangle width and height must be positive")
            points = (
                (x, y),
                (x + rectangle_width, y),
                (x + rectangle_width, y + rectangle_height),
                (x, y + rectangle_height),
            )
        else:
            raise ValueError(f"Slot {item.get('id', '')} requires points or rect")
        normalized_slots.append({"id": item["id"], "points": points})
    return slot_map_from_dict(
        {
            "schema_version": int(payload.get("schema_version", 1)),
            "source_width": int(width),
            "source_height": int(height),
            "source": str(payload.get("source", "")),
            "coordinate_system": str(payload.get("coordinate_system", "pixel")),
            "slots": normalized_slots,
        }
    )


def load_stage_v_slot_map(
    path: Path,
    *,
    frame_size: tuple[int, int] | None = None,
) -> SlotMap:
    payload = json.loads(path.read_text(encoding="utf-8"))
    slot_map = stage_v_slot_map_from_dict(payload)
    return slot_map if frame_size is None else slot_map.scaled_to(*frame_size)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_frozen_artifact(
    path: Path | None,
    *,
    label: str,
    expected: Mapping[str, Any],
) -> dict[str, Any]:
    if path is None:
        raise FrozenArtifactError(
            f"{label} is required but was not supplied; expected "
            f"bytes={expected['bytes']} sha256={expected['sha256']}"
        )
    path = path.resolve()
    if not path.is_file():
        raise FrozenArtifactError(
            f"{label} is missing: {path}; expected "
            f"bytes={expected['bytes']} sha256={expected['sha256']}"
        )
    actual_size = path.stat().st_size
    actual_hash = sha256_file(path)
    if actual_size != int(expected["bytes"]) or actual_hash != expected["sha256"]:
        raise FrozenArtifactError(
            f"{label} does not match the frozen artifact; "
            f"expected bytes={expected['bytes']} sha256={expected['sha256']}; "
            f"actual bytes={actual_size} sha256={actual_hash}"
        )
    return {
        "label": label,
        "filename": path.name,
        "bytes": actual_size,
        "sha256": actual_hash,
        "verified": True,
    }


class InputStream:
    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self.capture: cv2.VideoCapture | None = None
        self.image_paths: tuple[Path, ...] = ()
        self._image_index = 0
        self.first_frame: np.ndarray
        if self.path.is_dir():
            self.kind = "images"
            self.continuous = False
            self.image_paths = tuple(
                path
                for path in sorted(self.path.iterdir(), key=lambda item: item.name)
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
            if not self.image_paths:
                raise ValueError(f"Image directory is empty: {self.path}")
            first = cv2.imread(str(self.image_paths[0]), cv2.IMREAD_COLOR)
            if first is None:
                raise ValueError(f"Could not decode image: {self.image_paths[0]}")
            self.first_frame = first
            self.fps = 1.0
            self.source_id = self.path.name
        elif self.path.is_file():
            self.kind = "video"
            self.continuous = True
            self.capture = cv2.VideoCapture(str(self.path))
            if not self.capture.isOpened():
                raise ValueError(f"Could not open video: {self.path}")
            fps = float(self.capture.get(cv2.CAP_PROP_FPS))
            self.fps = fps if fps > 0 else 25.0
            ok, first = self.capture.read()
            if not ok or first is None:
                self.capture.release()
                raise ValueError(f"Video contains no decodable frames: {self.path}")
            self.first_frame = first
            self.source_id = self.path.stem
        else:
            raise FileNotFoundError(self.path)
        self.height, self.width = self.first_frame.shape[:2]
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Input reports an invalid frame size")

    def frames(self, max_frames: int | None = None):
        yielded = 0
        if max_frames is None or yielded < max_frames:
            yield self.first_frame
            yielded += 1
        if self.kind == "images":
            for image_path in self.image_paths[1:]:
                if max_frames is not None and yielded >= max_frames:
                    break
                image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f"Could not decode image: {image_path}")
                if image.shape[:2] != (self.height, self.width):
                    raise ValueError(
                        "All image-directory frames must have the same dimensions"
                    )
                yield image
                yielded += 1
        else:
            assert self.capture is not None
            while max_frames is None or yielded < max_frames:
                ok, frame = self.capture.read()
                if not ok:
                    break
                yield frame
                yielded += 1

    def close(self) -> None:
        if self.capture is not None:
            self.capture.release()


def _timing(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "frames": 0,
            "mean_ms": 0.0,
            "p50_ms": 0.0,
            "p95_ms": 0.0,
            "fps_from_mean": None,
        }
    ordered = sorted(float(value) for value in values)
    mean = statistics.fmean(ordered)
    return {
        "frames": len(ordered),
        "mean_ms": mean,
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
        "fps_from_mean": 1000.0 / mean if mean > 0 else None,
    }


@dataclass
class ModeRunRecord:
    mode: str
    output_dir: Path
    summary: dict[str, Any]
    runtime: dict[str, Any]
    metrics: dict[str, Any]


class ModeOutput:
    OCCUPANCY_FIELDS = (
        "video_id",
        "frame_index",
        "timestamp_s",
        "slot_id",
        "state",
        "occupied",
        "vacant",
        "evidence",
        "evidence_source",
        "track_id",
        "details_json",
        "warnings",
    )
    EVENT_FIELDS = (
        "video_id",
        "frame_index",
        "timestamp_s",
        "slot_id",
        "from_state",
        "to_state",
        "evidence",
        "evidence_source",
        "track_id",
    )

    def __init__(
        self,
        *,
        output_dir: Path,
        mode: str,
        source_id: str,
        input_kind: str,
        fps: float,
        frame_size: tuple[int, int],
        slots: Sequence[ParkingSlot],
        backend: OccupancyBackend,
        config_snapshot: Mapping[str, Any],
    ) -> None:
        self.output_dir = output_dir
        self.mode = mode
        self.source_id = source_id
        self.input_kind = input_kind
        self.fps = fps
        self.frame_size = frame_size
        self.slots = tuple(slots)
        self.backend = backend
        self.config_snapshot = dict(config_snapshot)
        self.previous: dict[str, bool] | None = None
        self.frame_count = 0
        self.event_count = 0
        self.timing: dict[str, list[float]] = {}
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.occupancy_handle = (self.output_dir / "occupancy.csv").open(
            "x", encoding="utf-8", newline=""
        )
        self.event_handle = (self.output_dir / "events.csv").open(
            "x", encoding="utf-8", newline=""
        )
        self.detection_handle = (self.output_dir / "detections.jsonl").open(
            "x", encoding="utf-8"
        )
        self.occupancy_writer = csv.DictWriter(
            self.occupancy_handle,
            fieldnames=list(self.OCCUPANCY_FIELDS),
            lineterminator="\n",
        )
        self.event_writer = csv.DictWriter(
            self.event_handle,
            fieldnames=list(self.EVENT_FIELDS),
            lineterminator="\n",
        )
        self.occupancy_writer.writeheader()
        self.event_writer.writeheader()
        self.video_writer: cv2.VideoWriter | None = None
        self.image_dir: Path | None = None
        if input_kind == "video":
            video_path = self.output_dir / "annotated.mp4"
            self.video_writer = cv2.VideoWriter(
                str(video_path),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                frame_size,
            )
            if not self.video_writer.isOpened():
                self._close_handles()
                raise RuntimeError(f"Could not create annotated video: {video_path}")
        else:
            self.image_dir = self.output_dir / "annotated_images"
            self.image_dir.mkdir()

    def write(self, frame: np.ndarray, result: FrameOccupancyResult) -> None:
        write_started = time.perf_counter()
        validate_frame_result(result, self.slots)
        by_slot = result.state_by_slot()
        for slot in self.slots:
            state = by_slot[slot.slot_id]
            self.occupancy_writer.writerow(
                {
                    "video_id": self.source_id,
                    "frame_index": result.frame_index,
                    "timestamp_s": f"{result.timestamp_s:.9f}",
                    "slot_id": slot.slot_id,
                    "state": int(state.occupied),
                    "occupied": int(state.occupied),
                    "vacant": int(not state.occupied),
                    "evidence": f"{state.evidence_score:.9f}",
                    "evidence_source": state.evidence_source,
                    "track_id": "" if state.track_id is None else state.track_id,
                    "details_json": json.dumps(
                        state.details or {}, sort_keys=True, separators=(",", ":")
                    ),
                    "warnings": " | ".join(result.warnings),
                }
            )
            old_state = (
                None
                if self.previous is None
                else self.previous[slot.slot_id]
            )
            if (
                self.input_kind == "video"
                and old_state is not None
                and state.occupied != old_state
            ):
                self.event_writer.writerow(
                    {
                        "video_id": self.source_id,
                        "frame_index": result.frame_index,
                        "timestamp_s": f"{result.timestamp_s:.9f}",
                        "slot_id": slot.slot_id,
                        "from_state": int(old_state),
                        "to_state": int(state.occupied),
                        "evidence": f"{state.evidence_score:.9f}",
                        "evidence_source": state.evidence_source,
                        "track_id": (
                            "" if state.track_id is None else state.track_id
                        ),
                    }
                )
                self.event_count += 1
        self.previous = {
            slot.slot_id: by_slot[slot.slot_id].occupied for slot in self.slots
        }
        self.detection_handle.write(
            json.dumps(
                {
                    "video_id": self.source_id,
                    "frame_index": result.frame_index,
                    "timestamp_s": result.timestamp_s,
                    "detections": [
                        {
                            "bbox": list(detection.bbox),
                            "confidence": detection.confidence,
                            "class_id": detection.class_id,
                            "class_name": detection.class_name,
                            "track_id": detection.track_id,
                        }
                        for detection in result.vehicle_detections
                    ],
                },
                sort_keys=True,
            )
            + "\n"
        )
        render_started = time.perf_counter()
        attributed_ms = float(
            result.timing_ms.get(
                "attributed_backend_total",
                result.timing_ms.get("backend_total", 0.0),
            )
        )
        cache_value = result.timing_ms.get("cache_hit")
        cache_status = (
            "not-used"
            if cache_value is None
            else ("hit" if bool(cache_value) else "miss")
        )
        tracker_enabled = (
            self.config_snapshot.get("tracking", {}).get("backend", "none")
            != "none"
        )
        temporal_enabled = bool(
            self.config_snapshot.get("temporal", {}).get("enabled", False)
        )
        annotated, rendered_slot_count = draw_stage_v_frame(
            frame=frame,
            detections=result.vehicle_detections,
            slots=self.slots,
            states=result.state_by_slot(),
            mode=self.mode,
            processing_fps=1000.0 / max(attributed_ms, 1e-9),
            cache_status=cache_status,
            temporal_enabled=temporal_enabled,
            tracker_enabled=tracker_enabled,
        )
        if rendered_slot_count != len(self.slots):
            raise RuntimeError("Rendered slot count does not match configuration")
        if self.video_writer is not None:
            self.video_writer.write(annotated)
        else:
            assert self.image_dir is not None
            output_image = self.image_dir / f"frame_{result.frame_index:06d}.png"
            if not cv2.imwrite(str(output_image), annotated):
                raise RuntimeError(f"Could not write annotated image: {output_image}")
        finished = time.perf_counter()
        for name, value in result.timing_ms.items():
            self.timing.setdefault(name, []).append(float(value))
        self.timing.setdefault("render_and_write", []).append(
            (finished - render_started) * 1000.0
        )
        self.timing.setdefault("end_to_end", []).append(
            (finished - write_started)
            * 1000.0
            + float(
                result.timing_ms.get(
                    "attributed_backend_total",
                    result.timing_ms.get("backend_total", 0.0),
                )
            )
        )
        self.frame_count += 1

    def _close_handles(self) -> None:
        if not self.occupancy_handle.closed:
            self.occupancy_handle.close()
        if not self.event_handle.closed:
            self.event_handle.close()
        if not self.detection_handle.closed:
            self.detection_handle.close()
        if self.video_writer is not None:
            self.video_writer.release()

    def finalize(
        self,
        *,
        input_record: Mapping[str, Any],
        slot_record: Mapping[str, Any],
        metrics: Mapping[str, Any],
        model_warmup_performed: bool,
    ) -> ModeRunRecord:
        self._close_handles()
        if self.frame_count == 0:
            raise ValueError("Input yielded no frames")
        timing = {name: _timing(values) for name, values in self.timing.items()}
        steady_values = self.timing.get("end_to_end", [])[1:]
        runtime = {
            "schema_version": 1,
            "stage": "V.1",
            "mode": self.mode,
            "frames": self.frame_count,
            "model_warmup_performed": model_warmup_performed,
            "model_load_in_steady_state_timing": False,
            "first_measured_frame_excluded_from_steady_state": bool(
                self.frame_count > 1
            ),
            "timing": timing,
            "steady_state_end_to_end": _timing(steady_values),
            "backend": dict(self.backend.metadata()),
        }
        output_names = [
            "occupancy.csv",
            "events.csv",
            "detections.jsonl",
            "summary.json",
            "metrics.json",
            "runtime_metadata.json",
            "configuration_snapshot.yaml",
            (
                "annotated.mp4"
                if self.input_kind == "video"
                else "annotated_images/"
            ),
        ]
        events_temporally_valid = self.input_kind == "video"
        temporal_enabled = bool(
            self.config_snapshot.get("temporal", {}).get("enabled", False)
        )
        summary = {
            "schema_version": 1,
            "stage": "V.1",
            "stage_name": "Multi-Backend Occupancy Integration Closure",
            "mode": self.mode,
            "method_id": dict(self.backend.metadata()).get("method_id"),
            "status": "executed",
            "source_id": self.source_id,
            "input_kind": self.input_kind,
            "frames": self.frame_count,
            "slots": len(self.slots),
            "slot_frame_predictions": self.frame_count * len(self.slots),
            "events": self.event_count,
            "events_temporally_valid": events_temporally_valid,
            "event_semantics": (
                "not_temporally_valid_noncontinuous_input"
                if not events_temporally_valid
                else (
                    "e4_stabilized_frame_state_changes"
                    if temporal_enabled
                    else "raw_frame_level_state_changes"
                )
            ),
            "initial_frame_emits_event": False,
            "all_slots_covered_every_frame": True,
            "input": dict(input_record),
            "slot_map": dict(slot_record),
            "truth_status": metrics.get("status"),
            "tracking_enabled": (
                self.config_snapshot.get("tracking", {}).get("backend", "none")
                != "none"
            ),
            "temporal_enabled": temporal_enabled,
            "output_files": output_names,
        }
        (self.output_dir / "metrics.json").write_text(
            json.dumps(dict(metrics), indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "runtime_metadata.json").write_text(
            json.dumps(runtime, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "configuration_snapshot.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "stage": "V.1",
                    "mode": self.mode,
                    **self.config_snapshot,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )
        return ModeRunRecord(
            mode=self.mode,
            output_dir=self.output_dir,
            summary=summary,
            runtime=runtime,
            metrics=dict(metrics),
        )


def _read_state_rows(path: Path) -> dict[tuple[str, int, str], int]:
    rows: dict[tuple[str, int, str], int] = {}
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["video_id"], int(row["frame_index"]), row["slot_id"])
            if key in rows:
                raise ValueError(f"Duplicate state row: {key}")
            state = int(row["state"])
            if state not in (0, 1):
                raise ValueError(f"State must be binary for {key}")
            rows[key] = state
    return rows


def static_metrics(
    truth_path: Path,
    prediction_path: Path,
    *,
    truth_role: str,
) -> dict[str, Any]:
    truth = _read_state_rows(truth_path)
    prediction = _read_state_rows(prediction_path)
    if set(truth) != set(prediction):
        missing = len(set(truth).difference(prediction))
        extra = len(set(prediction).difference(truth))
        raise ValueError(
            f"Truth/prediction key mismatch: missing={missing}, extra={extra}"
        )
    ordered = sorted(truth)
    result = binary_metrics(
        [truth[key] for key in ordered],
        [prediction[key] for key in ordered],
    )
    return {
        "status": "computed_static_slot_metrics",
        "truth_role": truth_role,
        "truth_sha256": sha256_file(truth_path),
        **result,
        "transition_latency_reported": False,
        "tracking_improvement_reported": False,
    }


def _empty_metrics(*, truth_role: str | None) -> dict[str, Any]:
    return {
        "status": "not_computed_no_truth",
        "truth_role": truth_role,
        "transition_latency_reported": False,
        "tracking_improvement_reported": False,
        "claim_boundary": (
            "No accuracy metric is reported without key-aligned per-slot truth."
        ),
    }


def _unique_detection_caches(
    backends: Mapping[str, OccupancyBackend],
) -> tuple[DetectionCache, ...]:
    caches: dict[int, DetectionCache] = {}
    for backend in backends.values():
        cache = getattr(backend, "cache", None)
        if isinstance(cache, DetectionCache):
            caches[id(cache)] = cache
    return tuple(caches.values())


def run_stage_v(
    *,
    input_path: Path,
    slots_path: Path,
    output_root: Path,
    mode: str,
    backends: Mapping[str, OccupancyBackend],
    config_snapshot: Mapping[str, Any],
    truth_path: Path | None = None,
    truth_role: str | None = None,
    warmup: bool = False,
    max_frames: int | None = None,
) -> dict[str, Any]:
    if mode not in (*MODES, "compare"):
        raise ValueError(f"Unsupported Stage V mode: {mode}")
    selected_modes = MODES if mode == "compare" else (mode,)
    missing_backends = set(selected_modes).difference(backends)
    if missing_backends:
        raise ValueError(f"Missing backends: {sorted(missing_backends)}")
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Stage V output: {output_root}")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    if truth_path is not None and not truth_role:
        raise ValueError("truth_role is required when truth is supplied")
    if truth_path is None and truth_role is not None:
        raise ValueError("truth_role requires truth")

    stream = InputStream(input_path)
    slots_path = slots_path.resolve()
    if not slots_path.is_file():
        stream.close()
        raise FileNotFoundError(slots_path)
    slots = load_stage_v_slot_map(
        slots_path,
        frame_size=(stream.width, stream.height),
    ).slots
    selected = {name: backends[name] for name in selected_modes}
    if not stream.continuous and any(
        dict(backend.metadata()).get("temporal", {}).get("enabled", False)
        for backend in selected.values()
    ):
        stream.close()
        raise ValueError("E4 temporal stabilization requires continuous video")
    input_record = {
        "filename": stream.path.name,
        "sha256": (
            sha256_file(stream.path)
            if stream.path.is_file()
            else hashlib.sha256(
                "\n".join(
                    f"{path.name}:{sha256_file(path)}"
                    for path in stream.image_paths
                ).encode("utf-8")
            ).hexdigest()
        ),
        "continuous": stream.continuous,
    }
    slot_record = {
        "filename": slots_path.name,
        "sha256": sha256_file(slots_path),
        "configured_slots": len(slots),
    }
    caches = _unique_detection_caches(selected)
    for cache in caches:
        cache.begin_source(stream.source_id, continuous=stream.continuous)
    for backend in selected.values():
        reset_state = getattr(backend, "reset_state", None)
        if reset_state is not None:
            reset_state(slots)

    if warmup and caches:
        warmup_backend = selected.get("fusion") or selected.get("detection")
        assert warmup_backend is not None
        warmup_backend.process_frame(stream.first_frame, slots, -1, 0.0)
        for cache in caches:
            cache.begin_source(stream.source_id, continuous=stream.continuous)
        for backend in selected.values():
            reset_state = getattr(backend, "reset_state", None)
            if reset_state is not None:
                reset_state(slots)

    sinks: dict[str, ModeOutput] = {}
    completed = False
    try:
        if mode == "compare":
            output_root.mkdir(parents=True)
            for selected_mode, backend in selected.items():
                sinks[selected_mode] = ModeOutput(
                    output_dir=output_root / selected_mode,
                    mode=selected_mode,
                    source_id=stream.source_id,
                    input_kind=stream.kind,
                    fps=stream.fps,
                    frame_size=(stream.width, stream.height),
                    slots=slots,
                    backend=backend,
                    config_snapshot=config_snapshot,
                )
        else:
            backend = selected[mode]
            sinks[mode] = ModeOutput(
                output_dir=output_root,
                mode=mode,
                source_id=stream.source_id,
                input_kind=stream.kind,
                fps=stream.fps,
                frame_size=(stream.width, stream.height),
                slots=slots,
                backend=backend,
                config_snapshot=config_snapshot,
            )
        frame_count = 0
        for frame_index, frame in enumerate(stream.frames(max_frames=max_frames)):
            timestamp_s = frame_index / stream.fps
            for selected_mode in selected_modes:
                result = selected[selected_mode].process_frame(
                    frame,
                    slots,
                    frame_index,
                    timestamp_s,
                )
                sinks[selected_mode].write(frame, result)
            frame_count += 1
        completed = True
    finally:
        stream.close()
        if not completed:
            for sink in sinks.values():
                sink._close_handles()

    records: dict[str, ModeRunRecord] = {}
    for selected_mode, sink in sinks.items():
        sink._close_handles()
        metrics = (
            static_metrics(
                truth_path.resolve(),
                sink.output_dir / "occupancy.csv",
                truth_role=str(truth_role),
            )
            if truth_path is not None
            else _empty_metrics(truth_role=truth_role)
        )
        records[selected_mode] = sink.finalize(
            input_record=input_record,
            slot_record=slot_record,
            metrics=metrics,
            model_warmup_performed=warmup and bool(caches),
        )

    result: dict[str, Any] = {
        "stage": "V.1",
        "mode": mode,
        "frames": frame_count,
        "slots": len(slots),
        "outputs": {
            name: str(record.output_dir) for name, record in records.items()
        },
    }
    if mode == "compare":
        _write_comparison(
            output_root=output_root,
            records=records,
            input_record=input_record,
            slot_record=slot_record,
            truth_role=truth_role,
        )
        result["comparison"] = str(output_root / "comparison.json")
    return result


def _metric_value(metrics: Mapping[str, Any], key: str) -> Any:
    return metrics.get(key, "")


def _write_comparison(
    *,
    output_root: Path,
    records: Mapping[str, ModeRunRecord],
    input_record: Mapping[str, Any],
    slot_record: Mapping[str, Any],
    truth_role: str | None,
) -> None:
    metric_fields = (
        "mode",
        "method_id",
        "status",
        "macro_f1",
        "occupied_recall",
        "vacant_recall",
        "false_free_rate",
        "false_occupied_rate",
        "mean_frame_latency_ms",
        "steady_state_fps",
    )
    runtime_fields = (
        "mode",
        "preprocessing_ms",
        "detector_ms",
        "mapping_ms",
        "classifier_ms",
        "fusion_ms",
        "render_and_write_ms",
        "end_to_end_ms",
        "steady_state_fps",
    )
    with (output_root / "method_metrics.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(metric_fields), lineterminator="\n"
        )
        writer.writeheader()
        for mode in MODES:
            record = records[mode]
            timing = record.runtime["timing"]
            writer.writerow(
                {
                    "mode": mode,
                    "method_id": record.summary["method_id"],
                    "status": record.metrics["status"],
                    "macro_f1": _metric_value(record.metrics, "macro_f1"),
                    "occupied_recall": _metric_value(
                        record.metrics, "occupied_recall"
                    ),
                    "vacant_recall": _metric_value(
                        record.metrics, "vacant_recall"
                    ),
                    "false_free_rate": _metric_value(
                        record.metrics, "false_free_rate"
                    ),
                    "false_occupied_rate": _metric_value(
                        record.metrics, "false_occupied_rate"
                    ),
                    "mean_frame_latency_ms": timing["end_to_end"]["mean_ms"],
                    "steady_state_fps": record.runtime[
                        "steady_state_end_to_end"
                    ]["fps_from_mean"],
                }
            )
    with (output_root / "runtime_comparison.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(runtime_fields), lineterminator="\n"
        )
        writer.writeheader()
        for mode in MODES:
            timing = records[mode].runtime["timing"]
            writer.writerow(
                {
                    "mode": mode,
                    "preprocessing_ms": timing.get("preprocessing", {}).get(
                        "mean_ms", ""
                    ),
                    "detector_ms": timing.get("detector", {}).get("mean_ms", ""),
                    "mapping_ms": timing.get("mapping", {}).get("mean_ms", ""),
                    "classifier_ms": timing.get("classifier", {}).get(
                        "mean_ms", ""
                    ),
                    "fusion_ms": timing.get("fusion", {}).get("mean_ms", ""),
                    "render_and_write_ms": timing["render_and_write"]["mean_ms"],
                    "end_to_end_ms": timing["end_to_end"]["mean_ms"],
                    "steady_state_fps": records[mode].runtime[
                        "steady_state_end_to_end"
                    ]["fps_from_mean"],
                }
            )
    method_ids = {
        mode: str(records[mode].summary["method_id"]) for mode in MODES
    }
    custom_identity = any(
        method_id.lower().startswith("custom")
        for method_id in method_ids.values()
    )
    comparison = {
        "schema_version": 1,
        "stage": "V.1",
        "comparison_id": (
            "CUSTOM_CONFIG_COMPARISON"
            if custom_identity
            else "C0_C1_C2"
        ),
        "methods": {
            method_ids["classic"]: "Classic OpenCV foreground-ratio baseline",
            method_ids["detection"]: "D1 + B1",
            method_ids["fusion"]: "D1 + B1 + E1b + F2",
        },
        "frozen_method_identity_retained": not custom_identity,
        "same_input": dict(input_record),
        "same_slot_map": dict(slot_record),
        "same_frame_count": len({record.summary["frames"] for record in records.values()})
        == 1,
        "truth_role": truth_role,
        "accuracy_status": (
            "computed_from_user_supplied_key_aligned_truth"
            if truth_role is not None
            else "not_computed_no_truth"
        ),
        "transition_latency_reported": False,
        "tracking_improvement_reported": False,
        "detection_cache_shared_between_C1_C2": True,
        "timing_definition": (
            "attributed backend processing plus render/write; cache hits retain "
            "the measured D1 detector cost"
        ),
        "negative_results_hidden": False,
        "side_by_side_comparison_video": None,
    }
    (output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def build_backends(
    *,
    mode: str,
    config_path: Path,
    d1_weights: Path | None,
    e1b_checkpoint: Path | None,
    device: str,
    tracker_backend: str,
    tracker_config_override: Path | None,
    classifier_batch_size: int,
    classic_threshold: float,
    temporal_enabled: bool,
    allow_custom_config: bool = False,
) -> tuple[dict[str, OccupancyBackend], dict[str, Any]]:
    selected_modes = MODES if mode == "compare" else (mode,)
    model_modes = set(selected_modes).intersection({"detection", "fusion"})
    if model_modes:
        config, config_identity = validate_stage_v_config(
            config_path,
            allow_custom_config=allow_custom_config,
        )
    else:
        config = load_integrated_config(config_path)
        config_identity = {
            "classification": "not_applicable_classic_only",
            "method_identity": "C0",
            "filename": config_path.name,
            "sha256": sha256_file(config_path),
            "expected_filename": EXPECTED_FROZEN_CONFIG["filename"],
            "expected_sha256": EXPECTED_FROZEN_CONFIG["sha256"],
            "exact_sha256_match": (
                sha256_file(config_path) == EXPECTED_FROZEN_CONFIG["sha256"]
            ),
            "hash_mismatch": (
                sha256_file(config_path) != EXPECTED_FROZEN_CONFIG["sha256"]
            ),
            "critical_parameters_match": None,
            "parameter_values_changed": None,
            "frozen_parameters_changed": False,
            "allow_custom_config": bool(allow_custom_config),
            "parameter_differences": [],
        }
    artifact_records: list[dict[str, Any]] = []
    backends: dict[str, OccupancyBackend] = {}
    custom_config = config_identity["classification"] == "custom"
    runtime_variant = tracker_backend != "none" or temporal_enabled
    detection_method_id = (
        "custom-detection"
        if custom_config
        else (
            f"C1+{tracker_backend}"
            if tracker_backend != "none"
            else "C1"
        )
    )
    fusion_suffixes = []
    if tracker_backend != "none":
        fusion_suffixes.append(tracker_backend)
    if temporal_enabled:
        fusion_suffixes.append("E4")
    fusion_method_id = (
        "custom-fusion"
        if custom_config
        else "C2" + (
            "+" + "+".join(fusion_suffixes) if fusion_suffixes else ""
        )
    )
    if "classic" in selected_modes:
        backends["classic"] = ClassicPixelBackend(
            ClassicPixelConfig(
                occupied_foreground_ratio=classic_threshold,
            )
        )
    if set(selected_modes).intersection({"detection", "fusion"}):
        d1_record = validate_frozen_artifact(
            d1_weights, label="D1 detector", expected=EXPECTED_D1
        )
        artifact_records.append(d1_record)
        assert d1_weights is not None
        tracker_path = resolve_tracker_config(
            config_path=config_path,
            config=config,
            backend=tracker_backend,
            override=tracker_config_override,
        )
        detector = create_detector(
            config=config,
            weights=d1_weights,
            device=device,
            tracker_config=tracker_path,
        )
        cache = DetectionCache(detector)
        if "detection" in selected_modes:
            backends["detection"] = DetectionBackend(
                cache,
                overlap_threshold=float(
                    config["mapping"]["minimum_slot_coverage"]
                ),
                method_id=detection_method_id,
            )
        if "fusion" in selected_modes:
            e1b_record = validate_frozen_artifact(
                e1b_checkpoint, label="E1b classifier", expected=EXPECTED_E1B
            )
            artifact_records.append(e1b_record)
            assert e1b_checkpoint is not None
            classifier = create_classifier(
                config=config,
                checkpoint=e1b_checkpoint,
                device=device,
                batch_size=classifier_batch_size,
            )
            backends["fusion"] = FusionBackend(
                cache,
                classifier,
                overlap_threshold=float(
                    config["mapping"]["minimum_slot_coverage"]
                ),
                classifier_threshold=float(
                    config["classifier"]["occupied_threshold"]
                ),
                temporal_config=(
                    config["temporal"] if temporal_enabled else None
                ),
                method_id=fusion_method_id,
            )
    snapshot = {
        "config_id": config["config_id"],
        "config_identity": config_identity,
        "method_identity": (
            "custom"
            if custom_config
            else ("explicit_runtime_variant" if runtime_variant else "frozen")
        ),
        "frozen_parameters_changed": bool(custom_config or runtime_variant),
        "runtime_variant_parameters_changed": runtime_variant,
        "classic": {
            "threshold": classic_threshold,
            "threshold_role": "uncalibrated_reference_default",
        },
        "detection": {
            "method_id": detection_method_id,
            "detector": "D1",
            "mapping": "B1",
            "minimum_slot_coverage": config["mapping"][
                "minimum_slot_coverage"
            ],
        },
        "fusion": {
            "method_id": fusion_method_id,
            "classifier": "E1b",
            "fusion": "F2",
            "classifier_threshold": config["classifier"][
                "occupied_threshold"
            ],
            "detector_negative_slots_only": True,
        },
        "temporal": {"enabled": temporal_enabled, "component": "E4"},
        "tracking": {
            "backend": tracker_backend,
            "default": "none",
        },
        "artifacts": artifact_records,
    }
    return backends, snapshot


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Stage V unified Classic/Detection/Fusion occupancy runner"
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument(
        "--mode",
        choices=(*MODES, "compare"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--d1-weights", type=Path)
    parser.add_argument("--e1b-checkpoint", type=Path)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FINAL_INTEGRATED_CONFIG,
    )
    parser.add_argument(
        "--allow-custom-config",
        action="store_true",
        help=(
            "Permit a non-frozen P3 configuration and label all affected "
            "Detection/Fusion results as custom."
        ),
    )
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--tracker",
        choices=("none", "bytetrack", "tracktrack"),
        default="none",
    )
    parser.add_argument("--tracker-config", type=Path)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument(
        "--classic-threshold",
        type=float,
        default=0.30,
        help="Uncalibrated foreground-ratio reference default",
    )
    parser.add_argument("--truth", type=Path)
    parser.add_argument(
        "--truth-role",
        choices=("development", "test", "consumed-demonstration"),
    )
    parser.add_argument(
        "--warmup",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--temporal",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optional E4 for continuous video in fusion mode; default off",
    )
    parser.add_argument("--max-frames", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.mode in {"classic", "compare"} and args.tracker != "none":
        raise ValueError("Tracking is not part of Classic or controlled comparison")
    if args.temporal and args.mode != "fusion":
        raise ValueError("E4 is optional only for single fusion-mode video runs")
    if args.tracker_config is not None and args.tracker == "none":
        raise ValueError("--tracker-config requires --tracker")
    if args.classifier_batch_size <= 0:
        raise ValueError("--classifier-batch-size must be positive")
    backends, snapshot = build_backends(
        mode=args.mode,
        config_path=args.config.resolve(),
        d1_weights=args.d1_weights,
        e1b_checkpoint=args.e1b_checkpoint,
        device=args.device,
        tracker_backend=args.tracker,
        tracker_config_override=args.tracker_config,
        classifier_batch_size=args.classifier_batch_size,
        classic_threshold=args.classic_threshold,
        temporal_enabled=args.temporal,
        allow_custom_config=args.allow_custom_config,
    )
    result = run_stage_v(
        input_path=args.input,
        slots_path=args.slots,
        output_root=args.output_dir,
        mode=args.mode,
        backends=backends,
        config_snapshot=snapshot,
        truth_path=args.truth,
        truth_role=args.truth_role,
        warmup=args.warmup,
        max_frames=args.max_frames,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
