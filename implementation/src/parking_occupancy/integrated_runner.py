from __future__ import annotations

import csv
import json
import statistics
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np
import yaml

from .detector_comparison import sha256_file
from .geometry import map_detections_to_slots
from .integrated_workflow import (
    IntegratedSlotDecision,
    UncertaintyGateConfig,
    uncertainty_gated_fusion,
)
from .models import Detection, ParkingSlot
from .slots import load_slot_map
from .stage_m_tracking import (
    InferenceSettings,
    UltralyticsSequenceAdapter,
    load_tracker_config,
)
from .temporal import FilteredSlotState
from .visualization import draw_frame


DEFAULT_INTEGRATED_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "p3_integrated_runtime_defaults_20260729.yaml"
)
OUTPUT_FILES = (
    "occupancy.csv",
    "events.csv",
    "detections.jsonl",
    "annotated.mp4",
    "metrics.json",
    "summary.json",
    "runtime_metadata.json",
)


class IntegratedDetector(Protocol):
    def begin_source(self, source_id: str, *, continuous: bool) -> None: ...

    def detect(self, frame: np.ndarray) -> Sequence[Detection]: ...

    def metadata(self) -> dict[str, Any]: ...


class IntegratedClassifier(Protocol):
    def predict(
        self,
        frame: np.ndarray,
        slots: Sequence[ParkingSlot],
    ) -> Mapping[str, float]: ...

    def metadata(self) -> dict[str, Any]: ...


@dataclass(frozen=True, slots=True)
class IntegratedFrameResult:
    frame_index: int
    detections: tuple[Detection, ...]
    decisions: dict[str, IntegratedSlotDecision]
    states: dict[str, FilteredSlotState]
    events: tuple[dict[str, Any], ...]
    timing_ms: dict[str, float]


def load_integrated_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("Unsupported integrated runtime config schema")
    if payload.get("method_id") != "P3":
        raise ValueError("Integrated runtime config must identify P3")
    detector = payload.get("detector", {})
    mapping = payload.get("mapping", {})
    classifier = payload.get("classifier", {})
    temporal = payload.get("temporal", {})
    if tuple(detector.get("class_ids", ())) != (0,):
        raise ValueError("P3 D1 runtime expects project vehicle class 0")
    if mapping.get("mode") != "overlap" or mapping.get("one_to_one") is not True:
        raise ValueError("P3 requires B1 one-to-one polygon overlap mapping")
    if classifier.get("detector_negative_slots_only") is not True:
        raise ValueError("E1b must review detector-negative slots only")
    for key, value in (
        ("detector.confidence", detector.get("confidence")),
        ("detector.nms_iou", detector.get("nms_iou")),
        (
            "mapping.minimum_slot_coverage",
            mapping.get("minimum_slot_coverage"),
        ),
        (
            "classifier.occupied_threshold",
            classifier.get("occupied_threshold"),
        ),
        ("temporal.rise_alpha", temporal.get("rise_alpha")),
        ("temporal.fall_alpha", temporal.get("fall_alpha")),
        (
            "temporal.occupied_threshold",
            temporal.get("occupied_threshold"),
        ),
        (
            "temporal.vacant_threshold",
            temporal.get("vacant_threshold"),
        ),
        ("temporal.raw_threshold", temporal.get("raw_threshold")),
    ):
        if value is None or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{key} must be in [0, 1]")
    if float(temporal["vacant_threshold"]) >= float(
        temporal["occupied_threshold"]
    ):
        raise ValueError("Temporal vacant threshold must be below occupied")
    return payload


def resolve_tracker_config(
    *,
    config_path: Path,
    config: Mapping[str, Any],
    backend: str,
    override: Path | None = None,
) -> Path | None:
    backend = backend.lower()
    if backend == "none":
        if override is not None:
            raise ValueError("--tracker-config requires a tracker backend")
        return None
    if backend not in {"bytetrack", "tracktrack"}:
        raise ValueError("Tracker must be none, bytetrack, or tracktrack")
    path = (
        override.resolve()
        if override is not None
        else (
            config_path.resolve().parent
            / str(config["tracking"][backend]["config_path"])
        ).resolve()
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    load_tracker_config(path, expected_type=backend)
    if override is None:
        expected = str(config["tracking"][backend]["config_sha256"])
        if sha256_file(path) != expected:
            raise ValueError(f"{backend} default config SHA-256 mismatch")
    return path


class E1bClassifierAdapter:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str,
        patch_size: tuple[int, int],
        perspective_warp: bool,
        batch_size: int,
    ) -> None:
        from literature_core.classifier import MobileNetSlotClassifier

        self.checkpoint = checkpoint.resolve()
        self.patch_size = patch_size
        self.perspective_warp = perspective_warp
        self.classifier = MobileNetSlotClassifier(
            self.checkpoint,
            device=device,
            batch_size=batch_size,
        )

    def predict(
        self,
        frame: np.ndarray,
        slots: Sequence[ParkingSlot],
    ) -> Mapping[str, float]:
        from literature_core.patches import extract_slot_patch

        if not slots:
            return {}
        patches = [
            extract_slot_patch(
                frame,
                slot.points,
                output_size=self.patch_size,
                perspective_warp=self.perspective_warp,
            )
            for slot in slots
        ]
        scores = self.classifier.predict_patches(patches)
        return {
            slot.slot_id: float(score)
            for slot, score in zip(slots, scores, strict=True)
        }

    def metadata(self) -> dict[str, Any]:
        return {
            **self.classifier.metadata(),
            "checkpoint_sha256": sha256_file(self.checkpoint),
            "perspective_warp": self.perspective_warp,
        }


def create_detector(
    *,
    config: Mapping[str, Any],
    weights: Path,
    device: str,
    tracker_config: Path | None,
) -> IntegratedDetector:
    detector = config["detector"]
    settings = InferenceSettings(
        weights=str(weights.resolve()),
        confidence=float(detector["confidence"]),
        nms_iou=float(detector["nms_iou"]),
        image_size=int(detector["image_size"]),
        class_ids=tuple(int(value) for value in detector["class_ids"]),
        max_detections=int(detector["max_detections"]),
        device=device,
        agnostic_nms=bool(detector["agnostic_nms"]),
    )
    return UltralyticsSequenceAdapter(
        settings,
        tracker_config=tracker_config,
    )


def create_classifier(
    *,
    config: Mapping[str, Any],
    checkpoint: Path,
    device: str,
    batch_size: int,
) -> IntegratedClassifier:
    classifier = config["classifier"]
    return E1bClassifierAdapter(
        checkpoint,
        device=device,
        patch_size=tuple(int(value) for value in classifier["patch_size"]),
        perspective_warp=bool(classifier["perspective_warp"]),
        batch_size=batch_size,
    )


class IntegratedFrameProcessor:
    """Stateful P3 frame processor with explicit per-source reset."""

    def __init__(
        self,
        *,
        slots: Sequence[ParkingSlot],
        detector: IntegratedDetector,
        classifier: IntegratedClassifier,
        config: Mapping[str, Any],
        temporal_enabled: bool,
    ) -> None:
        if not slots:
            raise ValueError("At least one parking slot is required")
        self.slots = tuple(slots)
        self.detector = detector
        self.classifier = classifier
        self.config = config
        self.temporal_enabled = temporal_enabled
        self._source_id: str | None = None
        self._frame_index = 0
        self._previous = {slot.slot_id: False for slot in self.slots}
        self._temporal_filter: Any | None = None

    def begin_source(self, source_id: str) -> None:
        if not source_id:
            raise ValueError("source_id must not be empty")
        self.detector.begin_source(source_id, continuous=True)
        self._source_id = source_id
        self._frame_index = 0
        self._previous = {slot.slot_id: False for slot in self.slots}
        self._temporal_filter = None
        if self.temporal_enabled:
            from literature_core.temporal import (
                TemporalConfig,
                TemporalFusionFilter,
            )

            temporal = self.config["temporal"]
            self._temporal_filter = TemporalFusionFilter(
                tuple(slot.slot_id for slot in self.slots),
                TemporalConfig(
                    rise_alpha=float(temporal["rise_alpha"]),
                    fall_alpha=float(temporal["fall_alpha"]),
                    occupied_threshold=float(
                        temporal["occupied_threshold"]
                    ),
                    vacant_threshold=float(temporal["vacant_threshold"]),
                    raw_threshold=float(temporal["raw_threshold"]),
                ),
            )

    def process(self, frame: np.ndarray, *, fps: float) -> IntegratedFrameResult:
        if self._source_id is None:
            raise RuntimeError("begin_source must be called before process")
        if fps <= 0:
            raise ValueError("fps must be positive")
        started = time.perf_counter()
        detections = tuple(self.detector.detect(frame))
        detector_end = time.perf_counter()
        evidence = map_detections_to_slots(
            list(detections),
            self.slots,
            mode="overlap",
            overlap_threshold=float(
                self.config["mapping"]["minimum_slot_coverage"]
            ),
        )
        mapping_end = time.perf_counter()
        uncertain_slots = [
            slot
            for slot in self.slots
            if not evidence[slot.slot_id].occupied
        ]
        classifier_scores = {
            str(key): float(value)
            for key, value in self.classifier.predict(
                frame, uncertain_slots
            ).items()
        }
        expected_uncertain = {slot.slot_id for slot in uncertain_slots}
        if set(classifier_scores) != expected_uncertain:
            raise ValueError(
                "Classifier must return every and only detector-negative slot"
            )
        decisions = uncertainty_gated_fusion(
            evidence,
            classifier_scores,
            UncertaintyGateConfig(
                classifier_occupied_threshold=float(
                    self.config["classifier"]["occupied_threshold"]
                )
            ),
        )
        classifier_end = time.perf_counter()

        states: dict[str, FilteredSlotState] = {}
        events: list[dict[str, Any]] = []
        for slot in self.slots:
            decision = decisions[slot.slot_id]
            if self._temporal_filter is None:
                occupied = decision.occupied
                filtered_score = decision.score
                changed = occupied != self._previous[slot.slot_id]
            else:
                from literature_core.models import FusedEvidence

                temporal_state = self._temporal_filter.update(
                    FusedEvidence(
                        slot_id=slot.slot_id,
                        p_cls=decision.classifier_probability,
                        p_det=decision.detector_score,
                        p_track=None,
                        probability=decision.score,
                        effective_weights=(0.0, 0.0, 0.0),
                    )
                )
                occupied = temporal_state.occupied
                filtered_score = temporal_state.filtered_probability
                changed = temporal_state.changed
            old_state = self._previous[slot.slot_id]
            if occupied != old_state:
                events.append(
                    {
                        "video_id": self._source_id,
                        "frame_index": self._frame_index,
                        "timestamp_s": self._frame_index / fps,
                        "slot_id": slot.slot_id,
                        "from_state": int(old_state),
                        "to_state": int(occupied),
                    }
                )
            self._previous[slot.slot_id] = occupied
            states[slot.slot_id] = FilteredSlotState(
                slot_id=slot.slot_id,
                occupied=occupied,
                filtered_score=filtered_score,
                raw_occupied=decision.occupied,
                raw_evidence_score=decision.score,
                changed=changed,
                track_id=decision.track_id,
            )
        finished = time.perf_counter()
        result = IntegratedFrameResult(
            frame_index=self._frame_index,
            detections=detections,
            decisions=decisions,
            states=states,
            events=tuple(events),
            timing_ms={
                "detector": (detector_end - started) * 1000.0,
                "mapping": (mapping_end - detector_end) * 1000.0,
                "classifier_and_fusion": (
                    classifier_end - mapping_end
                )
                * 1000.0,
                "temporal_and_events": (
                    finished - classifier_end
                )
                * 1000.0,
                "frame_processing": (finished - started) * 1000.0,
            },
        )
        self._frame_index += 1
        return result


def _timing_summary(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"frames": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(float(value) for value in values)
    mean = statistics.fmean(ordered)
    return {
        "frames": len(ordered),
        "mean_ms": mean,
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
        "fps_from_mean": 1000.0 / mean if mean > 0 else None,
    }


def run_integrated_video(
    *,
    input_path: Path,
    slots_path: Path,
    detector_weights: Path,
    classifier_checkpoint: Path,
    output_root: Path,
    config_path: Path = DEFAULT_INTEGRATED_CONFIG,
    device: str = "auto",
    source_id: str | None = None,
    truth_path: Path | None = None,
    temporal_enabled: bool | None = None,
    tracker_backend: str | None = None,
    tracker_config_override: Path | None = None,
    classifier_batch_size: int = 64,
    detector: IntegratedDetector | None = None,
    classifier: IntegratedClassifier | None = None,
) -> dict[str, Any]:
    input_path = input_path.resolve()
    slots_path = slots_path.resolve()
    detector_weights = detector_weights.resolve()
    classifier_checkpoint = classifier_checkpoint.resolve()
    output_root = output_root.resolve()
    config_path = config_path.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite integrated output: {output_root}"
        )
    for path in (
        input_path,
        slots_path,
        detector_weights,
        classifier_checkpoint,
        config_path,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if truth_path is not None and not truth_path.resolve().is_file():
        raise FileNotFoundError(truth_path)
    if classifier_batch_size <= 0:
        raise ValueError("classifier_batch_size must be positive")

    config = load_integrated_config(config_path)
    if temporal_enabled is None:
        temporal_enabled = bool(config["temporal"]["default_enabled"])
    if tracker_backend is None:
        tracker_backend = str(config["tracking"]["default_backend"])
    tracker_path = resolve_tracker_config(
        config_path=config_path,
        config=config,
        backend=tracker_backend,
        override=tracker_config_override,
    )

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input video: {input_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if fps <= 0 or width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input video has invalid FPS or dimensions")
    slots = load_slot_map(slots_path, frame_size=(width, height)).slots
    detector = detector or create_detector(
        config=config,
        weights=detector_weights,
        device=device,
        tracker_config=tracker_path,
    )
    classifier = classifier or create_classifier(
        config=config,
        checkpoint=classifier_checkpoint,
        device=device,
        batch_size=classifier_batch_size,
    )
    source_id = source_id or input_path.stem
    processor = IntegratedFrameProcessor(
        slots=slots,
        detector=detector,
        classifier=classifier,
        config=config,
        temporal_enabled=temporal_enabled,
    )
    processor.begin_source(source_id)

    output_root.mkdir(parents=True)
    video_path = output_root / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*str(config["output"]["codec"])),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create annotated video: {video_path}")

    occupancy_fields = (
        "video_id",
        "frame_index",
        "timestamp_s",
        "slot_id",
        "state",
        "evidence",
        "raw_state",
        "filtered_score",
        "detector_occupied",
        "detector_score",
        "classifier_probability",
        "classifier_consulted",
        "gate_branch",
        "track_id",
        "tracker_backend",
        "temporal_enabled",
    )
    event_fields = (
        "video_id",
        "frame_index",
        "timestamp_s",
        "slot_id",
        "from_state",
        "to_state",
    )
    timing: dict[str, list[float]] = {}
    frame_count = 0
    event_count = 0
    classifier_review_count = 0
    started = time.perf_counter()
    try:
        with (
            (output_root / "occupancy.csv").open(
                "x", encoding="utf-8", newline=""
            ) as occupancy_handle,
            (output_root / "events.csv").open(
                "x", encoding="utf-8", newline=""
            ) as event_handle,
            (output_root / "detections.jsonl").open(
                "x", encoding="utf-8"
            ) as detection_handle,
        ):
            occupancy_writer = csv.DictWriter(
                occupancy_handle,
                fieldnames=list(occupancy_fields),
                lineterminator="\n",
            )
            event_writer = csv.DictWriter(
                event_handle,
                fieldnames=list(event_fields),
                lineterminator="\n",
            )
            occupancy_writer.writeheader()
            event_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_started = time.perf_counter()
                result = processor.process(frame, fps=fps)
                for key, value in result.timing_ms.items():
                    timing.setdefault(key, []).append(value)
                classifier_review_count += sum(
                    decision.classifier_consulted
                    for decision in result.decisions.values()
                )
                for slot in slots:
                    decision = result.decisions[slot.slot_id]
                    state = result.states[slot.slot_id]
                    occupancy_writer.writerow(
                        {
                            "video_id": source_id,
                            "frame_index": result.frame_index,
                            "timestamp_s": f"{result.frame_index / fps:.9f}",
                            "slot_id": slot.slot_id,
                            "state": int(state.occupied),
                            "evidence": f"{state.filtered_score:.9f}",
                            "raw_state": int(decision.occupied),
                            "filtered_score": f"{state.filtered_score:.9f}",
                            "detector_occupied": int(
                                decision.detector_occupied
                            ),
                            "detector_score": f"{decision.detector_score:.9f}",
                            "classifier_probability": (
                                ""
                                if decision.classifier_probability is None
                                else (
                                    f"{decision.classifier_probability:.9f}"
                                )
                            ),
                            "classifier_consulted": int(
                                decision.classifier_consulted
                            ),
                            "gate_branch": decision.branch,
                            "track_id": (
                                ""
                                if state.track_id is None
                                else state.track_id
                            ),
                            "tracker_backend": tracker_backend,
                            "temporal_enabled": int(temporal_enabled),
                        }
                    )
                for event in result.events:
                    event_writer.writerow(event)
                event_count += len(result.events)
                detection_handle.write(
                    json.dumps(
                        {
                            "video_id": source_id,
                            "frame_index": result.frame_index,
                            "timestamp_s": result.frame_index / fps,
                            "detections": [
                                asdict(detection)
                                for detection in result.detections
                            ],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                annotated = draw_frame(
                    frame=frame,
                    detections=list(result.detections),
                    slots=slots,
                    states=result.states,
                    experiment="P3 integrated",
                    processing_fps=1.0
                    / max(time.perf_counter() - frame_started, 1e-9),
                )
                writer.write(annotated)
                timing.setdefault("end_to_end", []).append(
                    (time.perf_counter() - frame_started) * 1000.0
                )
                frame_count += 1
    finally:
        capture.release()
        writer.release()
    if frame_count == 0:
        raise ValueError("Input video contains no decodable frames")

    occupancy_path = output_root / "occupancy.csv"
    if truth_path is None:
        metrics = {
            "schema_version": 1,
            "method_id": "P3",
            "status": "not_computed_no_truth",
            "truth_required_for_inference": False,
            "frames": frame_count,
            "slots": len(slots),
        }
    else:
        from .evaluate import evaluate

        metrics = evaluate(
            truth_path.resolve(),
            occupancy_path,
            output_root / "evaluation",
            fps,
            stable_frames=int(
                config["temporal"]["stable_frames_for_evaluation"]
            ),
        )
        metrics["status"] = "computed_from_optional_truth"
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "method_id": "P3",
        "method_name": config["method_name"],
        "legacy_proposed_alias_used": False,
        "status": "executed_integrated_video",
        "source_id": source_id,
        "frames": frame_count,
        "slots": len(slots),
        "fps": fps,
        "events": event_count,
        "classifier_reviews": classifier_review_count,
        "truth_supplied": truth_path is not None,
        "truth_required_for_inference": False,
        "temporal_enabled": temporal_enabled,
        "tracker_backend": tracker_backend,
        "elapsed_s": elapsed,
        "parameter_selection_from_run": False,
        "parameter_provenance": config["parameter_provenance"],
        "inputs": {
            "video": str(input_path),
            "video_sha256": sha256_file(input_path),
            "slots": str(slots_path),
            "slots_sha256": sha256_file(slots_path),
            "D1_weights": str(detector_weights),
            "D1_weights_sha256": sha256_file(detector_weights),
            "E1b_checkpoint": str(classifier_checkpoint),
            "E1b_checkpoint_sha256": sha256_file(
                classifier_checkpoint
            ),
            "config": str(config_path),
            "config_sha256": sha256_file(config_path),
            "truth": None if truth_path is None else str(truth_path.resolve()),
            "tracker_config": (
                None if tracker_path is None else str(tracker_path)
            ),
            "tracker_config_sha256": (
                None if tracker_path is None else sha256_file(tracker_path)
            ),
        },
        "output_files": list(OUTPUT_FILES),
    }
    runtime = {
        "schema_version": 1,
        "source_state_reset": {
            "source_id": source_id,
            "detector_begin_source_called": True,
            "temporal_filter_reconstructed": bool(temporal_enabled),
            "event_state_reinitialized": True,
        },
        "timing": {
            key: _timing_summary(values) for key, values in timing.items()
        },
        "detector": detector.metadata(),
        "classifier": classifier.metadata(),
    }
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return summary
