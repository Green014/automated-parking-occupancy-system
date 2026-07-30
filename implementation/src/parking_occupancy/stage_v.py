from __future__ import annotations

import time
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import cv2
import numpy as np

from .geometry import map_detections_to_slots
from .integrated_runner import IntegratedClassifier, IntegratedDetector
from .integrated_workflow import (
    UncertaintyGateConfig,
    uncertainty_gated_fusion,
)
from .models import Detection, ParkingSlot


@dataclass(frozen=True, slots=True)
class SlotOccupancyState:
    """One and only one occupancy decision for one configured slot."""

    slot_id: str
    occupied: bool
    evidence_score: float
    evidence_source: str
    track_id: int | None = None
    details: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if not self.slot_id:
            raise ValueError("slot_id must not be empty")
        if not 0.0 <= self.evidence_score <= 1.0:
            raise ValueError("evidence_score must be in [0, 1]")


@dataclass(frozen=True, slots=True)
class FrameOccupancyResult:
    """Backend-neutral result for one frame."""

    frame_index: int
    timestamp_s: float
    slot_states: tuple[SlotOccupancyState, ...]
    vehicle_detections: tuple[Detection, ...]
    timing_ms: Mapping[str, float]
    warnings: tuple[str, ...] = ()

    def state_by_slot(self) -> dict[str, SlotOccupancyState]:
        return {state.slot_id: state for state in self.slot_states}


class OccupancyBackend(Protocol):
    mode: str

    def process_frame(
        self,
        frame: np.ndarray,
        slots: Sequence[ParkingSlot],
        frame_index: int,
        timestamp_s: float,
    ) -> FrameOccupancyResult: ...

    def metadata(self) -> Mapping[str, Any]: ...


OCCUPIED_COLOR = (40, 40, 230)
VACANT_COLOR = (40, 190, 40)
DETECTION_COLOR = (0, 210, 255)
TRACK_COLOR = (255, 220, 0)


def validate_slot_render_coverage(
    slots: Sequence[ParkingSlot],
    states: Mapping[str, SlotOccupancyState],
) -> int:
    configured = [slot.slot_id for slot in slots]
    if len(configured) != len(set(configured)):
        raise ValueError("Configured slots contain duplicate IDs")
    missing = sorted(set(configured).difference(states))
    extra = sorted(set(states).difference(configured))
    if missing or extra or len(configured) != len(states):
        raise ValueError(
            "Visualization state coverage mismatch; "
            f"missing={missing}, extra={extra}"
        )
    return len(configured)


def draw_stage_v_frame(
    *,
    frame: np.ndarray,
    detections: Sequence[Detection],
    slots: Sequence[ParkingSlot],
    states: Mapping[str, SlotOccupancyState],
    mode: str,
    processing_fps: float,
    cache_status: str = "not-used",
    temporal_enabled: bool = False,
    tracker_enabled: bool = False,
    stage_label: str = "STAGE V",
) -> tuple[np.ndarray, int]:
    rendered_slot_count = validate_slot_render_coverage(slots, states)
    canvas = frame.copy()
    overlay = canvas.copy()
    for slot in slots:
        state = states[slot.slot_id]
        color = OCCUPIED_COLOR if state.occupied else VACANT_COLOR
        polygon = np.asarray(slot.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        anchor_x, anchor_y = (
            np.asarray(slot.points, dtype=np.float32)
            .min(axis=0)
            .round()
            .astype(int)
        )
        numeric_suffix = re.search(r"(\d+)$", slot.slot_id)
        short_id = (
            str(int(numeric_suffix.group(1)))[-3:]
            if numeric_suffix
            else slot.slot_id[:4]
        )
        cv2.putText(
            canvas,
            short_id,
            (int(anchor_x), max(68, int(anchor_y) - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

    show_detection_labels = len(detections) <= 12
    show_track_labels = len(detections) <= 20
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), DETECTION_COLOR, 2)
        if show_detection_labels:
            cv2.putText(
                canvas,
                f"{detection.class_name} {detection.confidence:.2f}",
                (x1, max(64, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                DETECTION_COLOR,
                1,
                cv2.LINE_AA,
            )
        if tracker_enabled and detection.track_id is not None and show_track_labels:
            cv2.putText(
                canvas,
                f"track={detection.track_id}",
                (x1, min(canvas.shape[0] - 6, y2 + 16)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                TRACK_COLOR,
                1,
                cv2.LINE_AA,
            )

    occupied_count = sum(state.occupied for state in states.values())
    summary = (
        f"{stage_label} {mode.upper()}  occupied={occupied_count}/{len(states)}  "
        f"rendered={rendered_slot_count}  detections={len(detections)}  "
        f"attributed={processing_fps:.1f} FPS  cache={cache_status}"
    )
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 62), (0, 0, 0), -1)
    cv2.putText(
        canvas,
        summary,
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    legend: tuple[tuple[str, tuple[int, int, int]], ...] = (
        ("vacant", VACANT_COLOR),
        ("occupied", OCCUPIED_COLOR),
        ("vehicle", DETECTION_COLOR),
    )
    if tracker_enabled and any(
        detection.track_id is not None for detection in detections
    ):
        legend += (("track ID", TRACK_COLOR),)
    x = 8
    for name, color in legend:
        cv2.rectangle(canvas, (x, 38), (x + 10, 48), color, -1)
        cv2.putText(
            canvas,
            name,
            (x + 14, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.36,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        x += 30 + len(name) * 8
    flags = (
        f"temporal={'on' if temporal_enabled else 'off'}  "
        f"tracker={'on' if tracker_enabled else 'off'}"
    )
    cv2.putText(
        canvas,
        flags,
        (max(8, canvas.shape[1] - 245), 48),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.36,
        (210, 210, 210),
        1,
        cv2.LINE_AA,
    )
    return canvas, rendered_slot_count


def validate_frame_result(
    result: FrameOccupancyResult,
    slots: Sequence[ParkingSlot],
) -> None:
    """Enforce exactly one result row for every configured slot."""

    expected = [slot.slot_id for slot in slots]
    actual = [state.slot_id for state in result.slot_states]
    if len(actual) != len(set(actual)):
        raise ValueError("Backend returned duplicate slot states")
    if set(actual) != set(expected) or len(actual) != len(expected):
        missing = sorted(set(expected).difference(actual))
        extra = sorted(set(actual).difference(expected))
        raise ValueError(
            "Backend result does not cover the configured slot map exactly; "
            f"missing={missing}, extra={extra}"
        )


@dataclass(frozen=True, slots=True)
class ClassicPixelConfig:
    """Uncalibrated engineering defaults for the clean-room Classic backend."""

    gaussian_kernel: int = 3
    adaptive_block_size: int = 25
    adaptive_c: int = 16
    median_kernel: int = 5
    dilation_kernel: int = 3
    dilation_iterations: int = 1
    occupied_foreground_ratio: float = 0.30

    def __post_init__(self) -> None:
        for name in (
            "gaussian_kernel",
            "adaptive_block_size",
            "median_kernel",
            "dilation_kernel",
        ):
            value = int(getattr(self, name))
            if value <= 0 or value % 2 == 0:
                raise ValueError(f"{name} must be a positive odd integer")
        if self.adaptive_block_size <= 1:
            raise ValueError("adaptive_block_size must be greater than one")
        if self.dilation_iterations < 0:
            raise ValueError("dilation_iterations must not be negative")
        if not 0.0 <= self.occupied_foreground_ratio <= 1.0:
            raise ValueError("occupied_foreground_ratio must be in [0, 1]")


class ClassicPixelBackend:
    """Polygon-aware OpenCV foreground-pixel baseline.

    This is an independent implementation. It is not copied from the
    unlicensed reference repository and its defaults are not claimed optimal.
    """

    mode = "classic"

    def __init__(self, config: ClassicPixelConfig | None = None) -> None:
        self.config = config or ClassicPixelConfig()
        self._dilation_kernel = np.ones(
            (self.config.dilation_kernel, self.config.dilation_kernel),
            dtype=np.uint8,
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("Classic backend expects a BGR frame")
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(
            gray,
            (self.config.gaussian_kernel, self.config.gaussian_kernel),
            1,
        )
        thresholded = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            self.config.adaptive_block_size,
            self.config.adaptive_c,
        )
        filtered = cv2.medianBlur(
            thresholded,
            self.config.median_kernel,
        )
        return cv2.dilate(
            filtered,
            self._dilation_kernel,
            iterations=self.config.dilation_iterations,
        )

    @staticmethod
    def _slot_ratio(binary: np.ndarray, slot: ParkingSlot) -> tuple[int, int, float]:
        mask = np.zeros(binary.shape, dtype=np.uint8)
        polygon = np.asarray(slot.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(mask, [polygon], 255)
        area_pixels = cv2.countNonZero(mask)
        if area_pixels <= 0:
            raise ValueError(f"Slot {slot.slot_id} has no in-frame pixels")
        foreground = cv2.countNonZero(cv2.bitwise_and(binary, binary, mask=mask))
        return foreground, area_pixels, foreground / area_pixels

    def process_frame(
        self,
        frame: np.ndarray,
        slots: Sequence[ParkingSlot],
        frame_index: int,
        timestamp_s: float,
    ) -> FrameOccupancyResult:
        preprocess_start = time.perf_counter()
        binary = self._preprocess(frame)
        preprocess_end = time.perf_counter()
        states: list[SlotOccupancyState] = []
        for slot in slots:
            foreground, area, ratio = self._slot_ratio(binary, slot)
            states.append(
                SlotOccupancyState(
                    slot_id=slot.slot_id,
                    occupied=ratio >= self.config.occupied_foreground_ratio,
                    evidence_score=min(1.0, max(0.0, ratio)),
                    evidence_source="classic_pixel_foreground_ratio",
                    details={
                        "foreground_pixels": foreground,
                        "roi_pixels": area,
                        "threshold": self.config.occupied_foreground_ratio,
                    },
                )
            )
        finished = time.perf_counter()
        result = FrameOccupancyResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            slot_states=tuple(states),
            vehicle_detections=(),
            timing_ms={
                "preprocessing": (preprocess_end - preprocess_start) * 1000.0,
                "slot_classification": (finished - preprocess_end) * 1000.0,
                "backend_total": (finished - preprocess_start) * 1000.0,
            },
            warnings=(
                "Classic threshold is an uncalibrated reference default; "
                "it is not an optimality claim.",
            ),
        )
        validate_frame_result(result, slots)
        return result

    def metadata(self) -> Mapping[str, Any]:
        return {
            "mode": self.mode,
            "method_id": "C0",
            "name": "OpenCV pixel-count baseline inspired by the reference project",
            "implementation": "clean_room_independent",
            "license_boundary": "no reference source code or weights reused",
            "preprocessing": {
                "grayscale": True,
                "gaussian_blur": True,
                "adaptive_threshold": "gaussian_binary_inverse",
                "median_blur": True,
                "dilation": True,
            },
            "slot_representation": "polygon",
            "decision_evidence": "foreground_pixel_ratio",
            "config": {
                field: getattr(self.config, field)
                for field in self.config.__dataclass_fields__
            },
            "threshold_claim": "uncalibrated_reference_default",
        }


@dataclass(frozen=True, slots=True)
class CachedDetections:
    detections: tuple[Detection, ...]
    detector_ms: float


class DetectionCache:
    """Frame-indexed D1 cache shared by Detection and Fusion."""

    def __init__(self, detector: IntegratedDetector) -> None:
        self.detector = detector
        self._source_id: str | None = None
        self._cache: dict[int, CachedDetections] = {}
        self.hits = 0
        self.misses = 0

    def begin_source(self, source_id: str, *, continuous: bool) -> None:
        self.detector.begin_source(source_id, continuous=continuous)
        self._source_id = source_id
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def clear(self) -> None:
        self._cache.clear()
        self.hits = 0
        self.misses = 0

    def get(self, frame: np.ndarray, frame_index: int) -> tuple[CachedDetections, bool]:
        if self._source_id is None:
            raise RuntimeError("DetectionCache.begin_source must be called first")
        if frame_index in self._cache:
            self.hits += 1
            return self._cache[frame_index], True
        started = time.perf_counter()
        detections = tuple(self.detector.detect(frame))
        item = CachedDetections(
            detections=detections,
            detector_ms=(time.perf_counter() - started) * 1000.0,
        )
        self._cache[frame_index] = item
        self.misses += 1
        return item, False

    def metadata(self) -> Mapping[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "source_id": self._source_id,
            "detector": self.detector.metadata(),
        }


class DetectionBackend:
    mode = "detection"

    def __init__(
        self,
        cache: DetectionCache,
        *,
        overlap_threshold: float,
        method_id: str = "C1",
    ) -> None:
        self.cache = cache
        self.overlap_threshold = float(overlap_threshold)
        self.method_id = str(method_id)

    def process_frame(
        self,
        frame: np.ndarray,
        slots: Sequence[ParkingSlot],
        frame_index: int,
        timestamp_s: float,
    ) -> FrameOccupancyResult:
        started = time.perf_counter()
        cached, cache_hit = self.cache.get(frame, frame_index)
        mapping_start = time.perf_counter()
        evidence = map_detections_to_slots(
            list(cached.detections),
            slots,
            mode="overlap",
            overlap_threshold=self.overlap_threshold,
        )
        finished = time.perf_counter()
        states = tuple(
            SlotOccupancyState(
                slot_id=slot.slot_id,
                occupied=evidence[slot.slot_id].occupied,
                evidence_score=evidence[slot.slot_id].evidence_score,
                evidence_source=(
                    "D1_B1_detector_positive"
                    if evidence[slot.slot_id].occupied
                    else "D1_B1_detector_negative"
                ),
                track_id=evidence[slot.slot_id].track_id,
                details={
                    "geometric_score": evidence[slot.slot_id].geometric_score,
                    "detection_confidence": evidence[
                        slot.slot_id
                    ].detection_confidence,
                    "detection_index": evidence[slot.slot_id].detection_index,
                },
            )
            for slot in slots
        )
        result = FrameOccupancyResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            slot_states=states,
            vehicle_detections=cached.detections,
            timing_ms={
                "detector": cached.detector_ms,
                "detector_execution": 0.0 if cache_hit else cached.detector_ms,
                "mapping": (finished - mapping_start) * 1000.0,
                "backend_total": (finished - started) * 1000.0,
                "attributed_backend_total": (
                    (finished - started) * 1000.0
                    + (cached.detector_ms if cache_hit else 0.0)
                ),
                "cache_hit": float(cache_hit),
            },
            warnings=("D1 detections reused from frame cache",) if cache_hit else (),
        )
        validate_frame_result(result, slots)
        return result

    def metadata(self) -> Mapping[str, Any]:
        return {
            "mode": self.mode,
            "method_id": self.method_id,
            "components": ["D1", "B1"],
            "mapping": {
                "mode": "overlap",
                "one_to_one": True,
                "minimum_slot_coverage": self.overlap_threshold,
            },
            "cache": self.cache.metadata(),
        }


class FusionBackend:
    mode = "fusion"

    def __init__(
        self,
        cache: DetectionCache,
        classifier: IntegratedClassifier,
        *,
        overlap_threshold: float,
        classifier_threshold: float,
        temporal_config: Mapping[str, Any] | None = None,
        method_id: str = "C2",
    ) -> None:
        self.cache = cache
        self.classifier = classifier
        self.overlap_threshold = float(overlap_threshold)
        self.classifier_threshold = float(classifier_threshold)
        self.temporal_config = (
            None if temporal_config is None else dict(temporal_config)
        )
        self.method_id = str(method_id)
        self._temporal_filter: Any | None = None

    def reset_state(self, slots: Sequence[ParkingSlot]) -> None:
        self._temporal_filter = None
        if self.temporal_config is None:
            return
        from literature_core.temporal import TemporalConfig, TemporalFusionFilter

        temporal = self.temporal_config
        self._temporal_filter = TemporalFusionFilter(
            tuple(slot.slot_id for slot in slots),
            TemporalConfig(
                rise_alpha=float(temporal["rise_alpha"]),
                fall_alpha=float(temporal["fall_alpha"]),
                occupied_threshold=float(temporal["occupied_threshold"]),
                vacant_threshold=float(temporal["vacant_threshold"]),
                raw_threshold=float(temporal["raw_threshold"]),
            ),
        )

    def process_frame(
        self,
        frame: np.ndarray,
        slots: Sequence[ParkingSlot],
        frame_index: int,
        timestamp_s: float,
    ) -> FrameOccupancyResult:
        started = time.perf_counter()
        cached, cache_hit = self.cache.get(frame, frame_index)
        mapping_start = time.perf_counter()
        evidence = map_detections_to_slots(
            list(cached.detections),
            slots,
            mode="overlap",
            overlap_threshold=self.overlap_threshold,
        )
        mapping_end = time.perf_counter()
        review_slots = [
            slot for slot in slots if not evidence[slot.slot_id].occupied
        ]
        classifier_start = time.perf_counter()
        scores = {
            str(slot_id): float(score)
            for slot_id, score in self.classifier.predict(
                frame,
                review_slots,
            ).items()
        }
        expected = {slot.slot_id for slot in review_slots}
        if set(scores) != expected:
            raise ValueError(
                "E1b must return every and only detector-negative slot"
            )
        classifier_end = time.perf_counter()
        decisions = uncertainty_gated_fusion(
            evidence,
            scores,
            UncertaintyGateConfig(
                classifier_occupied_threshold=self.classifier_threshold,
            ),
        )
        fusion_end = time.perf_counter()
        if self.temporal_config is not None and self._temporal_filter is None:
            self.reset_state(slots)
        states: list[SlotOccupancyState] = []
        for slot in slots:
            decision = decisions[slot.slot_id]
            occupied = decision.occupied
            score = decision.score
            source = f"D1_B1_E1b_F2.{decision.branch}"
            details = {
                "detector_occupied": decision.detector_occupied,
                "detector_score": decision.detector_score,
                "classifier_probability": decision.classifier_probability,
                "classifier_consulted": decision.classifier_consulted,
                "gate_branch": decision.branch,
            }
            if self._temporal_filter is not None:
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
                score = temporal_state.filtered_probability
                source += ".E4"
                details["pre_temporal_occupied"] = decision.occupied
                details["pre_temporal_score"] = decision.score
            states.append(
                SlotOccupancyState(
                    slot_id=slot.slot_id,
                    occupied=occupied,
                    evidence_score=score,
                    evidence_source=source,
                    track_id=decision.track_id,
                    details=details,
                )
            )
        finished = time.perf_counter()
        result = FrameOccupancyResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            slot_states=tuple(states),
            vehicle_detections=cached.detections,
            timing_ms={
                "detector": cached.detector_ms,
                "detector_execution": 0.0 if cache_hit else cached.detector_ms,
                "mapping": (mapping_end - mapping_start) * 1000.0,
                "classifier": (classifier_end - classifier_start) * 1000.0,
                "fusion": (fusion_end - classifier_end) * 1000.0,
                "temporal": (finished - fusion_end) * 1000.0,
                "backend_total": (finished - started) * 1000.0,
                "attributed_backend_total": (
                    (finished - started) * 1000.0
                    + (cached.detector_ms if cache_hit else 0.0)
                ),
                "cache_hit": float(cache_hit),
            },
            warnings=("D1 detections reused from frame cache",) if cache_hit else (),
        )
        validate_frame_result(result, slots)
        return result

    def metadata(self) -> Mapping[str, Any]:
        return {
            "mode": self.mode,
            "method_id": self.method_id,
            "components": ["D1", "B1", "E1b", "F2"],
            "mapping": {
                "mode": "overlap",
                "one_to_one": True,
                "minimum_slot_coverage": self.overlap_threshold,
            },
            "fusion": {
                "mode": "asymmetric_uncertainty_gate",
                "detector_positive_is_final": True,
                "classifier_reviews_detector_negative_only": True,
                "classifier_occupied_threshold": self.classifier_threshold,
            },
            "temporal": {
                "component": "E4",
                "enabled": self.temporal_config is not None,
                "default": False,
            },
            "classifier": self.classifier.metadata(),
            "cache": self.cache.metadata(),
        }
