from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Point: TypeAlias = tuple[float, float]
BBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """One raw object detection in image pixel coordinates."""

    bbox: BBox
    confidence: float
    class_id: int
    label: str
    track_id: int | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if x2 <= x1 or y2 <= y1:
            raise ValueError(f"Detection box must have positive area: {self.bbox}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("confidence must be in [0, 1]")
        if not self.label:
            raise ValueError("label must not be empty")


@dataclass(frozen=True, slots=True)
class ParkingSlot:
    """A parking-space polygon in image pixel coordinates."""

    slot_id: str
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.slot_id:
            raise ValueError("slot_id must not be empty")
        if len(self.points) < 3:
            raise ValueError("A slot polygon needs at least three points")


@dataclass(frozen=True, slots=True)
class DetectionEvidence:
    """Detector-derived occupancy evidence for one slot."""

    slot_id: str
    probability: float
    coverage: float = 0.0
    detection_index: int | None = None
    detection_confidence: float = 0.0
    detection_label: str | None = None
    detection_bbox: BBox | None = None
    track_id: int | None = None


@dataclass(frozen=True, slots=True)
class FusedEvidence:
    """Auditable branch probabilities and their fused occupancy score."""

    slot_id: str
    p_cls: float | None
    p_det: float | None
    p_track: float | None
    probability: float
    effective_weights: tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class SlotDecision:
    """Final state for one slot at one frame."""

    slot_id: str
    occupied: bool
    probability: float
    filtered_probability: float
    p_cls: float | None
    p_det: float | None
    p_track: float | None
    raw_occupied: bool
    changed: bool


@dataclass(frozen=True, slots=True)
class FrameResult:
    """All intermediate and final outputs for one frame."""

    frame_index: int
    timestamp_s: float
    detections: tuple[Detection, ...]
    detector_evidence: tuple[DetectionEvidence, ...]
    decisions: tuple[SlotDecision, ...]
