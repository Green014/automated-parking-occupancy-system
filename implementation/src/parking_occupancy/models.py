from __future__ import annotations

from dataclasses import dataclass
from typing import TypeAlias

Point: TypeAlias = tuple[float, float]
BBox: TypeAlias = tuple[float, float, float, float]


@dataclass(frozen=True, slots=True)
class Detection:
    """One vehicle detection in image pixel coordinates."""

    bbox: BBox
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = self.bbox
        if x2 < x1 or y2 < y1:
            raise ValueError(f"Invalid detection box: {self.bbox}")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("Detection confidence must be in [0, 1]")

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


@dataclass(frozen=True, slots=True)
class ParkingSlot:
    """A predefined convex parking-space polygon."""

    slot_id: str
    points: tuple[Point, ...]

    def __post_init__(self) -> None:
        if not self.slot_id:
            raise ValueError("slot_id must not be empty")
        if len(self.points) < 3:
            raise ValueError(f"Slot {self.slot_id} needs at least three points")


@dataclass(frozen=True, slots=True)
class SlotEvidence:
    """Mapping evidence that turns detections into a raw slot state."""

    slot_id: str
    occupied: bool
    geometric_score: float
    evidence_score: float
    detection_index: int | None = None
    detection_confidence: float = 0.0
    track_id: int | None = None

