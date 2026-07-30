from __future__ import annotations

from collections.abc import Iterable
from typing import Literal

import cv2
import numpy as np

from .models import Detection, ParkingSlot, SlotEvidence

MappingMode = Literal["center", "overlap"]


def _contour(points: Iterable[tuple[float, float]]) -> np.ndarray:
    return np.asarray(tuple(points), dtype=np.float32).reshape((-1, 1, 2))


def point_in_slot(point: tuple[float, float], slot: ParkingSlot) -> bool:
    """Return True for points inside or on the polygon boundary."""

    return cv2.pointPolygonTest(_contour(slot.points), point, False) >= 0


def slot_area(slot: ParkingSlot) -> float:
    return abs(float(cv2.contourArea(_contour(slot.points))))


def bbox_slot_intersection_area(detection: Detection, slot: ParkingSlot) -> float:
    """Compute exact convex intersection area with OpenCV."""

    x1, y1, x2, y2 = detection.bbox
    if x2 <= x1 or y2 <= y1:
        return 0.0
    box = np.asarray(
        ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
        dtype=np.float32,
    ).reshape((-1, 1, 2))
    area, _ = cv2.intersectConvexConvex(_contour(slot.points), box)
    return max(0.0, float(area))


def slot_overlap_score(detection: Detection, slot: ParkingSlot) -> float:
    """Fraction of the parking-slot polygon covered by a detection box."""

    area = slot_area(slot)
    if area <= 1e-6:
        return 0.0
    return min(1.0, bbox_slot_intersection_area(detection, slot) / area)


def map_detections_to_slots(
    detections: list[Detection],
    slots: Iterable[ParkingSlot],
    mode: MappingMode,
    overlap_threshold: float = 0.30,
) -> dict[str, SlotEvidence]:
    """Map detections to slots with at most one detection assigned per slot.

    A detection may not mark two neighbouring slots occupied. Candidate
    assignments are therefore greedily selected by confidence-weighted
    geometric score.
    """

    if mode not in {"center", "overlap"}:
        raise ValueError(f"Unknown mapping mode: {mode}")
    if not 0.0 <= overlap_threshold <= 1.0:
        raise ValueError("overlap_threshold must be in [0, 1]")

    slot_list = list(slots)
    slot_geometry = []
    for slot in slot_list:
        contour = _contour(slot.points)
        x, y, width, height = cv2.boundingRect(contour)
        slot_geometry.append(
            (
                contour,
                abs(float(cv2.contourArea(contour))),
                (float(x), float(y), float(x + width), float(y + height)),
            )
        )
    detection_geometry = []
    for detection in detections:
        x1, y1, x2, y2 = detection.bbox
        contour = np.asarray(
            ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
            dtype=np.float32,
        ).reshape((-1, 1, 2))
        detection_geometry.append((contour, detection.center, detection.bbox))

    candidates: list[tuple[float, float, int, int]] = []
    for slot_index, (slot, geometry) in enumerate(
        zip(slot_list, slot_geometry, strict=True)
    ):
        slot_contour, area, (slot_x1, slot_y1, slot_x2, slot_y2) = geometry
        for detection_index, detection in enumerate(detections):
            detection_contour, center, (x1, y1, x2, y2) = detection_geometry[
                detection_index
            ]
            if x2 < slot_x1 or x1 > slot_x2 or y2 < slot_y1 or y1 > slot_y2:
                continue
            if mode == "center":
                if cv2.pointPolygonTest(slot_contour, center, False) < 0:
                    continue
                geometric_score = 1.0
            else:
                if area <= 1e-6:
                    continue
                intersection_area, _ = cv2.intersectConvexConvex(
                    slot_contour,
                    detection_contour,
                )
                geometric_score = min(
                    1.0,
                    max(0.0, float(intersection_area)) / area,
                )
                if geometric_score < overlap_threshold:
                    continue
            evidence_score = detection.confidence * geometric_score
            candidates.append(
                (evidence_score, geometric_score, slot_index, detection_index)
            )

    candidates.sort(reverse=True)
    assigned_slots: set[int] = set()
    assigned_detections: set[int] = set()
    result: dict[str, SlotEvidence] = {
        slot.slot_id: SlotEvidence(
            slot_id=slot.slot_id,
            occupied=False,
            geometric_score=0.0,
            evidence_score=0.0,
        )
        for slot in slot_list
    }

    for evidence_score, geometric_score, slot_index, detection_index in candidates:
        if slot_index in assigned_slots or detection_index in assigned_detections:
            continue
        assigned_slots.add(slot_index)
        assigned_detections.add(detection_index)
        slot = slot_list[slot_index]
        detection = detections[detection_index]
        result[slot.slot_id] = SlotEvidence(
            slot_id=slot.slot_id,
            occupied=True,
            geometric_score=geometric_score,
            evidence_score=evidence_score,
            detection_index=detection_index,
            detection_confidence=detection.confidence,
            track_id=detection.track_id,
        )

    return result
