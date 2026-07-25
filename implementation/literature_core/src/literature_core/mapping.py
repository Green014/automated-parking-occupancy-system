from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np

from .models import Detection, DetectionEvidence, ParkingSlot


def _contour(points: tuple[tuple[float, float], ...]) -> np.ndarray:
    contour = np.asarray(points, dtype=np.float32).reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour):
        raise ValueError("Detection mapping requires convex slot polygons")
    if abs(float(cv2.contourArea(contour))) <= 1e-6:
        raise ValueError("Slot polygon has zero area")
    return contour


def _box_contour(detection: Detection) -> np.ndarray:
    x1, y1, x2, y2 = detection.bbox
    return np.asarray(
        ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
        dtype=np.float32,
    ).reshape((-1, 1, 2))


def slot_coverage(detection: Detection, slot: ParkingSlot) -> float:
    """Return fraction of the slot polygon covered by a detection box."""

    slot_contour = _contour(slot.points)
    slot_area = abs(float(cv2.contourArea(slot_contour)))
    intersection, _ = cv2.intersectConvexConvex(
        slot_contour,
        _box_contour(detection),
    )
    return min(1.0, max(0.0, float(intersection)) / slot_area)


def map_detections_to_slots(
    detections: list[Detection],
    slots: Iterable[ParkingSlot],
    minimum_slot_coverage: float = 0.10,
    one_to_one: bool = True,
) -> tuple[DetectionEvidence, ...]:
    """Convert object boxes into P_det with auditable one-to-one assignments."""

    if not 0.0 <= minimum_slot_coverage <= 1.0:
        raise ValueError("minimum_slot_coverage must be in [0, 1]")
    slot_list = tuple(slots)
    if not slot_list:
        raise ValueError("At least one slot is required")

    candidates: list[tuple[float, float, int, int]] = []
    for slot_index, slot in enumerate(slot_list):
        for detection_index, detection in enumerate(detections):
            coverage = slot_coverage(detection, slot)
            if coverage < minimum_slot_coverage:
                continue
            probability = detection.confidence * coverage
            candidates.append(
                (probability, coverage, slot_index, detection_index)
            )
    candidates.sort(reverse=True)

    assigned_slots: set[int] = set()
    assigned_detections: set[int] = set()
    result = [
        DetectionEvidence(slot_id=slot.slot_id, probability=0.0)
        for slot in slot_list
    ]
    for probability, coverage, slot_index, detection_index in candidates:
        if slot_index in assigned_slots:
            continue
        if one_to_one and detection_index in assigned_detections:
            continue
        detection = detections[detection_index]
        result[slot_index] = DetectionEvidence(
            slot_id=slot_list[slot_index].slot_id,
            probability=probability,
            coverage=coverage,
            detection_index=detection_index,
            detection_confidence=detection.confidence,
            detection_label=detection.label,
            detection_bbox=detection.bbox,
        )
        assigned_slots.add(slot_index)
        assigned_detections.add(detection_index)
    return tuple(result)

