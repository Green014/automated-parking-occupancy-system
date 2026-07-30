from __future__ import annotations

import cv2
import numpy as np

from .models import Detection, ParkingSlot
from .temporal import FilteredSlotState

OCCUPIED_COLOR = (40, 40, 230)
VACANT_COLOR = (40, 190, 40)
DETECTION_COLOR = (0, 210, 255)


def draw_frame(
    frame: np.ndarray,
    detections: list[Detection],
    slots: tuple[ParkingSlot, ...],
    states: dict[str, FilteredSlotState],
    experiment: str,
    processing_fps: float,
) -> np.ndarray:
    canvas = frame.copy()
    overlay = canvas.copy()

    for slot in slots:
        state = states[slot.slot_id]
        color = OCCUPIED_COLOR if state.occupied else VACANT_COLOR
        polygon = np.asarray(slot.points, dtype=np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [polygon], color)
        cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
        center = tuple(
            np.mean(np.asarray(slot.points, dtype=np.float32), axis=0)
            .round()
            .astype(int)
        )
        label = f"{slot.slot_id}:{'OCC' if state.occupied else 'FREE'}"
        cv2.putText(
            canvas,
            label,
            center,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color,
            1,
            cv2.LINE_AA,
        )

    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)

    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), DETECTION_COLOR, 2)
        track = "" if detection.track_id is None else f" id={detection.track_id}"
        label = f"{detection.class_name} {detection.confidence:.2f}{track}"
        cv2.putText(
            canvas,
            label,
            (x1, max(16, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            DETECTION_COLOR,
            1,
            cv2.LINE_AA,
        )

    occupied_count = sum(state.occupied for state in states.values())
    summary = (
        f"{experiment.upper()}  occupied={occupied_count}/{len(states)}  "
        f"processing={processing_fps:.1f} FPS"
    )
    cv2.rectangle(canvas, (0, 0), (min(canvas.shape[1], 640), 32), (0, 0, 0), -1)
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
    return canvas

