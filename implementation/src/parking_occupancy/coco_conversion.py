from __future__ import annotations

import math
from dataclasses import dataclass
from collections.abc import Sequence


@dataclass(frozen=True)
class CocoBoxConversion:
    original_xywh: tuple[float, float, float, float]
    clipped_xyxy: tuple[float, float, float, float]
    yolo_xywh: tuple[float, float, float, float]
    clipped: bool


def validate_coco_bbox(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> CocoBoxConversion:
    """Validate, clip, and normalize one COCO ``xywh`` box."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    if len(bbox) != 4:
        raise ValueError("COCO bounding box must contain four values")
    x, y, width, height = (float(value) for value in bbox)
    if not all(math.isfinite(value) for value in (x, y, width, height)):
        raise ValueError("COCO bounding box values must be finite")
    if width <= 0 or height <= 0:
        raise ValueError("COCO bounding box width and height must be positive")

    original_x2 = x + width
    original_y2 = y + height
    x1 = min(max(x, 0.0), float(image_width))
    y1 = min(max(y, 0.0), float(image_height))
    x2 = min(max(original_x2, 0.0), float(image_width))
    y2 = min(max(original_y2, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box has no area after clipping")
    clipped_xyxy = (x1, y1, x2, y2)
    return CocoBoxConversion(
        original_xywh=(x, y, width, height),
        clipped_xyxy=clipped_xyxy,
        yolo_xywh=(
            ((x1 + x2) / 2.0) / image_width,
            ((y1 + y2) / 2.0) / image_height,
            (x2 - x1) / image_width,
            (y2 - y1) / image_height,
        ),
        clipped=clipped_xyxy != (x, y, original_x2, original_y2),
    )


def coco_bbox_to_yolo(
    bbox: Sequence[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Backward-compatible normalized YOLO coordinates."""

    return validate_coco_bbox(bbox, image_width, image_height).yolo_xywh
