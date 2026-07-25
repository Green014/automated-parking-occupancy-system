from __future__ import annotations


def coco_bbox_to_yolo(
    bbox: list[float],
    image_width: int,
    image_height: int,
) -> tuple[float, float, float, float]:
    """Convert and clip COCO xywh pixels to normalized YOLO xywh."""

    if image_width <= 0 or image_height <= 0:
        raise ValueError("Image dimensions must be positive")
    x, y, width, height = (float(value) for value in bbox)
    x1 = min(max(x, 0.0), float(image_width))
    y1 = min(max(y, 0.0), float(image_height))
    x2 = min(max(x + width, 0.0), float(image_width))
    y2 = min(max(y + height, 0.0), float(image_height))
    if x2 <= x1 or y2 <= y1:
        raise ValueError("Bounding box has no area after clipping")
    return (
        ((x1 + x2) / 2.0) / image_width,
        ((y1 + y2) / 2.0) / image_height,
        (x2 - x1) / image_width,
        (y2 - y1) / image_height,
    )
