from __future__ import annotations

from collections.abc import Iterable

import cv2
import numpy as np


def _points_array(points: Iterable[tuple[float, float]]) -> np.ndarray:
    array = np.asarray(tuple(points), dtype=np.float32)
    if array.ndim != 2 or array.shape[1] != 2 or len(array) < 3:
        raise ValueError("points must contain at least three (x, y) pairs")
    if not np.isfinite(array).all():
        raise ValueError("points must be finite")
    if len(np.unique(array, axis=0)) != len(array):
        raise ValueError("slot polygon contains duplicate points")
    return array


def order_quad(points: Iterable[tuple[float, float]]) -> np.ndarray:
    """Return quadrilateral corners as top-left, top-right, bottom-right, bottom-left."""

    array = _points_array(points)
    if len(array) != 4:
        raise ValueError("Perspective ordering requires exactly four points")
    hull = cv2.convexHull(array).reshape((-1, 2))
    if len(hull) != 4:
        raise ValueError("Perspective polygon must be a convex quadrilateral")
    center = hull.mean(axis=0)
    angles = np.arctan2(hull[:, 1] - center[1], hull[:, 0] - center[0])
    ordered = hull[np.argsort(angles)]
    start = int(np.argmin(ordered.sum(axis=1)))
    ordered = np.roll(ordered, -start, axis=0)
    signed_area = 0.5 * sum(
        ordered[index, 0] * ordered[(index + 1) % 4, 1]
        - ordered[(index + 1) % 4, 0] * ordered[index, 1]
        for index in range(4)
    )
    if signed_area < 0:
        ordered = ordered[[0, 3, 2, 1]]
    return ordered


def extract_slot_patch(
    image: np.ndarray,
    points: Iterable[tuple[float, float]],
    output_size: tuple[int, int] = (224, 224),
    perspective_warp: bool = True,
) -> np.ndarray:
    """Extract a fixed-size BGR patch from a slot polygon with OpenCV.

    Four-point polygons use a perspective transform. Other convex polygons are
    masked, cropped to their bounding rectangle, and resized.
    """

    if image.ndim != 3 or image.shape[2] != 3 or image.size == 0:
        raise ValueError("image must be a non-empty HxWx3 array")
    width, height = output_size
    if width <= 0 or height <= 0:
        raise ValueError("output_size must be positive")

    array = _points_array(points)
    frame_height, frame_width = image.shape[:2]
    array[:, 0] = np.clip(array[:, 0], 0, frame_width - 1)
    array[:, 1] = np.clip(array[:, 1], 0, frame_height - 1)

    if perspective_warp and len(array) == 4:
        source = order_quad(tuple(map(tuple, array)))
        if abs(float(cv2.contourArea(source.reshape((-1, 1, 2))))) <= 1e-6:
            raise ValueError("slot polygon has zero area")
        target = np.asarray(
            (
                (0, 0),
                (width - 1, 0),
                (width - 1, height - 1),
                (0, height - 1),
            ),
            dtype=np.float32,
        )
        transform = cv2.getPerspectiveTransform(source, target)
        return cv2.warpPerspective(
            image,
            transform,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    contour = array.reshape((-1, 1, 2))
    if abs(float(cv2.contourArea(contour))) <= 1e-6:
        raise ValueError("slot polygon has zero area")
    integer_contour = np.rint(array).astype(np.int32).reshape((-1, 1, 2))
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask, [integer_contour], 255)
    masked = cv2.bitwise_and(image, image, mask=mask)
    x, y, box_width, box_height = cv2.boundingRect(integer_contour)
    if box_width <= 0 or box_height <= 0:
        raise ValueError("slot polygon produced an empty crop")
    crop = masked[y : y + box_height, x : x + box_width]
    return cv2.resize(crop, (width, height), interpolation=cv2.INTER_AREA)
