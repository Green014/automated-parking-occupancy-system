import cv2
import numpy as np
import pytest

from literature_core.patches import extract_slot_patch, order_quad


def test_order_quad_accepts_unordered_points() -> None:
    ordered = order_quad(((9, 9), (1, 1), (1, 9), (9, 1)))
    assert ordered.tolist() == [[1, 1], [9, 1], [9, 9], [1, 9]]


def test_perspective_patch_has_requested_shape_and_content() -> None:
    image = np.zeros((20, 20, 3), dtype=np.uint8)
    cv2.rectangle(image, (5, 4), (14, 15), (10, 100, 240), -1)
    patch = extract_slot_patch(
        image,
        ((14, 15), (5, 4), (14, 4), (5, 15)),
        output_size=(32, 24),
    )
    assert patch.shape == (24, 32, 3)
    assert patch[12, 16, 2] > 200


def test_non_quad_uses_masked_crop() -> None:
    image = np.full((20, 20, 3), 200, dtype=np.uint8)
    patch = extract_slot_patch(
        image,
        ((2, 2), (17, 2), (17, 10), (10, 17), (2, 10)),
        output_size=(16, 16),
    )
    assert patch.shape == (16, 16, 3)
    assert patch.max() == 200
    assert patch.min() < 200


def test_duplicate_points_are_rejected() -> None:
    image = np.zeros((10, 10, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="duplicate"):
        extract_slot_patch(image, ((1, 1), (8, 1), (8, 1), (1, 8)))

