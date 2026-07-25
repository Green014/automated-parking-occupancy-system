import pytest

from parking_occupancy.coco_conversion import coco_bbox_to_yolo


def test_coco_bbox_to_yolo_normalizes_and_clips() -> None:
    assert coco_bbox_to_yolo([-10, 20, 40, 40], 100, 100) == pytest.approx(
        (0.15, 0.4, 0.3, 0.4)
    )


def test_coco_bbox_to_yolo_rejects_empty_clipped_box() -> None:
    with pytest.raises(ValueError):
        coco_bbox_to_yolo([110, 20, 10, 10], 100, 100)
