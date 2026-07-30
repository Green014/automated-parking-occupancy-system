import pytest

from parking_occupancy.coco_conversion import (
    coco_bbox_to_yolo,
    validate_coco_bbox,
)


def test_coco_bbox_to_yolo_normalizes_and_clips() -> None:
    assert coco_bbox_to_yolo([-10, 20, 40, 40], 100, 100) == pytest.approx(
        (0.15, 0.4, 0.3, 0.4)
    )


def test_coco_bbox_to_yolo_rejects_empty_clipped_box() -> None:
    with pytest.raises(ValueError):
        coco_bbox_to_yolo([110, 20, 10, 10], 100, 100)


def test_validate_coco_bbox_reports_boundary_clipping() -> None:
    result = validate_coco_bbox([-10, 20, 40, 40], 100, 100)

    assert result.clipped is True
    assert result.original_xywh == (-10.0, 20.0, 40.0, 40.0)
    assert result.clipped_xyxy == (0.0, 20.0, 30.0, 60.0)


@pytest.mark.parametrize(
    "bbox",
    [
        [0, 0, 0, 10],
        [0, 0, -1, 10],
        [0, 0, 10, float("nan")],
        [0, 0, 10],
    ],
)
def test_validate_coco_bbox_rejects_invalid_values(bbox) -> None:
    with pytest.raises(ValueError):
        validate_coco_bbox(bbox, 100, 100)
