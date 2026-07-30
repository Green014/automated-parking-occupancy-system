import numpy as np
import pytest

from parking_occupancy.detector import UltralyticsDetector, track_ids_from_output


def test_track_ids_from_output_keeps_unmatched_detections() -> None:
    tracks = np.asarray(
        [
            [10, 20, 30, 40, 101, 0.8, 2, 1],
            [50, 60, 70, 80, 205, 0.7, 5, 0],
        ],
        dtype=np.float32,
    )

    assert track_ids_from_output(tracks, 3) == [205, 101, None]


def test_track_ids_from_empty_output() -> None:
    assert track_ids_from_output(np.empty((0, 8)), 2) == [None, None]


def test_detector_validates_extended_inference_controls() -> None:
    with pytest.raises(ValueError, match="nms_iou"):
        UltralyticsDetector(nms_iou=1.1)
    with pytest.raises(ValueError, match="max_detections"):
        UltralyticsDetector(max_detections=0)

    detector = UltralyticsDetector(
        nms_iou=0.7,
        agnostic_nms=True,
        max_detections=300,
        augmentation=False,
        rect=False,
        half=False,
    )
    assert detector.nms_iou == 0.7
    assert detector.agnostic_nms is True
    assert detector.max_detections == 300
