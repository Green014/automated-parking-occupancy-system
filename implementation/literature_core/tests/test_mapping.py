import pytest

from literature_core.mapping import map_detections_to_slots, slot_coverage
from literature_core.models import Detection, ParkingSlot


def detection(box, confidence=1.0) -> Detection:
    return Detection(box, confidence, 0, "car")


def test_slot_coverage_is_fraction_of_slot_not_box() -> None:
    slot = ParkingSlot("A", ((0, 0), (10, 0), (10, 10), (0, 10)))
    assert slot_coverage(detection((0, 0, 5, 10)), slot) == pytest.approx(0.5)


def test_mapping_retains_raw_detection_details() -> None:
    slot = ParkingSlot("A", ((0, 0), (10, 0), (10, 10), (0, 10)))
    result = map_detections_to_slots(
        [Detection((0, 0, 5, 10), 0.8, 0, "car", track_id=17)],
        [slot],
        minimum_slot_coverage=0.4,
    )
    assert result[0].probability == pytest.approx(0.4)
    assert result[0].detection_confidence == pytest.approx(0.8)
    assert result[0].detection_bbox == (0, 0, 5, 10)
    assert result[0].track_id == 17


def test_one_detection_cannot_occupy_two_slots() -> None:
    slots = (
        ParkingSlot("A", ((0, 0), (10, 0), (10, 10), (0, 10))),
        ParkingSlot("B", ((10, 0), (20, 0), (20, 10), (10, 10))),
    )
    result = map_detections_to_slots(
        [detection((0, 0, 20, 10))],
        slots,
        minimum_slot_coverage=0.5,
        one_to_one=True,
    )
    assert sum(item.probability > 0 for item in result) == 1


def test_vacant_is_zero_when_no_object_evidence() -> None:
    slot = ParkingSlot("A", ((0, 0), (10, 0), (10, 10), (0, 10)))
    result = map_detections_to_slots([], [slot])
    assert result[0].probability == 0.0
    assert result[0].detection_index is None
