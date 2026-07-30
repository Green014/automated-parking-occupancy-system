from parking_occupancy.geometry import (
    map_detections_to_slots,
    point_in_slot,
    slot_overlap_score,
)
from parking_occupancy.models import Detection, ParkingSlot


def detection(
    bbox: tuple[float, float, float, float],
    confidence: float = 1.0,
) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=confidence,
        class_id=2,
        class_name="car",
    )


def test_point_inside_and_on_boundary() -> None:
    slot = ParkingSlot("s1", ((0, 0), (10, 0), (10, 10), (0, 10)))
    assert point_in_slot((5, 5), slot)
    assert point_in_slot((10, 5), slot)
    assert not point_in_slot((11, 5), slot)


def test_center_mapping_does_not_assign_lane_vehicle() -> None:
    slot = ParkingSlot("s1", ((0, 0), (10, 0), (10, 10), (0, 10)))
    result = map_detections_to_slots(
        [detection((12, 2, 20, 8))],
        [slot],
        mode="center",
    )
    assert not result["s1"].occupied


def test_overlap_score_uses_slot_coverage() -> None:
    slot = ParkingSlot("s1", ((0, 0), (10, 0), (10, 10), (0, 10)))
    assert slot_overlap_score(detection((0, 0, 5, 10)), slot) == 0.5


def test_one_detection_cannot_occupy_two_slots() -> None:
    slots = [
        ParkingSlot("left", ((0, 0), (10, 0), (10, 10), (0, 10))),
        ParkingSlot("right", ((10, 0), (20, 0), (20, 10), (10, 10))),
    ]
    result = map_detections_to_slots(
        [detection((5, 0, 15, 10), confidence=0.9)],
        slots,
        mode="overlap",
        overlap_threshold=0.3,
    )
    assert sum(item.occupied for item in result.values()) == 1


def test_overlap_threshold_rejects_small_intersection() -> None:
    slot = ParkingSlot("s1", ((0, 0), (10, 0), (10, 10), (0, 10)))
    result = map_detections_to_slots(
        [detection((9, 0, 20, 10))],
        [slot],
        mode="overlap",
        overlap_threshold=0.2,
    )
    assert not result["s1"].occupied

