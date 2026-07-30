from parking_occupancy.pklot_experiment import _error_cause, _slots_and_truth


def test_slots_and_truth_scales_points_and_keeps_unknown() -> None:
    record = {
        "sample": {
            "parking_spaces": {
                "polylines": [
                    {
                        "space_id": 1,
                        "points": [[[0.1, 0.2], [0.3, 0.2], [0.3, 0.4]]],
                        "occupancy_status": "occupied",
                    },
                    {
                        "space_id": 2,
                        "points": [[[0.5, 0.5], [0.6, 0.5], [0.6, 0.6]]],
                        "occupancy_status": "unknown",
                    },
                ]
            }
        }
    }

    slots, truth = _slots_and_truth(record, width=100, height=200)

    assert slots[0].points[0] == (10.0, 40.0)
    assert truth == {"slot_001": 1, "slot_002": None}


def test_error_cause_separates_detector_and_mapping_failures() -> None:
    assert (
        _error_cause("b0", 1, 0, False, 0.0, 0.3)
        == "detector_miss_or_severe_localization"
    )
    assert (
        _error_cause("b0", 1, 0, False, 0.5, 0.3)
        == "centre_mapping_failure"
    )
    assert (
        _error_cause("b1", 1, 0, True, 0.2, 0.3)
        == "overlap_threshold_failure"
    )
