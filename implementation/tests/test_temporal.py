from parking_occupancy.models import SlotEvidence
from parking_occupancy.temporal import HysteresisConfig, TemporalOccupancyFilter


def evidence(score: float) -> dict[str, SlotEvidence]:
    return {
        "s1": SlotEvidence(
            slot_id="s1",
            occupied=score > 0,
            geometric_score=score,
            evidence_score=score,
        )
    }


def test_hysteresis_suppresses_one_frame_detection() -> None:
    occupancy_filter = TemporalOccupancyFilter(
        ["s1"],
        HysteresisConfig(
            rise_alpha=0.5,
            fall_alpha=0.5,
            occupied_threshold=0.6,
            vacant_threshold=0.2,
        ),
    )
    assert not occupancy_filter.update(evidence(1.0))["s1"].occupied
    assert not occupancy_filter.update(evidence(0.0))["s1"].occupied


def test_hysteresis_transitions_after_sustained_evidence() -> None:
    occupancy_filter = TemporalOccupancyFilter(
        ["s1"],
        HysteresisConfig(
            rise_alpha=0.5,
            fall_alpha=0.5,
            occupied_threshold=0.6,
            vacant_threshold=0.2,
        ),
    )
    occupancy_filter.update(evidence(1.0))
    state = occupancy_filter.update(evidence(1.0))["s1"]
    assert state.occupied
    assert state.changed

    assert occupancy_filter.update(evidence(0.0))["s1"].occupied
    state = occupancy_filter.update(evidence(0.0))["s1"]
    assert not state.occupied
    assert state.changed


def test_filter_accepts_one_shot_slot_id_iterator() -> None:
    occupancy_filter = TemporalOccupancyFilter(
        (slot_id for slot_id in ["s1"]),
        HysteresisConfig(
            rise_alpha=1.0,
            fall_alpha=1.0,
            occupied_threshold=0.6,
            vacant_threshold=0.2,
        ),
    )

    assert occupancy_filter.update(evidence(1.0))["s1"].occupied
