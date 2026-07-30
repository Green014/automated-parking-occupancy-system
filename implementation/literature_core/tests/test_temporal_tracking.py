import pytest

from literature_core.temporal_tracking import (
    TrackSlotEvidence,
    TrackSlotGate,
    TrackSlotGateConfig,
    associate_tracks_to_slots,
)


def _e(
    slot: str,
    track: str,
    coverage: float = 0.6,
    displacement: float = 1.0,
) -> TrackSlotEvidence:
    return TrackSlotEvidence(slot, track, coverage, displacement)


def _config() -> TrackSlotGateConfig:
    # Synthetic-only values: they are not selected on VIRAT truth.
    return TrackSlotGateConfig(
        minimum_coverage=0.3,
        maximum_stationary_displacement_px=4.0,
        occupied_dwell_frames=3,
        vacant_dwell_frames=2,
    )


def test_track_to_slot_assignment_is_one_to_one_and_deterministic() -> None:
    assignments = associate_tracks_to_slots(
        (
            _e("A", "track-1", 0.8),
            _e("B", "track-1", 0.7),
            _e("B", "track-2", 0.6),
        ),
        slot_ids=("A", "B"),
        minimum_coverage=0.3,
        maximum_stationary_displacement_px=4.0,
    )
    assert assignments["A"].track_id == "track-1"
    assert assignments["B"].track_id == "track-2"


def test_stationary_track_requires_occupancy_and_vacancy_dwell() -> None:
    gate = TrackSlotGate(("A",), _config())

    assert not gate.update((_e("A", "parked"),))[0].occupied
    assert not gate.update((_e("A", "parked"),))[0].occupied
    confirmed = gate.update((_e("A", "parked"),))[0]
    assert confirmed.occupied
    assert confirmed.changed
    assert confirmed.confirmed_track_id == "parked"

    first_clear = gate.update(())[0]
    assert first_clear.occupied
    assert first_clear.clear_dwell_frames == 1
    vacant = gate.update(())[0]
    assert not vacant.occupied
    assert vacant.changed


def test_fast_passing_vehicle_is_suppressed_without_state_change() -> None:
    gate = TrackSlotGate(("A",), _config())

    for _ in range(8):
        state = gate.update((_e("A", "passing", 0.9, 12.0),))[0]
        assert not state.occupied
        assert state.assigned_track_id is None
        assert state.suppressed_moving_track_ids == ("passing",)


def test_track_id_switch_resets_preconfirmation_dwell() -> None:
    gate = TrackSlotGate(("A",), _config())

    gate.update((_e("A", "old-id"),))
    gate.update((_e("A", "old-id"),))
    switched = gate.update((_e("A", "new-id"),))[0]
    assert not switched.occupied
    assert switched.candidate_dwell_frames == 1
    gate.update((_e("A", "new-id"),))
    assert gate.update((_e("A", "new-id"),))[0].occupied


def test_confirmed_occupancy_survives_track_id_switch() -> None:
    gate = TrackSlotGate(("A",), _config())
    for _ in range(3):
        gate.update((_e("A", "original"),))

    switched = gate.update((_e("A", "replacement"),))[0]
    assert switched.occupied
    assert switched.confirmed_track_id == "original"
    gate.update((_e("A", "replacement"),))
    adopted = gate.update((_e("A", "replacement"),))[0]
    assert adopted.occupied
    assert adopted.confirmed_track_id == "replacement"


def test_invalid_or_unknown_track_evidence_is_rejected() -> None:
    with pytest.raises(ValueError, match="coverage"):
        _e("A", "track", 1.1)
    with pytest.raises(KeyError):
        associate_tracks_to_slots(
            (_e("unknown", "track"),),
            slot_ids=("A",),
            minimum_coverage=0.3,
            maximum_stationary_displacement_px=4.0,
        )
