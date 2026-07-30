from __future__ import annotations

import numpy as np

from parking_occupancy.models import Detection, ParkingSlot
from parking_occupancy.stage_v import (
    FrameOccupancyResult,
    SlotOccupancyState,
    draw_stage_v_frame,
)
from parking_occupancy.stage_w_ui_adapter import (
    StateEventTracker,
    frame_result_to_ui_payload,
    redact_source,
)


SLOTS = (
    ParkingSlot("A01", ((1, 1), (14, 1), (14, 14), (1, 14))),
    ParkingSlot("A02", ((17, 1), (30, 1), (30, 14), (17, 14))),
)


def _result(
    frame_index: int = 0,
    states: tuple[bool, bool] = (True, False),
    *,
    tracking: bool = False,
) -> FrameOccupancyResult:
    return FrameOccupancyResult(
        frame_index=frame_index,
        timestamp_s=frame_index / 5.0,
        slot_states=tuple(
            SlotOccupancyState(
                slot_id=slot.slot_id,
                occupied=states[index],
                evidence_score=0.91 if states[index] else 0.12,
                evidence_source=(
                    "D1_B1_E1b_F2.detector_confirmed"
                    if states[index]
                    else "D1_B1_E1b_F2.classifier_rejected"
                ),
                track_id=17 if tracking and index == 0 else None,
            )
            for index, slot in enumerate(SLOTS)
        ),
        vehicle_detections=(
            Detection(
                bbox=(2, 2, 13, 13),
                confidence=0.9,
                class_id=0,
                class_name="vehicle",
                track_id=17 if tracking else None,
            ),
        ),
        timing_ms={
            "backend_total": 5.0,
            "attributed_backend_total": 50.0,
            "cache_hit": 1.0,
        },
        warnings=(),
    )


def test_ui_adapter_schema_and_count_invariants() -> None:
    payload = frame_result_to_ui_payload(
        _result(),
        SLOTS,
        mode="fusion",
        temporal_enabled=False,
        tracker_enabled=False,
    )
    assert payload["frame_index"] == 0
    assert payload["occupied"] == 1
    assert payload["vacant"] == 1
    assert payload["total"] == 2
    assert payload["occupied"] + payload["vacant"] == payload["total"]
    assert len(payload["slots"]) == 2
    assert {row["slot_id"] for row in payload["slots"]} == {"A01", "A02"}
    assert payload["runtime"]["attributed_fps"] == 20.0
    assert payload["runtime"]["cache"] == "hit"
    assert payload["detections"][0]["bbox"] == [2.0, 2.0, 13.0, 13.0]


def test_ui_adapter_separates_slot_ids_and_tracking_ids() -> None:
    without_tracking = frame_result_to_ui_payload(
        _result(tracking=True),
        SLOTS,
        mode="fusion",
        temporal_enabled=False,
        tracker_enabled=False,
    )
    assert all(row["track_id"] is None for row in without_tracking["slots"])
    assert without_tracking["detections"][0]["track_id"] is None

    with_tracking = frame_result_to_ui_payload(
        _result(tracking=True),
        SLOTS,
        mode="fusion",
        temporal_enabled=False,
        tracker_enabled=True,
    )
    assert with_tracking["slots"][0]["slot_id"] == "A01"
    assert with_tracking["slots"][0]["track_id"] == 17


def test_ui_json_redacts_paths_and_rtsp_credentials() -> None:
    result = _result()
    result = FrameOccupancyResult(
        frame_index=result.frame_index,
        timestamp_s=result.timestamp_s,
        slot_states=result.slot_states,
        vehicle_detections=result.vehicle_detections,
        timing_ms=result.timing_ms,
        warnings=("failed " + r"C:" + r"\Users\person\models\best.pt",),
    )
    payload = frame_result_to_ui_payload(
        result,
        SLOTS,
        mode="fusion",
        temporal_enabled=False,
        tracker_enabled=False,
        source=(
            "rtsp://"
            + "fixture-user"
            + ":"
            + "fixture-token"
            + "@"
            + "example.test:8554/camera"
        ),
    )
    serialized = str(payload)
    assert "secret" not in serialized
    assert ("C:" + r"\Users") not in serialized
    assert payload["source"] == "rtsp://example.test:8554/<redacted>"
    assert redact_source(r"C:\video\parking.mp4") == "parking.mp4"


def test_event_tracker_has_no_initial_event_and_emits_real_transition() -> None:
    tracker = StateEventTracker(continuous=True, temporal_enabled=False)
    assert tracker.update(_result(0, (True, False))) == []
    events = tracker.update(_result(1, (False, False)))
    assert len(events) == 1
    assert events[0]["event_type"] == "departure"
    assert events[0]["slot_id"] == "A01"
    assert events[0]["temporal_semantics"] == "raw_frame_level_state_changes"


def test_noncontinuous_event_tracker_never_emits_events() -> None:
    tracker = StateEventTracker(continuous=False, temporal_enabled=False)
    assert tracker.update(_result(0, (False, False))) == []
    assert tracker.update(_result(1, (True, False))) == []
    assert tracker.events() == []
    assert tracker.semantics == "not_temporally_valid_noncontinuous_input"


def test_no_track_legend_or_labels_when_tracking_is_disabled(monkeypatch) -> None:
    texts: list[str] = []
    original = __import__("cv2").putText

    def capture_text(image, text, *args, **kwargs):
        texts.append(str(text))
        return original(image, text, *args, **kwargs)

    monkeypatch.setattr("parking_occupancy.stage_v.cv2.putText", capture_text)
    states = _result(tracking=True).state_by_slot()
    draw_stage_v_frame(
        frame=np.zeros((100, 200, 3), dtype=np.uint8),
        detections=_result(tracking=True).vehicle_detections,
        slots=SLOTS,
        states=states,
        mode="fusion",
        processing_fps=20.0,
        tracker_enabled=False,
    )
    assert "track ID" not in texts
    assert not any(text.startswith("track=") for text in texts)
