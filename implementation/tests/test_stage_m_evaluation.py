from __future__ import annotations

from pathlib import Path

import numpy as np

from parking_occupancy.models import Detection, ParkingSlot
from parking_occupancy.stage_m_evaluation import (
    METHODS,
    export_stage_m_run,
    run_t0_t3_sequence,
)


class FakeAdapter:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.started = []

    def begin_source(self, source_id: str, *, continuous: bool) -> None:
        self.started.append((source_id, continuous))

    def detect(self, _frame):
        value = self.frames[self.index]
        self.index += 1
        return value

    def metadata(self):
        return {"backend": "fake", "frames": self.index}


def _detection(
    bbox: tuple[float, float, float, float],
    track_id: int | None,
) -> Detection:
    return Detection(
        bbox=bbox,
        confidence=0.9,
        class_id=0,
        class_name="vehicle",
        track_id=track_id,
    )


def test_t0_t3_handle_static_motion_short_miss_and_common_schema(
    tmp_path: Path,
) -> None:
    slots = (
        ParkingSlot(
            slot_id="s1",
            points=((0, 0), (10, 0), (10, 10), (0, 10)),
        ),
    )
    plain = FakeAdapter(
        [
            (_detection((1, 1, 9, 9), None),),
            (_detection((2, 1, 10, 9), None),),
            (),
        ]
    )
    byte = FakeAdapter(
        [
            (_detection((1, 1, 9, 9), 11),),
            (_detection((2, 1, 10, 9), 11),),
            (),
        ]
    )
    track = FakeAdapter(
        [
            (_detection((1, 1, 9, 9), 21),),
            (_detection((2, 1, 10, 9), 21),),
            (),
        ]
    )
    frames = [
        np.zeros((16, 16, 3), dtype=np.uint8)
        for _ in range(3)
    ]
    result = run_t0_t3_sequence(
        frames=frames,
        fps=5.0,
        source_id="synthetic-interface-smoke",
        slots=slots,
        plain_adapter=plain,
        bytetrack_adapter=byte,
        tracktrack_adapter=track,
        classifier_scores=lambda _frame, frozen_slots: {
            slot.slot_id: 0.1 for slot in frozen_slots
        },
        mapping_coverage=0.4,
        classifier_threshold=0.76,
        temporal_config={
            "rise_alpha": 0.60,
            "fall_alpha": 0.15,
            "occupied_threshold": 0.58,
            "vacant_threshold": 0.42,
            "raw_threshold": 0.76,
        },
        claim_scope="smoke_test",
    )

    assert len(result.rows) == 3
    assert all(set(METHODS).issubset(row) for row in result.rows)
    assert result.rows[0]["T2_track_id"] == 11
    assert result.rows[0]["T3_track_id"] == 21
    assert result.rows[2]["T2_track_id"] is None
    assert result.rows[2]["T3_track_id"] is None
    assert result.metrics["status"] == "not_computed_no_truth"
    assert plain.started == byte.started == track.started == [
        ("synthetic-interface-smoke", True)
    ]

    output = tmp_path / "export"
    export_stage_m_run(result, output_root=output, fps=5.0)
    assert {
        "occupancy.csv",
        "events.csv",
        "detections.jsonl",
        "annotated.mp4",
        "metrics.json",
        "summary.json",
        "runtime_metadata.json",
    } == {path.name for path in output.iterdir()}
    assert (output / "annotated.mp4").stat().st_size > 0


def test_t0_t3_metrics_keep_occupancy_and_transition_scopes_separate() -> None:
    slots = (
        ParkingSlot(
            slot_id="s1",
            points=((0, 0), (10, 0), (10, 10), (0, 10)),
        ),
    )
    present = (_detection((1, 1, 9, 9), 1),)
    absent = ()
    adapters = [
        FakeAdapter([absent, present, present, absent])
        for _ in range(3)
    ]
    truth = {
        "slots": [
            {
                "slot_id": "s1",
                "intervals": [
                    {"start_frame": 0, "end_frame": 1, "state": "vacant"},
                    {
                        "start_frame": 1,
                        "end_frame": 3,
                        "state": "occupied",
                    },
                    {"start_frame": 3, "end_frame": 4, "state": "vacant"},
                ],
            }
        ]
    }
    result = run_t0_t3_sequence(
        frames=[
            np.zeros((16, 16, 3), dtype=np.uint8)
            for _ in range(4)
        ],
        fps=1.0,
        source_id="truth-smoke",
        slots=slots,
        plain_adapter=adapters[0],
        bytetrack_adapter=adapters[1],
        tracktrack_adapter=adapters[2],
        classifier_scores=lambda _frame, _slots: {"s1": 0.1},
        mapping_coverage=0.4,
        classifier_threshold=0.76,
        temporal_config={
            "rise_alpha": 1.0,
            "fall_alpha": 1.0,
            "occupied_threshold": 0.58,
            "vacant_threshold": 0.42,
            "raw_threshold": 0.76,
        },
        truth=truth,
        claim_scope="retrospective_diagnostic",
        stable_frames=1,
    )

    t0 = result.metrics["methods"]["T0"]
    assert t0["macro_f1"] == 1.0
    assert t0["occupied_recall"] == 1.0
    assert t0["temporal_aggregate"]["on_time_transitions"] == 2
    assert "HOTA" not in t0
