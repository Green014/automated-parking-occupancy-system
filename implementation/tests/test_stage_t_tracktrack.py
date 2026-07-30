from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from parking_occupancy.artifact_registry import (
    STAGE_T_FROZEN_REGISTRY_SHA256,
    verify_historical_artifact_registry,
)
from parking_occupancy.geometry import map_detections_to_slots
from parking_occupancy.integrated_runner import IntegratedFrameProcessor
from parking_occupancy.models import Detection, ParkingSlot
from parking_occupancy.stage_t_cli import DEFAULT_P3_TT_CONFIG, build_parser
from parking_occupancy.stage_t_tracktrack import (
    build_track_records,
    load_p3_tt_config,
    validate_tracks_schema,
)


def _detection(
    *,
    track_id: int | None,
    bbox: tuple[float, float, float, float] = (2.0, 2.0, 8.0, 8.0),
) -> dict[str, object]:
    return {
        "bbox": list(bbox),
        "confidence": 0.9,
        "class_id": 0,
        "class_name": "vehicle",
        "track_id": track_id,
    }


def _frame(
    source: str,
    index: int,
    detections: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "video_id": source,
        "frame_index": index,
        "timestamp_s": index / 10.0,
        "detections": detections,
    }


def test_p3_tt_config_reuses_stage_s_and_frozen_tracktrack() -> None:
    config = load_p3_tt_config(DEFAULT_P3_TT_CONFIG)
    assert config["detector"]["id"] == "D1"
    assert config["mapping"]["id"] == "B1"
    assert config["fusion"]["id"] == "F2"
    assert config["temporal"]["default_enabled"] is False
    assert config["tracking"]["default_backend"] == "tracktrack"
    assert config["claims"]["replaces_stage_s_default"] is False


def test_p3_tt_cli_is_explicit_and_has_no_temporal_switch(tmp_path: Path) -> None:
    args = build_parser().parse_args(
        [
            "--input",
            str(tmp_path / "input.mp4"),
            "--slots",
            str(tmp_path / "slots.json"),
            "--d1-weights",
            str(tmp_path / "d1.pt"),
            "--e1b-checkpoint",
            str(tmp_path / "e1b.pt"),
            "--output-dir",
            str(tmp_path / "output"),
            "--source-id",
            "video-a",
        ]
    )
    assert args.config.resolve() == DEFAULT_P3_TT_CONFIG.resolve()
    assert not hasattr(args, "temporal")
    assert not hasattr(args, "tracker")


def test_track_id_persists_across_short_miss_and_empty_frame_is_logged() -> None:
    records, summary = build_track_records(
        [
            _frame("a", 0, [_detection(track_id=7)]),
            _frame("a", 1, []),
            _frame("a", 2, [_detection(track_id=7)]),
        ],
        track_buffer=30,
    )
    validate_tracks_schema(records)
    assert records[1]["tracks"] == []
    assert records[2]["tracks"][0]["track_id"] == 7
    assert records[2]["tracks"][0]["gap_from_previous_observation"] == 2
    assert records[2]["tracks"][0]["reacquired_after_short_gap"] is True
    assert records[2]["tracks"][0]["expired_before_observation"] is False
    assert summary["short_gap_reacquisitions"] == 1


def test_expired_gap_is_flagged_and_source_switch_resets_history() -> None:
    records, summary = build_track_records(
        [
            _frame("a", 0, [_detection(track_id=3)]),
            _frame("a", 31, [_detection(track_id=3)]),
            _frame("b", 0, [_detection(track_id=3)]),
        ],
        track_buffer=30,
    )
    assert records[1]["tracks"][0]["expired_before_observation"] is True
    assert records[2]["tracks"][0]["gap_from_previous_observation"] is None
    assert records[2]["tracks"][0]["observation_index"] == 1
    assert summary["sources"] == 2
    assert summary["expired_id_reappearances"] == 1


def test_untracked_passing_vehicle_does_not_create_track_record() -> None:
    records, summary = build_track_records(
        [_frame("a", 0, [_detection(track_id=None)])],
        track_buffer=30,
    )
    assert records[0]["tracks"] == []
    assert summary["untracked_detections"] == 1
    assert summary["tracked_detections"] == 0


def test_passing_vehicle_and_one_to_one_slot_assignment() -> None:
    slots = (
        ParkingSlot(
            slot_id="s1",
            points=((0, 0), (10, 0), (10, 10), (0, 10)),
        ),
        ParkingSlot(
            slot_id="s2",
            points=((8, 0), (18, 0), (18, 10), (8, 10)),
        ),
    )
    passing = Detection(
        bbox=(30, 0, 40, 10),
        confidence=0.9,
        class_id=0,
        class_name="vehicle",
        track_id=5,
    )
    passing_evidence = map_detections_to_slots(
        [passing], slots, mode="overlap", overlap_threshold=0.4
    )
    assert not any(item.occupied for item in passing_evidence.values())

    overlapping = Detection(
        bbox=(4, 0, 14, 10),
        confidence=0.9,
        class_id=0,
        class_name="vehicle",
        track_id=6,
    )
    mapped = map_detections_to_slots(
        [overlapping], slots, mode="overlap", overlap_threshold=0.4
    )
    assert sum(item.occupied for item in mapped.values()) == 1
    assert {item.track_id for item in mapped.values() if item.occupied} == {6}


class _Detector:
    def __init__(self, track_id: int | None) -> None:
        self.track_id = track_id

    def begin_source(self, source_id: str, *, continuous: bool) -> None:
        self.source_id = source_id
        self.continuous = continuous

    def detect(self, frame: np.ndarray) -> tuple[Detection, ...]:
        return (
            Detection(
                bbox=(1, 1, 9, 9),
                confidence=0.9,
                class_id=0,
                class_name="vehicle",
                track_id=self.track_id,
            ),
        )


class _Classifier:
    def predict(
        self,
        frame: np.ndarray,
        slots: tuple[ParkingSlot, ...],
    ) -> dict[str, float]:
        return {slot.slot_id: 0.1 for slot in slots}


def _processor(track_id: int | None) -> IntegratedFrameProcessor:
    config = load_p3_tt_config(DEFAULT_P3_TT_CONFIG)
    return IntegratedFrameProcessor(
        slots=(
            ParkingSlot(
                slot_id="s1",
                points=((0, 0), (10, 0), (10, 10), (0, 10)),
            ),
        ),
        detector=_Detector(track_id),
        classifier=_Classifier(),
        config=config,
        temporal_enabled=False,
    )


def test_p3_and_p3_tt_occupancy_state_are_independent_of_id_label() -> None:
    plain = _processor(None)
    tracked = _processor(42)
    plain.begin_source("a")
    tracked.begin_source("a")
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    plain_result = plain.process(frame, fps=10.0)
    tracked_result = tracked.process(frame, fps=10.0)
    assert plain_result.states["s1"].occupied is True
    assert tracked_result.states["s1"].occupied is True
    assert plain_result.decisions["s1"].score == tracked_result.decisions["s1"].score
    assert plain_result.states["s1"].track_id is None
    assert tracked_result.states["s1"].track_id == 42


def test_tracks_output_schema_preserves_one_to_one_slot_reference() -> None:
    records, _ = build_track_records(
        [_frame("a", 0, [_detection(track_id=11)])],
        assignments={("a", 0, 11): ("slot-1",)},
        track_buffer=30,
    )
    validate_tracks_schema(records)
    assert records[0]["tracks"][0]["assigned_slot_ids"] == ["slot-1"]
    assert set(records[0]["tracks"][0]) == {
        "track_id",
        "bbox",
        "confidence",
        "class_id",
        "class_name",
        "assigned_slot_ids",
        "observation_index",
        "gap_from_previous_observation",
        "reacquired_after_short_gap",
        "expired_before_observation",
    }


def test_processor_reset_keeps_events_independent_between_videos() -> None:
    processor = _processor(9)
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    processor.begin_source("a")
    first = processor.process(frame, fps=10.0)
    processor.begin_source("b")
    second = processor.process(frame, fps=10.0)
    assert first.frame_index == second.frame_index == 0
    assert first.events[0]["video_id"] == "a"
    assert second.events[0]["video_id"] == "b"
    assert first.events[0]["from_state"] == second.events[0]["from_state"] == 0


def test_completed_tt0_tt1_output_contract_when_available() -> None:
    root = Path(__file__).resolve().parents[1]
    output_root = root / "outputs" / "stage_t_tracktrack_consumed_dev_20260729"
    if not output_root.exists():
        return
    required = {
        "occupancy.csv",
        "events.csv",
        "detections.jsonl",
        "tracks.jsonl",
        "annotated.mp4",
        "metrics.json",
        "summary.json",
        "runtime_metadata.json",
    }
    for variant_id, backend in (("tt0", "none"), ("tt1", "tracktrack")):
        variant_root = output_root / variant_id
        assert required <= {path.name for path in variant_root.iterdir()}
        summary = json.loads(
            (variant_root / "summary.json").read_text(encoding="utf-8")
        )
        assert summary["frames"] == 1974
        assert summary["temporal_enabled"] is False
        assert summary["tracker_backend"] == backend
        assert summary["formal_occupancy_improvement_conclusion"] == "blocked"
        capture = cv2.VideoCapture(str(variant_root / "annotated.mp4"))
        assert capture.isOpened()
        assert int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT))) == 1974
        capture.release()

    records = [
        json.loads(line)
        for line in (output_root / "tt1" / "tracks.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(records) == 1974
    validate_tracks_schema(records)
    assert any(frame["tracks"] for frame in records)


def test_stage_t_comparison_marks_formal_improvement_blocked() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "data" / "stage_t" / "STAGE_T_TT0_TT1_COMPARISON.json"
    if not path.exists():
        return
    comparison = json.loads(path.read_text(encoding="utf-8"))
    assert comparison["claim_class"] == "consumed-development diagnostic"
    assert comparison["untouched_test"] is False
    assert comparison["E4_included"] is False
    assert comparison["occupancy_predictions_identical"] is True
    assert comparison["TT1_minus_TT0"]["macro_f1"] == 0.0
    assert comparison["formal_occupancy_improvement_conclusion"] == "blocked"


def test_stage_s_registry_remains_valid_after_stage_t() -> None:
    from parking_occupancy.artifact_registry import (
        STAGE_S_FROZEN_REGISTRY_SHA256,
        verify_historical_artifact_registry,
    )

    implementation_root = Path(__file__).resolve().parents[1]
    result = verify_historical_artifact_registry(
        implementation_root
        / "data"
        / "stage_s"
        / "STAGE_S_ARTIFACT_REGISTRY_20260729.yaml",
        artifact_root=implementation_root.parent,
        expected_registry_sha256=STAGE_S_FROZEN_REGISTRY_SHA256,
        immutable_path_prefixes=("implementation/data/stage_s",),
    )
    assert result["verified"] is True


def test_stage_t_registry_when_frozen() -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    registry = (
        implementation_root
        / "data"
        / "stage_t"
        / "STAGE_T_ARTIFACT_REGISTRY_20260729.yaml"
    )
    if not registry.exists():
        return
    result = verify_historical_artifact_registry(
        registry,
        artifact_root=implementation_root.parent,
        expected_registry_sha256=STAGE_T_FROZEN_REGISTRY_SHA256,
        immutable_path_prefixes=("implementation/data/stage_t",),
    )
    assert result["verified"] is True
