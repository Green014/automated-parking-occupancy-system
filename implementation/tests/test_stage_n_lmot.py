from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from parking_occupancy.stage_n_lmot import (
    LmotAnnotation,
    OfficialTrackEvalAdapter,
    FrozenStageNTrackerAdapter,
    StageNInferenceSettings,
    StageNDataGateError,
    TrackPrediction,
    VerifiedLmotClassMap,
    audit_lmot_sequence,
    evaluate_motor_vehicle_detections,
    load_stage_n_protocol,
    parse_lmot_gt,
    split_motor_vehicle_truth,
    suppress_predictions_on_excluded_truth,
)


def _gt(
    frame: int,
    track_id: int,
    *,
    class_id: int = 1,
    ignore: int = 0,
    x: float = 10.0,
) -> LmotAnnotation:
    return LmotAnnotation(
        frame_number=frame,
        track_id=track_id,
        x=x,
        y=10.0,
        width=10.0,
        height=10.0,
        ignore=ignore,
        class_id=class_id,
        visibility=1.0,
    )


def _prediction(
    frame: int,
    track_id: int,
    *,
    x: float = 10.0,
) -> TrackPrediction:
    return TrackPrediction(
        frame_number=frame,
        track_id=track_id,
        xyxy=(x, 10.0, x + 10.0, 20.0),
    )


def _synthetic_map() -> VerifiedLmotClassMap:
    # Explicitly fixture-only: this is not a claim about official LMOT IDs.
    return VerifiedLmotClassMap(
        id_to_name={
            1: "car",
            2: "person",
            3: "bicycle",
            4: "motorcycle",
            5: "bus",
            6: "truck",
        },
        verification_status="synthetic_fixture",
        evidence="unit-test fixture only",
    )


def test_stage_n_protocol_preserves_frozen_inputs() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = load_stage_n_protocol(
        root
        / "configs"
        / "stage_n_lmot_tracking_diagnostic_frozen_20260728.yaml",
        verify_files=True,
    )

    assert list(protocol["methods"]) == ["L0", "L1", "L2", "L3"]
    assert protocol["lmot"]["data_gate"] == "blocked_before_download"
    assert (
        protocol["comparison_label"]
        == "controlled_end_to_end_tracker_backend_comparison"
    )


def test_parser_reads_exact_nine_column_schema(tmp_path: Path) -> None:
    gt_path = tmp_path / "gt.txt"
    gt_path.write_text(
        "1,7,10.5,20,30,40,0,4,0.75\n", encoding="utf-8"
    )

    rows = parse_lmot_gt(gt_path)

    assert rows == [
        LmotAnnotation(1, 7, 10.5, 20.0, 30.0, 40.0, 0, 4, 0.75)
    ]
    assert rows[0].xyxy == (10.5, 20.0, 40.5, 60.0)


def test_parser_rejects_bad_width_visibility_and_column_count(
    tmp_path: Path,
) -> None:
    for index, text in enumerate(
        (
            "1,1,0,0,0,10,0,1,1\n",
            "1,1,0,0,10,10,0,1,1.1\n",
            "1,1,0,0,10,10,0,1\n",
        )
    ):
        path = tmp_path / f"bad-{index}.txt"
        path.write_text(text, encoding="utf-8")
        with pytest.raises(StageNDataGateError):
            parse_lmot_gt(path)


def test_class_map_requires_evidence_and_unifies_four_vehicle_classes() -> None:
    mapping = _synthetic_map()

    assert {
        mapping.class_name(class_id)
        for class_id in (1, 4, 5, 6)
        if mapping.is_motor_vehicle(class_id)
    } == {"car", "motorcycle", "bus", "truck"}
    assert mapping.is_non_motor(2)
    assert mapping.is_non_motor(3)
    with pytest.raises(StageNDataGateError):
        VerifiedLmotClassMap(
            id_to_name={1: "car"},
            verification_status="unresolved",
            evidence="README order",
        )


def test_person_bicycle_and_ignore_are_prediction_suppression_regions() -> None:
    rows = [
        _gt(1, 1, class_id=1, x=10),
        _gt(1, 2, class_id=2, x=30),
        _gt(1, 3, class_id=3, x=50),
        _gt(1, 4, class_id=6, ignore=1, x=70),
    ]
    evaluated, suppression = split_motor_vehicle_truth(
        rows,
        class_map=_synthetic_map(),
        evaluated_ignore_values=frozenset({0}),
    )
    predictions = [
        _prediction(1, 10, x=10),
        _prediction(1, 20, x=30),
        _prediction(1, 30, x=50),
        _prediction(1, 40, x=70),
        _prediction(1, 50, x=90),
    ]

    kept, removed = suppress_predictions_on_excluded_truth(
        predictions, evaluated, suppression
    )

    assert [row.track_id for row in evaluated] == [1]
    assert [row.track_id for row in kept] == [10, 50]
    assert removed == 3


def test_official_trackeval_perfect_tracking() -> None:
    gt = [_gt(frame, 1) for frame in range(1, 5)]
    predictions = [_prediction(frame, 9) for frame in range(1, 5)]

    metrics = OfficialTrackEvalAdapter().evaluate_sequence(
        num_timesteps=4, gt=gt, predictions=predictions
    )

    assert metrics["HOTA"] == pytest.approx(100.0)
    assert metrics["DetA"] == pytest.approx(100.0)
    assert metrics["AssA"] == pytest.approx(100.0)
    assert metrics["IDF1"] == pytest.approx(100.0)
    assert metrics["ID_switches"] == 0
    assert metrics["MOTA"] == pytest.approx(100.0)


def test_official_trackeval_detects_id_switch() -> None:
    gt = [_gt(frame, 1) for frame in range(1, 5)]
    predictions = [
        _prediction(1, 9),
        _prediction(2, 9),
        _prediction(3, 10),
        _prediction(4, 10),
    ]

    metrics = OfficialTrackEvalAdapter().evaluate_sequence(
        num_timesteps=4, gt=gt, predictions=predictions
    )

    assert metrics["DetA"] == pytest.approx(100.0)
    assert metrics["AssA"] < 100.0
    assert metrics["IDF1"] < 100.0
    assert metrics["ID_switches"] == 1


def test_official_trackeval_counts_miss_and_false_positive() -> None:
    gt = [_gt(frame, 1) for frame in range(1, 4)]
    predictions = [
        _prediction(1, 9),
        _prediction(2, 9),
        _prediction(2, 10, x=100),
    ]

    metrics = OfficialTrackEvalAdapter().evaluate_sequence(
        num_timesteps=3, gt=gt, predictions=predictions
    )

    assert metrics["false_negatives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["DetA"] < 100.0
    assert metrics["MOTA"] < 100.0


def test_detection_metrics_perfect_and_missed_detection() -> None:
    gt = [_gt(1, 1), _gt(2, 1)]
    perfect = evaluate_motor_vehicle_detections(
        gt=gt, predictions=[_prediction(1, 4), _prediction(2, 4)]
    )
    missed = evaluate_motor_vehicle_detections(
        gt=gt, predictions=[_prediction(1, 4)]
    )

    assert perfect["AP50"] == pytest.approx(1.0)
    assert perfect["AP50-95"] == pytest.approx(1.0)
    assert perfect["precision"] == pytest.approx(1.0)
    assert perfect["recall"] == pytest.approx(1.0)
    assert missed["recall"] == pytest.approx(0.5)
    assert missed["AP50"] < 1.0


def test_detection_matching_falls_back_to_best_unused_gt() -> None:
    gt = [
        _gt(1, 1, x=0.0),
        _gt(1, 2, x=4.0),
    ]
    predictions = [
        TrackPrediction(
            frame_number=1,
            track_id=10,
            xyxy=(0.0, 10.0, 10.0, 20.0),
            confidence=0.9,
        ),
        TrackPrediction(
            frame_number=1,
            track_id=11,
            xyxy=(1.0, 10.0, 11.0, 20.0),
            confidence=0.8,
        ),
    ]

    metrics = evaluate_motor_vehicle_detections(
        gt=gt,
        predictions=predictions,
        iou_thresholds=(0.5,),
    )

    assert metrics["true_positives"] == 2
    assert metrics["false_positives"] == 0
    assert metrics["false_negatives"] == 0
    assert metrics["precision"] == pytest.approx(1.0)
    assert metrics["recall"] == pytest.approx(1.0)


def test_tracker_adapter_uses_complete_ultralytics_track_call(
    tmp_path: Path,
) -> None:
    tracker = tmp_path / "tracktrack.yaml"
    tracker.write_text("tracker_type: tracktrack\n", encoding="utf-8")

    class Tensor:
        def __init__(self, values):
            self.values = np.asarray(values)

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.values

    class Boxes:
        xyxy = Tensor([[1, 2, 11, 12]])
        conf = Tensor([0.8])
        id = Tensor([17])

        def __len__(self):
            return 1

    class Model:
        def __init__(self):
            self.calls = []

        def track(self, **kwargs):
            self.calls.append(kwargs)
            return [SimpleNamespace(boxes=Boxes())]

    model = Model()
    adapter = FrozenStageNTrackerAdapter(
        StageNInferenceSettings(weights="D1.pt", device="cpu"),
        tracker_config=tracker,
        model_factory=lambda _weights: model,
    )

    rows = adapter.track(np.zeros((20, 20, 3), dtype=np.uint8))

    assert rows[0].track_id == 17
    assert model.calls[0]["persist"] is True
    assert model.calls[0]["tracker"] == str(tracker.resolve())
    assert model.calls[0]["conf"] == 0.30
    assert model.calls[0]["iou"] == 0.70
    assert model.calls[0]["imgsz"] == 640
    assert model.calls[0]["agnostic_nms"] is True
    assert model.calls[0]["max_det"] == 300


def _write_sequence(root: Path, *, light_frames: tuple[int, ...]) -> Path:
    sequence = root / "LMOT-synthetic"
    for directory in ("img_dark_rgb", "img_light_rgb", "gt"):
        (sequence / directory).mkdir(parents=True, exist_ok=True)
    image = np.zeros((12, 12, 3), dtype=np.uint8)
    for frame in (1, 2):
        assert cv2.imwrite(
            str(sequence / "img_dark_rgb" / f"{frame:06d}.jpg"), image
        )
    for frame in light_frames:
        assert cv2.imwrite(
            str(sequence / "img_light_rgb" / f"{frame:06d}.jpg"), image
        )
    (sequence / "gt" / "gt.txt").write_text(
        "1,1,1,1,5,5,0,1,1\n2,1,1,1,5,5,0,1,1\n",
        encoding="utf-8",
    )
    (sequence / "seqinfo.ini").write_text(
        "[Sequence]\nname=LMOT-synthetic\nseqLength=2\n",
        encoding="utf-8",
    )
    return sequence


def test_dark_light_alignment_audit_passes_for_exact_pair(
    tmp_path: Path,
) -> None:
    result = audit_lmot_sequence(
        _write_sequence(tmp_path, light_frames=(1, 2))
    )

    assert result["passed"] is True
    assert result["dark_light_frame_numbers_aligned"] is True
    assert result["single_gt_file_for_aligned_pair"] is True


def test_dark_light_alignment_audit_blocks_missing_frame(
    tmp_path: Path,
) -> None:
    result = audit_lmot_sequence(
        _write_sequence(tmp_path, light_frames=(1,))
    )

    assert result["passed"] is False
    assert "dark_light_frame_sets_differ" in result["errors"]
    assert result["missing_light_frames"] == [2]
