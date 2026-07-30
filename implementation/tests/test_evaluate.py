import csv

import pytest

from parking_occupancy.evaluate import (
    binary_metrics,
    evaluate,
    precision_recall_curve,
    temporal_metrics,
)


def test_binary_metrics_and_false_rates() -> None:
    metrics = binary_metrics([1, 1, 0, 0], [1, 0, 1, 0])
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["fp"] == 1
    assert metrics["fn"] == 1
    assert metrics["f1"] == pytest.approx(0.5)
    assert metrics["false_free_rate"] == pytest.approx(0.5)
    assert metrics["false_occupied_rate"] == pytest.approx(0.5)


def test_pr_curve_returns_bounded_ap() -> None:
    precision, recall, thresholds, ap = precision_recall_curve(
        [1, 0, 1, 0],
        [0.9, 0.7, 0.6, 0.1],
    )
    assert len(precision) == len(recall) == len(thresholds)
    assert 0.0 <= ap <= 1.0


def _temporal_rows(
    truth_states: list[int],
    prediction_states: list[int],
) -> tuple[dict, dict]:
    truth = {}
    prediction = {}
    for frame, (truth_state, prediction_state) in enumerate(
        zip(truth_states, prediction_states, strict=True)
    ):
        key = ("video", frame, "slot")
        truth[key] = {"state": truth_state}
        prediction[key] = {"state": prediction_state}
    return truth, prediction


@pytest.mark.parametrize(
    ("truth_states", "prediction_states", "expected_outcome"),
    [
        ([0, 0, 1, 1, 1], [0, 0, 1, 1, 1], "on_time"),
        ([0, 0, 0, 1, 1], [0, 1, 1, 1, 1], "early"),
        ([0, 0, 1, 1, 1, 1], [0, 0, 0, 1, 1, 1], "delayed"),
        ([0, 0, 1, 1, 1], [0, 0, 0, 0, 0], "missed"),
    ],
)
def test_public_temporal_metrics_use_canonical_signed_outcomes(
    truth_states,
    prediction_states,
    expected_outcome,
) -> None:
    truth, prediction = _temporal_rows(truth_states, prediction_states)

    metrics = temporal_metrics(
        truth,
        prediction,
        fps=2.0,
        stable_frames=2,
    )

    assert metrics["definition"].endswith("sequence_temporal_metrics")
    assert metrics["transition_outcomes"][expected_outcome] == 1
    if expected_outcome == "early":
        assert metrics["signed_transition_error_s"]["all"]["median"] < 0
        assert metrics["transition_latency_s"]["count"] == 0
        assert metrics["transition_latency_s"]["median"] is None


def test_public_temporal_metrics_do_not_cross_next_truth_event_window() -> None:
    truth, prediction = _temporal_rows(
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
    )

    metrics = temporal_metrics(
        truth,
        prediction,
        fps=2.0,
        stable_frames=2,
    )

    first_event = metrics["transition_events"][0]
    assert first_event["event_window_end_frame_exclusive"] == 4
    assert first_event["predicted_transition_frame"] is None
    assert first_event["outcome"] == "missed"


def test_evaluate_writes_report_and_plots(tmp_path) -> None:
    fieldnames = [
        "video_id",
        "frame_index",
        "timestamp_s",
        "slot_id",
        "state",
        "evidence",
    ]
    rows = [
        {
            "video_id": "video",
            "frame_index": 0,
            "timestamp_s": 0.0,
            "slot_id": "slot_001",
            "state": 1,
            "evidence": 0.9,
        },
        {
            "video_id": "video",
            "frame_index": 0,
            "timestamp_s": 0.0,
            "slot_id": "slot_002",
            "state": 0,
            "evidence": 0.1,
        },
    ]
    truth_path = tmp_path / "truth.csv"
    prediction_path = tmp_path / "predictions.csv"
    for path in (truth_path, prediction_path):
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    report = evaluate(truth_path, prediction_path, tmp_path / "evaluation", fps=10)

    assert report["classification"]["f1"] == 1.0
    assert (tmp_path / "evaluation" / "metrics.json").is_file()
    assert (tmp_path / "evaluation" / "confusion_matrix.png").is_file()
    assert (tmp_path / "evaluation" / "pr_curve.png").is_file()
    assert (tmp_path / "evaluation" / "errors.csv").is_file()
