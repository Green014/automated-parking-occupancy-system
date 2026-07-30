from __future__ import annotations

import pytest

from parking_occupancy.count_metrics import (
    count_regression_metrics,
    evaluate_count_rows,
)


def test_count_metrics_keep_zero_truth_defined_without_mape() -> None:
    metrics = count_regression_metrics([0, 2], [1, 1])

    assert metrics["mae"] == pytest.approx(1.0)
    assert metrics["rmse"] == pytest.approx(1.0)
    assert metrics["mean_true_count"] == pytest.approx(1.0)
    assert metrics["mean_predicted_count"] == pytest.approx(1.0)
    assert "mape" not in metrics


def test_count_evaluation_reports_per_camera() -> None:
    truth = [
        {"file_name": "60_a.jpg", "camera_id": "60", "vehicle_count": 0},
        {"file_name": "60_b.jpg", "camera_id": "60", "vehicle_count": 2},
        {"file_name": "64_a.jpg", "camera_id": "64", "vehicle_count": 4},
    ]
    predictions = [
        {"file_name": "64_a.jpg", "predicted_count": 2},
        {"file_name": "60_b.jpg", "predicted_count": 1},
        {"file_name": "60_a.jpg", "predicted_count": 1},
    ]

    result = evaluate_count_rows(truth, predictions)

    assert result["metrics"]["mae"] == pytest.approx(4 / 3)
    assert result["per_camera"]["60"]["rmse"] == pytest.approx(1.0)
    assert result["per_camera"]["64"]["mae"] == pytest.approx(2.0)
    assert result["mape_reported"] is False


def test_count_evaluation_rejects_membership_mismatch() -> None:
    truth = [{"file_name": "a.jpg", "camera_id": "60", "vehicle_count": 1}]
    predictions = [{"file_name": "b.jpg", "predicted_count": 1}]

    with pytest.raises(ValueError, match="membership differs"):
        evaluate_count_rows(truth, predictions)


def test_count_metrics_reject_negative_prediction() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        count_regression_metrics([0], [-1])
