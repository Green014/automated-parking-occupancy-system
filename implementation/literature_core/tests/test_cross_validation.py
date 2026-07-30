import pytest

from literature_core.cross_validation import summarize_cross_camera_folds


def _payload(test_camera: str, e0: float, e2: float) -> dict:
    metric = {
        "occupied_recall": 0.8,
        "vacant_recall": 0.9,
        "false_free_rate": 0.2,
        "false_occupied_rate": 0.1,
    }
    return {
        "split": {
            "development_camera": "dev",
            "test_camera": test_camera,
            "development_samples": 10,
            "test_samples": 12,
        },
        "selected_parameters": {"test_used_for_selection": False},
        "test": {
            "E0_yolov8_polygon": {**metric, "macro_f1": e0},
            "E1_mobilenet": {**metric, "macro_f1": 0.7},
            "E2_yolo_world_polygon": {**metric, "macro_f1": e2},
            "E3_fusion": {**metric, "macro_f1": 0.75},
        },
    }


def test_cross_camera_summary_counts_wins_and_variability() -> None:
    report = summarize_cross_camera_folds(
        [
            ("A", _payload("cam_a", 0.8, 0.9)),
            ("B", _payload("cam_b", 0.8, 0.7)),
        ]
    )
    comparison = report["aggregate"]["E2"]["macro_f1_vs_E0"]
    assert comparison == {
        "wins": 1,
        "ties": 0,
        "losses": 1,
        "mean_delta": pytest.approx(0.0),
    }
    assert report["aggregate"]["E0"]["macro_f1"]["population_std"] == 0.0


def test_cross_camera_summary_requires_multiple_folds() -> None:
    with pytest.raises(ValueError):
        summarize_cross_camera_folds([("A", _payload("cam_a", 0.8, 0.9))])
