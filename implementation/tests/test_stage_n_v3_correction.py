from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest

from parking_occupancy import stage_n_v3_correction
from parking_occupancy.stage_n_v3_correction import (
    load_saved_detections,
    per_sequence_macro,
    summed_counts,
)


def test_saved_detection_loader_uses_jsonl_without_model(tmp_path: Path) -> None:
    path = tmp_path / "detections.jsonl"
    path.write_text(
        json.dumps(
            {
                "frame": 3,
                "track_id": 7,
                "bbox_xyxy": [1, 2, 11, 12],
                "confidence": 0.8,
                "class_id": 0,
                "class_name": "motor_vehicle",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    rows = load_saved_detections(path)

    assert len(rows) == 1
    assert rows[0].frame_number == 3
    assert rows[0].xyxy == (1.0, 2.0, 11.0, 12.0)
    source = inspect.getsource(stage_n_v3_correction)
    assert "FrozenStageNTrackerAdapter" not in source
    assert "ultralytics" not in source


def test_v3_aggregation_sums_counts_and_keeps_macro_rates_separate() -> None:
    rows = {
        "small": {
            "precision": 1.0,
            "recall": 1.0,
            "AP50": 1.0,
            "AP50-95": 0.9,
            "ground_truth_boxes": 1,
            "predicted_boxes": 1,
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
        },
        "large": {
            "precision": 0.5,
            "recall": 0.25,
            "AP50": 0.2,
            "AP50-95": 0.1,
            "ground_truth_boxes": 100,
            "predicted_boxes": 50,
            "true_positives": 25,
            "false_positives": 25,
            "false_negatives": 75,
        },
    }

    counts = summed_counts(rows)
    macro = per_sequence_macro(rows)

    assert counts == {
        "ground_truth_boxes": 101,
        "predicted_boxes": 51,
        "true_positives": 26,
        "false_positives": 25,
        "false_negatives": 75,
    }
    assert macro["precision"] == pytest.approx(0.75)
    assert macro["recall"] == pytest.approx(0.625)
