from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def count_regression_metrics(
    true_counts: Sequence[float],
    predicted_counts: Sequence[float],
) -> dict[str, float | int]:
    """Compute count metrics without undefined percentage errors."""

    if len(true_counts) != len(predicted_counts):
        raise ValueError("True and predicted count lengths differ")
    if not true_counts:
        raise ValueError("At least one count pair is required")
    truth = [float(value) for value in true_counts]
    predictions = [float(value) for value in predicted_counts]
    if not all(math.isfinite(value) and value >= 0 for value in truth):
        raise ValueError("True counts must be finite and non-negative")
    if not all(
        math.isfinite(value) and value >= 0 for value in predictions
    ):
        raise ValueError("Predicted counts must be finite and non-negative")

    errors = [
        prediction - actual
        for actual, prediction in zip(truth, predictions, strict=True)
    ]
    return {
        "images": len(errors),
        "mae": sum(abs(error) for error in errors) / len(errors),
        "rmse": math.sqrt(
            sum(error * error for error in errors) / len(errors)
        ),
        "mean_predicted_count": sum(predictions) / len(predictions),
        "mean_true_count": sum(truth) / len(truth),
    }


def evaluate_count_rows(
    truth_rows: Sequence[dict[str, Any]],
    prediction_rows: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Validate one-to-one membership and report overall/per-camera counts."""

    truth_by_name: dict[str, dict[str, Any]] = {}
    for row in truth_rows:
        file_name = str(row["file_name"])
        if file_name in truth_by_name:
            raise ValueError(f"Duplicate truth image: {file_name}")
        truth_by_name[file_name] = row

    prediction_by_name: dict[str, dict[str, Any]] = {}
    for row in prediction_rows:
        file_name = str(row["file_name"])
        if file_name in prediction_by_name:
            raise ValueError(f"Duplicate prediction image: {file_name}")
        prediction_by_name[file_name] = row

    missing_predictions = sorted(set(truth_by_name) - set(prediction_by_name))
    extra_predictions = sorted(set(prediction_by_name) - set(truth_by_name))
    if missing_predictions or extra_predictions:
        raise ValueError(
            "Prediction membership differs from count truth: "
            f"missing={missing_predictions}, extra={extra_predictions}"
        )

    ordered_names = sorted(truth_by_name)
    true_counts = [
        float(truth_by_name[name]["vehicle_count"]) for name in ordered_names
    ]
    predicted_counts = [
        float(prediction_by_name[name]["predicted_count"])
        for name in ordered_names
    ]
    overall = count_regression_metrics(true_counts, predicted_counts)

    camera_pairs: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for name, actual, predicted in zip(
        ordered_names,
        true_counts,
        predicted_counts,
        strict=True,
    ):
        camera_id = str(truth_by_name[name]["camera_id"])
        camera_pairs[camera_id].append((actual, predicted))
    per_camera = {
        camera_id: count_regression_metrics(
            [pair[0] for pair in pairs],
            [pair[1] for pair in pairs],
        )
        for camera_id, pairs in sorted(camera_pairs.items())
    }
    return {
        "task": "vehicle_counting",
        "metrics": overall,
        "per_camera": per_camera,
        "mape_reported": False,
        "mape_reason": (
            "MAPE is not reported by default because percentage error is "
            "undefined when a true count is zero."
        ),
        "detection_metric_warning": (
            "Count MAE/RMSE are not detector mAP, box precision, or box recall."
        ),
    }


def evaluate_count_files(
    *,
    truth_manifest: Path,
    predictions_csv: Path,
    output_path: Path,
) -> dict[str, Any]:
    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite count evaluation: {output_path}"
        )
    with truth_manifest.open("r", encoding="utf-8-sig", newline="") as handle:
        truth_rows = list(csv.DictReader(handle))
    with predictions_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        prediction_rows = list(csv.DictReader(handle))
    if not truth_rows or "vehicle_count" not in truth_rows[0]:
        raise ValueError("Truth manifest lacks vehicle_count")
    if not prediction_rows or "predicted_count" not in prediction_rows[0]:
        raise ValueError("Prediction CSV lacks predicted_count")

    report = evaluate_count_rows(truth_rows, prediction_rows)
    report["truth_manifest"] = truth_manifest.name
    report["predictions"] = predictions_csv.name
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
