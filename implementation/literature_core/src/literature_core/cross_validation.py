from __future__ import annotations

import statistics
from collections.abc import Mapping, Sequence
from typing import Any


METHODS = {
    "E0": "E0_yolov8_polygon",
    "E1": "E1_mobilenet",
    "E2": "E2_yolo_world_polygon",
    "E3": "E3_fusion",
}
METRICS = (
    "macro_f1",
    "occupied_recall",
    "vacant_recall",
    "false_free_rate",
    "false_occupied_rate",
)


def summarize_cross_camera_folds(
    folds: Sequence[tuple[str, Mapping[str, Any]]],
) -> dict[str, Any]:
    """Aggregate already-selected fold test results without new tuning."""

    if len(folds) < 2:
        raise ValueError("At least two folds are required")
    fold_rows: list[dict[str, Any]] = []
    for fold_name, payload in folds:
        test = payload["test"]
        development_camera = payload["split"]["development_camera"]
        test_camera = payload["split"]["test_camera"]
        known_cameras = {"pucpr", "ufpr04", "ufpr05"}
        remaining_cameras = known_cameras - {
            development_camera,
            test_camera,
        }
        train_camera = (
            payload["split"].get("train_camera")
            or (
                next(iter(remaining_cameras))
                if len(remaining_cameras) == 1
                else "not_recorded"
            )
        )
        method_metrics = {
            short_name: {
                metric: float(test[full_name][metric])
                for metric in METRICS
            }
            for short_name, full_name in METHODS.items()
        }
        baseline_f1 = method_metrics["E0"]["macro_f1"]
        best_f1 = max(
            metrics["macro_f1"] for metrics in method_metrics.values()
        )
        fold_rows.append(
            {
                "fold": fold_name,
                "train_camera": train_camera,
                "development_camera": development_camera,
                "test_camera": test_camera,
                "development_samples": int(
                    payload["split"]["development_samples"]
                ),
                "test_samples": int(payload["split"]["test_samples"]),
                "selected_parameters": dict(payload["selected_parameters"]),
                "methods": {
                    method: {
                        **metrics,
                        "macro_f1_delta_vs_E0": (
                            metrics["macro_f1"] - baseline_f1
                        ),
                        "is_fold_best_macro_f1": (
                            abs(metrics["macro_f1"] - best_f1) <= 1e-12
                        ),
                    }
                    for method, metrics in method_metrics.items()
                },
            }
        )

    aggregate: dict[str, Any] = {}
    for method in METHODS:
        aggregate[method] = {}
        for metric in METRICS:
            values = [
                row["methods"][method][metric] for row in fold_rows
            ]
            aggregate[method][metric] = {
                "mean": statistics.fmean(values),
                "population_std": statistics.pstdev(values),
                "minimum": min(values),
                "maximum": max(values),
                "values": values,
            }
        deltas = [
            row["methods"][method]["macro_f1_delta_vs_E0"]
            for row in fold_rows
        ]
        aggregate[method]["macro_f1_vs_E0"] = {
            "wins": sum(delta > 1e-12 for delta in deltas),
            "ties": sum(abs(delta) <= 1e-12 for delta in deltas),
            "losses": sum(delta < -1e-12 for delta in deltas),
            "mean_delta": statistics.fmean(deltas),
        }
        aggregate[method]["fold_best_count"] = sum(
            row["methods"][method]["is_fold_best_macro_f1"]
            for row in fold_rows
        )

    return {
        "protocol": {
            "study_type": "post_hoc_three_camera_rotation",
            "selection": "within-fold development camera only",
            "test_used_for_selection": False,
            "folds": len(fold_rows),
        },
        "folds": fold_rows,
        "aggregate": aggregate,
    }
