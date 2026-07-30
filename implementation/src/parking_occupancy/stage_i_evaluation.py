from __future__ import annotations

import csv
import json
import os
from collections.abc import Callable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

from .count_metrics import count_regression_metrics, evaluate_count_rows
from .detector_comparison import (
    ComparisonDetectorAdapter,
    DetectorSpec,
    _load_ground_truth_boxes,
    _split_paths,
    load_comparison_protocol,
    pairwise_iou_xyxy,
    sha256_file,
    V2_COMPARISON_PROTOCOLS,
)
from .image_io import read_image, write_image


class StageIProtocolError(ValueError):
    """Raised when Stage I evidence conflicts with the frozen protocol."""


METHOD_IDS = ("D0", "D1", "D2")
SELECTION_ID = "D-SELECT-NDISPARK-DEV-20260727-01"
COUNT_PROTOCOL_ID = "D-COUNT-NDISPARK-TEST-20260727-01"
STAGE_I_V2_SELECTION_ID = "D-SELECT-NDISPARK-DEV-V2-20260727-01"
STAGE_I_V2_MAXDET_DECISION_ID = (
    "D-MAXDET-NDISPARK-DEV-V2-20260727-01"
)
STAGE_I_V2_COUNT_PROTOCOL_ID = (
    "D-COUNT-NDISPARK-POSTHOC-V2-20260727-01"
)
STAGE_I_V2_RECORD_ID = "D-STAGE-I-V2-RECORD-NDISPARK-20260727-01"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise StageIProtocolError(
                f"Expected object at {path}:{line_number}"
            )
        rows.append(row)
    return rows


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "name": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _validate_development_comparison(
    comparison_root: Path,
    *,
    expected_protocol_id: str = "D-COMP-NDISPARK-DEV-20260727-01",
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    comparison_path = comparison_root / "comparison.json"
    if not comparison_path.is_file():
        raise StageIProtocolError(f"Missing comparison: {comparison_path}")
    comparison = _read_json(comparison_path)
    if comparison.get("comparison_protocol_id") != expected_protocol_id:
        raise StageIProtocolError("Unexpected detector comparison ID")
    if comparison.get("dataset_role") != "consumed_development_validation":
        raise StageIProtocolError("Detector selection must use development")
    if tuple(comparison.get("models", {})) != METHOD_IDS:
        raise StageIProtocolError("Comparison must contain D0/D1/D2 in order")

    detections_by_method: dict[str, list[dict[str, Any]]] = {}
    expected_membership: list[tuple[str, int]] | None = None
    for method_id in METHOD_IDS:
        metrics = comparison["models"][method_id]
        if metrics.get("dataset_role") != (
            "consumed_development_validation"
        ):
            raise StageIProtocolError(
                f"{method_id} metrics do not have the development role"
            )
        detections_path = comparison_root / method_id / "detections.jsonl"
        rows = _read_jsonl(detections_path)
        membership = [
            (str(row["image_name"]), int(row["ground_truth_box_count"]))
            for row in rows
        ]
        if len(set(membership)) != len(membership):
            raise StageIProtocolError(
                f"{method_id} has duplicate development image rows"
            )
        if expected_membership is None:
            expected_membership = membership
        elif membership != expected_membership:
            raise StageIProtocolError(
                "D0/D1/D2 development memberships or truth counts differ"
            )
        if len(rows) != int(metrics["images"]):
            raise StageIProtocolError(
                f"{method_id} detection row count differs from metrics"
            )
        detections_by_method[method_id] = rows
    return comparison, detections_by_method


def select_detector_and_count_rule(
    *,
    comparison_root: Path,
    output_path: Path,
    candidate_thresholds: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Select D0/D1/D2 and one shared count threshold using development only."""

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage I selection: {output_path}"
        )
    comparison_root = comparison_root.resolve()
    comparison, detections_by_method = _validate_development_comparison(
        comparison_root
    )
    thresholds = (
        [round(index * 0.05, 2) for index in range(1, 20)]
        if candidate_thresholds is None
        else [float(value) for value in candidate_thresholds]
    )
    if (
        not thresholds
        or thresholds != sorted(set(thresholds))
        or any(value <= 0.001 or value > 1.0 for value in thresholds)
    ):
        raise StageIProtocolError(
            "Candidate thresholds must be unique, sorted, and in (0.001, 1]"
        )

    detector_ranking = sorted(
        METHOD_IDS,
        key=lambda method_id: (
            -float(comparison["models"][method_id]["map_50_95"]),
            -float(comparison["models"][method_id]["recall"]),
            -float(
                comparison["models"][method_id][
                    "framework_pipeline_fps"
                ]
            ),
            method_id,
        ),
    )
    threshold_rows = []
    for threshold in thresholds:
        per_model: dict[str, Any] = {}
        for method_id in METHOD_IDS:
            rows = detections_by_method[method_id]
            truth = [
                int(row["ground_truth_box_count"])
                for row in rows
            ]
            predicted = [
                sum(
                    float(detection["confidence"]) >= threshold
                    for detection in row["detections"]
                )
                for row in rows
            ]
            per_model[method_id] = count_regression_metrics(
                truth,
                predicted,
            )
        model_maes = [
            float(per_model[method_id]["mae"]) for method_id in METHOD_IDS
        ]
        threshold_rows.append(
            {
                "confidence": threshold,
                "aggregate_mean_mae": sum(model_maes) / len(model_maes),
                "worst_model_mae": max(model_maes),
                "per_model": per_model,
            }
        )
    selected_threshold = min(
        threshold_rows,
        key=lambda row: (
            float(row["aggregate_mean_mae"]),
            float(row["worst_model_mae"]),
            -float(row["confidence"]),
        ),
    )

    comparison_path = comparison_root / "comparison.json"
    source_artifacts = {
        "comparison": _artifact(comparison_path),
        "detections": {
            method_id: _artifact(
                comparison_root / method_id / "detections.jsonl"
            )
            for method_id in METHOD_IDS
        },
    }
    selected_method = detector_ranking[0]
    report = {
        "schema_version": 1,
        "selection_id": SELECTION_ID,
        "status": "selected_on_consumed_development_before_count_test",
        "development_role": "consumed_development_validation",
        "test_counts_read": False,
        "test_predictions_run": False,
        "detector_selection": {
            "selected_method_id": selected_method,
            "selected_name": comparison["models"][selected_method]["name"],
            "primary_order": ["map_50_95", "recall"],
            "deployment_constraint_checked": (
                "framework_pipeline_fps_recorded"
            ),
            "ranking": detector_ranking,
            "metrics": {
                method_id: {
                    "map_50_95": comparison["models"][method_id][
                        "map_50_95"
                    ],
                    "recall": comparison["models"][method_id]["recall"],
                    "framework_pipeline_fps": comparison["models"][method_id][
                        "framework_pipeline_fps"
                    ],
                }
                for method_id in METHOD_IDS
            },
        },
        "shared_count_rule": {
            "rule": "count canonical NMS detections at confidence >= threshold",
            "selected_confidence": selected_threshold["confidence"],
            "candidate_grid": thresholds,
            "objective": (
                "minimize mean development count MAE across D0/D1/D2"
            ),
            "tie_breaks": [
                "lower worst-model development count MAE",
                "higher confidence threshold",
            ],
            "selected_development_result": selected_threshold,
            "all_development_results": threshold_rows,
            "applies_identically_to": list(METHOD_IDS),
        },
        "source_artifacts": source_artifacts,
        "negative_results_retained": True,
        "no_claim": (
            "Detector box metrics and count metrics do not establish slot "
            "occupancy Macro F1 improvement."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def select_stage_i_v2_operating_points(
    *,
    comparison_root: Path,
    comparison_config: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Calibrate corrected Stage I operating points using development only."""

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage I-v2 selection: {output_path}"
        )
    comparison_root = comparison_root.resolve()
    comparison_config = comparison_config.resolve()
    protocol, _ = load_comparison_protocol(comparison_config)
    protocol_id = str(protocol["comparison_protocol_id"])
    if protocol_id not in V2_COMPARISON_PROTOCOLS:
        raise StageIProtocolError(
            "Stage I-v2 calibration requires a corrected v2 protocol"
        )
    comparison, detections_by_method = _validate_development_comparison(
        comparison_root,
        expected_protocol_id=protocol_id,
    )
    for method_id in METHOD_IDS:
        common = comparison["models"][method_id].get("common_inference", {})
        if common.get("agnostic_nms") is not True:
            raise StageIProtocolError(
                f"{method_id} result does not record agnostic_nms=true"
            )

    thresholds = [
        float(value)
        for value in protocol["threshold_evaluation"][
            "candidate_thresholds"
        ]
    ]
    if (
        not thresholds
        or thresholds != sorted(set(thresholds))
        or any(value <= 0.001 or value > 1.0 for value in thresholds)
    ):
        raise StageIProtocolError(
            "Frozen v2 candidate thresholds must be unique, sorted, "
            "and in (0.001, 1]"
        )

    threshold_rows = []
    for threshold in thresholds:
        per_model: dict[str, Any] = {}
        for method_id in METHOD_IDS:
            rows = detections_by_method[method_id]
            truth = [
                int(row["ground_truth_box_count"])
                for row in rows
            ]
            predicted = [
                sum(
                    float(detection["confidence"]) >= threshold
                    for detection in row["detections"]
                )
                for row in rows
            ]
            metrics = count_regression_metrics(truth, predicted)
            per_model[method_id] = {
                "mae": metrics["mae"],
                "rmse": metrics["rmse"],
                "mean_predicted_count": metrics["mean_predicted_count"],
                "mean_true_count": metrics["mean_true_count"],
            }
        threshold_rows.append(
            {
                "confidence": threshold,
                "mean_mae": float(
                    np.mean(
                        [
                            per_model[method_id]["mae"]
                            for method_id in METHOD_IDS
                        ]
                    )
                ),
                "mean_rmse": float(
                    np.mean(
                        [
                            per_model[method_id]["rmse"]
                            for method_id in METHOD_IDS
                        ]
                    )
                ),
                "per_model": per_model,
            }
        )

    common_selected = min(
        threshold_rows,
        key=lambda row: (
            float(row["mean_mae"]),
            float(row["mean_rmse"]),
            -float(row["confidence"]),
        ),
    )
    per_model_selected = {
        method_id: min(
            threshold_rows,
            key=lambda row: (
                float(row["per_model"][method_id]["mae"]),
                float(row["per_model"][method_id]["rmse"]),
                -float(row["confidence"]),
            ),
        )
        for method_id in METHOD_IDS
    }
    detector_ranking = sorted(
        METHOD_IDS,
        key=lambda method_id: (
            -float(comparison["models"][method_id]["map_50_95"]),
            -float(comparison["models"][method_id]["recall"]),
            -float(
                comparison["models"][method_id][
                    "framework_pipeline_fps"
                ]
            ),
            method_id,
        ),
    )

    comparison_path = comparison_root / "comparison.json"
    report = {
        "schema_version": 2,
        "selection_id": STAGE_I_V2_SELECTION_ID,
        "comparison_protocol_id": protocol_id,
        "status": "corrected_development_calibration_complete",
        "selected_at": datetime.now().astimezone().isoformat(),
        "data_role": "consumed_development_validation",
        "test_labels_read": False,
        "test_predictions_run": False,
        "detector_ranking": {
            "primary_order": ["map_50_95", "recall"],
            "ranking": detector_ranking,
            "selected_before_posthoc_test": detector_ranking[0],
        },
        "common_threshold_diagnostic": {
            "role": "controlled_sensitivity_only",
            "deployment_selection_allowed": False,
            "objective": "lowest mean development count MAE",
            "tie_breaks": ["lower mean RMSE", "higher threshold"],
            "selected_confidence": common_selected["confidence"],
            "selected_development_result": common_selected,
        },
        "per_model_development_calibration": {
            "role": "primary_deployment_operating_point",
            "objective": "lowest per-model development count MAE",
            "tie_breaks": ["lower RMSE", "higher threshold"],
            "same_membership_and_grid": True,
            "selected": {
                method_id: {
                    "confidence": per_model_selected[method_id][
                        "confidence"
                    ],
                    "development_metrics": per_model_selected[method_id][
                        "per_model"
                    ][method_id],
                }
                for method_id in METHOD_IDS
            },
        },
        "candidate_grid": thresholds,
        "all_candidate_results": threshold_rows,
        "source_artifacts": {
            "comparison_config": _artifact(comparison_config),
            "comparison": _artifact(comparison_path),
            "detections": {
                method_id: _artifact(
                    comparison_root / method_id / "detections.jsonl"
                )
                for method_id in METHOD_IDS
            },
        },
        "negative_results_retained": True,
        "no_claim": (
            "Vehicle count MAE and detector box metrics are not "
            "parking-slot occupancy accuracy."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def analyze_stage_i_v2_max_det_sensitivity(
    *,
    max_det_300_root: Path,
    max_det_1000_root: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Compare the two frozen development-only max_det sensitivity arms."""

    if output_path.exists():
        raise FileExistsError(
            f"Refusing to overwrite max_det decision: {output_path}"
        )
    roots = {
        300: max_det_300_root.resolve(),
        1000: max_det_1000_root.resolve(),
    }
    expected_ids = {
        300: "D-COMP-NDISPARK-DEV-V2-MAXDET300-20260727-01",
        1000: "D-COMP-NDISPARK-DEV-V2-MAXDET1000-20260727-01",
    }
    comparisons: dict[int, dict[str, Any]] = {}
    for max_det, root in roots.items():
        comparison_path = root / "comparison.json"
        if not comparison_path.is_file():
            raise StageIProtocolError(
                f"Missing max_det={max_det} comparison"
            )
        comparison = _read_json(comparison_path)
        if comparison.get("comparison_protocol_id") != expected_ids[max_det]:
            raise StageIProtocolError(
                f"Unexpected max_det={max_det} comparison protocol"
            )
        if comparison.get("dataset_role") != (
            "consumed_development_validation"
        ):
            raise StageIProtocolError(
                "max_det sensitivity must use development validation"
            )
        if tuple(comparison.get("models", {})) != METHOD_IDS:
            raise StageIProtocolError(
                "max_det sensitivity must contain D0/D1/D2"
            )
        for method_id in METHOD_IDS:
            model = comparison["models"][method_id]
            common = model.get("common_inference", {})
            if (
                common.get("agnostic_nms") is not True
                or int(common.get("max_detections", -1)) != max_det
            ):
                raise StageIProtocolError(
                    f"{method_id} max_det={max_det} settings mismatch"
                )
            required = {
                "precision",
                "recall",
                "map_50",
                "map_50_95",
                "images_reaching_max_det",
                "framework_pipeline_latency_ms_per_image",
                "framework_pipeline_fps",
                "peak_cuda_memory_allocated_bytes",
                "peak_cuda_memory_reserved_bytes",
            }
            if not required.issubset(model):
                raise StageIProtocolError(
                    f"{method_id} max_det={max_det} evidence is incomplete"
                )
        comparisons[max_det] = comparison

    def ranking(comparison: dict[str, Any]) -> list[str]:
        return sorted(
            METHOD_IDS,
            key=lambda method_id: (
                -float(comparison["models"][method_id]["map_50_95"]),
                -float(comparison["models"][method_id]["recall"]),
                method_id,
            ),
        )

    rankings = {
        max_det: ranking(comparison)
        for max_det, comparison in comparisons.items()
    }
    ranking_unchanged = rankings[300] == rankings[1000]
    fields = (
        "precision",
        "recall",
        "map_50",
        "map_50_95",
        "images_reaching_max_det",
        "framework_pipeline_latency_ms_per_image",
        "framework_pipeline_fps",
        "peak_cuda_memory_allocated_bytes",
        "peak_cuda_memory_reserved_bytes",
    )
    per_model = {
        method_id: {
            str(max_det): {
                field: comparisons[max_det]["models"][method_id][field]
                for field in fields
            }
            for max_det in (300, 1000)
        }
        for method_id in METHOD_IDS
    }
    report = {
        "schema_version": 1,
        "decision_id": STAGE_I_V2_MAXDET_DECISION_ID,
        "status": (
            "max_det_300_retained"
            if ranking_unchanged
            else "blocked_pending_explicit_methodology_review"
        ),
        "decided_at": datetime.now().astimezone().isoformat(),
        "data_role": "consumed_development_validation",
        "test_labels_read": False,
        "rankings": {str(key): value for key, value in rankings.items()},
        "ranking_unchanged": ranking_unchanged,
        "final_max_detections": 300 if ranking_unchanged else None,
        "decision_reason": (
            "The development ranking is unchanged; retain the smaller "
            "max_det=300 setting and record saturation as a limitation."
            if ranking_unchanged
            else "The development ranking changed; no max_det setting is "
            "silently substituted and downstream execution is blocked "
            "until the methodology choice is explicitly frozen."
        ),
        "per_model": per_model,
        "source_artifacts": {
            str(max_det): _artifact(root / "comparison.json")
            for max_det, root in roots.items()
        },
        "negative_results_retained": True,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def load_stage_i_v2_posthoc_count_protocol(
    config_path: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("count_protocol_id") != STAGE_I_V2_COUNT_PROTOCOL_ID:
        raise StageIProtocolError(
            "Unexpected Stage I-v2 post-hoc count protocol ID"
        )
    if payload.get("status") != (
        "frozen_before_posthoc_corrected_predictions"
    ):
        raise StageIProtocolError("Stage I-v2 post-hoc protocol is not frozen")
    if payload.get("result_boundary", {}).get("role") != (
        "consumed_test_posthoc_sensitivity"
    ):
        raise StageIProtocolError("Stage I-v2 test role is not post-hoc")
    if tuple(payload.get("models", {})) != METHOD_IDS:
        raise StageIProtocolError(
            "Stage I-v2 post-hoc protocol must contain D0/D1/D2"
        )
    common = payload.get("common_inference", {})
    if (
        common.get("agnostic_nms") is not True
        or int(common.get("max_detections", -1)) != 300
        or float(common.get("confidence_floor", -1.0)) != 0.001
    ):
        raise StageIProtocolError(
            "Stage I-v2 post-hoc inference settings are not corrected/frozen"
        )
    points = payload.get("operating_points", {})
    common_rule = points.get("common_threshold_diagnostic", {})
    if (
        common_rule.get("applies_identically_to") != list(METHOD_IDS)
        or common_rule.get("deployment_selection_allowed") is not False
    ):
        raise StageIProtocolError(
            "Common threshold must remain diagnostic only"
        )
    calibrated = points.get(
        "per_model_development_calibration",
        {},
    )
    if calibrated.get("selection_data_role") != (
        "consumed_development_validation"
    ):
        raise StageIProtocolError(
            "Per-model thresholds were not selected on development"
        )
    if tuple(calibrated.get("thresholds", {})) != METHOD_IDS:
        raise StageIProtocolError(
            "Per-model thresholds must contain D0/D1/D2 in order"
        )
    if points.get("test_labels_used_for_selection") is not False:
        raise StageIProtocolError(
            "Post-hoc test labels must not select operating points"
        )
    return payload


def load_count_test_protocol(config_path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("count_protocol_id") != COUNT_PROTOCOL_ID:
        raise StageIProtocolError("Unexpected Stage I count protocol ID")
    if payload.get("status") != "frozen_before_count_test_predictions":
        raise StageIProtocolError("Count protocol is not frozen")
    if tuple(payload.get("models", {})) != METHOD_IDS:
        raise StageIProtocolError("Count protocol must contain D0/D1/D2")
    if payload.get("selected_detector", {}).get("method_id") != "D1":
        raise StageIProtocolError("Frozen selected detector must be D1")
    rule = payload.get("shared_count_rule", {})
    if rule.get("applies_identically_to") != list(METHOD_IDS):
        raise StageIProtocolError("Shared count rule must apply to D0/D1/D2")
    if rule.get("selection_data_role") != (
        "consumed_development_validation"
    ):
        raise StageIProtocolError("Count threshold was not selected on dev")
    return payload


def _verify_bound_artifact(
    *,
    path: Path,
    binding: dict[str, Any],
    label: str,
) -> None:
    if not path.is_file():
        raise StageIProtocolError(f"Missing {label}: {path}")
    if path.name != str(binding["name"]):
        raise StageIProtocolError(f"{label} filename mismatch")
    if path.stat().st_size != int(binding["bytes"]):
        raise StageIProtocolError(f"{label} byte size mismatch")
    if sha256_file(path) != str(binding["sha256"]):
        raise StageIProtocolError(f"{label} SHA-256 mismatch")


def _load_test_truth(
    *,
    truth_manifest: Path,
    test_images_root: Path,
    expected_rows: int,
) -> tuple[list[dict[str, Any]], list[Path]]:
    with truth_manifest.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != expected_rows:
        raise StageIProtocolError("Count-test manifest row count mismatch")
    required = {
        "file_name",
        "camera_id",
        "vehicle_count",
        "truth_type",
        "role",
        "sha256",
        "width",
        "height",
    }
    if not rows or not required.issubset(rows[0]):
        raise StageIProtocolError("Count-test manifest schema mismatch")
    names = [str(row["file_name"]) for row in rows]
    if len(set(names)) != len(names):
        raise StageIProtocolError("Duplicate count-test image")

    image_paths = []
    for row in rows:
        if (
            row["role"] != "count_only_test"
            or row["truth_type"] != "vehicle_count"
        ):
            raise StageIProtocolError("Unexpected count-test data role")
        image_path = test_images_root / row["file_name"]
        if not image_path.is_file():
            raise StageIProtocolError(
                f"Missing count-test image: {image_path}"
            )
        if sha256_file(image_path) != row["sha256"]:
            raise StageIProtocolError(
                f"Count-test image SHA-256 mismatch: {image_path.name}"
            )
        try:
            image = read_image(image_path)
        except RuntimeError as exc:
            raise StageIProtocolError(
                f"Unreadable count-test image: {image_path.name}"
            ) from exc
        if (
            image.shape[1] != int(row["width"])
            or image.shape[0] != int(row["height"])
        ):
            raise StageIProtocolError(
                f"Count-test dimensions mismatch: {image_path.name}"
            )
        image_paths.append(image_path)
    return rows, image_paths


def count_test_preflight(
    *,
    config_path: Path,
    selection_record: Path,
    comparison_path: Path,
    comparison_config: Path,
    truth_manifest: Path,
    test_images_root: Path,
    weight_paths: dict[str, Path],
) -> dict[str, Any]:
    protocol = load_count_test_protocol(config_path)
    _verify_bound_artifact(
        path=selection_record,
        binding=protocol["evidence"]["selection_record"],
        label="selection record",
    )
    selection = _read_json(selection_record)
    if selection.get("selection_id") != SELECTION_ID:
        raise StageIProtocolError("Unexpected selection evidence ID")
    frozen_threshold = float(
        protocol["shared_count_rule"]["confidence_threshold"]
    )
    if float(
        selection["shared_count_rule"]["selected_confidence"]
    ) != frozen_threshold:
        raise StageIProtocolError("Frozen count threshold differs from evidence")
    if selection["detector_selection"]["selected_method_id"] != "D1":
        raise StageIProtocolError("Selection evidence does not select D1")

    _verify_bound_artifact(
        path=comparison_path,
        binding=protocol["evidence"]["comparison"],
        label="comparison",
    )
    _verify_bound_artifact(
        path=comparison_config,
        binding=protocol["evidence"]["comparison_config"],
        label="comparison config",
    )
    comparison_protocol, specs = load_comparison_protocol(comparison_config)
    if comparison_protocol["comparison_protocol_id"] != (
        protocol["comparison_protocol_id"]
    ):
        raise StageIProtocolError("Comparison protocol binding mismatch")

    _verify_bound_artifact(
        path=truth_manifest,
        binding=protocol["count_test"]["manifest"],
        label="count-test manifest",
    )
    truth_rows, image_paths = _load_test_truth(
        truth_manifest=truth_manifest,
        test_images_root=test_images_root,
        expected_rows=int(protocol["count_test"]["images"]),
    )

    model_checks = {}
    for method_id in METHOD_IDS:
        weights = weight_paths[method_id]
        expected_hash = str(protocol["models"][method_id]["weights_sha256"])
        actual_hash = sha256_file(weights) if weights.is_file() else None
        if actual_hash != expected_hash:
            raise StageIProtocolError(
                f"{method_id} count-test weights SHA-256 mismatch"
            )
        model_checks[method_id] = {
            "name": specs[method_id].name,
            "weights_name": weights.name,
            "weights_sha256": actual_hash,
            "ready": True,
        }
    return {
        "schema_version": 1,
        "count_protocol_id": protocol["count_protocol_id"],
        "selection_id": selection["selection_id"],
        "selected_detector": "D1",
        "shared_confidence_threshold": frozen_threshold,
        "count_test": {
            "role": "count_only_test",
            "images": len(image_paths),
            "cameras": sorted({str(row["camera_id"]) for row in truth_rows}),
            "truth_type": "vehicle_count",
            "box_ground_truth_available": False,
        },
        "models": model_checks,
        "predictions_run": False,
        "execution_gate": "open",
    }


def stage_i_v2_posthoc_count_preflight(
    *,
    config_path: Path,
    operating_points_record: Path,
    max_det_decision: Path,
    comparison_path: Path,
    comparison_config: Path,
    truth_manifest: Path,
    test_images_root: Path,
    weight_paths: dict[str, Path],
) -> dict[str, Any]:
    """Verify all corrected settings before the one post-hoc prediction pass."""

    protocol = load_stage_i_v2_posthoc_count_protocol(config_path)
    for label, path, binding_key in (
        (
            "operating-points record",
            operating_points_record,
            "operating_points",
        ),
        ("max_det decision", max_det_decision, "max_det_decision"),
        ("comparison", comparison_path, "comparison"),
        ("comparison config", comparison_config, "comparison_config"),
    ):
        _verify_bound_artifact(
            path=path,
            binding=protocol["evidence"][binding_key],
            label=label,
        )

    operating_points = _read_json(operating_points_record)
    if operating_points.get("selection_id") != (
        STAGE_I_V2_SELECTION_ID
    ):
        raise StageIProtocolError("Unexpected v2 operating-points evidence")
    if operating_points.get("test_labels_read") is not False:
        raise StageIProtocolError(
            "Operating points were not frozen before reading post-hoc labels"
        )
    if operating_points["detector_ranking"][
        "selected_before_posthoc_test"
    ] != "D1":
        raise StageIProtocolError(
            "Corrected development evidence does not select D1"
        )
    expected_common = float(
        protocol["operating_points"]["common_threshold_diagnostic"][
            "confidence"
        ]
    )
    if float(
        operating_points["common_threshold_diagnostic"][
            "selected_confidence"
        ]
    ) != expected_common:
        raise StageIProtocolError("Common diagnostic threshold mismatch")
    expected_thresholds = {
        method_id: float(value)
        for method_id, value in protocol["operating_points"][
            "per_model_development_calibration"
        ]["thresholds"].items()
    }
    observed_thresholds = {
        method_id: float(
            operating_points["per_model_development_calibration"][
                "selected"
            ][method_id]["confidence"]
        )
        for method_id in METHOD_IDS
    }
    if observed_thresholds != expected_thresholds:
        raise StageIProtocolError(
            "Per-model thresholds differ from development evidence"
        )

    decision = _read_json(max_det_decision)
    if (
        decision.get("decision_id") != STAGE_I_V2_MAXDET_DECISION_ID
        or decision.get("ranking_unchanged") is not True
        or int(decision.get("final_max_detections", -1)) != 300
        or decision.get("test_labels_read") is not False
    ):
        raise StageIProtocolError(
            "max_det=300 was not frozen entirely on development"
        )

    comparison_protocol, specs = load_comparison_protocol(comparison_config)
    if comparison_protocol["comparison_protocol_id"] != (
        protocol["comparison_protocol_id"]
    ):
        raise StageIProtocolError("Corrected comparison protocol mismatch")
    comparison = _read_json(comparison_path)
    if comparison.get("comparison_protocol_id") != (
        protocol["comparison_protocol_id"]
    ):
        raise StageIProtocolError("Corrected comparison evidence mismatch")

    _verify_bound_artifact(
        path=truth_manifest,
        binding=protocol["count_data"]["manifest"],
        label="consumed count-truth manifest",
    )
    truth_rows, image_paths = _load_test_truth(
        truth_manifest=truth_manifest,
        test_images_root=test_images_root,
        expected_rows=int(protocol["count_data"]["images"]),
    )

    model_checks = {}
    for method_id in METHOD_IDS:
        weights = weight_paths[method_id]
        expected_hash = str(protocol["models"][method_id]["weights_sha256"])
        actual_hash = sha256_file(weights) if weights.is_file() else None
        if actual_hash != expected_hash:
            raise StageIProtocolError(
                f"{method_id} post-hoc weights SHA-256 mismatch"
            )
        model_checks[method_id] = {
            "name": specs[method_id].name,
            "weights_name": weights.name,
            "weights_sha256": actual_hash,
            "ready": True,
        }
    return {
        "schema_version": 2,
        "count_protocol_id": protocol["count_protocol_id"],
        "selection_id": operating_points["selection_id"],
        "max_det_decision_id": decision["decision_id"],
        "dataset_role": "consumed_test_posthoc_sensitivity",
        "settings_frozen_before_this_execution": True,
        "test_labels_used_for_selection": False,
        "test_truth_manifest_verified": True,
        "count_data": {
            "images": len(image_paths),
            "cameras": sorted({str(row["camera_id"]) for row in truth_rows}),
            "truth_type": "vehicle_count",
            "vehicle_box_ground_truth_available": False,
        },
        "common_diagnostic_threshold": expected_common,
        "per_model_thresholds": expected_thresholds,
        "models": model_checks,
        "predictions_run": False,
        "execution_gate": "open",
    }


def run_count_test(
    *,
    config_path: Path,
    selection_record: Path,
    comparison_path: Path,
    comparison_config: Path,
    truth_manifest: Path,
    test_images_root: Path,
    output_root: Path,
    weight_paths: dict[str, Path],
    device: str,
    adapter_factory: Callable[
        [DetectorSpec, Path, dict[str, Any], str],
        ComparisonDetectorAdapter,
    ]
    | None = None,
) -> dict[str, Any]:
    preflight = count_test_preflight(
        config_path=config_path,
        selection_record=selection_record,
        comparison_path=comparison_path,
        comparison_config=comparison_config,
        truth_manifest=truth_manifest,
        test_images_root=test_images_root,
        weight_paths=weight_paths,
    )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite count-test output: {output_root}"
        )
    protocol = load_count_test_protocol(config_path)
    _, specs = load_comparison_protocol(comparison_config)
    with truth_manifest.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        truth_rows = list(csv.DictReader(handle))
    image_paths = [
        test_images_root / str(row["file_name"]) for row in truth_rows
    ]
    common = dict(protocol["common_inference"])
    common["confidence_floor"] = float(
        protocol["shared_count_rule"]["confidence_threshold"]
    )
    factory = adapter_factory or (
        lambda spec, weights, settings, requested_device: (
            ComparisonDetectorAdapter(
                spec=spec,
                weights_path=weights,
                common=settings,
                device=requested_device,
            )
        )
    )

    output_root.mkdir(parents=True)
    ultralytics_config = output_root / "_ultralytics_config"
    ultralytics_config.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config.resolve())
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )
    model_reports = {}
    runtime_rows = []
    for method_id in METHOD_IDS:
        method_root = output_root / method_id
        method_root.mkdir()
        adapter = factory(
            specs[method_id],
            weight_paths[method_id],
            common,
            device,
        )
        results = adapter.predict_images(image_paths)
        if len(results) != len(image_paths):
            raise RuntimeError(
                f"{method_id} returned an unexpected count-test result count"
            )
        prediction_rows = []
        detection_lines = []
        speed_totals: dict[str, float] = {}
        for row, result in zip(truth_rows, results, strict=True):
            detections = adapter._canonicalize_result(result)
            prediction_rows.append(
                {
                    "file_name": row["file_name"],
                    "camera_id": row["camera_id"],
                    "predicted_count": len(detections),
                }
            )
            detection_lines.append(
                json.dumps(
                    {
                        "image_name": row["file_name"],
                        "camera_id": row["camera_id"],
                        "predicted_count": len(detections),
                        "detections": [
                            {
                                "bbox_xyxy": list(detection.bbox),
                                "confidence": detection.confidence,
                                "class_id": 0,
                                "class_name": "vehicle",
                            }
                            for detection in detections
                        ],
                    }
                )
            )
            for key, value in result.speed.items():
                speed_totals[str(key)] = (
                    speed_totals.get(str(key), 0.0) + float(value)
                )
        prediction_path = method_root / "count_predictions.csv"
        with prediction_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file_name",
                    "camera_id",
                    "predicted_count",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(prediction_rows)
        (method_root / "detections.jsonl").write_text(
            "\n".join(detection_lines) + "\n",
            encoding="utf-8",
        )
        report = evaluate_count_rows(truth_rows, prediction_rows)
        speed = {
            key: value / len(image_paths)
            for key, value in speed_totals.items()
        }
        latency = sum(
            speed.get(key, 0.0)
            for key in ("preprocess", "inference", "postprocess")
        )
        report.update(
            {
                "method_id": method_id,
                "name": specs[method_id].name,
                "dataset_role": "count_only_test",
                "confidence_threshold": common["confidence_floor"],
                "box_ground_truth_available": False,
                "box_metrics_reported": False,
                "speed_ms_per_image": speed,
                "framework_pipeline_latency_ms_per_image": latency,
                "framework_pipeline_fps": (
                    1000.0 / latency if latency > 0 else None
                ),
                "runtime_metadata": adapter.model_metadata(),
            }
        )
        (method_root / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        model_reports[method_id] = report
        runtime_rows.append(
            {
                "method_id": method_id,
                "latency_ms_per_image": latency,
                "fps": report["framework_pipeline_fps"],
            }
        )

    comparison = {
        "schema_version": 1,
        "count_protocol_id": protocol["count_protocol_id"],
        "selection_id": protocol["selection_id"],
        "dataset_role": "count_only_test",
        "truth_type": "vehicle_count",
        "shared_confidence_threshold": common["confidence_floor"],
        "selected_detector_before_test": "D1",
        "models": model_reports,
        "box_metrics_reported": False,
        "test_used_for_retuning": False,
        "negative_results_retained": True,
    }
    (output_root / "count_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "count_runtime_table.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(runtime_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(runtime_rows)
    return comparison


def run_stage_i_v2_posthoc_count_sensitivity(
    *,
    config_path: Path,
    operating_points_record: Path,
    max_det_decision: Path,
    comparison_path: Path,
    comparison_config: Path,
    truth_manifest: Path,
    test_images_root: Path,
    output_root: Path,
    weight_paths: dict[str, Path],
    device: str,
    adapter_factory: Callable[
        [DetectorSpec, Path, dict[str, Any], str],
        ComparisonDetectorAdapter,
    ]
    | None = None,
) -> dict[str, Any]:
    """Run one consumed-test sensitivity under settings frozen on development."""

    preflight = stage_i_v2_posthoc_count_preflight(
        config_path=config_path,
        operating_points_record=operating_points_record,
        max_det_decision=max_det_decision,
        comparison_path=comparison_path,
        comparison_config=comparison_config,
        truth_manifest=truth_manifest,
        test_images_root=test_images_root,
        weight_paths=weight_paths,
    )
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite post-hoc output: {output_root}"
        )
    protocol = load_stage_i_v2_posthoc_count_protocol(config_path)
    _, specs = load_comparison_protocol(comparison_config)
    with truth_manifest.open(
        "r",
        encoding="utf-8-sig",
        newline="",
    ) as handle:
        truth_rows = list(csv.DictReader(handle))
    image_paths = [
        test_images_root / str(row["file_name"]) for row in truth_rows
    ]
    common = dict(protocol["common_inference"])
    common_threshold = float(
        protocol["operating_points"]["common_threshold_diagnostic"][
            "confidence"
        ]
    )
    calibrated_thresholds = {
        method_id: float(value)
        for method_id, value in protocol["operating_points"][
            "per_model_development_calibration"
        ]["thresholds"].items()
    }
    factory = adapter_factory or (
        lambda spec, weights, settings, requested_device: (
            ComparisonDetectorAdapter(
                spec=spec,
                weights_path=weights,
                common=settings,
                device=requested_device,
            )
        )
    )

    output_root.mkdir(parents=True)
    ultralytics_config = output_root / "_ultralytics_config"
    ultralytics_config.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config.resolve())
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )

    model_reports = {}
    runtime_rows = []
    for method_id in METHOD_IDS:
        method_root = output_root / method_id
        method_root.mkdir()
        adapter = factory(
            specs[method_id],
            weight_paths[method_id],
            common,
            device,
        )
        results = adapter.predict_images(image_paths)
        if len(results) != len(image_paths):
            raise RuntimeError(
                f"{method_id} returned an unexpected post-hoc result count"
            )

        common_rows = []
        calibrated_rows = []
        export_rows = []
        detection_lines = []
        speed_totals: dict[str, float] = {}
        calibrated_threshold = calibrated_thresholds[method_id]
        for row, result in zip(truth_rows, results, strict=True):
            detections = adapter._canonicalize_result(result)
            common_count = sum(
                detection.confidence >= common_threshold
                for detection in detections
            )
            calibrated_count = sum(
                detection.confidence >= calibrated_threshold
                for detection in detections
            )
            base = {
                "file_name": row["file_name"],
                "camera_id": row["camera_id"],
            }
            common_rows.append(
                {**base, "predicted_count": common_count}
            )
            calibrated_rows.append(
                {**base, "predicted_count": calibrated_count}
            )
            export_rows.append(
                {
                    **base,
                    "predicted_count_common": common_count,
                    "predicted_count_calibrated": calibrated_count,
                }
            )
            detection_lines.append(
                json.dumps(
                    {
                        "image_name": row["file_name"],
                        "camera_id": row["camera_id"],
                        "confidence_floor": common["confidence_floor"],
                        "common_threshold": common_threshold,
                        "per_model_threshold": calibrated_threshold,
                        "predicted_count_common": common_count,
                        "predicted_count_calibrated": calibrated_count,
                        "detections": [
                            {
                                "bbox_xyxy": list(detection.bbox),
                                "confidence": detection.confidence,
                                "class_id": 0,
                                "class_name": "vehicle",
                            }
                            for detection in detections
                        ],
                    }
                )
            )
            for key, value in result.speed.items():
                speed_totals[str(key)] = (
                    speed_totals.get(str(key), 0.0) + float(value)
                )

        prediction_path = method_root / "count_predictions.csv"
        with prediction_path.open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "file_name",
                    "camera_id",
                    "predicted_count_common",
                    "predicted_count_calibrated",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerows(export_rows)
        (method_root / "detections.jsonl").write_text(
            "\n".join(detection_lines) + "\n",
            encoding="utf-8",
        )

        speed = {
            key: value / len(image_paths)
            for key, value in speed_totals.items()
        }
        latency = sum(
            speed.get(key, 0.0)
            for key in ("preprocess", "inference", "postprocess")
        )
        report = {
            "method_id": method_id,
            "name": specs[method_id].name,
            "dataset_role": "consumed_test_posthoc_sensitivity",
            "truth_type": "vehicle_count",
            "vehicle_box_ground_truth_available": False,
            "box_metrics_reported": False,
            "confidence_floor": common["confidence_floor"],
            "agnostic_nms": common["agnostic_nms"],
            "max_detections": common["max_detections"],
            "regimes": {
                "common_threshold_diagnostic": {
                    "role": "controlled_sensitivity_only",
                    "confidence_threshold": common_threshold,
                    **evaluate_count_rows(truth_rows, common_rows),
                },
                "per_model_development_calibration": {
                    "role": "primary_deployment_operating_point",
                    "confidence_threshold": calibrated_threshold,
                    **evaluate_count_rows(truth_rows, calibrated_rows),
                },
            },
            "speed_ms_per_image": speed,
            "framework_pipeline_latency_ms_per_image": latency,
            "framework_pipeline_fps": (
                1000.0 / latency if latency > 0 else None
            ),
            "runtime_metadata": adapter.model_metadata(),
        }
        (method_root / "metrics.json").write_text(
            json.dumps(report, indent=2) + "\n",
            encoding="utf-8",
        )
        model_reports[method_id] = report
        runtime_rows.append(
            {
                "method_id": method_id,
                "latency_ms_per_image": latency,
                "fps": report["framework_pipeline_fps"],
            }
        )
        if hasattr(adapter, "release"):
            adapter.release()

    comparison = {
        "schema_version": 2,
        "count_protocol_id": protocol["count_protocol_id"],
        "selection_id": protocol["selection_id"],
        "dataset_role": "consumed_test_posthoc_sensitivity",
        "truth_type": "vehicle_count",
        "selected_detector_before_posthoc": "D1",
        "detector_reselected_after_posthoc": False,
        "test_used_for_retuning": False,
        "settings_frozen_before_execution": True,
        "models": model_reports,
        "box_metrics_reported": False,
        "negative_results_retained": True,
        "no_claim": (
            "Vehicle count sensitivity is not parking-slot occupancy "
            "accuracy."
        ),
    }
    (output_root / "count_comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "count_runtime_table.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(runtime_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(runtime_rows)
    return comparison


def classify_detection_errors(
    *,
    ground_truth_boxes: np.ndarray,
    predicted_boxes: np.ndarray,
    iou_threshold: float = 0.5,
) -> tuple[set[int], set[int]]:
    """Return matched prediction and truth indices using evaluator matching."""

    iou = pairwise_iou_xyxy(ground_truth_boxes, predicted_boxes)
    candidates = np.argwhere(iou >= float(iou_threshold))
    if not len(candidates):
        return set(), set()
    scores = iou[candidates[:, 0], candidates[:, 1]]
    matches = np.column_stack((candidates, scores))
    if len(matches) > 1:
        matches = matches[np.argsort(matches[:, 2])[::-1]]
        matches = matches[
            np.unique(matches[:, 1], return_index=True)[1]
        ]
        matches = matches[np.argsort(matches[:, 2])[::-1]]
        matches = matches[
            np.unique(matches[:, 0], return_index=True)[1]
        ]
    return (
        {int(value) for value in matches[:, 1]},
        {int(value) for value in matches[:, 0]},
    )


def _draw_error_overlay(
    *,
    image: np.ndarray,
    ground_truth_boxes: np.ndarray,
    predictions: list[dict[str, Any]],
    title: str,
) -> tuple[np.ndarray, dict[str, int]]:
    predicted_boxes = np.asarray(
        [prediction["bbox_xyxy"] for prediction in predictions],
        dtype=np.float32,
    ).reshape(-1, 4)
    matched_predictions, matched_truth = classify_detection_errors(
        ground_truth_boxes=ground_truth_boxes,
        predicted_boxes=predicted_boxes,
    )
    overlay = image.copy()
    thickness = max(2, round(min(image.shape[:2]) / 600))
    font_scale = max(0.55, min(image.shape[:2]) / 1400)
    for index, box in enumerate(ground_truth_boxes):
        if index in matched_truth:
            continue
        x1, y1, x2, y2 = (round(float(value)) for value in box)
        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 165, 255),
            thickness,
        )
        cv2.putText(
            overlay,
            "FN",
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (0, 165, 255),
            thickness,
            cv2.LINE_AA,
        )
    for index, prediction in enumerate(predictions):
        x1, y1, x2, y2 = (
            round(float(value)) for value in prediction["bbox_xyxy"]
        )
        matched = index in matched_predictions
        color = (0, 200, 0) if matched else (0, 0, 255)
        label = (
            f"TP {float(prediction['confidence']):.2f}"
            if matched
            else f"FP {float(prediction['confidence']):.2f}"
        )
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(
            overlay,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            color,
            thickness,
            cv2.LINE_AA,
        )
    counts = {
        "tp": len(matched_predictions),
        "fp": len(predictions) - len(matched_predictions),
        "fn": len(ground_truth_boxes) - len(matched_truth),
    }
    banner_height = max(42, round(55 * font_scale))
    cv2.rectangle(
        overlay,
        (0, 0),
        (overlay.shape[1], banner_height),
        (20, 20, 20),
        -1,
    )
    cv2.putText(
        overlay,
        (
            f"{title}  TP={counts['tp']} FP={counts['fp']} "
            f"FN={counts['fn']}"
        ),
        (12, round(banner_height * 0.72)),
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        (255, 255, 255),
        thickness,
        cv2.LINE_AA,
    )
    return overlay, counts


def _tile(
    image: np.ndarray,
    *,
    width: int = 640,
    height: int = 440,
) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    y = (height - resized.shape[0]) // 2
    x = (width - resized.shape[1]) // 2
    canvas[y : y + resized.shape[0], x : x + resized.shape[1]] = resized
    return canvas


def _montage(rows: list[list[np.ndarray]]) -> np.ndarray:
    return np.vstack([np.hstack([_tile(image) for image in row]) for row in rows])


def export_development_qualitative_evidence(
    *,
    comparison_root: Path,
    data_yaml: Path,
    output_root: Path,
    confidence_threshold: float,
) -> dict[str, Any]:
    """Export error montages only where development box truth exists."""

    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite qualitative output: {output_root}"
        )
    comparison, detections_by_method = _validate_development_comparison(
        comparison_root.resolve()
    )
    dataset = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    image_root, label_root = _split_paths(data_yaml, dataset, "val")
    image_by_name = {
        path.name: path
        for suffix in ("*.jpg", "*.jpeg", "*.png")
        for path in image_root.glob(suffix)
    }
    expected_names = [
        str(row["image_name"]) for row in detections_by_method["D0"]
    ]
    if set(image_by_name) != set(expected_names):
        raise StageIProtocolError(
            "Prepared validation images differ from comparison detections"
        )

    evidence: dict[str, dict[str, dict[str, Any]]] = {
        method_id: {} for method_id in METHOD_IDS
    }
    overlays: dict[tuple[str, str], np.ndarray] = {}
    for method_id in METHOD_IDS:
        for row in detections_by_method[method_id]:
            image_name = str(row["image_name"])
            image = read_image(image_by_name[image_name])
            ground_truth = _load_ground_truth_boxes(
                label_path=label_root / f"{Path(image_name).stem}.txt",
                image_width=image.shape[1],
                image_height=image.shape[0],
            )
            predictions = [
                item
                for item in row["detections"]
                if float(item["confidence"]) >= confidence_threshold
            ]
            overlay, counts = _draw_error_overlay(
                image=image,
                ground_truth_boxes=ground_truth,
                predictions=predictions,
                title=f"{method_id} | {image_name} | conf>={confidence_threshold:.2f}",
            )
            overlays[(method_id, image_name)] = overlay
            areas = (
                (ground_truth[:, 2] - ground_truth[:, 0])
                * (ground_truth[:, 3] - ground_truth[:, 1])
            )
            evidence[method_id][image_name] = {
                **counts,
                "ground_truth_boxes": len(ground_truth),
                "median_ground_truth_area_fraction": (
                    float(np.median(areas) / (image.shape[0] * image.shape[1]))
                    if len(areas)
                    else None
                ),
            }

    total_errors = {
        image_name: sum(
            evidence[method_id][image_name]["fp"]
            + evidence[method_id][image_name]["fn"]
            for method_id in METHOD_IDS
        )
        for image_name in expected_names
    }
    densest = max(
        expected_names,
        key=lambda name: (
            evidence["D0"][name]["ground_truth_boxes"],
            name,
        ),
    )
    hardest = max(
        expected_names,
        key=lambda name: (total_errors[name], name),
    )
    small_object = min(
        (
            name
            for name in expected_names
            if evidence["D0"][name]["ground_truth_boxes"] > 0
        ),
        key=lambda name: (
            evidence["D0"][name]["median_ground_truth_area_fraction"],
            name,
        ),
    )
    representative = list(dict.fromkeys([hardest, densest, small_object]))
    while len(representative) < 3:
        representative.append(
            next(name for name in expected_names if name not in representative)
        )

    output_root.mkdir(parents=True)
    annotated_root = output_root / "annotated"
    for method_id in METHOD_IDS:
        for image_name in representative:
            write_image(
                annotated_root / method_id / image_name,
                overlays[(method_id, image_name)],
            )
    write_image(
        output_root / "representative_comparison.png",
        _montage(
            [
                [
                    overlays[(method_id, image_name)]
                    for image_name in representative
                ]
                for method_id in METHOD_IDS
            ]
        ),
    )

    fp_rows = []
    fn_rows = []
    fp_selection = {}
    fn_selection = {}
    for method_id in METHOD_IDS:
        top_fp = sorted(
            expected_names,
            key=lambda name: (
                -evidence[method_id][name]["fp"],
                name,
            ),
        )[:3]
        top_fn = sorted(
            expected_names,
            key=lambda name: (
                -evidence[method_id][name]["fn"],
                name,
            ),
        )[:3]
        fp_selection[method_id] = top_fp
        fn_selection[method_id] = top_fn
        fp_rows.append(
            [overlays[(method_id, image_name)] for image_name in top_fp]
        )
        fn_rows.append(
            [overlays[(method_id, image_name)] for image_name in top_fn]
        )
    write_image(
        output_root / "false_positive_montage.png",
        _montage(fp_rows),
    )
    write_image(
        output_root / "false_negative_montage.png",
        _montage(fn_rows),
    )
    write_image(
        output_root / "night_occlusion_candidate_comparison.png",
        _montage(
            [
                [overlays[(method_id, densest)]]
                for method_id in METHOD_IDS
            ]
        ),
    )

    report = {
        "schema_version": 1,
        "task": "development_box_error_qualitative_evidence",
        "dataset_role": "consumed_development_validation",
        "lighting": "night",
        "confidence_threshold": confidence_threshold,
        "iou_threshold": 0.5,
        "legend": {
            "green": "matched prediction (TP)",
            "red": "unmatched prediction (FP)",
            "orange": "unmatched ground-truth box (FN)",
        },
        "representative_images": {
            "hardest_aggregate_fp_plus_fn": hardest,
            "densest_ground_truth": densest,
            "smallest_median_ground_truth_box": small_object,
            "exported": representative,
        },
        "false_positive_montage_selection": fp_selection,
        "false_negative_montage_selection": fn_selection,
        "night_occlusion_candidate": {
            "image_name": densest,
            "selection_rule": "highest ground-truth box count",
            "official_occlusion_tag_available": False,
            "interpretation": (
                "Dense/overlap visual-review candidate only; it is not "
                "occlusion ground truth."
            ),
        },
        "per_image_error_counts": evidence,
        "source_comparison": _artifact(comparison_root / "comparison.json"),
        "box_truth_used_on_count_test": False,
        "no_claim": (
            "Qualitative development evidence is not an untouched test and "
            "does not establish slot occupancy improvement."
        ),
    }
    (output_root / "qualitative_manifest.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report


def verify_stage_i_record(
    *,
    record_path: Path,
    implementation_root: Path,
) -> dict[str, Any]:
    """Verify frozen Stage I generated evidence without modifying it."""

    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    if record.get("record_id") != (
        "D-STAGE-I-RECORD-NDISPARK-20260727-01"
    ):
        raise StageIProtocolError("Unexpected Stage I result record ID")
    checks = []
    for artifact in record["artifacts"]:
        path = implementation_root / str(artifact["path"])
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha256 = sha256_file(path) if path.is_file() else None
        passed = (
            actual_bytes == int(artifact["bytes"])
            and actual_sha256 == str(artifact["sha256"])
        )
        checks.append(
            {
                "role": artifact["role"],
                "path": artifact["path"],
                "expected_bytes": artifact["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": artifact["sha256"],
                "actual_sha256": actual_sha256,
                "passed": passed,
            }
        )
    return {
        "schema_version": 1,
        "record_id": record["record_id"],
        "artifact_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }


def verify_stage_i_v2_record(
    *,
    record_path: Path,
    source_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    """Verify corrected v2 source and generated artifacts across two roots."""

    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    if record.get("record_id") != STAGE_I_V2_RECORD_ID:
        raise StageIProtocolError("Unexpected Stage I-v2 result record ID")
    roots = {
        "source": source_root.resolve(),
        "external": external_root.resolve(),
    }
    checks = []
    for artifact in record["artifacts"]:
        root_name = str(artifact["root"])
        if root_name not in roots:
            raise StageIProtocolError(
                f"Unexpected Stage I-v2 artifact root: {root_name}"
            )
        path = roots[root_name] / str(artifact["path"])
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha256 = sha256_file(path) if path.is_file() else None
        passed = (
            actual_bytes == int(artifact["bytes"])
            and actual_sha256 == str(artifact["sha256"])
        )
        checks.append(
            {
                "role": artifact["role"],
                "root": root_name,
                "path": artifact["path"],
                "expected_bytes": artifact["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": artifact["sha256"],
                "actual_sha256": actual_sha256,
                "passed": passed,
            }
        )
    return {
        "schema_version": 1,
        "record_id": record["record_id"],
        "artifact_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
