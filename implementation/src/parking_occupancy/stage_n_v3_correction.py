from __future__ import annotations

import json
import platform
import statistics
from dataclasses import replace
from pathlib import Path
from typing import Any, Sequence

from .stage_n_lmot import (
    LmotAnnotation,
    TrackPrediction,
    evaluate_motor_vehicle_detections,
    parse_lmot_gt,
    sha256_file,
    split_motor_vehicle_truth,
)
from .stage_n_lmot_v2 import load_lmot_class_map_v2


STAGE_N_V3_CORRECTION_ID = (
    "STAGE-N-V3-EMITTED-BOX-CORRECTION-20260729-01"
)
EXPECTED_METHODS = ("L0", "L1", "L2", "L3")
EXPECTED_SEQUENCES = ("LMOT-05", "LMOT-13", "LMOT-14", "LMOT-25")
RATE_KEYS = ("precision", "recall", "AP50", "AP50-95")
COUNT_KEYS = (
    "ground_truth_boxes",
    "predicted_boxes",
    "true_positives",
    "false_positives",
    "false_negatives",
)


def load_saved_detections(path: Path) -> list[TrackPrediction]:
    """Load emitted boxes without constructing or invoking a model."""

    predictions: list[TrackPrediction] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            try:
                box = tuple(float(value) for value in payload["bbox_xyxy"])
                if len(box) != 4:
                    raise ValueError("bbox_xyxy must contain four values")
                if payload.get("class_name") != "motor_vehicle":
                    raise ValueError("Expected emitted motor_vehicle boxes")
                predictions.append(
                    TrackPrediction(
                        frame_number=int(payload["frame"]),
                        track_id=int(payload["track_id"]),
                        xyxy=box,
                        confidence=float(payload["confidence"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid saved detection at {path}:{line_number}"
                ) from error
    return predictions


def per_sequence_macro(
    rows: dict[str, dict[str, float | int]],
) -> dict[str, float]:
    if not rows:
        raise ValueError("At least one sequence metric is required")
    return {
        key: statistics.fmean(float(row[key]) for row in rows.values())
        for key in RATE_KEYS
    }


def summed_counts(
    rows: dict[str, dict[str, float | int]],
) -> dict[str, int]:
    if not rows:
        raise ValueError("At least one sequence metric is required")
    return {
        key: sum(int(row[key]) for row in rows.values())
        for key in COUNT_KEYS
    }


def _offset_inputs(
    *,
    gt: Sequence[LmotAnnotation],
    predictions: Sequence[TrackPrediction],
    offset: int,
) -> tuple[list[LmotAnnotation], list[TrackPrediction]]:
    return (
        [
            replace(row, frame_number=row.frame_number + offset)
            for row in gt
        ],
        [
            replace(row, frame_number=row.frame_number + offset)
            for row in predictions
        ],
    )


def _artifact_record(path: Path, *, role: str) -> dict[str, Any]:
    return {
        "role": role,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def recompute_emitted_box_metrics(
    *,
    v2_output_root: Path,
    validation_root: Path,
    class_map_path: Path,
    output_root: Path,
    methods: Sequence[str] = EXPECTED_METHODS,
    sequences: Sequence[str] = EXPECTED_SEQUENCES,
) -> dict[str, Any]:
    """Offline Stage N-v3 correction from saved JSONL and released GT only."""

    v2_output_root = v2_output_root.resolve()
    validation_root = validation_root.resolve()
    class_map_path = class_map_path.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    if not v2_output_root.is_dir() or not validation_root.is_dir():
        raise FileNotFoundError("Stage N-v2 output or LMOT validation root")

    old_aggregate_path = v2_output_root / "aggregate_metrics.json"
    old_sequence_path = v2_output_root / "sequence_metrics.json"
    old_aggregate = json.loads(
        old_aggregate_path.read_text(encoding="utf-8")
    )
    old_sequences = json.loads(
        old_sequence_path.read_text(encoding="utf-8")
    )
    class_map = load_lmot_class_map_v2(class_map_path)

    input_records = [
        _artifact_record(old_aggregate_path, role="v2_aggregate_metrics"),
        _artifact_record(old_sequence_path, role="v2_sequence_metrics"),
        _artifact_record(class_map_path, role="frozen_class_map"),
        _artifact_record(
            Path(class_map.evidence), role="frozen_class_map_evidence"
        ),
    ]
    gt_by_sequence: dict[str, list[LmotAnnotation]] = {}
    for sequence in sequences:
        gt_path = validation_root / sequence / "gt" / "gt.txt"
        input_records.append(_artifact_record(gt_path, role="LMOT_GT"))
        annotations = parse_lmot_gt(gt_path)
        evaluated_gt, _suppression_gt = split_motor_vehicle_truth(
            annotations,
            class_map=class_map,
            evaluated_ignore_values=class_map.evaluated_mark_values,
        )
        gt_by_sequence[sequence] = evaluated_gt

    method_results: dict[str, Any] = {}
    for method in methods:
        sequence_results: dict[str, dict[str, float | int]] = {}
        pooled_gt: list[LmotAnnotation] = []
        pooled_predictions: list[TrackPrediction] = []
        for sequence_index, sequence in enumerate(sequences):
            detection_path = (
                v2_output_root
                / "detections"
                / method
                / f"{sequence}.jsonl"
            )
            input_records.append(
                _artifact_record(
                    detection_path, role="v2_emitted_detection_JSONL"
                )
            )
            predictions = load_saved_detections(detection_path)
            gt = gt_by_sequence[sequence]
            old_detection = old_sequences[method][sequence]["detection"]
            if len(gt) != int(old_detection["ground_truth_boxes"]):
                raise ValueError(
                    f"GT count differs from v2 for {method}/{sequence}"
                )
            if len(predictions) != int(old_detection["predicted_boxes"]):
                raise ValueError(
                    f"Prediction count differs from v2 for {method}/{sequence}"
                )
            sequence_results[sequence] = (
                evaluate_motor_vehicle_detections(
                    gt=gt, predictions=predictions
                )
            )
            offset_gt, offset_predictions = _offset_inputs(
                gt=gt,
                predictions=predictions,
                offset=sequence_index * 10_000,
            )
            pooled_gt.extend(offset_gt)
            pooled_predictions.extend(offset_predictions)

        aggregate = evaluate_motor_vehicle_detections(
            gt=pooled_gt,
            predictions=pooled_predictions,
        )
        count_sums = summed_counts(sequence_results)
        for key, value in count_sums.items():
            if int(aggregate[key]) != value:
                raise AssertionError(
                    f"Pooled {key} differs from sequence sum for {method}"
                )
        corrected_macro = per_sequence_macro(sequence_results)
        recorded_macro = {
            key: float(
                old_aggregate["methods"][method]["detection"][key]
            )
            for key in RATE_KEYS
        }
        method_results[method] = {
            "aggregate": aggregate,
            "aggregate_definition": (
                "all-data pooled/micro emitted-box evaluation with one "
                "global confidence ordering and sequence-isolated frame keys"
            ),
            "per_sequence_macro": corrected_macro,
            "per_sequence_macro_definition": (
                "unweighted arithmetic mean of the four sequence metrics"
            ),
            "v2_recorded_per_sequence_macro": recorded_macro,
            "matching_correction_delta_per_sequence_macro": {
                key: corrected_macro[key] - recorded_macro[key]
                for key in RATE_KEYS
            },
            "summed_counts_cross_check": count_sums,
            "per_sequence": sequence_results,
        }

    metrics = {
        "schema_version": 1,
        "correction_id": STAGE_N_V3_CORRECTION_ID,
        "status": "complete_offline_emitted_box_metric_correction",
        "primary_table_definition": (
            "all-data pooled/micro emitted-box metrics; per-sequence macro "
            "metrics are retained as secondary diagnostics"
        ),
        "metric_scope": (
            "boxes emitted by the saved complete model.track(...) outputs "
            "after excluded-class suppression; not raw detector-only metrics"
        ),
        "methods": method_results,
        "official_tracking_metrics": {
            "status": "not_recomputed_and_not_rewritten",
            "unaffected_metrics": [
                "HOTA",
                "DetA",
                "AssA",
                "IDF1",
                "MOTA",
                "ID_switches",
            ],
            "reason": (
                "The corrected bug exists only in the local emitted-box "
                "AP/precision/recall matcher; official TrackEval consumed "
                "saved tracks through independent matching code."
            ),
            "v2_aggregate_reference": str(old_aggregate_path),
            "v2_aggregate_sha256": sha256_file(old_aggregate_path),
        },
    }

    output_root.mkdir(parents=True)
    metrics_path = output_root / "emitted_box_metrics.json"
    metrics_path.write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    input_manifest = {
        "correction_id": STAGE_N_V3_CORRECTION_ID,
        "input_count": len(input_records),
        "inputs": input_records,
    }
    (output_root / "input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    runtime = {
        "correction_id": STAGE_N_V3_CORRECTION_ID,
        "python": platform.python_version(),
        "execution_mode": "offline_saved_predictions_and_gt_only",
        "inference_performed": False,
        "model_loaded": False,
        "model_track_called": False,
        "training_performed": False,
        "trackeval_called": False,
        "inputs": {
            "saved_detection_jsonl_count": len(methods) * len(sequences),
            "LMOT_gt_file_count": len(sequences),
        },
        "outputs": [
            "emitted_box_metrics.json",
            "input_manifest.json",
            "runtime_metadata.json",
        ],
    }
    (output_root / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metrics
