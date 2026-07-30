from __future__ import annotations

import csv
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .evaluate import binary_metrics
from .integrated_runner import (
    IntegratedClassifier,
    IntegratedDetector,
    IntegratedFrameProcessor,
    _timing_summary,
    create_classifier,
    create_detector,
    load_integrated_config,
)
from .slots import load_slot_map
from .stage_n_lmot import sha256_file
from .stage_q_external import ManifestRecord, verify_manifest_records
from .stage_q_v2_upm import (
    STAGE_Q_V2_PROTOCOL_ID,
    UPM_SLOT_IDS,
    validate_upm_slot_map,
)
from .visualization import draw_frame


EXPECTED_SHARED_INFERENCE = {
    "api": "ultralytics.YOLO.predict",
    "tracking_backend": "none",
    "imgsz": 640,
    "confidence": 0.30,
    "nms_iou": 0.70,
    "agnostic_nms": True,
    "max_det": 300,
    "classes": [0],
    "model_track_prohibited": True,
    "methods_must_be_identical_except_detector_weights": True,
}
EXPECTED_COMPONENTS = {
    "B1": {
        "mode": "overlap",
        "minimum_slot_coverage": 0.40,
        "one_to_one": True,
    },
    "E1b_F2": {
        "occupied_threshold": 0.76,
        "detector_negative_slots_only": True,
        "patch_size": [224, 224],
        "perspective_warp": True,
    },
    "E4": {
        "enabled": True,
        "rise_alpha": 0.60,
        "fall_alpha": 0.15,
        "occupied_threshold": 0.58,
        "vacant_threshold": 0.42,
        "raw_threshold": 0.76,
        "stable_frames_for_evaluation": 3,
    },
}
METHODS = (
    ("QV2-0", "P3-D1", "D1", "primary"),
    (
        "QV2-1",
        "P3-D1-LL",
        "D1_LL",
        "secondary_frozen_comparison",
    ),
)
REQUIRED_METHOD_OUTPUTS = {
    "annotated_frames",
    "annotated.mp4",
    "occupancy.csv",
    "events.csv",
    "detections.jsonl",
    "summary.json",
    "metrics.json",
    "runtime_metadata.json",
    "confusion_matrix.png",
    "qualitative_contact_sheet.jpg",
    "failure_cases.json",
}
OCCUPANCY_FIELDS = (
    "video_id",
    "frame_index",
    "timestamp_s",
    "slot_id",
    "state",
    "evidence",
    "raw_state",
    "filtered_score",
    "detector_occupied",
    "detector_score",
    "classifier_probability",
    "classifier_consulted",
    "gate_branch",
    "track_id",
    "tracker_backend",
    "temporal_enabled",
)
EVENT_FIELDS = (
    "video_id",
    "frame_index",
    "timestamp_s",
    "slot_id",
    "from_state",
    "to_state",
)


class StageQV2EvaluationError(ValueError):
    """Raised when the frozen Stage Q-v2 evaluation contract is violated."""


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (config_path.resolve().parent / path).resolve()


def _require_file_binding(
    *,
    config_path: Path,
    record: Mapping[str, Any],
    label: str,
) -> Path:
    path = _resolve(config_path, str(record["path"]))
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise StageQV2EvaluationError(f"{label} byte-size mismatch")
    if sha256_file(path) != str(record["sha256"]):
        raise StageQV2EvaluationError(f"{label} SHA-256 mismatch")
    return path


def load_frozen_stage_q_v2_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    retry = payload.get("additive_retry_of")
    if retry is not None:
        if payload.get("correction_type") != (
            "additive_runtime_environment_retry"
        ):
            raise StageQV2EvaluationError(
                "Unexpected Stage Q-v2 correction type"
            )
        base_path = _require_file_binding(
            config_path=path,
            record=retry,
            label="Stage Q-v2 base frozen config",
        )
        _require_file_binding(
            config_path=path,
            record=payload["failed_attempt_evidence"],
            label="Stage Q-v2 failed-attempt evidence",
        )
        base = yaml.safe_load(base_path.read_text(encoding="utf-8"))
        invariants = payload.get("invariants", {})
        if not all(
            invariants.get(key) is False
            for key in (
                "detector_weights_changed",
                "E1b_checkpoint_changed",
                "manifest_changed",
                "truth_changed",
                "polygons_changed",
                "P3_parameters_changed",
                "method_order_changed",
            )
        ):
            raise StageQV2EvaluationError(
                "Additive retry changes a frozen experimental input"
            )
        correction = payload["corrections"]
        if (
            correction["output_root"]["old"]
            != base["formal_runs"]["output_root"]
        ):
            raise StageQV2EvaluationError(
                "Retry output-root history does not match base config"
            )
        base["formal_runs"]["output_root"] = correction["output_root"]["new"]
        base["runtime_environment"] = {
            "YOLO_CONFIG_DIR": correction["runtime_environment"][
                "YOLO_CONFIG_DIR"
            ]
        }
        base["additive_retry"] = {
            "config_id": payload["config_id"],
            "base_config": str(base_path),
            "failed_attempt_evidence": str(
                _resolve(
                    path,
                    str(payload["failed_attempt_evidence"]["path"]),
                )
            ),
            "invariants": invariants,
        }
        payload = base
    if payload.get("schema_version") != 1:
        raise StageQV2EvaluationError("Unsupported Stage Q-v2 config schema")
    if payload.get("protocol_id") != STAGE_Q_V2_PROTOCOL_ID:
        raise StageQV2EvaluationError("Unexpected Stage Q-v2 protocol")
    if payload.get("status") != "frozen_before_formal_model_inference":
        raise StageQV2EvaluationError("Formal protocol was not pre-frozen")
    scope = payload.get("scope", {})
    if scope.get("primary_method") != "QV2-0_P3-D1":
        raise StageQV2EvaluationError("D1 primary role changed")
    if scope.get("D1_remains_project_default") is not True:
        raise StageQV2EvaluationError("D1 default role changed")
    if scope.get("D1_LL_role") != "secondary_frozen_comparison":
        raise StageQV2EvaluationError("D1-LL role changed")
    if scope.get("stage_p2_decision_remains") != "FAIL":
        raise StageQV2EvaluationError("Stage P2 FAIL history changed")

    shared = payload.get("shared_inference", {})
    for key, expected in EXPECTED_SHARED_INFERENCE.items():
        if shared.get(key) != expected:
            raise StageQV2EvaluationError(
                f"Frozen shared inference differs at {key}"
            )
    for component, expected_values in EXPECTED_COMPONENTS.items():
        actual = shared.get(component, {})
        for key, expected in expected_values.items():
            if actual.get(key) != expected:
                raise StageQV2EvaluationError(
                    f"Frozen {component} differs at {key}"
                )
    temporal = payload.get("temporal_semantics", {})
    if temporal.get("seconds_level_transition_latency") != "prohibited":
        raise StageQV2EvaluationError("Seconds latency must remain prohibited")
    if temporal.get("reliable_source_fps_available") is not False:
        raise StageQV2EvaluationError("UPM source FPS must remain unavailable")

    method_rows = payload.get("formal_runs", {}).get("methods")
    expected_methods = [
        {
            "method_id": method_id,
            "name": name,
            "detector": detector,
            "role": role,
        }
        for method_id, name, detector, role in METHODS
    ]
    if method_rows != expected_methods:
        raise StageQV2EvaluationError("Formal method roles/order changed")
    if payload["formal_runs"].get("run_count_per_method") != 1:
        raise StageQV2EvaluationError("Each method must run exactly once")
    return payload


def shared_method_signature(config: Mapping[str, Any]) -> dict[str, Any]:
    """Return the settings that must be identical for D1 and D1-LL."""

    return {
        "shared_inference": dict(config["shared_inference"]),
        "manifest_sha256": config["inputs"]["manifest"]["sha256"],
        "truth_sha256": config["inputs"]["occupancy_truth"]["sha256"],
        "polygon_sha256": config["inputs"]["polygons"]["sha256"],
        "E1b_sha256": config["models"]["E1b"]["sha256"],
    }


def preflight_stage_q_v2(
    config_path: Path,
    *,
    check_output: bool = True,
) -> dict[str, Any]:
    """Verify every frozen binding before constructing either model."""

    config_path = config_path.resolve()
    config = load_frozen_stage_q_v2_config(config_path)
    source_audit = _require_file_binding(
        config_path=config_path,
        record=config["gates"]["source_archive_audit"],
        label="source archive audit",
    )
    night_gate = _require_file_binding(
        config_path=config_path,
        record=config["gates"]["night_test_gate"],
        label="night test gate",
    )
    confirmation = _require_file_binding(
        config_path=config_path,
        record=config["gates"]["polygon_confirmation"],
        label="polygon confirmation",
    )
    confirmation_payload = yaml.safe_load(
        confirmation.read_text(encoding="utf-8")
    )
    if confirmation_payload.get("status") != "PASS":
        raise StageQV2EvaluationError("Polygon confirmation gate is not PASS")
    if (
        confirmation_payload.get("authorization", {}).get(
            "formal_inference_authorized"
        )
        is not True
    ):
        raise StageQV2EvaluationError(
            "Polygon confirmation does not authorize inference"
        )

    manifest = _require_file_binding(
        config_path=config_path,
        record=config["inputs"]["manifest"],
        label="test image manifest",
    )
    truth = _require_file_binding(
        config_path=config_path,
        record=config["inputs"]["occupancy_truth"],
        label="occupancy truth",
    )
    polygons = _require_file_binding(
        config_path=config_path,
        record=config["inputs"]["polygons"],
        label="slot polygons",
    )
    p3_runtime = _require_file_binding(
        config_path=config_path,
        record=config["inputs"]["p3_runtime"],
        label="P3 runtime defaults",
    )
    models = {
        model_id: _require_file_binding(
            config_path=config_path,
            record=config["models"][model_id],
            label=f"{model_id} model",
        )
        for model_id in ("D1", "D1_LL", "E1b")
    }
    test_root = _resolve(
        config_path, str(config["inputs"]["extracted_test_root"])
    )
    if not test_root.is_dir():
        raise FileNotFoundError(test_root)
    output_root = _resolve(
        config_path, str(config["formal_runs"]["output_root"])
    )
    if check_output and output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage Q-v2 output: {output_root}"
        )

    with manifest.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    if len(manifest_rows) != int(config["inputs"]["manifest"]["rows"]):
        raise StageQV2EvaluationError("Manifest row count mismatch")
    records = [
        ManifestRecord(
            relative_path=row["relative_path"],
            bytes=int(row["bytes"]),
            sha256=row["sha256"],
        )
        for row in manifest_rows
    ]
    manifest_verification = verify_manifest_records(
        test_root,
        records,
        expected_manifest_sha256=str(
            config["inputs"]["manifest"]["logical_sha256"]
        ),
    )
    if not manifest_verification["verified"]:
        raise StageQV2EvaluationError("Manifest image verification failed")

    with truth.open("r", encoding="utf-8", newline="") as handle:
        truth_rows = list(csv.DictReader(handle))
    if len(truth_rows) != int(config["inputs"]["occupancy_truth"]["rows"]):
        raise StageQV2EvaluationError("Occupancy truth row count mismatch")
    truth_keys = {
        (row["video_id"], int(row["frame_index"]), row["slot_id"])
        for row in truth_rows
    }
    manifest_keys = {
        (
            row["sequence_id"],
            int(row["frame_index"]),
            slot_id,
        )
        for row in manifest_rows
        for slot_id in UPM_SLOT_IDS
    }
    if truth_keys != manifest_keys:
        raise StageQV2EvaluationError(
            "Manifest and occupancy truth keys differ"
        )

    polygon_payload = json.loads(polygons.read_text(encoding="utf-8"))
    validate_upm_slot_map(polygon_payload)
    p3 = load_integrated_config(p3_runtime)
    if p3["tracking"]["default_backend"] != "none":
        raise StageQV2EvaluationError("Stage Q-v2 prohibits trackers")
    if (
        p3["detector"]["confidence"],
        p3["detector"]["image_size"],
        p3["detector"]["nms_iou"],
        p3["detector"]["agnostic_nms"],
        p3["detector"]["max_detections"],
    ) != (0.30, 640, 0.70, True, 300):
        raise StageQV2EvaluationError("P3 detector defaults changed")
    return {
        "config": config,
        "config_path": config_path,
        "source_audit": source_audit,
        "night_gate": night_gate,
        "confirmation": confirmation,
        "manifest": manifest,
        "manifest_rows": manifest_rows,
        "manifest_verification": manifest_verification,
        "truth": truth,
        "truth_rows": truth_rows,
        "polygons": polygons,
        "p3_runtime": p3_runtime,
        "p3": p3,
        "models": models,
        "test_root": test_root,
        "output_root": output_root,
    }


def _read_image(path: Path) -> np.ndarray:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise StageQV2EvaluationError(f"Could not decode image: {path}")
    return image


def _write_image(path: Path, image: np.ndarray, *, quality: int = 92) -> None:
    suffix = path.suffix.lower()
    extension = ".jpg" if suffix in {".jpg", ".jpeg"} else suffix
    parameters = (
        [cv2.IMWRITE_JPEG_QUALITY, quality]
        if extension == ".jpg"
        else []
    )
    ok, encoded = cv2.imencode(extension, image, parameters)
    if not ok:
        raise StageQV2EvaluationError(f"Could not encode image: {path}")
    encoded.tofile(path)


def _summary(values: Sequence[float | int]) -> dict[str, Any]:
    if not values:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p90": None,
            "maximum": None,
        }
    ordered = sorted(float(value) for value in values)
    return {
        "count": len(ordered),
        "mean": statistics.fmean(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[round(0.9 * (len(ordered) - 1))],
        "maximum": ordered[-1],
    }


def _truth_prediction_rows(
    truth_path: Path,
    prediction_path: Path,
) -> tuple[
    dict[tuple[str, int, str], dict[str, str]],
    dict[tuple[str, int, str], dict[str, str]],
]:
    with truth_path.open("r", encoding="utf-8", newline="") as handle:
        truth = {
            (row["video_id"], int(row["frame_index"]), row["slot_id"]): row
            for row in csv.DictReader(handle)
        }
    with prediction_path.open("r", encoding="utf-8", newline="") as handle:
        prediction = {
            (row["video_id"], int(row["frame_index"]), row["slot_id"]): row
            for row in csv.DictReader(handle)
        }
    if set(truth) != set(prediction):
        raise StageQV2EvaluationError(
            "Prediction and truth keys differ after formal inference"
        )
    return truth, prediction


def _frame_only_temporal_metrics(
    truth: Mapping[tuple[str, int, str], Mapping[str, str]],
    prediction: Mapping[tuple[str, int, str], Mapping[str, str]],
    *,
    stable_frames: int,
) -> dict[str, Any]:
    from literature_core.metrics import sequence_temporal_metrics

    groups: dict[tuple[str, str], list[int]] = defaultdict(list)
    for video_id, frame_index, slot_id in truth:
        groups[(video_id, slot_id)].append(frame_index)
    events: list[dict[str, Any]] = []
    outcomes = {"early": 0, "on_time": 0, "delayed": 0, "missed": 0}
    selected_frame_errors: list[int] = []
    source_index_errors: list[int] = []
    steady_state_latency: list[int] = []
    ground_truth_transitions = 0
    matched_transitions = 0
    unsupported_flicker = 0
    transition_instability = 0
    for (video_id, slot_id), frame_indices in sorted(groups.items()):
        frame_indices.sort()
        y_true = [
            int(truth[(video_id, frame, slot_id)]["state"])
            for frame in frame_indices
        ]
        y_pred = [
            int(prediction[(video_id, frame, slot_id)]["state"])
            for frame in frame_indices
        ]
        row = sequence_temporal_metrics(
            y_true,
            y_pred,
            fps=1.0,
            stable_frames=stable_frames,
            tolerance_frames=0,
        )
        ground_truth_transitions += int(row["ground_truth_transitions"])
        matched_transitions += int(row["matched_transitions"])
        unsupported_flicker += int(row["unsupported_flicker_count"])
        transition_instability += int(
            row["transition_instability_changes"]
        )
        for outcome, count in row["transition_outcomes"].items():
            outcomes[outcome] += int(count)
        for event in row["transition_events"]:
            truth_position = int(event["truth_transition_frame"])
            predicted_position = event["predicted_transition_frame"]
            translated = {
                "video_id": video_id,
                "slot_id": slot_id,
                "direction": event["direction"],
                "from_state": event["from_state"],
                "to_state": event["to_state"],
                "outcome": event["outcome"],
                "truth_selected_position": truth_position,
                "truth_source_frame_index": frame_indices[truth_position],
                "predicted_selected_position": predicted_position,
                "predicted_source_frame_index": (
                    None
                    if predicted_position is None
                    else frame_indices[int(predicted_position)]
                ),
                "signed_error_selected_frames": (
                    None
                    if predicted_position is None
                    else int(predicted_position) - truth_position
                ),
                "signed_error_source_frame_index": (
                    None
                    if predicted_position is None
                    else frame_indices[int(predicted_position)]
                    - frame_indices[truth_position]
                ),
                "event_window_units": "selected_ordered_frames",
            }
            events.append(translated)
            if predicted_position is not None:
                selected_error = int(predicted_position) - truth_position
                source_error = (
                    frame_indices[int(predicted_position)]
                    - frame_indices[truth_position]
                )
                selected_frame_errors.append(selected_error)
                source_index_errors.append(source_error)
                if selected_error >= 0:
                    steady_state_latency.append(selected_error)
    return {
        "definition": (
            "canonical adjacent-ground-truth event windows; frame units only"
        ),
        "source_timestamp_available": False,
        "reliable_source_fps_available": False,
        "seconds_level_transition_latency_computed": False,
        "ground_truth_transitions": ground_truth_transitions,
        "matched_transitions": matched_transitions,
        "missed_transitions": outcomes["missed"],
        "state_change_agreement": (
            matched_transitions / ground_truth_transitions
            if ground_truth_transitions
            else None
        ),
        "transition_outcomes": outcomes,
        "signed_transition_error_selected_frames": _summary(
            selected_frame_errors
        ),
        "signed_transition_error_source_frame_index": _summary(
            source_index_errors
        ),
        "steady_state_latency_frames": _summary(steady_state_latency),
        "unsupported_flicker_count": unsupported_flicker,
        "transition_instability_changes": transition_instability,
        "stable_frames": stable_frames,
        "transition_events": events,
    }


def _failure_category(
    truth_state: int,
    prediction: Mapping[str, str],
) -> str:
    state = int(prediction["state"])
    raw_state = int(prediction["raw_state"])
    detector_occupied = bool(int(prediction["detector_occupied"]))
    if truth_state == state:
        return "correct"
    if truth_state == 1:
        if raw_state == 1:
            return "E4_temporal_lag_or_carryover"
        if not detector_occupied:
            return "detector_negative_not_recovered_by_E1b_F2"
        return "B1_or_fusion_false_free"
    if raw_state == 0 and state == 1:
        return "E4_temporal_lag_or_carryover"
    if detector_occupied:
        return "detector_or_B1_geometry_false_occupied"
    if bool(int(prediction["classifier_consulted"])):
        return "E1b_F2_classifier_override"
    return "fusion_false_occupied"


def evaluate_stage_q_v2_method(
    *,
    truth_path: Path,
    prediction_path: Path,
    stable_frames: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth, prediction = _truth_prediction_rows(truth_path, prediction_path)
    keys = sorted(truth)
    y_true = [int(truth[key]["state"]) for key in keys]
    y_pred = [int(prediction[key]["state"]) for key in keys]
    classification = binary_metrics(y_true, y_pred)

    frame_counts: dict[tuple[str, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    per_sequence_values: dict[str, tuple[list[int], list[int]]] = {}
    failures: list[dict[str, Any]] = []
    for key in keys:
        video_id, frame_index, slot_id = key
        expected = int(truth[key]["state"])
        actual = int(prediction[key]["state"])
        frame_counts[(video_id, frame_index)][0] += expected
        frame_counts[(video_id, frame_index)][1] += actual
        true_values, pred_values = per_sequence_values.setdefault(
            video_id, ([], [])
        )
        true_values.append(expected)
        pred_values.append(actual)
        if expected != actual:
            failures.append(
                {
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "slot_id": slot_id,
                    "truth": expected,
                    "prediction": actual,
                    "raw_state": int(prediction[key]["raw_state"]),
                    "detector_occupied": int(
                        prediction[key]["detector_occupied"]
                    ),
                    "detector_score": float(
                        prediction[key]["detector_score"]
                    ),
                    "classifier_probability": (
                        None
                        if prediction[key]["classifier_probability"] == ""
                        else float(
                            prediction[key]["classifier_probability"]
                        )
                    ),
                    "classifier_consulted": int(
                        prediction[key]["classifier_consulted"]
                    ),
                    "gate_branch": prediction[key]["gate_branch"],
                    "category": _failure_category(
                        expected, prediction[key]
                    ),
                }
            )
    count_errors = [
        predicted - expected
        for expected, predicted in frame_counts.values()
    ]
    count_metrics = {
        "frames": len(frame_counts),
        "mae": statistics.fmean(abs(value) for value in count_errors),
        "rmse": math.sqrt(
            statistics.fmean(value * value for value in count_errors)
        ),
        "mean_signed_error": statistics.fmean(count_errors),
    }
    per_sequence = {
        video_id: binary_metrics(true_values, pred_values)
        for video_id, (true_values, pred_values) in per_sequence_values.items()
    }
    categories: dict[str, int] = defaultdict(int)
    for row in failures:
        categories[row["category"]] += 1
    temporal = _frame_only_temporal_metrics(
        truth,
        prediction,
        stable_frames=stable_frames,
    )
    return (
        {
            "schema_version": 1,
            "status": "computed_from_frozen_external_occupancy_truth",
            "classification": classification,
            "occupancy_count": count_metrics,
            "temporal_frame_only": temporal,
            "per_sequence": per_sequence,
            "failure_category_counts": dict(sorted(categories.items())),
            "metric_scope": "slot_occupancy_not_detector_AP",
            "unknown_truth_excluded": 0,
        },
        failures,
    )


def _render_confusion_matrix(
    path: Path,
    classification: Mapping[str, Any],
) -> None:
    canvas = np.full((500, 600, 3), 250, dtype=np.uint8)
    cv2.putText(
        canvas,
        "Slot-level confusion matrix",
        (120, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        (20, 20, 20),
        2,
        cv2.LINE_AA,
    )
    labels = (
        ("TN", int(classification["tn"]), 120, 150),
        ("FP", int(classification["fp"]), 340, 150),
        ("FN", int(classification["fn"]), 120, 330),
        ("TP", int(classification["tp"]), 340, 330),
    )
    maximum = max(1, *(value for _, value, _, _ in labels))
    for label, value, x, y in labels:
        intensity = int(245 - 175 * value / maximum)
        cv2.rectangle(
            canvas,
            (x, y),
            (x + 150, y + 120),
            (255, intensity, intensity),
            -1,
        )
        cv2.rectangle(canvas, (x, y), (x + 150, y + 120), (40, 40, 40), 2)
        cv2.putText(
            canvas,
            f"{label}: {value}",
            (x + 18, y + 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (20, 20, 20),
            2,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        "Predicted: vacant / occupied",
        (155, 480),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    _write_image(path, canvas)


def _render_contact_sheet(
    *,
    method_root: Path,
    failures: Sequence[Mapping[str, Any]],
    manifest_rows: Sequence[Mapping[str, str]],
    method_name: str,
) -> None:
    frame_lookup = {
        (row["sequence_id"], int(row["frame_index"])): index
        for index, row in enumerate(manifest_rows)
    }
    representatives: list[Mapping[str, Any]] = []
    seen_categories: set[str] = set()
    seen_frames: set[tuple[str, int]] = set()
    for failure in failures:
        category = str(failure["category"])
        frame_key = (
            str(failure["video_id"]),
            int(failure["frame_index"]),
        )
        if category not in seen_categories and frame_key not in seen_frames:
            representatives.append(failure)
            seen_categories.add(category)
            seen_frames.add(frame_key)
    for failure in failures:
        frame_key = (
            str(failure["video_id"]),
            int(failure["frame_index"]),
        )
        if frame_key not in seen_frames:
            representatives.append(failure)
            seen_frames.add(frame_key)
        if len(representatives) >= 12:
            break
    if not representatives:
        representatives = [
            {
                "video_id": row["sequence_id"],
                "frame_index": int(row["frame_index"]),
                "slot_id": "none",
                "category": "no_errors",
            }
            for row in manifest_rows[:12]
        ]
    representatives = representatives[:12]

    tile_width, tile_height, label_height = 400, 300, 55
    columns = 3
    rows = math.ceil(len(representatives) / columns)
    canvas = np.full(
        (rows * (tile_height + label_height), columns * tile_width, 3),
        245,
        dtype=np.uint8,
    )
    for index, item in enumerate(representatives):
        key = (str(item["video_id"]), int(item["frame_index"]))
        order_index = frame_lookup[key]
        frame_path = (
            method_root
            / "annotated_frames"
            / f"{order_index:06d}_{key[0]}_f{key[1]:06d}.jpg"
        )
        image = _read_image(frame_path)
        tile = cv2.resize(
            image, (tile_width, tile_height), interpolation=cv2.INTER_AREA
        )
        row, column = divmod(index, columns)
        x0, y0 = column * tile_width, row * (tile_height + label_height)
        canvas[y0 : y0 + tile_height, x0 : x0 + tile_width] = tile
        label = (
            f"{key[0]} f={key[1]} {item.get('slot_id')} "
            f"{item.get('category')}"
        )
        cv2.putText(
            canvas,
            label[:62],
            (x0 + 5, y0 + tile_height + 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        canvas,
        f"{method_name} representative frozen-test outcomes",
        (8, 18),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    _write_image(method_root / "qualitative_contact_sheet.jpg", canvas)


def _write_json_exclusive(path: Path, payload: Any) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False)
        handle.write("\n")


def _run_one_method(
    *,
    method_id: str,
    method_name: str,
    role: str,
    weights: Path,
    preflight: Mapping[str, Any],
    method_root: Path,
    device: str,
    classifier_batch_size: int,
    detector: IntegratedDetector | None = None,
    classifier: IntegratedClassifier | None = None,
) -> dict[str, Any]:
    if method_root.exists():
        raise FileExistsError(f"Refusing to overwrite {method_root}")
    p3 = preflight["p3"]
    slots = load_slot_map(
        preflight["polygons"], frame_size=(800, 600)
    ).slots
    detector = detector or create_detector(
        config=p3,
        weights=weights,
        device=device,
        tracker_config=None,
    )
    classifier = classifier or create_classifier(
        config=p3,
        checkpoint=preflight["models"]["E1b"],
        device=device,
        batch_size=classifier_batch_size,
    )
    processor = IntegratedFrameProcessor(
        slots=slots,
        detector=detector,
        classifier=classifier,
        config=p3,
        temporal_enabled=True,
    )

    method_root.mkdir(parents=True)
    annotated_root = method_root / "annotated_frames"
    annotated_root.mkdir()
    visualization_fps = float(
        preflight["config"]["temporal_semantics"][
            "visualization_reconstruction_fps"
        ]
    )
    video_path = method_root / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*str(p3["output"]["codec"])),
        visualization_fps,
        (800, 600),
    )
    if not writer.isOpened():
        raise StageQV2EvaluationError(
            f"Could not create visualization video: {video_path}"
        )

    sequence_resets: list[dict[str, Any]] = []
    timing: dict[str, list[float]] = defaultdict(list)
    event_count = 0
    classifier_reviews = 0
    started = time.perf_counter()
    previous_sequence: str | None = None
    try:
        with (
            (method_root / "occupancy.csv").open(
                "x", encoding="utf-8", newline=""
            ) as occupancy_handle,
            (method_root / "events.csv").open(
                "x", encoding="utf-8", newline=""
            ) as event_handle,
            (method_root / "detections.jsonl").open(
                "x", encoding="utf-8"
            ) as detection_handle,
        ):
            occupancy_writer = csv.DictWriter(
                occupancy_handle,
                fieldnames=list(OCCUPANCY_FIELDS),
                lineterminator="\n",
            )
            event_writer = csv.DictWriter(
                event_handle,
                fieldnames=list(EVENT_FIELDS),
                lineterminator="\n",
            )
            occupancy_writer.writeheader()
            event_writer.writeheader()
            for order_index, row in enumerate(preflight["manifest_rows"]):
                sequence_id = row["sequence_id"]
                source_frame_index = int(row["frame_index"])
                if sequence_id != previous_sequence:
                    processor.begin_source(sequence_id)
                    sequence_resets.append(
                        {
                            "source_id": sequence_id,
                            "manifest_order_index": order_index,
                            "detector_begin_source_called": True,
                            "temporal_filter_reconstructed": True,
                            "event_state_reinitialized": True,
                        }
                    )
                    previous_sequence = sequence_id
                image_path = (
                    preflight["test_root"] / row["relative_path"]
                )
                frame = _read_image(image_path)
                if frame.shape[:2] != (600, 800):
                    raise StageQV2EvaluationError(
                        f"Unexpected image size: {image_path}"
                    )
                frame_started = time.perf_counter()
                result = processor.process(frame, fps=visualization_fps)
                for key, value in result.timing_ms.items():
                    timing[key].append(value)
                classifier_reviews += sum(
                    decision.classifier_consulted
                    for decision in result.decisions.values()
                )
                for slot in slots:
                    decision = result.decisions[slot.slot_id]
                    state = result.states[slot.slot_id]
                    occupancy_writer.writerow(
                        {
                            "video_id": sequence_id,
                            "frame_index": source_frame_index,
                            "timestamp_s": "",
                            "slot_id": slot.slot_id,
                            "state": int(state.occupied),
                            "evidence": f"{state.filtered_score:.9f}",
                            "raw_state": int(decision.occupied),
                            "filtered_score": (
                                f"{state.filtered_score:.9f}"
                            ),
                            "detector_occupied": int(
                                decision.detector_occupied
                            ),
                            "detector_score": (
                                f"{decision.detector_score:.9f}"
                            ),
                            "classifier_probability": (
                                ""
                                if decision.classifier_probability is None
                                else (
                                    f"{decision.classifier_probability:.9f}"
                                )
                            ),
                            "classifier_consulted": int(
                                decision.classifier_consulted
                            ),
                            "gate_branch": decision.branch,
                            "track_id": "",
                            "tracker_backend": "none",
                            "temporal_enabled": 1,
                        }
                    )
                for event in result.events:
                    event_writer.writerow(
                        {
                            **event,
                            "video_id": sequence_id,
                            "frame_index": source_frame_index,
                            "timestamp_s": "",
                        }
                    )
                event_count += len(result.events)
                detection_handle.write(
                    json.dumps(
                        {
                            "video_id": sequence_id,
                            "frame_index": source_frame_index,
                            "timestamp_s": None,
                            "source_relative_path": row["relative_path"],
                            "detections": [
                                asdict(detection)
                                for detection in result.detections
                            ],
                        },
                        sort_keys=True,
                    )
                    + "\n"
                )
                processing_fps = 1.0 / max(
                    time.perf_counter() - frame_started, 1e-9
                )
                annotated = draw_frame(
                    frame=frame,
                    detections=list(result.detections),
                    slots=slots,
                    states=result.states,
                    experiment=method_name,
                    processing_fps=processing_fps,
                )
                cv2.putText(
                    annotated,
                    (
                        f"{sequence_id} source_frame={source_frame_index} "
                        "reconstructed visualization; source FPS unknown"
                    ),
                    (8, 52),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                frame_path = (
                    annotated_root
                    / (
                        f"{order_index:06d}_{sequence_id}_"
                        f"f{source_frame_index:06d}.jpg"
                    )
                )
                _write_image(frame_path, annotated)
                writer.write(annotated)
                timing["end_to_end"].append(
                    (time.perf_counter() - frame_started) * 1000.0
                )
    finally:
        writer.release()

    metrics, failures = evaluate_stage_q_v2_method(
        truth_path=preflight["truth"],
        prediction_path=method_root / "occupancy.csv",
        stable_frames=int(
            p3["temporal"]["stable_frames_for_evaluation"]
        ),
    )
    metrics["method_id"] = method_id
    metrics["method_name"] = method_name
    metrics["role"] = role
    _write_json_exclusive(method_root / "metrics.json", metrics)
    _write_json_exclusive(
        method_root / "failure_cases.json",
        {
            "schema_version": 1,
            "method_id": method_id,
            "failure_count": len(failures),
            "categories": metrics["failure_category_counts"],
            "cases": failures,
            "interpretation_boundary": (
                "categories are fixed diagnostic labels, not proof of "
                "causal component attribution"
            ),
        },
    )
    _render_confusion_matrix(
        method_root / "confusion_matrix.png",
        metrics["classification"],
    )
    _render_contact_sheet(
        method_root=method_root,
        failures=failures,
        manifest_rows=preflight["manifest_rows"],
        method_name=method_name,
    )

    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "method_id": method_id,
        "method_name": method_name,
        "role": role,
        "status": "executed_once_on_frozen_manifest",
        "frames": len(preflight["manifest_rows"]),
        "slots": len(slots),
        "sequences": len(sequence_resets),
        "events": event_count,
        "classifier_reviews": classifier_reviews,
        "truth_supplied": True,
        "temporal_enabled": True,
        "tracker_backend": "none",
        "elapsed_s": elapsed,
        "parameter_selection_from_run": False,
        "D1_remains_project_default": True,
        "stage_p2_decision_remains": "FAIL",
        "visualization": {
            "annotated_mp4_is_reconstructed": True,
            "reconstruction_fps": visualization_fps,
            "source_fps_known": False,
        },
        "inputs": {
            "manifest": str(preflight["manifest"]),
            "manifest_sha256": sha256_file(preflight["manifest"]),
            "truth": str(preflight["truth"]),
            "truth_sha256": sha256_file(preflight["truth"]),
            "polygons": str(preflight["polygons"]),
            "polygons_sha256": sha256_file(preflight["polygons"]),
            "detector_weights": str(weights),
            "detector_weights_sha256": sha256_file(weights),
            "E1b_checkpoint": str(preflight["models"]["E1b"]),
            "E1b_checkpoint_sha256": sha256_file(
                preflight["models"]["E1b"]
            ),
            "P3_runtime": str(preflight["p3_runtime"]),
            "P3_runtime_sha256": sha256_file(preflight["p3_runtime"]),
            "frozen_stage_q_v2_config": str(preflight["config_path"]),
            "frozen_stage_q_v2_config_sha256": sha256_file(
                preflight["config_path"]
            ),
        },
        "output_files": sorted(REQUIRED_METHOD_OUTPUTS),
    }
    _write_json_exclusive(method_root / "summary.json", summary)
    runtime = {
        "schema_version": 1,
        "method_id": method_id,
        "device_request": device,
        "python": sys.version,
        "platform": platform.platform(),
        "opencv": cv2.__version__,
        "source_state_resets": sequence_resets,
        "timing": {
            key: _timing_summary(values)
            for key, values in sorted(timing.items())
        },
        "descriptive_processing_fps": (
            len(preflight["manifest_rows"]) / elapsed if elapsed > 0 else None
        ),
        "runtime_not_used_for_model_selection": True,
        "detector": detector.metadata(),
        "classifier": classifier.metadata(),
        "model_track_called": False,
    }
    _write_json_exclusive(method_root / "runtime_metadata.json", runtime)
    actual = {path.name for path in method_root.iterdir()}
    if not REQUIRED_METHOD_OUTPUTS <= actual:
        raise StageQV2EvaluationError(
            f"Missing method outputs: {sorted(REQUIRED_METHOD_OUTPUTS - actual)}"
        )
    return {
        "summary": summary,
        "metrics": metrics,
        "runtime": runtime,
    }


def run_stage_q_v2_formal(
    *,
    config_path: Path,
    device: str | None = None,
    classifier_batch_size: int = 64,
    detector_overrides: Mapping[str, IntegratedDetector] | None = None,
    classifier_overrides: Mapping[str, IntegratedClassifier] | None = None,
) -> dict[str, Any]:
    """Run QV2-0 and QV2-1 once, in the frozen order, without tracking."""

    if classifier_batch_size <= 0:
        raise ValueError("classifier_batch_size must be positive")
    preflight = preflight_stage_q_v2(config_path)
    config = preflight["config"]
    device = str(device or config["shared_inference"]["device"])
    output_root = preflight["output_root"]
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage Q-v2 output: {output_root}"
        )
    output_root.mkdir(parents=True)
    runtime_environment = config.get("runtime_environment", {})
    yolo_config_value = runtime_environment.get("YOLO_CONFIG_DIR")
    if yolo_config_value is not None:
        yolo_config_dir = _resolve(
            preflight["config_path"], str(yolo_config_value)
        )
        try:
            yolo_config_dir.relative_to(output_root)
        except ValueError as exc:
            raise StageQV2EvaluationError(
                "Retry YOLO_CONFIG_DIR must stay inside the output root"
            ) from exc
        yolo_config_dir.mkdir(parents=True, exist_ok=False)
        os.environ["YOLO_CONFIG_DIR"] = str(yolo_config_dir)
    signature = shared_method_signature(config)
    results: dict[str, Any] = {}
    for method_id, method_name, detector_id, role in METHODS:
        method_root = output_root / method_id
        results[method_id] = _run_one_method(
            method_id=method_id,
            method_name=method_name,
            role=role,
            weights=preflight["models"][detector_id],
            preflight=preflight,
            method_root=method_root,
            device=device,
            classifier_batch_size=classifier_batch_size,
            detector=(
                None
                if detector_overrides is None
                else detector_overrides.get(method_id)
            ),
            classifier=(
                None
                if classifier_overrides is None
                else classifier_overrides.get(method_id)
            ),
        )
    comparison = {
        "schema_version": 1,
        "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
        "status": "FORMAL_RUNS_COMPLETE",
        "run_count_per_method": 1,
        "method_order": ["QV2-0", "QV2-1"],
        "shared_method_signature": signature,
        "methods_identical_except_detector_weights": True,
        "D1_remains_project_default": True,
        "D1_LL_role": "secondary_frozen_comparison",
        "stage_p2_decision_remains": "FAIL",
        "results": {
            method_id: {
                "classification": result["metrics"]["classification"],
                "occupancy_count": result["metrics"]["occupancy_count"],
                "temporal_frame_only": result["metrics"][
                    "temporal_frame_only"
                ],
                "descriptive_processing_fps": result["runtime"][
                    "descriptive_processing_fps"
                ],
            }
            for method_id, result in results.items()
        },
        "interpretation_boundary": {
            "single_external_camera_proves_generalization": False,
            "occupancy_metrics_are_detector_AP": False,
            "source_seconds_latency_available": False,
            "runtime_used_for_selection": False,
        },
    }
    _write_json_exclusive(output_root / "comparison.json", comparison)
    _write_json_exclusive(
        output_root / "formal_run_audit.json",
        {
            "schema_version": 1,
            "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
            "polygon_confirmation": str(preflight["confirmation"]),
            "polygon_confirmation_sha256": sha256_file(
                preflight["confirmation"]
            ),
            "model_predictions_viewed_before_polygon_confirmation": False,
            "model_track_called": False,
            "predict_calls_only": True,
            "formal_method_runs": [
                {
                    "method_id": method_id,
                    "run_count": 1,
                    "completed": True,
                }
                for method_id, *_ in METHODS
            ],
        },
    )
    return comparison
