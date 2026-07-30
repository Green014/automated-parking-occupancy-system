from __future__ import annotations

import csv
import json
import math
import os
import platform
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .count_metrics import evaluate_count_rows
from .stage_n_lmot import (
    LmotAnnotation,
    TrackPrediction,
    evaluate_motor_vehicle_detections,
    read_image,
    sha256_file,
    write_image,
)
from .stage_o_low_light import DetectorOnlySettings, FrozenRawDetectorAdapter


STAGE_P_PROTOCOL_ID = "STAGE-P-PARKING-DOMAIN-RETENTION-20260729-01"
AP_SEMANTICS = (
    "confidence-truncated AP at the frozen confidence threshold of 0.30"
)
AP50_KEY = (
    "confidence-truncated AP50 at the frozen confidence threshold of 0.30"
)
AP50_95_KEY = (
    "confidence-truncated AP50-95 at the frozen confidence threshold of 0.30"
)
MODEL_IDS = ("D1", "D1_LL")
BOX_SPLITS = ("train", "validation")
SPLIT_DIRECTORY = {
    "train": "train",
    "validation": "val",
    "test": "test",
}


class StagePDataGateError(ValueError):
    """Raised when a Stage P comparison would cross a frozen data boundary."""


@dataclass(frozen=True, slots=True)
class NdisparkManifestRow:
    split: str
    role: str
    image_id: str
    file_name: str
    camera_id: str
    width: int
    height: int
    truth_type: str
    vehicle_box_count: int | None
    vehicle_count: int | None
    sha256: str


def load_stage_p_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StagePDataGateError("Stage P YAML root must be a mapping")
    if payload.get("protocol_id") != STAGE_P_PROTOCOL_ID:
        raise StagePDataGateError("Unexpected Stage P protocol ID")
    inference = payload.get("shared_inference", {})
    expected = {
        "api": "ultralytics.YOLO.predict",
        "imgsz": 640,
        "confidence": 0.30,
        "nms_iou": 0.70,
        "agnostic_nms": True,
        "max_detections": 300,
        "classes": [0],
        "model_track_prohibited": True,
    }
    for key, value in expected.items():
        if inference.get(key) != value:
            raise StagePDataGateError(
                f"Stage P shared inference differs at {key!r}"
            )
    if payload.get("metrics", {}).get("ap_semantics") != AP_SEMANTICS:
        raise StagePDataGateError("Stage P AP semantics are not frozen")
    return payload


def verify_bound_file(
    path: Path, *, expected_bytes: int, expected_sha256: str
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != int(expected_bytes):
        raise StagePDataGateError(
            f"Size mismatch for {path}: {actual_bytes} != {expected_bytes}"
        )
    if actual_sha256 != str(expected_sha256):
        raise StagePDataGateError(f"SHA-256 mismatch for {path}")
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "verified": True,
    }


def shared_detector_settings(
    model_paths: Mapping[str, Path], *, device: str
) -> dict[str, DetectorOnlySettings]:
    if set(model_paths) != set(MODEL_IDS):
        raise StagePDataGateError(
            f"Exactly {list(MODEL_IDS)} must be compared"
        )
    settings = {
        model_id: DetectorOnlySettings(
            weights=Path(model_paths[model_id]).resolve(),
            imgsz=640,
            confidence=0.30,
            nms_iou=0.70,
            agnostic_nms=True,
            max_detections=300,
            device=device,
        )
        for model_id in MODEL_IDS
    }
    comparable = {
        (
            row.imgsz,
            row.confidence,
            row.nms_iou,
            row.agnostic_nms,
            row.max_detections,
            row.device,
        )
        for row in settings.values()
    }
    if len(comparable) != 1:
        raise StagePDataGateError("D1 and D1-LL settings are not identical")
    return settings


def _optional_int(value: str | None) -> int | None:
    if value is None or not str(value).strip():
        return None
    return int(value)


def load_manifest(path: Path, *, expected_split: str) -> list[NdisparkManifestRow]:
    if expected_split not in SPLIT_DIRECTORY:
        raise StagePDataGateError(f"Unsupported split: {expected_split}")
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise StagePDataGateError(f"Empty manifest: {path}")
    rows: list[NdisparkManifestRow] = []
    seen: set[str] = set()
    for raw in raw_rows:
        split = str(raw.get("split", "")).strip()
        if split != expected_split:
            raise StagePDataGateError(
                f"Unexpected split {split!r} in {path}"
            )
        file_name = str(raw.get("file_name", "")).strip()
        digest = str(raw.get("sha256", "")).strip().lower()
        if not file_name or len(digest) != 64:
            raise StagePDataGateError(f"Incomplete manifest row in {path}")
        if file_name in seen:
            raise StagePDataGateError(
                f"Duplicate manifest image name: {file_name}"
            )
        seen.add(file_name)
        rows.append(
            NdisparkManifestRow(
                split=split,
                role=str(raw.get("role", "")).strip(),
                image_id=str(raw.get("image_id", "")).strip(),
                file_name=file_name,
                camera_id=str(raw.get("camera_id", "")).strip(),
                width=int(raw["width"]),
                height=int(raw["height"]),
                truth_type=str(raw.get("truth_type", "")).strip().lower(),
                vehicle_box_count=_optional_int(raw.get("vehicle_box_count")),
                vehicle_count=_optional_int(raw.get("vehicle_count")),
                sha256=digest,
            )
        )
    return rows


def partition_known_truth(
    rows: Sequence[NdisparkManifestRow],
) -> tuple[list[NdisparkManifestRow], int]:
    known_types = {"vehicle_boxes", "vehicle_count"}
    known = [row for row in rows if row.truth_type in known_types]
    unknown = len(rows) - len(known)
    return known, unknown


def audit_manifest_groups(
    rows_by_split: Mapping[str, Sequence[NdisparkManifestRow]],
) -> dict[str, Any]:
    if set(rows_by_split) != set(SPLIT_DIRECTORY):
        raise StagePDataGateError("train, validation, and test are required")
    digest_owner: dict[str, str] = {}
    duplicate_digests: list[dict[str, str]] = []
    for split, rows in rows_by_split.items():
        if not rows:
            raise StagePDataGateError(f"Empty Stage P split: {split}")
        for row in rows:
            previous = digest_owner.setdefault(row.sha256, split)
            if previous != split:
                duplicate_digests.append(
                    {
                        "sha256": row.sha256,
                        "first_split": previous,
                        "second_split": split,
                    }
                )
    if duplicate_digests:
        raise StagePDataGateError(
            "Exact image leakage across data groups: "
            f"{duplicate_digests[:3]}"
        )
    return {
        "split_counts": {
            split: len(rows) for split, rows in rows_by_split.items()
        },
        "exact_cross_split_duplicate_images": 0,
        "camera_independent_split": False,
        "note": (
            "Exact image hashes are disjoint; cameras overlap and the split "
            "is not a camera-independent generalization test."
        ),
    }


def resolve_image_path(
    prepared_root: Path, row: NdisparkManifestRow
) -> Path:
    return (
        prepared_root.resolve()
        / "images"
        / SPLIT_DIRECTORY[row.split]
        / row.file_name
    )


def resolve_label_path(
    prepared_root: Path, row: NdisparkManifestRow
) -> Path:
    return (
        prepared_root.resolve()
        / "labels"
        / SPLIT_DIRECTORY[row.split]
        / f"{Path(row.file_name).stem}.txt"
    )


def verify_source_rows(
    prepared_root: Path,
    rows_by_split: Mapping[str, Sequence[NdisparkManifestRow]],
) -> dict[str, Any]:
    verified: list[dict[str, Any]] = []
    for split, rows in rows_by_split.items():
        for row in rows:
            image_path = resolve_image_path(prepared_root, row)
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            actual = sha256_file(image_path)
            if actual != row.sha256:
                raise StagePDataGateError(
                    f"Source image hash mismatch: {image_path}"
                )
            if row.truth_type == "vehicle_boxes":
                label_path = resolve_label_path(prepared_root, row)
                if not label_path.is_file():
                    raise StagePDataGateError(
                        f"Missing box annotation: {label_path}"
                    )
            verified.append(
                {
                    "split": split,
                    "file_name": row.file_name,
                    "path": str(image_path),
                    "bytes": image_path.stat().st_size,
                    "sha256": actual,
                }
            )
    return {
        "protocol_id": STAGE_P_PROTOCOL_ID,
        "verified": True,
        "verified_image_count": len(verified),
        "images": verified,
    }


def parse_yolo_vehicle_truth(
    label_path: Path,
    *,
    row: NdisparkManifestRow,
    frame_number: int,
) -> list[LmotAnnotation]:
    if row.truth_type != "vehicle_boxes":
        return []
    if not label_path.is_file():
        raise StagePDataGateError(f"Missing annotation: {label_path}")
    annotations: list[LmotAnnotation] = []
    text = label_path.read_text(encoding="utf-8").strip()
    for index, line in enumerate(text.splitlines(), start=1):
        parts = line.split()
        if len(parts) != 5:
            raise StagePDataGateError(
                f"Invalid YOLO label at {label_path}:{index}"
            )
        class_id = int(parts[0])
        if class_id != 0:
            raise StagePDataGateError(
                f"Unexpected NDISPark class {class_id} at {label_path}:{index}"
            )
        cx, cy, width, height = (float(value) for value in parts[1:])
        if (
            not all(math.isfinite(value) for value in (cx, cy, width, height))
            or width <= 0
            or height <= 0
            or not all(0.0 <= value <= 1.0 for value in (cx, cy, width, height))
        ):
            raise StagePDataGateError(
                f"Invalid normalized box at {label_path}:{index}"
            )
        x = (cx - width / 2.0) * row.width
        y = (cy - height / 2.0) * row.height
        annotations.append(
            LmotAnnotation(
                frame_number=frame_number,
                track_id=index,
                x=x,
                y=y,
                width=width * row.width,
                height=height * row.height,
                ignore=1,
                class_id=3,
                visibility=1.0,
            )
        )
    if row.vehicle_box_count is not None and len(annotations) != row.vehicle_box_count:
        raise StagePDataGateError(
            f"Box count mismatch for {row.file_name}: "
            f"{len(annotations)} != {row.vehicle_box_count}"
        )
    return annotations


def _public_box_metrics(
    metrics: Mapping[str, float | int],
) -> dict[str, float | int | str | bool]:
    return {
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        AP50_KEY: metrics["AP50"],
        AP50_95_KEY: metrics["AP50-95"],
        "ground_truth_boxes": metrics["ground_truth_boxes"],
        "predicted_boxes": metrics["predicted_boxes"],
        "true_positives": metrics["true_positives"],
        "false_positives": metrics["false_positives"],
        "false_negatives": metrics["false_negatives"],
        "AP_semantics": AP_SEMANTICS,
        "standard_COCO_AP": False,
    }


def _evaluate_box_group(
    rows: Sequence[
        tuple[
            NdisparkManifestRow,
            int,
            Sequence[LmotAnnotation],
            Sequence[TrackPrediction],
        ]
    ],
) -> dict[str, Any]:
    if not rows:
        raise StagePDataGateError("Cannot evaluate an empty box group")
    gt: list[LmotAnnotation] = []
    predictions: list[TrackPrediction] = []
    per_camera_inputs: dict[
        str, tuple[list[LmotAnnotation], list[TrackPrediction]]
    ] = {}
    for row, frame_number, frame_gt, frame_predictions in rows:
        gt.extend(frame_gt)
        predictions.extend(frame_predictions)
        camera_gt, camera_predictions = per_camera_inputs.setdefault(
            row.camera_id, ([], [])
        )
        camera_gt.extend(frame_gt)
        camera_predictions.extend(frame_predictions)
    pooled = _public_box_metrics(
        evaluate_motor_vehicle_detections(gt=gt, predictions=predictions)
    )
    per_camera = {
        camera_id: _public_box_metrics(
            evaluate_motor_vehicle_detections(
                gt=camera_gt,
                predictions=camera_predictions,
            )
        )
        for camera_id, (camera_gt, camera_predictions) in sorted(
            per_camera_inputs.items()
        )
    }
    macro = {
        key: float(np.mean([float(row[key]) for row in per_camera.values()]))
        for key in ("precision", "recall", AP50_KEY, AP50_95_KEY)
    }
    macro["definition"] = "unweighted_mean_of_per_camera_rates"
    return {
        "pooled_micro": {
            **pooled,
            "definition": (
                "all_boxes_and_predictions_in_the_group_with_summed_counts"
            ),
        },
        "per_camera_macro": macro,
        "per_camera": per_camera,
    }


def _per_image_counts(
    gt: Sequence[LmotAnnotation],
    predictions: Sequence[TrackPrediction],
) -> tuple[int, int, int]:
    metrics = evaluate_motor_vehicle_detections(gt=gt, predictions=predictions)
    return (
        int(metrics["true_positives"]),
        int(metrics["false_positives"]),
        int(metrics["false_negatives"]),
    )


def _runtime_environment(device: str) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "device_argument": device,
    }
    try:
        import torch

        environment.update(
            {
                "torch": torch.__version__,
                "cuda_available": torch.cuda.is_available(),
                "cuda_version": torch.version.cuda,
                "gpu_name": (
                    torch.cuda.get_device_name(0)
                    if torch.cuda.is_available()
                    else None
                ),
            }
        )
    except ImportError:
        environment["torch"] = None
    try:
        import ultralytics

        environment["ultralytics"] = ultralytics.__version__
    except ImportError:
        environment["ultralytics"] = None
    return environment


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    if not rows:
        raise StagePDataGateError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _model_metrics(
    *,
    model_id: str,
    box_rows: Sequence[
        tuple[
            NdisparkManifestRow,
            int,
            Sequence[LmotAnnotation],
            Sequence[TrackPrediction],
        ]
    ],
    count_truth: Sequence[dict[str, Any]],
    count_predictions: Sequence[dict[str, Any]],
    runtime: Mapping[str, Any],
    unknown_truth_excluded: int,
) -> dict[str, Any]:
    train = [item for item in box_rows if item[0].split == "train"]
    validation = [
        item for item in box_rows if item[0].split == "validation"
    ]
    return {
        "schema_version": 1,
        "protocol_id": STAGE_P_PROTOCOL_ID,
        "model_id": model_id,
        "task": "raw_detector_only_parking_domain_vehicle_detection",
        "raw_detector_only": True,
        "tracker_emitted_boxes": False,
        "parking_slot_occupancy_evaluation": False,
        "AP_semantics": AP_SEMANTICS,
        "standard_COCO_AP": False,
        "ignored_class_suppression": {
            "enabled": False,
            "ground_truth_derived": False,
        },
        "unknown_truth_excluded": unknown_truth_excluded,
        "groups": {
            "daytime_train_training_resubstitution": _evaluate_box_group(train),
            "night_validation_consumed_development": _evaluate_box_group(
                validation
            ),
            "all_box_labelled_descriptive": _evaluate_box_group(box_rows),
            "night_test_consumed_posthoc_count": evaluate_count_rows(
                count_truth, count_predictions
            ),
        },
        "runtime": dict(runtime),
    }


def _metric_delta(
    candidate: Mapping[str, Any], baseline: Mapping[str, Any]
) -> dict[str, float]:
    return {
        key: float(candidate[key]) - float(baseline[key])
        for key in ("precision", "recall", AP50_KEY, AP50_95_KEY)
    }


def build_comparison_metrics(
    reports: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    if set(reports) != set(MODEL_IDS):
        raise StagePDataGateError("Comparison requires D1 and D1_LL reports")
    d1_groups = reports["D1"]["groups"]
    ll_groups = reports["D1_LL"]["groups"]
    box_deltas: dict[str, Any] = {}
    for group in (
        "daytime_train_training_resubstitution",
        "night_validation_consumed_development",
        "all_box_labelled_descriptive",
    ):
        box_deltas[group] = _metric_delta(
            ll_groups[group]["pooled_micro"],
            d1_groups[group]["pooled_micro"],
        )
    d1_count = d1_groups["night_test_consumed_posthoc_count"]["metrics"]
    ll_count = ll_groups["night_test_consumed_posthoc_count"]["metrics"]
    count_deltas = {
        key: float(ll_count[key]) - float(d1_count[key])
        for key in ("mae", "rmse", "mean_predicted_count")
    }
    return {
        "schema_version": 1,
        "protocol_id": STAGE_P_PROTOCOL_ID,
        "comparison": "D1_LL_minus_D1",
        "AP_semantics": AP_SEMANTICS,
        "standard_COCO_AP": False,
        "same_inference_settings": True,
        "ignored_class_suppression_enabled": False,
        "models": {key: reports[key] for key in MODEL_IDS},
        "deltas": {
            "box": box_deltas,
            "night_test_count": count_deltas,
        },
        "claim_scope": "consumed_development_retrospective_diagnostic",
        "occupancy_claim_supported": False,
    }


def decide_parking_retention(
    comparison: Mapping[str, Any],
    decision_rule: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply the result-blind Stage P2 rule frozen before inference."""

    try:
        deltas = comparison["deltas"]
        night = deltas["box"]["night_validation_consumed_development"]
        day = deltas["box"]["daytime_train_training_resubstitution"]
        all_box = deltas["box"]["all_box_labelled_descriptive"]
        count = deltas["night_test_count"]
        required = decision_rule["required_night_improvement"]
        day_limits = decision_rule["daytime_training_resubstitution_max_drop"]
        count_limits = decision_rule["count_test_max_increase"]
    except (KeyError, TypeError) as exc:
        return {
            "status": "BLOCKED",
            "reason": f"required_metrics_missing:{exc}",
            "checks": {},
        }
    checks = {
        "night_recall_or_AP50_improvement": (
            float(night["recall"])
            >= float(required["any_of"][0]["recall_gain_at_least"])
            or float(night[AP50_KEY])
            >= float(
                required["any_of"][1][
                    "confidence_truncated_AP50_gain_at_least"
                ]
            )
        ),
        "night_precision_retained": float(night["precision"])
        >= -float(decision_rule["night_precision_drop_no_worse_than"]),
        "night_AP50_95_retained": float(night[AP50_95_KEY])
        >= -float(decision_rule["night_AP50_95_drop_no_worse_than"]),
        "day_precision_retained": float(day["precision"])
        >= -float(day_limits["precision"]),
        "day_recall_retained": float(day["recall"])
        >= -float(day_limits["recall"]),
        "day_AP50_retained": float(day[AP50_KEY])
        >= -float(day_limits["confidence_truncated_AP50"]),
        "day_AP50_95_retained": float(day[AP50_95_KEY])
        >= -float(day_limits["confidence_truncated_AP50_95"]),
        "all_box_AP50_95_retained": float(all_box[AP50_95_KEY])
        >= -float(decision_rule["all_box_labelled_max_AP50_95_drop"]),
        "night_count_MAE_retained": float(count["mae"])
        <= float(count_limits["MAE"]),
        "night_count_RMSE_retained": float(count["rmse"])
        <= float(count_limits["RMSE"]),
    }
    night_required = (
        checks["night_recall_or_AP50_improvement"]
        and checks["night_precision_retained"]
        and checks["night_AP50_95_retained"]
    )
    if not night_required:
        status = "FAIL"
    elif all(checks.values()):
        status = "PASS"
    else:
        status = "CONDITIONAL"
    return {
        "status": status,
        "checks": checks,
        "frozen_rule_applied_without_reselection": True,
        "claim_scope": "consumed_development_retrospective_diagnostic",
        "D1_LL_role_after_decision": "selected low-light detector candidate",
        "P3_LL_default_authorized": False,
    }


def _case_score(record: Mapping[str, Any]) -> tuple[int, int, str]:
    return (
        int(record.get("absolute_error_change", 0)),
        int(record.get("D1_error", 0)),
        str(record["file_name"]),
    )


def classify_qualitative_cases(
    per_model_rows: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, list[dict[str, Any]]]:
    by_model = {
        model: {
            (str(row["split"]), str(row["file_name"])): row for row in rows
        }
        for model, rows in per_model_rows.items()
    }
    if set(by_model) != set(MODEL_IDS):
        raise StagePDataGateError("Qualitative comparison requires two models")
    if set(by_model["D1"]) != set(by_model["D1_LL"]):
        raise StagePDataGateError("Per-image memberships differ")
    cases = {
        "typical_improvements": [],
        "shared_failures": [],
        "potential_regressions": [],
    }
    for key in sorted(by_model["D1"]):
        d1 = by_model["D1"][key]
        ll = by_model["D1_LL"][key]
        if d1["truth_type"] == "vehicle_boxes":
            d1_error = int(d1["false_positives"]) + int(d1["false_negatives"])
            ll_error = int(ll["false_positives"]) + int(ll["false_negatives"])
        elif d1["truth_type"] == "vehicle_count":
            d1_error = abs(int(d1["predicted_count"]) - int(d1["true_count"]))
            ll_error = abs(int(ll["predicted_count"]) - int(ll["true_count"]))
        else:
            continue
        record = {
            "split": key[0],
            "file_name": key[1],
            "camera_id": d1["camera_id"],
            "truth_type": d1["truth_type"],
            "D1_error": d1_error,
            "D1_LL_error": ll_error,
            "absolute_error_change": d1_error - ll_error,
        }
        if ll_error < d1_error:
            cases["typical_improvements"].append(record)
        elif ll_error > d1_error:
            cases["potential_regressions"].append(record)
        elif d1_error > 0:
            cases["shared_failures"].append(record)
    cases["typical_improvements"].sort(key=_case_score, reverse=True)
    cases["potential_regressions"].sort(
        key=lambda row: (-int(row["absolute_error_change"]), str(row["file_name"]))
    )
    cases["shared_failures"].sort(
        key=lambda row: (int(row["D1_error"]), str(row["file_name"])),
        reverse=True,
    )
    return {key: rows[:12] for key, rows in cases.items()}


def render_comparison_contact_sheet(
    *,
    output_path: Path,
    prepared_root: Path,
    cases: Mapping[str, Sequence[Mapping[str, Any]]],
    detections: Mapping[
        str, Mapping[tuple[str, str], Sequence[TrackPrediction]]
    ],
    truths: Mapping[tuple[str, str], Sequence[LmotAnnotation]],
) -> None:
    selected: list[tuple[str, Mapping[str, Any]]] = []
    for category in (
        "typical_improvements",
        "shared_failures",
        "potential_regressions",
    ):
        selected.extend((category, row) for row in cases.get(category, [])[:2])
    tile_width, tile_height = 640, 280
    sheet = np.full(
        (max(1, len(selected)) * tile_height, tile_width * 2, 3),
        28,
        dtype=np.uint8,
    )
    for row_index, (category, case) in enumerate(selected):
        split = str(case["split"])
        file_name = str(case["file_name"])
        path = (
            prepared_root
            / "images"
            / SPLIT_DIRECTORY[split]
            / file_name
        )
        original = read_image(path)
        if original is None:
            continue
        for column, model_id in enumerate(MODEL_IDS):
            image = original.copy()
            for truth in truths.get((split, file_name), []):
                x1, y1, x2, y2 = (int(value) for value in truth.xyxy)
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 210, 0), 3)
            for prediction in detections[model_id].get((split, file_name), []):
                x1, y1, x2, y2 = (
                    int(value) for value in prediction.xyxy
                )
                cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 230), 3)
            image = cv2.resize(image, (tile_width, tile_height))
            label = (
                f"{category} | {model_id} | {split}/{file_name} | "
                f"error={case[f'{model_id}_error']}"
            )
            cv2.rectangle(image, (0, 0), (tile_width, 30), (0, 0, 0), -1)
            cv2.putText(
                image,
                label,
                (7, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            y1 = row_index * tile_height
            sheet[y1 : y1 + tile_height, column * tile_width : (column + 1) * tile_width] = image
    write_image(output_path, sheet)


def run_stage_p_retention(
    *,
    protocol_path: Path,
    prepared_root: Path,
    manifest_paths: Mapping[str, Path],
    model_paths: Mapping[str, Path],
    output_root: Path,
    device: str = "0",
    model_factories: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the additive, non-overwriting Stage P1 D1/D1-LL comparison."""

    protocol_path = protocol_path.resolve()
    prepared_root = prepared_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    protocol = load_stage_p_protocol(protocol_path)
    if set(manifest_paths) != set(SPLIT_DIRECTORY):
        raise StagePDataGateError("All three NDISPark manifests are required")

    verified_inputs = {
        "protocol": verify_bound_file(
            protocol_path,
            expected_bytes=protocol_path.stat().st_size,
            expected_sha256=sha256_file(protocol_path),
        )
    }
    rows_by_split: dict[str, list[NdisparkManifestRow]] = {}
    for split, path in manifest_paths.items():
        expected = protocol["inputs"]["manifests"][split]
        verified_inputs[f"manifest_{split}"] = verify_bound_file(
            Path(path),
            expected_bytes=int(expected["bytes"]),
            expected_sha256=str(expected["sha256"]),
        )
        rows_by_split[split] = load_manifest(
            Path(path), expected_split=split
        )
    membership_audit = audit_manifest_groups(rows_by_split)
    source_verification = verify_source_rows(prepared_root, rows_by_split)

    for model_id, path in model_paths.items():
        expected = protocol["inputs"]["models"][model_id]
        verified_inputs[f"weights_{model_id}"] = verify_bound_file(
            Path(path),
            expected_bytes=int(expected["bytes"]),
            expected_sha256=str(expected["sha256"]),
        )
    settings = shared_detector_settings(model_paths, device=device)

    output_root.mkdir(parents=True)
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(output_root / "_ultralytics_config")
    )
    _write_json(
        output_root / "source_image_verification.json",
        {
            **source_verification,
            "membership_audit": membership_audit,
        },
    )
    snapshot = {
        "protocol": protocol,
        "execution_inputs": {
            "prepared_root": str(prepared_root),
            "verified_inputs": verified_inputs,
        },
    }
    (output_root / "config_snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )

    all_rows = [
        row
        for split in ("train", "validation", "test")
        for row in rows_by_split[split]
    ]
    known_rows, unknown_truth_excluded = partition_known_truth(all_rows)
    if not known_rows:
        raise StagePDataGateError("No known Stage P truth is available")

    truths: dict[tuple[str, str], list[LmotAnnotation]] = {}
    frame_numbers: dict[tuple[str, str], int] = {}
    for frame_number, row in enumerate(all_rows, start=1):
        key = (row.split, row.file_name)
        frame_numbers[key] = frame_number
        truths[key] = parse_yolo_vehicle_truth(
            resolve_label_path(prepared_root, row),
            row=row,
            frame_number=frame_number,
        )

    reports: dict[str, dict[str, Any]] = {}
    per_model_rows: dict[str, list[dict[str, Any]]] = {}
    detections_by_model: dict[
        str, dict[tuple[str, str], list[TrackPrediction]]
    ] = {}
    for model_id in MODEL_IDS:
        method_root = output_root / model_id
        method_root.mkdir()
        factory = (
            None
            if model_factories is None
            else model_factories.get(model_id)
        )
        adapter = FrozenRawDetectorAdapter(
            settings[model_id], model_factory=factory
        )
        try:
            import torch
        except ImportError:  # pragma: no cover
            torch = None
        measure_cuda = (
            torch is not None
            and torch.cuda.is_available()
            and device.lower() not in {"cpu", "mps"}
        )
        if measure_cuda:
            torch.cuda.reset_peak_memory_stats()

        detection_lines: list[str] = []
        image_rows: list[dict[str, Any]] = []
        count_truth: list[dict[str, Any]] = []
        count_predictions: list[dict[str, Any]] = []
        box_rows: list[
            tuple[
                NdisparkManifestRow,
                int,
                Sequence[LmotAnnotation],
                Sequence[TrackPrediction],
            ]
        ] = []
        model_detections: dict[
            tuple[str, str], list[TrackPrediction]
        ] = {}
        timing_totals: defaultdict[str, float] = defaultdict(float)
        wall_started = time.perf_counter()
        for row in all_rows:
            frame_number = frame_numbers[(row.split, row.file_name)]
            image_path = resolve_image_path(prepared_root, row)
            image = read_image(image_path)
            if image is None:
                raise StagePDataGateError(f"Could not decode {image_path}")
            started = time.perf_counter()
            predictions, speed = adapter.predict(
                image, frame_number=frame_number
            )
            timing_totals["wall_predict_ms"] += (
                time.perf_counter() - started
            ) * 1000.0
            for key in ("preprocess", "inference", "postprocess"):
                timing_totals[f"framework_{key}_ms"] += float(
                    speed.get(key, 0.0)
                )
            model_detections[(row.split, row.file_name)] = predictions
            gt = truths[(row.split, row.file_name)]
            tp = fp = fn = None
            if row.truth_type == "vehicle_boxes":
                tp, fp, fn = _per_image_counts(gt, predictions)
                box_rows.append((row, frame_number, gt, predictions))
            predicted_count = len(predictions)
            if row.truth_type == "vehicle_count":
                if row.vehicle_count is None:
                    raise StagePDataGateError(
                        f"Missing count truth: {row.file_name}"
                    )
                count_truth.append(
                    {
                        "file_name": row.file_name,
                        "camera_id": row.camera_id,
                        "vehicle_count": row.vehicle_count,
                    }
                )
                count_predictions.append(
                    {
                        "file_name": row.file_name,
                        "camera_id": row.camera_id,
                        "predicted_count": predicted_count,
                    }
                )
            record = {
                "protocol_id": STAGE_P_PROTOCOL_ID,
                "model_id": model_id,
                "split": row.split,
                "role": row.role,
                "lighting": (
                    "daytime" if row.split == "train" else "nighttime"
                ),
                "file_name": row.file_name,
                "camera_id": row.camera_id,
                "truth_type": row.truth_type,
                "true_count": (
                    row.vehicle_box_count
                    if row.truth_type == "vehicle_boxes"
                    else row.vehicle_count
                ),
                "predicted_count": predicted_count,
                "true_positives": tp,
                "false_positives": fp,
                "false_negatives": fn,
                "source_image_sha256": row.sha256,
            }
            image_rows.append(record)
            detection_lines.append(
                json.dumps(
                    {
                        **{
                            key: record[key]
                            for key in (
                                "protocol_id",
                                "model_id",
                                "split",
                                "role",
                                "lighting",
                                "file_name",
                                "camera_id",
                                "source_image_sha256",
                            )
                        },
                        "inference": {
                            "imgsz": 640,
                            "confidence": 0.30,
                            "nms_iou": 0.70,
                            "agnostic_nms": True,
                            "max_detections": 300,
                            "classes": [0],
                        },
                        "detections": [
                            {
                                "bbox_xyxy": list(prediction.xyxy),
                                "confidence": prediction.confidence,
                                "class_id": 0,
                                "class_name": "vehicle",
                            }
                            for prediction in predictions
                        ],
                    },
                    ensure_ascii=False,
                )
            )
        wall_seconds = time.perf_counter() - wall_started
        runtime = {
            "protocol_id": STAGE_P_PROTOCOL_ID,
            "model_id": model_id,
            "inference_api": "ultralytics.YOLO.predict",
            "inference_performed": True,
            "model_loaded": True,
            "model_predict_called": True,
            "model_predict_call_count": adapter.predict_calls,
            "model_track_called": False,
            "tracker_loaded": False,
            "training_performed": False,
            "settings": {
                "imgsz": 640,
                "confidence": 0.30,
                "nms_iou": 0.70,
                "agnostic_nms": True,
                "max_detections": 300,
                "classes": [0],
                "augment": False,
                "rect": False,
                "batch": 1,
                "device": device,
            },
            "evaluated_images": len(all_rows),
            "wall_seconds": wall_seconds,
            "wall_ms_per_image": wall_seconds * 1000.0 / len(all_rows),
            "wall_fps": len(all_rows) / wall_seconds,
            "framework_ms_per_image": {
                key: value / len(all_rows)
                for key, value in sorted(timing_totals.items())
            },
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated())
                if measure_cuda
                else None
            ),
            "runtime_is_descriptive_not_a_selection_metric": True,
            "environment": _runtime_environment(device),
            "weights": verified_inputs[f"weights_{model_id}"],
        }
        report = _model_metrics(
            model_id=model_id,
            box_rows=box_rows,
            count_truth=count_truth,
            count_predictions=count_predictions,
            runtime=runtime,
            unknown_truth_excluded=unknown_truth_excluded,
        )
        (method_root / "detections.jsonl").write_text(
            "\n".join(detection_lines) + "\n", encoding="utf-8"
        )
        _write_csv(method_root / "per_image_statistics.csv", image_rows)
        _write_csv(method_root / "count_predictions.csv", count_predictions)
        _write_json(method_root / "metrics.json", report)
        _write_json(method_root / "runtime_metadata.json", runtime)
        reports[model_id] = report
        per_model_rows[model_id] = image_rows
        detections_by_model[model_id] = model_detections

    comparison = build_comparison_metrics(reports)
    _write_json(output_root / "comparison_metrics.json", comparison)
    summary = {
        "protocol_id": STAGE_P_PROTOCOL_ID,
        "experiment": "NDISPark_D1_vs_D1_LL_parking_domain_retention",
        "status": "complete_consumed_development_retrospective_diagnostic",
        "same_inference_settings": True,
        "AP_semantics": AP_SEMANTICS,
        "standard_COCO_AP": False,
        "ignored_class_suppression_enabled": False,
        "P4_executed": False,
        "occupancy_claim_supported": False,
        "D1_LL_role": "selected low-light detector candidate",
    }
    _write_json(output_root / "comparison_summary.json", summary)
    cases = classify_qualitative_cases(per_model_rows)
    _write_json(
        output_root / "failure_cases.json",
        {
            "protocol_id": STAGE_P_PROTOCOL_ID,
            "box_matching_iou": 0.50,
            "categories": cases,
            "interpretation": (
                "Retrospective examples only; green boxes are ground truth "
                "when available and red boxes are frozen-threshold predictions."
            ),
        },
    )
    render_comparison_contact_sheet(
        output_path=output_root / "D1_vs_D1_LL_contact_sheet.jpg",
        prepared_root=prepared_root,
        cases=cases,
        detections=detections_by_model,
        truths=truths,
    )
    return comparison
