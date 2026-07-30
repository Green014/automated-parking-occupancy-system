from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .stage_n_lmot import sha256_file
from .stage_o_low_light import STAGE_O_PROTOCOL_ID


DETECTOR_ONLY_OUTPUT_FILES = (
    "detections.jsonl",
    "metrics.json",
    "runtime_metadata.json",
    "config_snapshot.yaml",
    "qualitative_contact_sheet.jpg",
    "failure_cases.json",
)


def artifact_record(
    *,
    label: str,
    path: Path,
    role: str,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "label": label,
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_artifact_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    count = 0
    for count, record in enumerate(records, start=1):
        label = str(record["label"])
        path = Path(str(record["path"]))
        if not path.is_file():
            errors.append(f"missing:{label}")
            continue
        if path.stat().st_size != int(record["bytes"]):
            errors.append(f"bytes:{label}")
            continue
        if sha256_file(path) != str(record["sha256"]):
            errors.append(f"sha256:{label}")
    return {
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "artifact_count": count,
        "verified": not errors,
        "errors": errors,
    }


def verify_stage_o_registry(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_O_PROTOCOL_ID:
        raise ValueError("Unexpected Stage O registry protocol")
    result = verify_artifact_records(payload["artifacts"])
    expected = int(payload["artifact_count"])
    if result["artifact_count"] != expected:
        result["errors"].append("artifact_count")
        result["verified"] = False
    result["registry_path"] = str(path)
    result["registry_sha256"] = sha256_file(path)
    return result


def verify_detector_only_output(
    output_root: Path,
    *,
    expected_method: str,
) -> dict[str, Any]:
    """Verify Stage O output scope, count identities, and no-tracker flags."""

    output_root = output_root.resolve()
    errors: list[str] = []
    if not output_root.is_dir():
        return {
            "output_root": str(output_root),
            "method_id": expected_method,
            "verified": False,
            "errors": ["missing_output_root"],
            "artifacts": [],
        }
    missing = [
        name
        for name in DETECTOR_ONLY_OUTPUT_FILES
        if not (output_root / name).is_file()
    ]
    errors.extend(f"missing:{name}" for name in missing)
    if missing:
        return {
            "output_root": str(output_root),
            "method_id": expected_method,
            "verified": False,
            "errors": errors,
            "artifacts": [],
        }

    metrics = json.loads(
        (output_root / "metrics.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (output_root / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    expected_metric_values = {
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "method_id": expected_method,
        "task": "raw_detector_only_unified_motor_vehicle_box_detection",
        "tracker_emitted_boxes": False,
        "model_track_called": False,
    }
    for key, expected in expected_metric_values.items():
        if metrics.get(key) != expected:
            errors.append(f"metrics:{key}")
    expected_runtime_values = {
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "method_id": expected_method,
        "inference_api": "ultralytics.YOLO.predict",
        "inference_performed": True,
        "model_loaded": True,
        "model_predict_called": True,
        "model_track_called": False,
        "tracker_loaded": False,
        "training_performed": False,
    }
    for key, expected in expected_runtime_values.items():
        if runtime.get(key) != expected:
            errors.append(f"runtime:{key}")
    frozen_settings = {
        "imgsz": 640,
        "confidence": 0.30,
        "nms_iou": 0.70,
        "agnostic_nms": True,
        "max_detections": 300,
    }
    for key, expected in frozen_settings.items():
        if runtime.get("settings", {}).get(key) != expected:
            errors.append(f"settings:{key}")
    if runtime.get("model_predict_call_count") != runtime.get(
        "evaluated_frames"
    ):
        errors.append("runtime:predict_call_count")
    if set(metrics.get("illumination", {})) != {"light", "dark"}:
        errors.append("metrics:formal_illumination_streams")

    for illumination, aggregate in metrics.get("illumination", {}).items():
        rows = [
            (f"{illumination}:pooled_micro", aggregate["pooled_micro"]),
            *[
                (f"{illumination}:{sequence}", row)
                for sequence, row in aggregate["per_sequence"].items()
            ],
        ]
        for label, row in rows:
            gt = int(row["ground_truth_boxes"])
            pred = int(row["predicted_boxes"])
            tp = int(row["true_positives"])
            fp = int(row["false_positives"])
            fn = int(row["false_negatives"])
            if gt != tp + fn:
                errors.append(f"counts:GT:{label}")
            if pred != tp + fp:
                errors.append(f"counts:Pred:{label}")
    artifacts = [
        artifact_record(
            label=name,
            path=output_root / name,
            role="formal_detector_only_output",
        )
        for name in DETECTOR_ONLY_OUTPUT_FILES
    ]
    return {
        "output_root": str(output_root),
        "method_id": expected_method,
        "verified": not errors,
        "errors": errors,
        "artifacts": artifacts,
    }
