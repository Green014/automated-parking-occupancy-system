import json
from pathlib import Path

import yaml

from parking_occupancy.stage_o_artifacts import (
    artifact_record,
    DETECTOR_ONLY_OUTPUT_FILES,
    verify_artifact_records,
    verify_detector_only_output,
    verify_stage_o_registry,
)
from parking_occupancy.stage_o_low_light import STAGE_O_PROTOCOL_ID


def test_artifact_validation_detects_additive_output_corruption(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "metrics.json"
    artifact.write_text("{}\n", encoding="utf-8")
    record = artifact_record(
        label="metrics", path=artifact, role="formal_output"
    )

    assert verify_artifact_records([record])["verified"] is True
    artifact.write_text('{"changed": true}\n', encoding="utf-8")
    result = verify_artifact_records([record])
    assert result["verified"] is False
    assert result["errors"] == ["bytes:metrics"]


def test_registry_verifies_exact_count_and_hashes(tmp_path: Path) -> None:
    artifact = tmp_path / "report.md"
    artifact.write_text("# report\n", encoding="utf-8")
    record = artifact_record(label="report", path=artifact, role="report")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "protocol_id": STAGE_O_PROTOCOL_ID,
                "artifact_count": 1,
                "artifacts": [record],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    assert verify_stage_o_registry(registry)["verified"] is True


def test_detector_only_output_verifier_checks_counts_and_no_tracker(
    tmp_path: Path,
) -> None:
    output = tmp_path / "O0"
    output.mkdir()
    row = {
        "ground_truth_boxes": 2,
        "predicted_boxes": 2,
        "true_positives": 1,
        "false_positives": 1,
        "false_negatives": 1,
        "precision": 0.5,
        "recall": 0.5,
        "AP50": 0.5,
        "AP50-95": 0.25,
    }
    metrics = {
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "method_id": "O0",
        "task": "raw_detector_only_unified_motor_vehicle_box_detection",
        "tracker_emitted_boxes": False,
        "model_track_called": False,
        "illumination": {
            "dark": {
                "pooled_micro": row,
                "per_sequence": {"LMOT-05": row},
            }
        },
    }
    runtime = {
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "method_id": "O0",
        "inference_api": "ultralytics.YOLO.predict",
        "model_loaded": True,
        "inference_performed": True,
        "model_predict_called": True,
        "model_predict_call_count": 2,
        "model_track_called": False,
        "tracker_loaded": False,
        "training_performed": False,
        "evaluated_frames": 2,
        "settings": {
            "imgsz": 640,
            "confidence": 0.30,
            "nms_iou": 0.70,
            "agnostic_nms": True,
            "max_detections": 300,
        },
    }
    metrics["illumination"]["light"] = {
        "pooled_micro": row,
        "per_sequence": {"LMOT-05": row},
    }
    for name in DETECTOR_ONLY_OUTPUT_FILES:
        path = output / name
        if name == "metrics.json":
            path.write_text(json.dumps(metrics), encoding="utf-8")
        elif name == "runtime_metadata.json":
            path.write_text(json.dumps(runtime), encoding="utf-8")
        else:
            path.write_bytes(b"fixture")

    assert verify_detector_only_output(
        output, expected_method="O0"
    )["verified"] is True
    metrics["illumination"]["dark"]["pooled_micro"][
        "false_negatives"
    ] = 2
    (output / "metrics.json").write_text(
        json.dumps(metrics), encoding="utf-8"
    )
    result = verify_detector_only_output(output, expected_method="O0")
    assert result["verified"] is False
    assert "counts:GT:dark:pooled_micro" in result["errors"]
