from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from parking_occupancy.stage_p_retention import (
    AP50_95_KEY,
    AP50_KEY,
    AP_SEMANTICS,
    NdisparkManifestRow,
    StagePDataGateError,
    _public_box_metrics,
    audit_manifest_groups,
    decide_parking_retention,
    load_manifest,
    load_stage_p_protocol,
    partition_known_truth,
    shared_detector_settings,
    verify_bound_file,
)


def _row(
    *,
    split: str,
    name: str,
    digest: str,
    truth_type: str = "vehicle_boxes",
) -> NdisparkManifestRow:
    return NdisparkManifestRow(
        split=split,
        role="diagnostic",
        image_id=name,
        file_name=name,
        camera_id="60",
        width=100,
        height=80,
        truth_type=truth_type,
        vehicle_box_count=1 if truth_type == "vehicle_boxes" else None,
        vehicle_count=1 if truth_type == "vehicle_count" else None,
        sha256=digest,
    )


def _comparison(
    *,
    night_recall: float = 0.03,
    night_precision: float = -0.01,
    night_ap50_95: float = 0.0,
    day_drop: float = 0.0,
    count_mae: float = 0.0,
) -> dict:
    box = {
        "daytime_train_training_resubstitution": {
            "precision": day_drop,
            "recall": day_drop,
            AP50_KEY: day_drop,
            AP50_95_KEY: day_drop,
        },
        "night_validation_consumed_development": {
            "precision": night_precision,
            "recall": night_recall,
            AP50_KEY: 0.0,
            AP50_95_KEY: night_ap50_95,
        },
        "all_box_labelled_descriptive": {
            "precision": 0.0,
            "recall": 0.0,
            AP50_KEY: 0.0,
            AP50_95_KEY: 0.0,
        },
    }
    return {
        "deltas": {
            "box": box,
            "night_test_count": {
                "mae": count_mae,
                "rmse": 0.0,
                "mean_predicted_count": 0.0,
            },
        }
    }


def _rule() -> dict:
    return {
        "required_night_improvement": {
            "any_of": [
                {"recall_gain_at_least": 0.02},
                {"confidence_truncated_AP50_gain_at_least": 0.02},
            ]
        },
        "night_precision_drop_no_worse_than": 0.05,
        "night_AP50_95_drop_no_worse_than": 0.02,
        "daytime_training_resubstitution_max_drop": {
            "precision": 0.05,
            "recall": 0.05,
            "confidence_truncated_AP50": 0.05,
            "confidence_truncated_AP50_95": 0.05,
        },
        "all_box_labelled_max_AP50_95_drop": 0.05,
        "count_test_max_increase": {"MAE": 0.50, "RMSE": 1.00},
    }


def test_d1_and_d1_ll_have_identical_frozen_inference_settings(
    tmp_path: Path,
) -> None:
    settings = shared_detector_settings(
        {
            "D1": tmp_path / "d1.pt",
            "D1_LL": tmp_path / "d1_ll.pt",
        },
        device="cpu",
    )
    left = settings["D1"]
    right = settings["D1_LL"]
    assert (
        left.imgsz,
        left.confidence,
        left.nms_iou,
        left.agnostic_nms,
        left.max_detections,
        left.device,
    ) == (
        right.imgsz,
        right.confidence,
        right.nms_iou,
        right.agnostic_nms,
        right.max_detections,
        right.device,
    )
    assert left.confidence == 0.30


def test_protocol_requires_explicit_confidence_truncated_ap_name(
    tmp_path: Path,
) -> None:
    protocol = {
        "protocol_id": "STAGE-P-PARKING-DOMAIN-RETENTION-20260729-01",
        "shared_inference": {
            "api": "ultralytics.YOLO.predict",
            "imgsz": 640,
            "confidence": 0.30,
            "nms_iou": 0.70,
            "agnostic_nms": True,
            "max_detections": 300,
            "classes": [0],
            "model_track_prohibited": True,
        },
        "metrics": {"ap_semantics": AP_SEMANTICS},
    }
    path = tmp_path / "protocol.yaml"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    assert load_stage_p_protocol(path)["metrics"]["ap_semantics"] == AP_SEMANTICS
    protocol["metrics"]["ap_semantics"] = "AP"
    path.write_text(yaml.safe_dump(protocol), encoding="utf-8")
    with pytest.raises(StagePDataGateError, match="AP semantics"):
        load_stage_p_protocol(path)


def test_public_ap_keys_include_frozen_confidence_semantics() -> None:
    metrics = _public_box_metrics(
        {
            "precision": 1.0,
            "recall": 1.0,
            "AP50": 1.0,
            "AP50-95": 1.0,
            "ground_truth_boxes": 1,
            "predicted_boxes": 1,
            "true_positives": 1,
            "false_positives": 0,
            "false_negatives": 0,
        }
    )
    assert AP50_KEY in metrics
    assert AP50_95_KEY in metrics
    assert "AP50" not in metrics
    assert metrics["standard_COCO_AP"] is False


def test_unknown_truth_is_excluded_and_counted() -> None:
    rows = [
        _row(split="train", name="known.jpg", digest="a" * 64),
        _row(
            split="train",
            name="unknown.jpg",
            digest="b" * 64,
            truth_type="unknown",
        ),
    ]
    known, excluded = partition_known_truth(rows)
    assert [row.file_name for row in known] == ["known.jpg"]
    assert excluded == 1


def test_empty_manifest_and_missing_fields_block_safely(
    tmp_path: Path,
) -> None:
    empty = tmp_path / "empty.csv"
    empty.write_text(
        "split,file_name,width,height,truth_type,sha256\n",
        encoding="utf-8",
    )
    with pytest.raises(StagePDataGateError, match="Empty manifest"):
        load_manifest(empty, expected_split="train")

    incomplete = tmp_path / "incomplete.csv"
    incomplete.write_text(
        "split,file_name,width,height,truth_type,sha256\n"
        "train,a.jpg,10,10,vehicle_boxes,\n",
        encoding="utf-8",
    )
    with pytest.raises(StagePDataGateError, match="Incomplete"):
        load_manifest(incomplete, expected_split="train")


def test_daytime_nighttime_groups_are_disjoint_by_exact_hash() -> None:
    audit = audit_manifest_groups(
        {
            "train": [_row(split="train", name="day.jpg", digest="a" * 64)],
            "validation": [
                _row(split="validation", name="night.jpg", digest="b" * 64)
            ],
            "test": [
                _row(
                    split="test",
                    name="count.jpg",
                    digest="c" * 64,
                    truth_type="vehicle_count",
                )
            ],
        }
    )
    assert audit["exact_cross_split_duplicate_images"] == 0
    assert audit["split_counts"] == {
        "train": 1,
        "validation": 1,
        "test": 1,
    }


def test_exact_train_validation_leakage_is_rejected() -> None:
    with pytest.raises(StagePDataGateError, match="leakage"):
        audit_manifest_groups(
            {
                "train": [
                    _row(split="train", name="day.jpg", digest="a" * 64)
                ],
                "validation": [
                    _row(
                        split="validation",
                        name="night.jpg",
                        digest="a" * 64,
                    )
                ],
                "test": [
                    _row(
                        split="test",
                        name="count.jpg",
                        digest="c" * 64,
                        truth_type="vehicle_count",
                    )
                ],
            }
        )


@pytest.mark.parametrize(
    ("comparison", "expected"),
    [
        (_comparison(), "PASS"),
        (_comparison(day_drop=-0.06), "CONDITIONAL"),
        (_comparison(night_recall=0.0), "FAIL"),
        ({"deltas": {}}, "BLOCKED"),
    ],
)
def test_retention_decision_has_only_predeclared_outcomes(
    comparison: dict, expected: str
) -> None:
    assert decide_parking_retention(comparison, _rule())["status"] == expected


def test_bound_file_checks_hash_and_size(tmp_path: Path) -> None:
    path = tmp_path / "bound.json"
    path.write_text(json.dumps({"ok": True}) + "\n", encoding="utf-8")
    import hashlib

    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert verify_bound_file(
        path,
        expected_bytes=path.stat().st_size,
        expected_sha256=digest,
    )["verified"]
    with pytest.raises(StagePDataGateError, match="SHA-256"):
        verify_bound_file(
            path,
            expected_bytes=path.stat().st_size,
            expected_sha256="0" * 64,
        )
