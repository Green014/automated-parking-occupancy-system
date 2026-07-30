from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import pytest
import yaml

from parking_occupancy.image_io import write_image
from parking_occupancy.training_smoke import (
    SmokeProtocolError,
    TrainingResourceProbe,
    analyze_results_rows,
    load_smoke_settings,
    smoke_preflight,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _fixture(
    tmp_path: Path,
) -> tuple[Path, Path, Path, Path, Path]:
    dataset_root = tmp_path / "dataset"
    for split in ("train", "val"):
        write_image(
            dataset_root / "images" / split / f"{split}.jpg",
            np.zeros((10, 20, 3), dtype=np.uint8),
        )
        (dataset_root / "labels" / split).mkdir(parents=True)
    (dataset_root / "labels" / "train" / "train.txt").write_text(
        "",
        encoding="utf-8",
    )
    (dataset_root / "labels" / "val" / "val.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n",
        encoding="utf-8",
    )
    data_yaml = dataset_root / "dataset.yaml"
    data_yaml.write_text(
        yaml.safe_dump(
            {
                "path": str(dataset_root),
                "train": "images/train",
                "val": "images/val",
                "names": {0: "vehicle"},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    preparation = tmp_path / "preparation.yaml"
    preparation.write_text(
        yaml.safe_dump(
            {
                "generated_artifacts": {
                    "dataset_yaml": {
                        "bytes": data_yaml.stat().st_size,
                        "sha256": _sha256(data_yaml),
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    validation_manifest = tmp_path / "validation.csv"
    validation_manifest.write_text("image\nval.jpg\n", encoding="utf-8")
    weights = tmp_path / "yolov8n.pt"
    weights.write_bytes(b"pretrained")

    dataset_protocol = {
        "schema_version": 1,
        "protocol_id": "DPROTO-NDISPARK-ONLY-20260727-01",
        "status": "frozen",
        "training_configuration_space": {
            "fixed": {
                "imgsz": 640,
                "seed": 20260727,
                "amp": True,
                "deterministic": True,
                "optimizer": "AdamW",
                "lr0": 0.001,
                "weight_decay": 0.0005,
            },
            "smoke": {
                "experiment_id": "D1-NDISPARK-SMOKE-20260727-01",
                "epochs": 3,
                "batch": 4,
            },
        },
        "detector_comparison": {
            "models": {
                "D0": {"weights_sha256": _sha256(weights)},
            }
        },
    }
    dataset_protocol_path = tmp_path / "dataset_protocol.yaml"
    dataset_protocol_path.write_text(
        yaml.safe_dump(dataset_protocol, sort_keys=False),
        encoding="utf-8",
    )

    models = {
        "D0": {
            "name": "COCO-pretrained YOLOv8n",
            "backend": "ultralytics_yolo",
            "status": "ready",
            "weights_name": weights.name,
            "weights_sha256": _sha256(weights),
            "source_class_ids": [2, 3, 5, 7],
            "source_class_names": ["car", "motorcycle", "bus", "truck"],
            "project_class_id": 0,
            "project_class_name": "vehicle",
        },
        "D1": {
            "name": "NDISPark-fine-tuned YOLOv8n",
            "backend": "ultralytics_yolo",
            "status": "not_trained",
            "weights_name": None,
            "weights_sha256": None,
            "source_class_ids": [0],
            "source_class_names": ["vehicle"],
            "project_class_id": 0,
            "project_class_name": "vehicle",
        },
        "D2": {
            "name": "YOLO-World zero-shot",
            "backend": "ultralytics_yolo_world",
            "status": "ready",
            "weights_name": "world.pt",
            "weights_sha256": "0" * 64,
            "prompts": ["car", "motorcycle", "bus", "truck"],
            "source_class_ids": [0, 1, 2, 3],
            "source_class_names": ["car", "motorcycle", "bus", "truck"],
            "project_class_id": 0,
            "project_class_name": "vehicle",
        },
        "D3": {
            "name": "fine-tuned YOLO-World",
            "backend": "ultralytics_yolo_world",
            "status": "deferred",
        },
    }
    comparison_protocol = {
        "comparison_protocol_id": "D-COMP-NDISPARK-DEV-20260727-01",
        "status": "frozen_before_new_predictions",
        "data": {
            "dataset_protocol_id": "DPROTO-NDISPARK-ONLY-20260727-01",
            "preparation_record": {
                "path": str(preparation),
                "bytes": preparation.stat().st_size,
                "sha256": _sha256(preparation),
            },
            "validation_manifest": {
                "path": str(validation_manifest),
                "bytes": validation_manifest.stat().st_size,
                "sha256": _sha256(validation_manifest),
            },
            "expected": {
                "train_images": 1,
                "train_boxes": 0,
                "validation_images": 1,
                "validation_boxes": 1,
            },
        },
        "models": models,
        "common_inference": {
            "split": "val",
            "imgsz": 640,
            "confidence_floor": 0.001,
            "nms_iou": 0.7,
            "max_detections": 300,
            "batch": 1,
            "single_class_evaluation": True,
            "augmentation": False,
            "rect": False,
            "half": False,
        },
    }
    comparison_protocol_path = tmp_path / "comparison_protocol.yaml"
    comparison_protocol_path.write_text(
        yaml.safe_dump(comparison_protocol, sort_keys=False),
        encoding="utf-8",
    )
    return (
        dataset_protocol_path,
        comparison_protocol_path,
        data_yaml,
        weights,
        tmp_path / "output",
    )


def test_load_committed_smoke_settings() -> None:
    path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "ndispark_only_dataset_frozen_20260727.yaml"
    )

    _, settings = load_smoke_settings(path)

    assert settings.experiment_id == "D1-NDISPARK-SMOKE-20260727-01"
    assert settings.epochs == 3
    assert settings.batch == 4
    assert settings.imgsz == 640
    assert settings.amp is True


def test_preflight_accepts_exact_inputs_without_creating_output(
    tmp_path: Path,
) -> None:
    dataset_protocol, comparison, data, weights, output = _fixture(tmp_path)

    report, settings = smoke_preflight(
        dataset_protocol_path=dataset_protocol,
        comparison_protocol_path=comparison,
        data_yaml=data,
        initial_weights=weights,
        output_dir=output,
        device="0",
        workers=2,
    )

    assert settings.epochs == 3
    assert report["status"] == "ready"
    assert report["data"]["count_test_accessed"] is False
    assert report["data"]["label_summary"]["train"]["background_images"] == 1
    assert report["training_run"] is False
    assert not output.exists()


def test_preflight_rejects_existing_output(tmp_path: Path) -> None:
    dataset_protocol, comparison, data, weights, output = _fixture(tmp_path)
    output.mkdir()

    with pytest.raises(FileExistsError, match="overwrite"):
        smoke_preflight(
            dataset_protocol_path=dataset_protocol,
            comparison_protocol_path=comparison,
            data_yaml=data,
            initial_weights=weights,
            output_dir=output,
            device="0",
            workers=2,
        )


def test_preflight_rejects_changed_initial_weights(tmp_path: Path) -> None:
    dataset_protocol, comparison, data, weights, output = _fixture(tmp_path)
    weights.write_bytes(b"changed")

    with pytest.raises(SmokeProtocolError, match="weight SHA"):
        smoke_preflight(
            dataset_protocol_path=dataset_protocol,
            comparison_protocol_path=comparison,
            data_yaml=data,
            initial_weights=weights,
            output_dir=output,
            device="0",
            workers=2,
        )


def test_results_analysis_requires_finite_changing_losses() -> None:
    rows = [
        {
            "epoch": str(epoch),
            "time": str(epoch * 2),
            "train/box_loss": str(2.0 / epoch),
            "train/cls_loss": str(1.0 / epoch),
            "metrics/precision(B)": str(0.1 * epoch),
        }
        for epoch in (1, 2, 3)
    ]

    report = analyze_results_rows(rows, expected_epochs=3)

    assert report["epochs_recorded"] == 3
    assert report["any_loss_changed"] is True
    assert report["all_numeric_values_finite"] is True
    assert report["final_validation_metrics"]["metrics/precision(B)"] == (
        pytest.approx(0.3)
    )


def test_results_analysis_rejects_nan() -> None:
    rows = [
        {
            "epoch": str(epoch),
            "time": str(epoch),
            "train/box_loss": "nan" if epoch == 2 else "1.0",
            "metrics/precision(B)": "0.1",
        }
        for epoch in (1, 2, 3)
    ]

    with pytest.raises(RuntimeError, match="Non-finite"):
        analyze_results_rows(rows, expected_epochs=3)


def test_resource_probe_dataloader_assessment() -> None:
    probe = TrainingResourceProbe()
    probe.epochs = [
        {"train_batch_wait_fraction": 0.10},
        {"train_batch_wait_fraction": 0.20},
    ]

    report = probe.dataloader_assessment()

    assert report["mean_wait_fraction"] == pytest.approx(0.15)
    assert report["finding"] == "moderate_wait_or_loader_overhead"
