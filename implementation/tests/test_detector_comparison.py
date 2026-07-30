from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

from parking_occupancy.detector_comparison import (
    ComparisonDetectorAdapter,
    ComparisonNotReady,
    ComparisonProtocolError,
    DetectorSpec,
    _evaluate_adapter,
    canonicalize_detections,
    comparison_preflight,
    load_comparison_protocol,
    match_detection_boxes,
    pairwise_iou_xyxy,
    run_comparison,
)
from parking_occupancy.image_io import write_image


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _comparison_fixture(
    tmp_path: Path,
) -> tuple[Path, Path, dict[str, Path | None], Path]:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    preparation = evidence / "preparation.yaml"
    preparation.write_text("status: complete\n", encoding="utf-8")
    validation_manifest = evidence / "validation.csv"
    validation_manifest.write_text("file_name\nval.jpg\n", encoding="utf-8")

    dataset_root = tmp_path / "dataset"
    write_image(
        dataset_root / "images" / "train" / "train.jpg",
        np.zeros((10, 20, 3), dtype=np.uint8),
    )
    write_image(
        dataset_root / "images" / "val" / "val.jpg",
        np.zeros((10, 20, 3), dtype=np.uint8),
    )
    (dataset_root / "labels" / "train").mkdir(parents=True)
    (dataset_root / "labels" / "val").mkdir(parents=True)
    (dataset_root / "labels" / "train" / "train.txt").write_text(
        "0 0.5 0.5 0.5 0.5\n",
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

    weights = {}
    for method_id in ("D0", "D2"):
        path = tmp_path / f"{method_id.lower()}.pt"
        path.write_bytes(method_id.encode())
        weights[method_id] = path
    weights["D1"] = None

    models = {
        "D0": {
            "name": "COCO-pretrained YOLOv8n",
            "backend": "ultralytics_yolo",
            "status": "ready",
            "weights_name": "d0.pt",
            "weights_sha256": _sha256(weights["D0"]),  # type: ignore[arg-type]
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
            "weights_name": "d2.pt",
            "weights_sha256": _sha256(weights["D2"]),  # type: ignore[arg-type]
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
    config = {
        "comparison_protocol_id": "D-COMP-NDISPARK-DEV-20260727-01",
        "status": "frozen_before_new_predictions",
        "data": {
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
                "train_boxes": 1,
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
    config_path = tmp_path / "comparison.yaml"
    config_path.write_text(
        yaml.safe_dump(config, sort_keys=False),
        encoding="utf-8",
    )
    return config_path, data_yaml, weights, tmp_path / "output"


def test_preflight_blocks_missing_d1_without_loading_models(tmp_path: Path) -> None:
    config, data, weights, _ = _comparison_fixture(tmp_path)

    report, specs = comparison_preflight(
        config_path=config,
        data_yaml=data,
        weight_paths=weights,
    )

    assert list(specs) == ["D0", "D1", "D2"]
    assert report["models"]["D0"]["ready"] is True
    assert report["models"]["D1"]["ready"] is False
    assert "model_not_trained" in report["models"]["D1"]["reasons"]
    assert report["models"]["D2"]["ready"] is True
    assert report["all_required_models_ready"] is False
    assert report["predictions_run"] is False


def test_preflight_accepts_runtime_bound_d1_hash(tmp_path: Path) -> None:
    config, data, weights, _ = _comparison_fixture(tmp_path)
    d1_path = tmp_path / "d1.pt"
    d1_path.write_bytes(b"fine tuned")
    weights["D1"] = d1_path

    report, _ = comparison_preflight(
        config_path=config,
        data_yaml=data,
        weight_paths=weights,
        runtime_weight_hashes={"D1": _sha256(d1_path)},
    )

    assert report["all_required_models_ready"] is True
    assert report["execution_gate"] == "open"


def test_run_comparison_fails_before_creating_output_when_d1_missing(
    tmp_path: Path,
) -> None:
    config, data, weights, output = _comparison_fixture(tmp_path)

    with pytest.raises(ComparisonNotReady, match="no inference"):
        run_comparison(
            config_path=config,
            data_yaml=data,
            output_root=output,
            weight_paths=weights,
            runtime_weight_hashes={},
            device="cpu",
        )

    assert not output.exists()


def test_canonicalize_detections_collapses_allowed_classes_to_vehicle() -> None:
    detections = canonicalize_detections(
        boxes_xyxy=np.asarray(
            [[0, 0, 10, 10], [1, 1, 4, 4], [5, 5, 9, 9]],
            dtype=np.float32,
        ),
        confidences=np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
        source_class_ids=np.asarray([2, 0, 7]),
        allowed_source_class_ids=(2, 3, 5, 7),
    )

    assert len(detections) == 2
    assert {detection.class_id for detection in detections} == {0}
    assert {detection.class_name for detection in detections} == {"vehicle"}


@pytest.mark.parametrize(
    ("source_class_ids", "allowed_source_class_ids"),
    [
        ([2, 7], (2, 3, 5, 7)),
        ([0, 3], (0, 1, 2, 3)),
    ],
    ids=["D0", "D2"],
)
def test_class_agnostic_nms_removes_cross_class_vehicle_duplicate(
    source_class_ids: list[int],
    allowed_source_class_ids: tuple[int, ...],
) -> None:
    detections = canonicalize_detections(
        boxes_xyxy=np.asarray(
            [[0, 0, 10, 10], [0.25, 0.25, 10.25, 10.25]],
            dtype=np.float32,
        ),
        confidences=np.asarray([0.9, 0.8], dtype=np.float32),
        source_class_ids=np.asarray(source_class_ids),
        allowed_source_class_ids=allowed_source_class_ids,
        agnostic_nms_iou=0.7,
    )

    assert len(detections) == 1
    assert detections[0].confidence == pytest.approx(0.9)
    assert detections[0].class_id == 0
    assert detections[0].class_name == "vehicle"


def test_d1_single_class_canonicalization_is_unchanged() -> None:
    arguments = {
        "boxes_xyxy": np.asarray(
            [[0, 0, 10, 10], [20, 20, 30, 30]],
            dtype=np.float32,
        ),
        "confidences": np.asarray([0.9, 0.8], dtype=np.float32),
        "source_class_ids": np.asarray([0, 0]),
        "allowed_source_class_ids": (0,),
    }

    legacy = canonicalize_detections(**arguments)
    corrected = canonicalize_detections(
        **arguments,
        agnostic_nms_iou=0.7,
    )

    assert corrected == legacy
    assert {detection.class_id for detection in corrected} == {0}


def test_adapter_passes_agnostic_nms_to_ultralytics_predict(
    tmp_path: Path,
) -> None:
    weights = tmp_path / "d0.pt"
    weights.write_bytes(b"weights")
    calls: list[dict[str, object]] = []

    class FakeModel:
        def predict(self, **kwargs: object) -> list[SimpleNamespace]:
            calls.append(kwargs)
            return [SimpleNamespace(boxes=None)]

    adapter = ComparisonDetectorAdapter(
        spec=DetectorSpec(
            method_id="D0",
            name="synthetic D0",
            backend="ultralytics_yolo",
            status="ready",
            weights_name=weights.name,
            weights_sha256=_sha256(weights),
            source_class_ids=(2, 3, 5, 7),
            source_class_names=("car", "motorcycle", "bus", "truck"),
            prompts=(),
            project_class_id=0,
            project_class_name="vehicle",
        ),
        weights_path=weights,
        common={
            "confidence_floor": 0.001,
            "nms_iou": 0.7,
            "agnostic_nms": True,
            "imgsz": 640,
            "max_detections": 300,
            "batch": 1,
            "augmentation": False,
            "rect": False,
            "half": False,
        },
        device="cpu",
    )
    adapter._model = FakeModel()
    adapter._resolved_device = "cpu"

    assert adapter.detect(np.zeros((10, 10, 3), dtype=np.uint8)) == []
    adapter.predict_images([tmp_path / "image.jpg"])

    assert len(calls) == 2
    assert all(call["agnostic_nms"] is True for call in calls)


def test_one_to_one_matching_does_not_double_count_ground_truth() -> None:
    ground_truth = np.asarray([[0, 0, 10, 10]], dtype=np.float32)
    predictions = np.asarray(
        [[0, 0, 10, 10], [0, 0, 9, 9]],
        dtype=np.float32,
    )

    iou = pairwise_iou_xyxy(ground_truth, predictions)
    correct = match_detection_boxes(
        ground_truth_boxes=ground_truth,
        predicted_boxes=predictions,
        iou_thresholds=np.asarray([0.5, 0.9]),
    )

    assert iou.tolist() == [[1.0, pytest.approx(0.81)]]
    assert correct.tolist() == [[True, True], [False, False]]


def test_canonical_evaluator_keeps_class_zero_truth(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torch

    config, data_yaml, _, _ = _comparison_fixture(tmp_path)
    protocol, _ = load_comparison_protocol(config)
    weights = tmp_path / "unused.pt"
    weights.write_bytes(b"not loaded")
    adapter = ComparisonDetectorAdapter(
        spec=DetectorSpec(
            method_id="D0",
            name="synthetic D0",
            backend="ultralytics_yolo",
            status="ready",
            weights_name=weights.name,
            weights_sha256=_sha256(weights),
            source_class_ids=(2, 3, 5, 7),
            source_class_names=("car", "motorcycle", "bus", "truck"),
            prompts=(),
            project_class_id=0,
            project_class_name="vehicle",
        ),
        weights_path=weights,
        common=protocol["common_inference"],
        device="cpu",
    )

    class FakeBoxes:
        xyxy = torch.tensor([[5.0, 2.5, 15.0, 7.5]])
        conf = torch.tensor([0.9])
        cls = torch.tensor([2.0])

        def __len__(self) -> int:
            return 1

    class FakeResult:
        boxes = FakeBoxes()
        orig_shape = (10, 20)
        speed = {
            "preprocess": 1.0,
            "inference": 2.0,
            "postprocess": 1.0,
        }

    monkeypatch.setattr(
        adapter,
        "predict_images",
        lambda image_paths: [FakeResult() for _ in image_paths],
    )
    monkeypatch.setattr(
        adapter,
        "model_metadata",
        lambda: {
            "weights_bytes": weights.stat().st_size,
            "parameter_count": 1,
        },
    )
    output = tmp_path / "evaluation"

    report = _evaluate_adapter(
        adapter=adapter,
        data_yaml=data_yaml,
        method_root=output,
        protocol=protocol,
    )

    assert report["ground_truth_boxes"] == 1
    assert report["predictions"] == 1
    assert report["images_reaching_max_det"] == 0
    assert report["peak_cuda_memory_allocated_bytes"] is None
    assert report["map_50"] > 0.99
    assert report["source_class_filter"] == [2, 3, 5, 7]
    for name in (
        "detections.jsonl",
        "PR_curve.png",
        "confusion_matrix.png",
        "confusion_matrix_normalized.png",
    ):
        assert (output / name).is_file()


def test_protocol_rejects_changed_common_image_size(tmp_path: Path) -> None:
    config, _, _, _ = _comparison_fixture(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload["common_inference"]["imgsz"] = 1280
    config.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ComparisonProtocolError, match="imgsz"):
        load_comparison_protocol(config)


def test_stage_i_v2_protocol_requires_explicit_agnostic_nms(
    tmp_path: Path,
) -> None:
    config, _, _, _ = _comparison_fixture(tmp_path)
    payload = yaml.safe_load(config.read_text(encoding="utf-8"))
    payload.update(
        {
            "comparison_protocol_id": (
                "D-COMP-NDISPARK-DEV-V2-MAXDET300-20260727-01"
            ),
            "protocol_generation": "stage_i_v2",
            "status": "frozen_before_corrected_predictions",
            "evaluation_role": "corrected_development_evaluation",
        }
    )
    config.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )

    with pytest.raises(ComparisonProtocolError, match="agnostic_nms"):
        load_comparison_protocol(config)

    payload["common_inference"]["agnostic_nms"] = True
    config.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    loaded, _ = load_comparison_protocol(config)

    assert loaded["common_inference"]["agnostic_nms"] is True


def test_committed_comparison_protocol_has_required_model_roles() -> None:
    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "detector_comparison_frozen_20260727.yaml"
    )

    payload, specs = load_comparison_protocol(config)

    assert list(specs) == ["D0", "D1", "D2"]
    assert specs["D0"].source_class_ids == (2, 3, 5, 7)
    assert specs["D1"].status == "not_trained"
    assert specs["D2"].prompts == ("car", "motorcycle", "bus", "truck")
    assert payload["models"]["D3"]["status"] == "deferred"


def test_committed_stage_i_v2_protocol_is_explicitly_agnostic() -> None:
    config = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "detector_comparison_stage_i_v2_maxdet300_frozen_20260727.yaml"
    )

    payload, specs = load_comparison_protocol(config)

    assert payload["protocol_generation"] == "stage_i_v2"
    assert payload["evaluation_role"] == "corrected_development_evaluation"
    assert payload["common_inference"]["agnostic_nms"] is True
    assert payload["common_inference"]["max_detections"] == 300
    assert specs["D1"].status == "ready"
    assert specs["D1"].weights_sha256 == (
        "0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64"
    )


def test_committed_maxdet_1000_protocol_changes_only_sensitivity_arm() -> None:
    project_root = Path(__file__).resolve().parents[1]
    config_300, specs_300 = load_comparison_protocol(
        project_root
        / "configs"
        / "detector_comparison_stage_i_v2_maxdet300_frozen_20260727.yaml"
    )
    config_1000, specs_1000 = load_comparison_protocol(
        project_root
        / "configs"
        / "detector_comparison_stage_i_v2_maxdet1000_frozen_20260727.yaml"
    )

    assert config_300["common_inference"]["max_detections"] == 300
    assert config_1000["common_inference"]["max_detections"] == 1000
    for key in (
        "split",
        "imgsz",
        "confidence_floor",
        "nms_iou",
        "agnostic_nms",
        "batch",
        "single_class_evaluation",
        "augmentation",
        "rect",
        "half",
    ):
        assert config_300["common_inference"][key] == (
            config_1000["common_inference"][key]
        )
    assert specs_300 == specs_1000


def test_detector_comparison_frozen_checksums() -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load(
        (
            project_root
            / "data"
            / "comparisons"
            / "detector_comparison_frozen_checksums.yaml"
        ).read_text(encoding="utf-8")
    )

    assert registry["registry_id"] == (
        "D-COMP-FREEZE-NDISPARK-20260727-01"
    )
    for artifact in registry["artifacts"]:
        path = project_root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]
