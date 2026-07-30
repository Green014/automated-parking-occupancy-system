import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import yaml

from parking_occupancy.stage_o_low_light import (
    FrozenRawDetectorAdapter,
    O1Parameters,
    DetectorOnlySettings,
    bt601_mean_luma,
    discover_paired_frames,
    gamma_clahe,
    pooled_detection_metrics,
    run_detector_only_evaluation,
    select_stage_o_candidate,
    sequence_brightness,
)
from parking_occupancy.stage_o_enhancement import retinex_tensor_to_rgb
from parking_occupancy.stage_n_lmot import LmotAnnotation, TrackPrediction


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(
        str(path), np.full((24, 32, 3), value, dtype=np.uint8)
    )


def test_pair_discovery_keeps_light_and_dark_same_group(tmp_path: Path) -> None:
    sequence = tmp_path / "LMOT-06"
    for frame in (1, 11):
        _write_image(sequence / "img_light_rgb" / f"{frame:06d}.jpg", 120)
        _write_image(sequence / "img_dark_rgb" / f"{frame:06d}.png", 20)

    pairs = discover_paired_frames(sequence, frame_numbers=(1, 11))

    assert [(row.sequence, row.frame_number) for row in pairs] == [
        ("LMOT-06", 1),
        ("LMOT-06", 11),
    ]
    (sequence / "img_dark_rgb" / "000011.png").unlink()
    with pytest.raises(ValueError, match="sets differ"):
        discover_paired_frames(sequence)


def test_brightness_gate_is_one_sequence_statistic_not_frame_adaptive(
    tmp_path: Path,
) -> None:
    paths = []
    for index, value in enumerate((5, 10, 200, 220), start=1):
        path = tmp_path / f"{index:06d}.jpg"
        _write_image(path, value)
        paths.append(path)

    brightness = sequence_brightness(paths, calibration_frames=2)

    assert brightness == pytest.approx(7.5, abs=0.6)
    assert brightness < 45
    assert bt601_mean_luma(np.full((2, 2, 3), 100, np.uint8)) == pytest.approx(
        100.0
    )


def test_fixed_gamma_clahe_is_deterministic_and_brightens_dark_input() -> None:
    image = np.tile(
        np.arange(0, 64, dtype=np.uint8)[None, :, None], (32, 1, 3)
    )
    parameters = O1Parameters(
        threshold=45,
        gamma=0.5,
        clahe_clip_limit=2.0,
    )

    first = gamma_clahe(image, parameters)
    second = gamma_clahe(image, parameters)

    assert np.array_equal(first, second)
    assert float(first.mean()) > float(image.mean())


def _gt(sequence_offset: int, frame: int) -> LmotAnnotation:
    return LmotAnnotation(
        frame_number=frame,
        track_id=sequence_offset,
        x=1,
        y=1,
        width=10,
        height=10,
        ignore=1,
        class_id=3,
        visibility=1,
    )


def _prediction(frame: int, confidence: float = 0.9) -> TrackPrediction:
    return TrackPrediction(
        frame_number=frame,
        track_id=-1,
        xyxy=(1, 1, 11, 11),
        confidence=confidence,
    )


def test_pooled_metrics_sum_counts_and_keep_macro_separate() -> None:
    metrics = pooled_detection_metrics(
        {
            "A": ([_gt(1, 1)], [_prediction(1)]),
            "B": ([_gt(2, 1), _gt(3, 2)], [_prediction(1)]),
        }
    )

    pooled = metrics["pooled_micro"]
    macro = metrics["per_sequence_macro"]
    assert pooled["ground_truth_boxes"] == 3
    assert pooled["predicted_boxes"] == 2
    assert pooled["true_positives"] == 2
    assert pooled["false_negatives"] == 1
    assert pooled["recall"] == pytest.approx(2 / 3)
    assert macro["recall"] == pytest.approx(0.75)


def _selection_metrics(
    *,
    precision: float,
    recall: float,
    ap50: float,
    ap5095: float,
    fps: float = 10.0,
) -> dict:
    return {
        "illumination": {
            "dark": {
                "pooled_micro": {
                    "precision": precision,
                    "recall": recall,
                    "AP50": ap50,
                    "AP50-95": ap5095,
                    "ground_truth_boxes": 100,
                    "predicted_boxes": 50,
                    "true_positives": 25,
                    "false_positives": 25,
                    "false_negatives": 75,
                }
            }
        },
        "runtime": {"wall_fps": fps},
    }


def test_frozen_selection_rule_rejects_partial_gain_and_selects_eligible() -> None:
    root = Path(__file__).resolve().parents[1]
    protocol = yaml.safe_load(
        (
            root
            / "configs"
            / "stage_o_low_light_adaptation_frozen_20260729.yaml"
        ).read_text(encoding="utf-8")
    )
    baseline = _selection_metrics(
        precision=0.72, recall=0.03, ap50=0.03, ap5095=0.018
    )
    o1 = _selection_metrics(
        precision=0.45, recall=0.05, ap50=0.029, ap5095=0.017
    )
    o3 = _selection_metrics(
        precision=0.70, recall=0.26, ap50=0.23, ap5095=0.11
    )

    result = select_stage_o_candidate(
        protocol=protocol,
        baseline_metrics=baseline,
        candidate_metrics={"O1": o1, "O3": o3},
        blocked_methods={"O2": "predeclared enhancer path blocked"},
    )

    assert result["selected_method"] == "O3"
    assert result["selected_detector_role"] == "D1-LL"
    assert result["thresholds_reselected"] is False
    by_method = {row["method_id"]: row for row in result["candidate_rows"]}
    assert by_method["O1"]["eligible"] is False
    assert by_method["O3"]["eligible"] is True


class _Tensor:
    def __init__(self, values):
        self.values = np.asarray(values)

    def detach(self):
        return self

    def cpu(self):
        return self

    def numpy(self):
        return self.values


class _Boxes:
    xyxy = _Tensor([[1, 1, 11, 11]])
    conf = _Tensor([0.9])

    def __len__(self):
        return 1


class _FakeModel:
    def __init__(self):
        self.predict_calls = []
        self.track_calls = 0

    def predict(self, **kwargs):
        self.predict_calls.append(kwargs)
        return [
            SimpleNamespace(
                boxes=_Boxes(),
                speed={"preprocess": 1.0, "inference": 2.0, "postprocess": 0.5},
            )
        ]

    def track(self, **_kwargs):
        self.track_calls += 1
        raise AssertionError("Stage O must not call model.track")


def test_raw_adapter_calls_predict_with_frozen_settings_only(tmp_path: Path) -> None:
    weights = tmp_path / "D1.pt"
    weights.write_bytes(b"fixture")
    model = _FakeModel()
    adapter = FrozenRawDetectorAdapter(
        DetectorOnlySettings(weights=weights, device="cpu"),
        model_factory=lambda _path: model,
    )

    rows, speed = adapter.predict(
        np.zeros((20, 20, 3), dtype=np.uint8), frame_number=1
    )

    assert len(rows) == 1
    assert speed["inference"] == 2.0
    assert model.track_calls == 0
    call = model.predict_calls[0]
    assert call["conf"] == 0.30
    assert call["iou"] == 0.70
    assert call["imgsz"] == 640
    assert call["agnostic_nms"] is True
    assert call["max_det"] == 300


def test_detector_only_runner_writes_complete_contract_without_tracker(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    sequence_root = tmp_path / "data" / "LMOT-05"
    _write_image(sequence_root / "img_light_rgb" / "000001.jpg", 100)
    _write_image(sequence_root / "img_dark_rgb" / "000001.png", 10)
    (sequence_root / "gt").mkdir()
    (sequence_root / "gt" / "gt.txt").write_text(
        "1,1,1,1,10,10,1,3,1\n", encoding="utf-8"
    )
    weights = tmp_path / "D1.pt"
    weights.write_bytes(b"fixture")
    model = _FakeModel()
    output = tmp_path / "output"

    metrics = run_detector_only_evaluation(
        protocol_path=(
            root
            / "configs"
            / "stage_o_low_light_adaptation_frozen_20260729.yaml"
        ),
        validation_root=tmp_path / "data",
        class_map_path=(
            root / "data" / "stage_n_v2" / "LMOT_CLASS_MAP_FROZEN_20260729.yaml"
        ),
        weights_path=weights,
        output_dir=output,
        method_id="O0",
        sequences=("LMOT-05",),
        device="cpu",
        model_factory=lambda _path: model,
    )

    assert metrics["tracker_emitted_boxes"] is False
    assert model.track_calls == 0
    assert {
        "detections.jsonl",
        "metrics.json",
        "runtime_metadata.json",
        "config_snapshot.yaml",
        "qualitative_contact_sheet.jpg",
        "failure_cases.json",
    }.issubset(path.name for path in output.iterdir())
    with pytest.raises(FileExistsError):
        run_detector_only_evaluation(
            protocol_path=(
                root
                / "configs"
                / "stage_o_low_light_adaptation_frozen_20260729.yaml"
            ),
            validation_root=tmp_path / "data",
            class_map_path=(
                root
                / "data"
                / "stage_n_v2"
                / "LMOT_CLASS_MAP_FROZEN_20260729.yaml"
            ),
            weights_path=weights,
            output_dir=output,
            method_id="O0",
            sequences=("LMOT-05",),
            device="cpu",
            model_factory=lambda _path: model,
        )


def test_o2_preprocessor_is_applied_to_dark_stream_only(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[1]
    sequence_root = tmp_path / "data" / "LMOT-05"
    _write_image(sequence_root / "img_light_rgb" / "000001.jpg", 100)
    _write_image(sequence_root / "img_dark_rgb" / "000001.png", 10)
    (sequence_root / "gt").mkdir()
    (sequence_root / "gt" / "gt.txt").write_text(
        "1,1,1,1,10,10,1,3,1\n", encoding="utf-8"
    )
    weights = tmp_path / "D1.pt"
    weights.write_bytes(b"fixture")
    model = _FakeModel()

    class FakeEnhancer:
        def __init__(self) -> None:
            self.calls = 0

        def __call__(self, image: np.ndarray) -> np.ndarray:
            self.calls += 1
            return np.full_like(image, 80)

        def metadata(self) -> dict:
            return {"method": "fake", "calls": self.calls}

    enhancer = FakeEnhancer()
    output = tmp_path / "o2"
    metrics = run_detector_only_evaluation(
        protocol_path=(
            root
            / "configs"
            / "stage_o_low_light_adaptation_frozen_20260729.yaml"
        ),
        validation_root=tmp_path / "data",
        class_map_path=(
            root / "data" / "stage_n_v2" / "LMOT_CLASS_MAP_FROZEN_20260729.yaml"
        ),
        weights_path=weights,
        output_dir=output,
        method_id="O2",
        sequences=("LMOT-05",),
        device="cpu",
        image_preprocessor=enhancer,
        model_factory=lambda _path: model,
    )

    assert enhancer.calls >= 1
    assert metrics["method_id"] == "O2"
    records = [
        json.loads(line)
        for line in (output / "detections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    by_illumination = {row["illumination"]: row for row in records}
    assert by_illumination["light"]["enhancement_applied"] is False
    assert by_illumination["dark"]["enhancement_applied"] is True


def test_retinex_tensor_conversion_does_not_mutate_inference_tensor() -> None:
    torch = pytest.importorskip("torch")
    with torch.inference_mode():
        restored = torch.tensor(
            [[[[1.2, -0.1]], [[0.5, 0.5]], [[0.0, 1.0]]]],
            dtype=torch.float32,
        )

    rgb = retinex_tensor_to_rgb(restored)

    assert rgb.shape == (1, 2, 3)
    assert rgb.tolist() == [[[255, 127, 0], [0, 127, 255]]]
