from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import statistics
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml

from .stage_n_lmot import (
    LmotAnnotation,
    TrackPrediction,
    evaluate_motor_vehicle_detections,
    parse_lmot_gt,
    read_image,
    sha256_file,
    split_motor_vehicle_truth,
    suppress_predictions_on_excluded_truth,
    write_image,
)
from .stage_n_lmot_v2 import LmotClassMapV2, load_lmot_class_map_v2


STAGE_O_PROTOCOL_ID = "STAGE-O-LOW-LIGHT-DETECTOR-ADAPTATION-20260729-01"
VALIDATION_SEQUENCES = ("LMOT-05", "LMOT-13", "LMOT-14", "LMOT-25")
TRAIN_SEQUENCES = (
    "LMOT-02",
    "LMOT-04",
    "LMOT-06",
    "LMOT-08",
    "LMOT-09",
    "LMOT-12",
    "LMOT-16",
    "LMOT-18",
    "LMOT-20",
    "LMOT-21",
    "LMOT-26",
)
RATE_KEYS = ("precision", "recall", "AP50", "AP50-95")
COUNT_KEYS = (
    "ground_truth_boxes",
    "predicted_boxes",
    "true_positives",
    "false_positives",
    "false_negatives",
)


class StageOProtocolError(ValueError):
    """Raised when a Stage O input violates a frozen boundary."""


@dataclass(frozen=True, slots=True)
class O1Parameters:
    threshold: float
    gamma: float
    clahe_clip_limit: float
    clahe_tile_grid: tuple[int, int] = (8, 8)
    calibration_frames: int = 32

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 255.0:
            raise ValueError("brightness threshold must be in [0, 255]")
        if not 0.0 < self.gamma <= 2.0:
            raise ValueError("gamma must be in (0, 2]")
        if self.clahe_clip_limit <= 0:
            raise ValueError("CLAHE clip limit must be positive")
        if (
            len(self.clahe_tile_grid) != 2
            or min(self.clahe_tile_grid) <= 0
            or self.calibration_frames <= 0
        ):
            raise ValueError("invalid CLAHE grid or calibration frame count")


@dataclass(frozen=True, slots=True)
class PairedFrame:
    sequence: str
    frame_number: int
    light_path: Path
    dark_path: Path


@dataclass(frozen=True, slots=True)
class DetectorOnlySettings:
    weights: Path
    imgsz: int = 640
    confidence: float = 0.30
    nms_iou: float = 0.70
    agnostic_nms: bool = True
    max_detections: int = 300
    device: str = "0"

    def __post_init__(self) -> None:
        if (
            self.imgsz != 640
            or not math.isclose(self.confidence, 0.30)
            or not math.isclose(self.nms_iou, 0.70)
            or self.agnostic_nms is not True
            or self.max_detections != 300
        ):
            raise StageOProtocolError(
                "Detector-only settings differ from the Stage O freeze"
            )


def load_stage_o_protocol(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageOProtocolError("Stage O YAML root must be a mapping")
    if payload.get("protocol_id") != STAGE_O_PROTOCOL_ID:
        raise StageOProtocolError("Unexpected Stage O protocol ID")
    inference = payload.get("shared_detector_only_inference", {})
    expected = {
        "api": "ultralytics.YOLO.predict",
        "model_track_prohibited": True,
        "tracker_loading_prohibited": True,
        "imgsz": 640,
        "confidence": 0.30,
        "nms_iou": 0.70,
        "agnostic_nms": True,
        "max_detections": 300,
        "classes": [0],
        "augment": False,
        "rect": False,
        "half": False,
        "batch": 1,
        "threshold_reselection": "prohibited",
    }
    for key, value in expected.items():
        if inference.get(key) != value:
            raise StageOProtocolError(
                f"Frozen Stage O inference setting changed: {key}"
            )
    data = payload.get("data", {})
    if tuple(data.get("lmot_train_sequences", ())) != TRAIN_SEQUENCES:
        raise StageOProtocolError("LMOT train membership changed")
    if data.get("internal_development_sequences") != ["LMOT-06", "LMOT-26"]:
        raise StageOProtocolError("Internal development membership changed")
    if (
        data.get("paired_group_key")
        != "LMOT_sequence_id_plus_frame_number"
        or data.get("pair_splitting_prohibited") is not True
    ):
        raise StageOProtocolError("Paired grouping is not frozen")
    return payload


def selected_o1_parameters(protocol: Mapping[str, Any]) -> O1Parameters:
    method = protocol["methods"]["O1"]
    selected = method["selected"]
    if selected.get("status") != "frozen_after_internal_development":
        raise StageOProtocolError(
            "O1 internal-development selection is not frozen"
        )
    return O1Parameters(
        threshold=float(selected["threshold"]),
        gamma=float(selected["gamma"]),
        clahe_clip_limit=float(selected["clahe_clip_limit"]),
        clahe_tile_grid=tuple(
            int(value) for value in selected["clahe_tile_grid"]
        ),
        calibration_frames=int(method["gate_calibration_frames"]),
    )


def _numbered_files(directory: Path) -> dict[int, Path]:
    result: dict[int, Path] = {}
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    for path in sorted(row for row in directory.iterdir() if row.is_file()):
        try:
            frame_number = int(path.stem)
        except ValueError as exc:
            raise StageOProtocolError(
                f"Non-numeric LMOT frame name: {path}"
            ) from exc
        if frame_number in result:
            raise StageOProtocolError(
                f"Duplicate LMOT frame number {frame_number}: {directory}"
            )
        result[frame_number] = path
    return result


def discover_paired_frames(
    sequence_root: Path,
    *,
    frame_numbers: Iterable[int] | None = None,
) -> list[PairedFrame]:
    """Return exact light/dark pairs and reject any cross-stream mismatch."""

    sequence_root = sequence_root.resolve()
    light = _numbered_files(sequence_root / "img_light_rgb")
    dark = _numbered_files(sequence_root / "img_dark_rgb")
    if set(light) != set(dark):
        raise StageOProtocolError(
            f"Light/dark frame sets differ for {sequence_root.name}"
        )
    selected = set(light) if frame_numbers is None else set(frame_numbers)
    missing = selected - set(light)
    if missing:
        raise StageOProtocolError(
            f"Missing paired frames in {sequence_root.name}: {sorted(missing)}"
        )
    return [
        PairedFrame(
            sequence=sequence_root.name,
            frame_number=frame_number,
            light_path=light[frame_number],
            dark_path=dark[frame_number],
        )
        for frame_number in sorted(selected)
    ]


def bt601_mean_luma(image: np.ndarray) -> float:
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("brightness gate expects a BGR image")
    means = image.reshape(-1, 3).mean(axis=0)
    return float(0.114 * means[0] + 0.587 * means[1] + 0.299 * means[2])


def sequence_brightness(
    image_paths: Sequence[Path],
    *,
    calibration_frames: int,
) -> float:
    """Compute one source-level decision statistic, never a per-frame gate."""

    if not image_paths:
        raise ValueError("at least one calibration image is required")
    if calibration_frames <= 0:
        raise ValueError("calibration_frames must be positive")
    selected = list(image_paths[:calibration_frames])
    values: list[float] = []
    for path in selected:
        image = read_image(path)
        if image is None:
            raise StageOProtocolError(f"Could not decode {path}")
        values.append(bt601_mean_luma(image))
    return float(statistics.median(values))


def gamma_clahe(image: np.ndarray, parameters: O1Parameters) -> np.ndarray:
    """Apply fixed Gamma followed by fixed LAB-luminance CLAHE."""

    values = np.arange(256, dtype=np.float32) / 255.0
    lut = np.clip(np.power(values, parameters.gamma) * 255.0, 0, 255).astype(
        np.uint8
    )
    corrected = cv2.LUT(image, lut)
    lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(
        clipLimit=parameters.clahe_clip_limit,
        tileGridSize=parameters.clahe_tile_grid,
    )
    enhanced = cv2.merge((clahe.apply(l_channel), a_channel, b_channel))
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


class FrozenRawDetectorAdapter:
    """Use only ``YOLO.predict``; no tracker is constructed or called."""

    def __init__(
        self,
        settings: DetectorOnlySettings,
        *,
        model_factory: Any | None = None,
    ) -> None:
        self.settings = settings
        if not settings.weights.is_file() and model_factory is None:
            raise FileNotFoundError(settings.weights)
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        self.model = model_factory(str(settings.weights))
        self.predict_calls = 0

    def predict(
        self, image: np.ndarray, *, frame_number: int
    ) -> tuple[list[TrackPrediction], dict[str, float]]:
        self.predict_calls += 1
        results = self.model.predict(
            source=image,
            imgsz=self.settings.imgsz,
            conf=self.settings.confidence,
            iou=self.settings.nms_iou,
            agnostic_nms=self.settings.agnostic_nms,
            max_det=self.settings.max_detections,
            classes=[0],
            device=self.settings.device,
            augment=False,
            rect=False,
            verbose=False,
        )
        if not results:
            return [], {}
        result = results[0]
        speed = {
            str(key): float(value)
            for key, value in getattr(result, "speed", {}).items()
        }
        boxes = getattr(result, "boxes", None)
        if boxes is None or len(boxes) == 0:
            return [], speed
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidence = boxes.conf.detach().cpu().numpy()
        rows = [
            TrackPrediction(
                frame_number=frame_number,
                track_id=-(frame_number * 100000 + index + 1),
                xyxy=tuple(float(value) for value in box),
                confidence=float(score),
            )
            for index, (box, score) in enumerate(
                zip(xyxy, confidence, strict=True)
            )
        ]
        return rows, speed


def pooled_detection_metrics(
    rows: Mapping[str, tuple[Sequence[LmotAnnotation], Sequence[TrackPrediction]]],
) -> dict[str, Any]:
    """Evaluate all sequences with isolated frame IDs and one score ordering."""

    pooled_gt: list[LmotAnnotation] = []
    pooled_predictions: list[TrackPrediction] = []
    per_sequence: dict[str, dict[str, float | int]] = {}
    for sequence_index, (sequence, (gt, predictions)) in enumerate(
        sorted(rows.items())
    ):
        per_sequence[sequence] = evaluate_motor_vehicle_detections(
            gt=gt, predictions=predictions
        )
        frame_offset = sequence_index * 1_000_000
        pooled_gt.extend(
            replace(item, frame_number=frame_offset + item.frame_number)
            for item in gt
        )
        pooled_predictions.extend(
            replace(item, frame_number=frame_offset + item.frame_number)
            for item in predictions
        )
    pooled = evaluate_motor_vehicle_detections(
        gt=pooled_gt, predictions=pooled_predictions
    )
    macro = {
        key: float(
            np.mean(
                [
                    float(metrics[key])
                    for metrics in per_sequence.values()
                ]
            )
        )
        for key in RATE_KEYS
    }
    macro["definition"] = "unweighted_mean_of_per_sequence_rates"
    return {
        "pooled_micro": {
            **pooled,
            "definition": (
                "all_predictions_one_confidence_ordering_with_isolated_"
                "sequence_frame_keys_and_summed_counts"
            ),
        },
        "per_sequence_macro": macro,
        "per_sequence": per_sequence,
    }


def light_to_dark_retention(
    light: Mapping[str, Any], dark: Mapping[str, Any]
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for key in RATE_KEYS:
        denominator = float(light[key])
        result[key] = (
            float(dark[key]) / denominator if denominator > 0.0 else None
        )
    return result


def select_stage_o_candidate(
    *,
    protocol: Mapping[str, Any],
    baseline_metrics: Mapping[str, Any],
    candidate_metrics: Mapping[str, Mapping[str, Any]],
    blocked_methods: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Apply the predeclared Stage O rule without changing any threshold."""

    rule = protocol["selection_rule_frozen_before_formal_comparison"]
    baseline = baseline_metrics["illumination"]["dark"]["pooled_micro"]
    rows: list[dict[str, Any]] = []
    for method_id, metrics in sorted(candidate_metrics.items()):
        dark = metrics["illumination"]["dark"]["pooled_micro"]
        deltas = {
            key: float(dark[key]) - float(baseline[key])
            for key in RATE_KEYS
        }
        checks = {
            "minimum_dark_recall_gain": (
                deltas["recall"]
                >= float(rule["eligibility"]["minimum_dark_recall_gain_over_O0"])
            ),
            "minimum_dark_AP50_gain": (
                deltas["AP50"]
                >= float(rule["eligibility"]["minimum_dark_AP50_gain_over_O0"])
            ),
            "maximum_precision_drop": (
                deltas["precision"]
                >= -float(rule["eligibility"]["maximum_precision_drop_from_O0"])
            ),
            "nonnegative_AP50_95_delta": (
                deltas["AP50-95"] >= 0.0
                if rule["eligibility"]["require_nonnegative_AP50_95_delta"]
                else True
            ),
        }
        rows.append(
            {
                "method_id": method_id,
                "eligible": all(checks.values()),
                "checks": checks,
                "dark_pooled_micro": {
                    key: dark[key] for key in (*RATE_KEYS, *COUNT_KEYS)
                },
                "delta_from_O0": deltas,
                "wall_fps": metrics["runtime"]["wall_fps"],
            }
        )
    eligible = [row for row in rows if row["eligible"]]
    eligible.sort(
        key=lambda row: (
            -float(row["dark_pooled_micro"]["AP50"]),
            -float(row["dark_pooled_micro"]["recall"]),
            -float(row["dark_pooled_micro"]["AP50-95"]),
            -float(row["dark_pooled_micro"]["precision"]),
            -float(row["wall_fps"]),
            str(row["method_id"]),
        )
    )
    selected = eligible[0]["method_id"] if eligible else "O0"
    return {
        "schema_version": 1,
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "selection_rule_source": "frozen_before_formal_comparison",
        "baseline_method": "O0",
        "selected_method": selected,
        "selected_detector_role": "D1-LL" if selected == "O3" else "D1",
        "fallback_to_O0": not eligible,
        "thresholds_reselected": False,
        "sequences_or_frames_removed": False,
        "candidate_rows": rows,
        "blocked_methods": dict(blocked_methods or {}),
    }


def _frame_failures(
    gt: Sequence[LmotAnnotation],
    predictions: Sequence[TrackPrediction],
    *,
    iou_threshold: float = 0.5,
) -> tuple[int, int, int]:
    used: set[int] = set()
    tp = 0
    for prediction in sorted(
        predictions, key=lambda row: row.confidence, reverse=True
    ):
        available = [
            (index, candidate)
            for index, candidate in enumerate(gt)
            if index not in used
        ]
        if not available:
            continue
        box = np.asarray(prediction.xyxy)
        scored = []
        for index, candidate in available:
            other = np.asarray(candidate.xyxy)
            left_top = np.maximum(box[:2], other[:2])
            right_bottom = np.minimum(box[2:], other[2:])
            wh = np.maximum(0.0, right_bottom - left_top)
            intersection = float(wh[0] * wh[1])
            area_a = float(np.prod(np.maximum(0.0, box[2:] - box[:2])))
            area_b = float(np.prod(np.maximum(0.0, other[2:] - other[:2])))
            union = area_a + area_b - intersection
            scored.append((intersection / union if union > 0 else 0.0, index))
        overlap, best = max(scored)
        if overlap >= iou_threshold:
            used.add(best)
            tp += 1
    return tp, len(predictions) - tp, len(gt) - tp


def _render_contact_sheet(
    *,
    failures: Sequence[dict[str, Any]],
    output_path: Path,
    rows_by_key: Mapping[
        tuple[str, str, int],
        tuple[Path, Sequence[LmotAnnotation], Sequence[TrackPrediction], bool],
    ],
    o1_parameters: O1Parameters | None,
    image_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
) -> None:
    selected = list(failures[:8])
    tile_w, tile_h = 450, 280
    sheet = np.full((tile_h * 2, tile_w * 4, 3), 28, dtype=np.uint8)
    for index, record in enumerate(selected):
        key = (
            str(record["sequence"]),
            str(record["illumination"]),
            int(record["frame_number"]),
        )
        path, gt, predictions, enhanced = rows_by_key[key]
        image = read_image(path)
        if image is None:
            continue
        if enhanced and o1_parameters is not None:
            image = gamma_clahe(image, o1_parameters)
        elif enhanced and image_preprocessor is not None:
            image = image_preprocessor(image)
        for row in gt:
            x1, y1, x2, y2 = (int(value) for value in row.xyxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 0), 2)
        for row in predictions:
            x1, y1, x2, y2 = (int(value) for value in row.xyxy)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 0, 240), 2)
        image = cv2.resize(image, (tile_w, tile_h))
        label = (
            f"{key[0]} {key[1]} f={key[2]} "
            f"FP={record['false_positives']} FN={record['false_negatives']}"
        )
        cv2.rectangle(image, (0, 0), (tile_w, 30), (0, 0, 0), -1)
        cv2.putText(
            image,
            label,
            (7, 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.49,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        row_index, column = divmod(index, 4)
        sheet[
            row_index * tile_h : (row_index + 1) * tile_h,
            column * tile_w : (column + 1) * tile_w,
        ] = image
    write_image(output_path, sheet)


def run_detector_only_evaluation(
    *,
    protocol_path: Path,
    validation_root: Path,
    class_map_path: Path,
    weights_path: Path,
    output_dir: Path,
    method_id: str,
    sequences: Sequence[str],
    illumination_directories: Sequence[str] = (
        "img_light_rgb",
        "img_dark_rgb",
    ),
    device: str = "0",
    o1_parameters_override: O1Parameters | None = None,
    image_preprocessor: Callable[[np.ndarray], np.ndarray] | None = None,
    model_factory: Any | None = None,
) -> dict[str, Any]:
    """Run one non-overwriting Stage O raw detector-only evaluation."""

    protocol_path = protocol_path.resolve()
    validation_root = validation_root.resolve()
    class_map_path = class_map_path.resolve()
    weights_path = weights_path.resolve()
    output_dir = output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {output_dir}")
    if method_id not in {"O0", "O1", "O2", "O3"}:
        raise StageOProtocolError(f"Unsupported detector-only arm: {method_id}")
    if any(
        item not in {"img_light_rgb", "img_dark_rgb"}
        for item in illumination_directories
    ):
        raise StageOProtocolError("Unsupported illumination directory")

    protocol = load_stage_o_protocol(protocol_path)
    o1_parameters = (
        o1_parameters_override
        if method_id == "O1" and o1_parameters_override is not None
        else selected_o1_parameters(protocol)
        if method_id == "O1"
        else None
    )
    if method_id != "O1" and o1_parameters_override is not None:
        raise StageOProtocolError("O1 override supplied to a non-O1 arm")
    if method_id == "O2" and image_preprocessor is None:
        raise StageOProtocolError("O2 requires the frozen image preprocessor")
    if method_id != "O2" and image_preprocessor is not None:
        raise StageOProtocolError(
            "External image preprocessor supplied to a non-O2 arm"
        )
    class_map: LmotClassMapV2 = load_lmot_class_map_v2(class_map_path)
    settings = DetectorOnlySettings(weights=weights_path, device=device)

    output_dir.mkdir(parents=True)
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(output_dir / "_ultralytics_config")
    )
    (output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    adapter = FrozenRawDetectorAdapter(settings, model_factory=model_factory)

    try:
        import torch
    except ImportError:  # pragma: no cover - package is present in formal runs
        torch = None
    measure_cuda = (
        torch is not None
        and torch.cuda.is_available()
        and device.lower() not in {"cpu", "mps"}
    )
    if measure_cuda:
        torch.cuda.reset_peak_memory_stats()

    result_rows: dict[
        str, dict[str, tuple[list[LmotAnnotation], list[TrackPrediction]]]
    ] = {directory: {} for directory in illumination_directories}
    detection_records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    qualitative: dict[
        tuple[str, str, int],
        tuple[Path, Sequence[LmotAnnotation], Sequence[TrackPrediction], bool],
    ] = {}
    timing = {
        "read_and_preprocess_ms": 0.0,
        "wall_predict_ms": 0.0,
        "framework_preprocess_ms": 0.0,
        "framework_inference_ms": 0.0,
        "framework_postprocess_ms": 0.0,
    }
    frame_count = 0
    sequence_gate: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()

    for sequence in sequences:
        sequence_root = validation_root / sequence
        pairs = discover_paired_frames(sequence_root)
        frame_numbers = {pair.frame_number for pair in pairs}
        all_annotations = [
            row
            for row in parse_lmot_gt(sequence_root / "gt" / "gt.txt")
            if row.frame_number in frame_numbers
        ]
        evaluated, suppression = split_motor_vehicle_truth(
            all_annotations,
            class_map=class_map,
            evaluated_ignore_values=class_map.evaluated_mark_values,
        )
        evaluated_by_frame: dict[int, list[LmotAnnotation]] = {}
        suppression_by_frame: dict[int, list[LmotAnnotation]] = {}
        for row in evaluated:
            evaluated_by_frame.setdefault(row.frame_number, []).append(row)
        for row in suppression:
            suppression_by_frame.setdefault(row.frame_number, []).append(row)

        for directory in illumination_directories:
            paths = [
                pair.light_path
                if directory == "img_light_rgb"
                else pair.dark_path
                for pair in pairs
            ]
            brightness = sequence_brightness(
                paths,
                calibration_frames=(
                    o1_parameters.calibration_frames
                    if o1_parameters is not None
                    else 32
                ),
            )
            apply_enhancement = bool(
                o1_parameters is not None
                and brightness < o1_parameters.threshold
                or method_id == "O2"
                and directory == "img_dark_rgb"
            )
            sequence_gate[f"{sequence}:{directory}"] = {
                "brightness": brightness,
                "threshold": (
                    o1_parameters.threshold
                    if o1_parameters is not None
                    else None
                ),
                "enhancement_applied_for_entire_sequence": apply_enhancement,
                "decision_scope": "once_per_sequence",
            }
            kept_all: list[TrackPrediction] = []
            for pair, path in zip(pairs, paths, strict=True):
                frame_started = time.perf_counter()
                image = read_image(path)
                if image is None:
                    raise StageOProtocolError(f"Could not decode {path}")
                if apply_enhancement and o1_parameters is not None:
                    image = gamma_clahe(image, o1_parameters)
                elif apply_enhancement and image_preprocessor is not None:
                    image = image_preprocessor(image)
                timing["read_and_preprocess_ms"] += (
                    time.perf_counter() - frame_started
                ) * 1000.0
                predict_started = time.perf_counter()
                raw_predictions, speed = adapter.predict(
                    image, frame_number=pair.frame_number
                )
                timing["wall_predict_ms"] += (
                    time.perf_counter() - predict_started
                ) * 1000.0
                for key in ("preprocess", "inference", "postprocess"):
                    timing[f"framework_{key}_ms"] += float(speed.get(key, 0.0))
                kept, removed = suppress_predictions_on_excluded_truth(
                    raw_predictions,
                    evaluated_by_frame.get(pair.frame_number, []),
                    suppression_by_frame.get(pair.frame_number, []),
                )
                kept_all.extend(kept)
                frame_gt = evaluated_by_frame.get(pair.frame_number, [])
                tp, fp, fn = _frame_failures(frame_gt, kept)
                record = {
                    "method_id": method_id,
                    "sequence": sequence,
                    "illumination": (
                        "light" if directory == "img_light_rgb" else "dark"
                    ),
                    "image_directory": directory,
                    "frame_number": pair.frame_number,
                    "image_name": path.name,
                    "source_image_sha256": sha256_file(path),
                    "sequence_brightness": brightness,
                    "enhancement_applied": apply_enhancement,
                    "raw_prediction_count": len(raw_predictions),
                    "suppressed_prediction_count": removed,
                    "ground_truth_box_count": len(frame_gt),
                    "detections": [
                        {
                            "bbox_xyxy": list(row.xyxy),
                            "confidence": row.confidence,
                            "class_id": 0,
                            "class_name": "vehicle",
                        }
                        for row in kept
                    ],
                }
                detection_records.append(record)
                failure = {
                    "sequence": sequence,
                    "illumination": record["illumination"],
                    "frame_number": pair.frame_number,
                    "true_positives": tp,
                    "false_positives": fp,
                    "false_negatives": fn,
                    "severity": fp + fn,
                    "source_image": str(path),
                }
                if fp or fn:
                    failures.append(failure)
                    qualitative[
                        (
                            sequence,
                            str(record["illumination"]),
                            pair.frame_number,
                        )
                    ] = (path, frame_gt, kept, apply_enhancement)
                frame_count += 1
            result_rows[directory][sequence] = (evaluated, kept_all)

    if measure_cuda:
        torch.cuda.synchronize()
        peak_allocated = int(torch.cuda.max_memory_allocated())
        peak_reserved = int(torch.cuda.max_memory_reserved())
    else:
        peak_allocated = None
        peak_reserved = None
    elapsed = time.perf_counter() - started

    metrics_by_illumination = {
        ("light" if directory == "img_light_rgb" else "dark"):
        pooled_detection_metrics(result_rows[directory])
        for directory in illumination_directories
    }
    metrics: dict[str, Any] = {
        "schema_version": 1,
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "method_id": method_id,
        "task": "raw_detector_only_unified_motor_vehicle_box_detection",
        "dataset_role": (
            "consumed_development_diagnostic"
            if tuple(sequences) == VALIDATION_SEQUENCES
            else "LMOT_train_internal_development"
        ),
        "tracker_emitted_boxes": False,
        "model_track_called": False,
        "thresholds_reselected_from_validation": False,
        "illumination": metrics_by_illumination,
    }
    if {"light", "dark"}.issubset(metrics_by_illumination):
        metrics["light_to_dark_retention"] = light_to_dark_retention(
            metrics_by_illumination["light"]["pooled_micro"],
            metrics_by_illumination["dark"]["pooled_micro"],
        )
    metrics["runtime"] = {
        "evaluated_frames": frame_count,
        "elapsed_seconds": elapsed,
        "wall_ms_per_frame": elapsed * 1000.0 / max(frame_count, 1),
        "wall_fps": frame_count / elapsed if elapsed > 0 else None,
        "framework_speed_ms_per_frame": {
            key.removeprefix("framework_"): value / max(frame_count, 1)
            for key, value in timing.items()
            if key.startswith("framework_")
        },
        "peak_cuda_memory_allocated_bytes": peak_allocated,
        "peak_cuda_memory_reserved_bytes": peak_reserved,
    }
    runtime = {
        "schema_version": 1,
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "method_id": method_id,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "weights": {
            "path": str(weights_path),
            "bytes": weights_path.stat().st_size,
            "sha256": sha256_file(weights_path),
        },
        "inference_api": "ultralytics.YOLO.predict",
        "inference_performed": True,
        "model_loaded": True,
        "model_predict_called": adapter.predict_calls > 0,
        "model_predict_call_count": adapter.predict_calls,
        "model_track_called": False,
        "tracker_loaded": False,
        "training_performed": False,
        "settings": asdict(settings),
        "O1_parameters": asdict(o1_parameters) if o1_parameters else None,
        "O2_preprocessor": (
            image_preprocessor.metadata()
            if image_preprocessor is not None
            and hasattr(image_preprocessor, "metadata")
            else None
        ),
        "sequence_gate": sequence_gate,
        **metrics["runtime"],
    }

    failures.sort(
        key=lambda row: (
            -int(row["severity"]),
            str(row["sequence"]),
            str(row["illumination"]),
            int(row["frame_number"]),
        )
    )
    (output_dir / "detections.jsonl").write_text(
        "\n".join(
            json.dumps(row, sort_keys=True) for row in detection_records
        )
        + "\n",
        encoding="utf-8",
    )
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (output_dir / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    (output_dir / "failure_cases.json").write_text(
        json.dumps(failures, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    _render_contact_sheet(
        failures=failures,
        output_path=output_dir / "qualitative_contact_sheet.jpg",
        rows_by_key=qualitative,
        o1_parameters=o1_parameters,
        image_preprocessor=image_preprocessor,
    )
    return metrics


def canonical_records_sha256(records: Sequence[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(records), sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
