from __future__ import annotations

import configparser
import csv
import hashlib
import importlib.metadata
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import cv2
import numpy as np
import yaml


STAGE_N_PROTOCOL_ID = "STAGE-N-LMOT-TRACKING-DIAGNOSTIC-20260728-01"
MOTOR_VEHICLE_CLASSES = frozenset(
    {"car", "motorcycle", "bus", "truck"}
)
NON_MOTOR_CLASSES = frozenset({"person", "bicycle"})
APPROVED_VALIDATION_ENTRIES = frozenset(
    {"img_dark_rgb", "img_light_rgb", "gt", "seqinfo.ini"}
)


class StageNDataGateError(ValueError):
    """Raised when a Stage N acquisition or truth invariant is not met."""


@dataclass(frozen=True, slots=True)
class LmotAnnotation:
    """One LMOT nine-column MOT-format annotation."""

    frame_number: int
    track_id: int
    x: float
    y: float
    width: float
    height: float
    ignore: int
    class_id: int
    visibility: float

    @property
    def xyxy(self) -> tuple[float, float, float, float]:
        return (
            self.x,
            self.y,
            self.x + self.width,
            self.y + self.height,
        )


@dataclass(frozen=True, slots=True)
class TrackPrediction:
    frame_number: int
    track_id: int
    xyxy: tuple[float, float, float, float]
    confidence: float = 1.0


@dataclass(frozen=True, slots=True)
class StageNInferenceSettings:
    weights: str
    confidence: float = 0.30
    nms_iou: float = 0.70
    image_size: int = 640
    max_detections: int = 300
    device: str = "0"
    agnostic_nms: bool = True

    def __post_init__(self) -> None:
        frozen = (
            self.confidence == 0.30
            and self.nms_iou == 0.70
            and self.image_size == 640
            and self.max_detections == 300
            and self.agnostic_nms is True
        )
        if not frozen:
            raise StageNDataGateError(
                "Stage N inference settings differ from the frozen design"
            )


@dataclass(frozen=True, slots=True)
class VerifiedLmotClassMap:
    """Numeric LMOT class IDs backed by explicit source evidence.

    The official README names six classes but does not state numeric IDs.
    Production conversion therefore accepts only ``official_verified`` maps.
    Synthetic fixtures can opt into ``synthetic_fixture`` without turning
    that fixture into a claim about LMOT.
    """

    id_to_name: Mapping[int, str]
    verification_status: str
    evidence: str
    evidence_sha256: str | None = None

    def __post_init__(self) -> None:
        allowed = {"official_verified", "synthetic_fixture"}
        if self.verification_status not in allowed:
            raise StageNDataGateError(
                "LMOT numeric class mapping is unresolved; do not infer IDs "
                "from the category order in the README"
            )
        names = {str(value).lower() for value in self.id_to_name.values()}
        unexpected = names - MOTOR_VEHICLE_CLASSES - NON_MOTOR_CLASSES
        if unexpected:
            raise StageNDataGateError(
                f"Unexpected LMOT class names: {sorted(unexpected)}"
            )
        if self.verification_status == "official_verified":
            expected = MOTOR_VEHICLE_CLASSES | NON_MOTOR_CLASSES
            if names != expected:
                raise StageNDataGateError(
                    "An official class map must cover all six LMOT classes"
                )
            if not self.evidence_sha256:
                raise StageNDataGateError(
                    "Official class mapping requires a frozen evidence hash"
                )

    def class_name(self, class_id: int) -> str:
        try:
            return str(self.id_to_name[int(class_id)]).lower()
        except KeyError as exc:
            raise StageNDataGateError(
                f"Class ID {class_id} is absent from the verified map"
            ) from exc

    def is_motor_vehicle(self, class_id: int) -> bool:
        return self.class_name(class_id) in MOTOR_VEHICLE_CLASSES

    def is_non_motor(self, class_id: int) -> bool:
        return self.class_name(class_id) in NON_MOTOR_CLASSES


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_image(path: Path, flags: int = cv2.IMREAD_COLOR) -> np.ndarray | None:
    """Read images through bytes so Windows Unicode paths remain supported."""

    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def write_image(path: Path, image: np.ndarray) -> None:
    """Write an image without relying on OpenCV's Windows path handling."""

    extension = path.suffix.lower()
    if extension not in {".jpg", ".jpeg", ".png", ".bmp"}:
        raise ValueError(f"Unsupported qualitative image extension: {extension}")
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise RuntimeError(f"Could not encode image for {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded.tofile(path)


def load_stage_n_protocol(
    config_path: Path,
    *,
    verify_files: bool = False,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise StageNDataGateError("Unsupported Stage N schema")
    if payload.get("protocol_id") != STAGE_N_PROTOCOL_ID:
        raise StageNDataGateError("Unexpected Stage N protocol ID")
    shared = payload.get("shared_inference", {})
    expected = {
        "detector_id": "D1",
        "confidence": 0.30,
        "nms_iou": 0.70,
        "imgsz": 640,
        "agnostic_nms": True,
        "max_detections": 300,
        "training": "prohibited",
        "fine_tuning": "prohibited",
        "threshold_reselection": "prohibited",
    }
    for key, value in expected.items():
        if shared.get(key) != value:
            raise StageNDataGateError(
                f"Stage N shared_inference.{key} changed"
            )
    methods = payload.get("methods", {})
    expected_methods = {
        "L0": ("img_light_rgb", "ByteTrack"),
        "L1": ("img_light_rgb", "TrackTrack"),
        "L2": ("img_dark_rgb", "ByteTrack"),
        "L3": ("img_dark_rgb", "TrackTrack"),
    }
    if list(methods) != list(expected_methods):
        raise StageNDataGateError("Stage N must contain L0-L3 in order")
    for method_id, (image_directory, tracker) in expected_methods.items():
        if (
            methods[method_id].get("image_directory") != image_directory
            or methods[method_id].get("tracker") != tracker
        ):
            raise StageNDataGateError(f"Invalid {method_id} freeze")
    if (
        payload.get("comparison_label")
        != "controlled_end_to_end_tracker_backend_comparison"
    ):
        raise StageNDataGateError("Stage N comparison label changed")
    if payload["lmot"].get("data_gate") != "blocked_before_download":
        raise StageNDataGateError(
            "This frozen Stage N record must preserve the blocked download"
        )
    if (
        payload["parking_occupancy_formal_gate"].get("status")
        != "blocked_no_qualifying_new_data"
    ):
        raise StageNDataGateError(
            "This Stage N formal parking gate must remain blocked"
        )
    if verify_files:
        records = [
            (
                Path(shared["weights_path"]),
                int(shared["weights_bytes"]),
                shared["weights_sha256"],
                "D1 weights",
            )
        ]
        for key in ("bytetrack", "tracktrack"):
            tracker = payload["trackers"][key]
            records.append(
                (
                    (config_path.parent / tracker["config_path"]).resolve(),
                    int(tracker["config_bytes"]),
                    tracker["config_sha256"],
                    f"{key} config",
                )
            )
        for key in ("stage_m_report", "stage_m_registry"):
            record = payload["preservation"][key]
            records.append(
                (
                    (config_path.parent / record["path"]).resolve(),
                    int(record["bytes"]),
                    record["sha256"],
                    key,
                )
            )
        for path, expected_bytes, expected_hash, label in records:
            if not path.is_file():
                raise StageNDataGateError(f"Missing {label}: {path}")
            if path.stat().st_size != expected_bytes:
                raise StageNDataGateError(f"{label} byte count mismatch")
            if sha256_file(path) != expected_hash:
                raise StageNDataGateError(f"{label} SHA-256 mismatch")
    return payload


def parse_lmot_gt(path: Path) -> list[LmotAnnotation]:
    """Parse the exact LMOT row schema.

    Expected columns:
    fn,id,x,y,width,height,ignore,classid,visibility
    """

    rows: list[LmotAnnotation] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.reader(handle)
        for line_number, values in enumerate(reader, start=1):
            if not values or all(not value.strip() for value in values):
                continue
            if len(values) != 9:
                raise StageNDataGateError(
                    f"{path}:{line_number}: expected 9 columns, got "
                    f"{len(values)}"
                )
            try:
                numeric = [float(value.strip()) for value in values]
            except ValueError as exc:
                raise StageNDataGateError(
                    f"{path}:{line_number}: non-numeric MOT field"
                ) from exc
            integer_indices = (0, 1, 6, 7)
            if any(not numeric[index].is_integer() for index in integer_indices):
                raise StageNDataGateError(
                    f"{path}:{line_number}: fn/id/ignore/classid must be "
                    "integers"
                )
            row = LmotAnnotation(
                frame_number=int(numeric[0]),
                track_id=int(numeric[1]),
                x=numeric[2],
                y=numeric[3],
                width=numeric[4],
                height=numeric[5],
                ignore=int(numeric[6]),
                class_id=int(numeric[7]),
                visibility=numeric[8],
            )
            if row.frame_number <= 0 or row.track_id <= 0:
                raise StageNDataGateError(
                    f"{path}:{line_number}: frame and track IDs must be "
                    "positive"
                )
            if not all(
                math.isfinite(value)
                for value in (
                    row.x,
                    row.y,
                    row.width,
                    row.height,
                    row.visibility,
                )
            ):
                raise StageNDataGateError(
                    f"{path}:{line_number}: non-finite value"
                )
            if row.width <= 0 or row.height <= 0:
                raise StageNDataGateError(
                    f"{path}:{line_number}: box size must be positive"
                )
            if not 0.0 <= row.visibility <= 1.0:
                raise StageNDataGateError(
                    f"{path}:{line_number}: visibility must be in [0, 1]"
                )
            rows.append(row)
    return rows


def _numbered_images(path: Path) -> tuple[dict[int, Path], list[str]]:
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images: dict[int, Path] = {}
    errors: list[str] = []
    for candidate in sorted(path.iterdir()):
        if not candidate.is_file() or candidate.suffix.lower() not in extensions:
            continue
        try:
            frame_number = int(candidate.stem)
        except ValueError:
            errors.append(f"non_numeric_frame_name:{candidate.name}")
            continue
        if frame_number in images:
            errors.append(
                f"duplicate_frame_number:{frame_number}:"
                f"{images[frame_number].name}:{candidate.name}"
            )
            continue
        images[frame_number] = candidate
    return images, errors


def audit_lmot_sequence(
    sequence_root: Path,
    *,
    decode_images: bool = True,
) -> dict[str, Any]:
    """Audit paired validation RGB streams and the shared GT without guessing.

    A sequence passes only when both RGB frame-number sets are identical,
    ``seqLength`` agrees, every image decodes, GT keys are unique, and every
    GT frame exists in the paired streams.
    """

    sequence_root = sequence_root.resolve()
    required = {
        "img_dark_rgb": sequence_root / "img_dark_rgb",
        "img_light_rgb": sequence_root / "img_light_rgb",
        "gt": sequence_root / "gt" / "gt.txt",
        "seqinfo": sequence_root / "seqinfo.ini",
    }
    missing = [
        key
        for key, path in required.items()
        if not (path.is_dir() if key.endswith("_rgb") else path.is_file())
    ]
    if missing:
        raise StageNDataGateError(
            f"{sequence_root}: missing required entries {missing}"
        )

    unexpected = sorted(
        item.name
        for item in sequence_root.iterdir()
        if item.name not in APPROVED_VALIDATION_ENTRIES
    )
    dark, dark_errors = _numbered_images(required["img_dark_rgb"])
    light, light_errors = _numbered_images(required["img_light_rgb"])
    corrupt: list[str] = []
    if decode_images:
        for stream_name, images in (("dark", dark), ("light", light)):
            for frame_number, image_path in images.items():
                if read_image(image_path, cv2.IMREAD_UNCHANGED) is None:
                    corrupt.append(f"{stream_name}:{frame_number}:{image_path.name}")

    parser = configparser.ConfigParser()
    parser.read(required["seqinfo"], encoding="utf-8-sig")
    if "Sequence" not in parser:
        raise StageNDataGateError(
            f"{required['seqinfo']}: missing [Sequence]"
        )
    section = parser["Sequence"]
    try:
        seq_length = section.getint("seqLength")
    except (ValueError, TypeError) as exc:
        raise StageNDataGateError(
            f"{required['seqinfo']}: invalid seqLength"
        ) from exc

    annotations = parse_lmot_gt(required["gt"])
    duplicate_gt_keys: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    frames_by_track: dict[int, list[int]] = {}
    for row in annotations:
        key = (row.frame_number, row.track_id)
        if key in seen:
            duplicate_gt_keys.append(key)
        seen.add(key)
        frames_by_track.setdefault(row.track_id, []).append(row.frame_number)
    id_gaps = {
        track_id: [
            [left, right]
            for left, right in zip(
                sorted(set(frames))[:-1], sorted(set(frames))[1:]
            )
            if right != left + 1
        ]
        for track_id, frames in frames_by_track.items()
    }
    id_gaps = {key: value for key, value in id_gaps.items() if value}

    dark_frames = set(dark)
    light_frames = set(light)
    gt_frames = {row.frame_number for row in annotations}
    paired = dark_frames == light_frames
    expected_frames = set(range(1, seq_length + 1))
    errors = dark_errors + light_errors
    if unexpected:
        errors.append(f"unexpected_sequence_entries:{unexpected}")
    if not paired:
        errors.append("dark_light_frame_sets_differ")
    if dark_frames != expected_frames:
        errors.append("dark_frames_do_not_match_seqLength")
    if light_frames != expected_frames:
        errors.append("light_frames_do_not_match_seqLength")
    if not gt_frames.issubset(dark_frames & light_frames):
        errors.append("gt_references_unpaired_or_missing_frame")
    if duplicate_gt_keys:
        errors.append("duplicate_gt_frame_track_keys")
    if corrupt:
        errors.append("corrupt_images")

    return {
        "sequence": sequence_root.name,
        "seq_length": seq_length,
        "dark_frame_count": len(dark),
        "light_frame_count": len(light),
        "gt_rows": len(annotations),
        "gt_tracks": len(frames_by_track),
        "dark_light_frame_numbers_aligned": paired,
        "single_gt_file_for_aligned_pair": paired
        and gt_frames.issubset(dark_frames),
        "missing_dark_frames": sorted(expected_frames - dark_frames),
        "missing_light_frames": sorted(expected_frames - light_frames),
        "dark_only_frames": sorted(dark_frames - light_frames),
        "light_only_frames": sorted(light_frames - dark_frames),
        "duplicate_gt_keys": [list(key) for key in duplicate_gt_keys],
        "track_id_frame_gaps": id_gaps,
        "ignore_values_observed": sorted({row.ignore for row in annotations}),
        "class_ids_observed": sorted({row.class_id for row in annotations}),
        "visibility": {
            "minimum": min((row.visibility for row in annotations), default=None),
            "maximum": max((row.visibility for row in annotations), default=None),
        },
        "corrupt_images": corrupt,
        "unexpected_entries": unexpected,
        "errors": errors,
        "passed": not errors,
    }


def split_motor_vehicle_truth(
    annotations: Iterable[LmotAnnotation],
    *,
    class_map: VerifiedLmotClassMap,
    evaluated_ignore_values: frozenset[int],
) -> tuple[list[LmotAnnotation], list[LmotAnnotation]]:
    """Split evaluated motor vehicles from all prediction-suppressing GT.

    Person, bicycle, and explicitly ignored GT are retained as suppression
    regions. This prevents a unified D1 prediction on an excluded object from
    becoming a motor-vehicle false positive.
    """

    if not evaluated_ignore_values:
        raise StageNDataGateError(
            "Verified semantics for the LMOT ignore field are required"
        )
    evaluated: list[LmotAnnotation] = []
    suppression: list[LmotAnnotation] = []
    for row in annotations:
        if (
            row.ignore in evaluated_ignore_values
            and class_map.is_motor_vehicle(row.class_id)
        ):
            evaluated.append(row)
        else:
            # This includes person, bicycle, and dataset-marked ignored GT.
            class_map.class_name(row.class_id)
            suppression.append(row)
    return evaluated, suppression


def _iou_matrix(
    boxes_a: np.ndarray,
    boxes_b: np.ndarray,
) -> np.ndarray:
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float64)
    top_left = np.maximum(boxes_a[:, None, :2], boxes_b[None, :, :2])
    bottom_right = np.minimum(boxes_a[:, None, 2:], boxes_b[None, :, 2:])
    widths = np.maximum(0.0, bottom_right - top_left)
    intersection = widths[..., 0] * widths[..., 1]
    area_a = np.maximum(0.0, boxes_a[:, 2] - boxes_a[:, 0]) * np.maximum(
        0.0, boxes_a[:, 3] - boxes_a[:, 1]
    )
    area_b = np.maximum(0.0, boxes_b[:, 2] - boxes_b[:, 0]) * np.maximum(
        0.0, boxes_b[:, 3] - boxes_b[:, 1]
    )
    union = area_a[:, None] + area_b[None, :] - intersection
    return np.divide(
        intersection,
        union,
        out=np.zeros_like(intersection),
        where=union > 0,
    )


def suppress_predictions_on_excluded_truth(
    predictions: Sequence[TrackPrediction],
    evaluated_gt: Sequence[LmotAnnotation],
    suppression_gt: Sequence[LmotAnnotation],
    *,
    iou_threshold: float = 0.5,
) -> tuple[list[TrackPrediction], int]:
    """Remove predictions attributable only to excluded/ignored GT.

    Predictions that overlap evaluated motor-vehicle truth at the threshold
    are always retained. The rule is frozen before any LMOT result is seen.
    """

    if not 0.0 < iou_threshold <= 1.0:
        raise ValueError("iou_threshold must be in (0, 1]")
    kept: list[TrackPrediction] = []
    removed = 0
    gt_by_frame: dict[int, list[LmotAnnotation]] = {}
    ignored_by_frame: dict[int, list[LmotAnnotation]] = {}
    for row in evaluated_gt:
        gt_by_frame.setdefault(row.frame_number, []).append(row)
    for row in suppression_gt:
        ignored_by_frame.setdefault(row.frame_number, []).append(row)
    for prediction in predictions:
        box = np.asarray([prediction.xyxy], dtype=np.float64)
        active_boxes = np.asarray(
            [
                row.xyxy
                for row in gt_by_frame.get(prediction.frame_number, [])
            ],
            dtype=np.float64,
        ).reshape((-1, 4))
        ignored_boxes = np.asarray(
            [
                row.xyxy
                for row in ignored_by_frame.get(
                    prediction.frame_number, []
                )
            ],
            dtype=np.float64,
        ).reshape((-1, 4))
        active_overlap = (
            float(_iou_matrix(box, active_boxes).max())
            if len(active_boxes)
            else 0.0
        )
        ignored_overlap = (
            float(_iou_matrix(box, ignored_boxes).max())
            if len(ignored_boxes)
            else 0.0
        )
        if active_overlap < iou_threshold <= ignored_overlap:
            removed += 1
        else:
            kept.append(prediction)
    return kept, removed


def evaluate_motor_vehicle_detections(
    *,
    gt: Sequence[LmotAnnotation],
    predictions: Sequence[TrackPrediction],
    iou_thresholds: Sequence[float] = tuple(
        float(value) for value in np.linspace(0.5, 0.95, 10)
    ),
) -> dict[str, float | int]:
    """Evaluate frozen-confidence, single-class motor-vehicle boxes.

    HOTA/IDF1 are always delegated to TrackEval. Detection AP is calculated
    here because TrackEval is a tracking evaluator and does not provide COCO
    box AP. Predictions must already have passed excluded-GT suppression.
    """

    gt_by_frame: dict[int, list[LmotAnnotation]] = {}
    for row in gt:
        gt_by_frame.setdefault(row.frame_number, []).append(row)
    ordered_predictions = sorted(
        predictions, key=lambda row: row.confidence, reverse=True
    )
    if not gt:
        return {
            "precision": 0.0 if predictions else 1.0,
            "recall": 1.0,
            "AP50": 0.0 if predictions else 1.0,
            "AP50-95": 0.0 if predictions else 1.0,
            "ground_truth_boxes": 0,
            "predicted_boxes": len(predictions),
            "true_positives": 0,
            "false_positives": len(predictions),
            "false_negatives": 0,
        }

    aps: list[float] = []
    operating_tp: np.ndarray | None = None
    operating_fp: np.ndarray | None = None
    for threshold in iou_thresholds:
        used: dict[int, set[int]] = {}
        tp = np.zeros(len(ordered_predictions), dtype=np.float64)
        fp = np.zeros(len(ordered_predictions), dtype=np.float64)
        for index, prediction in enumerate(ordered_predictions):
            candidates = gt_by_frame.get(prediction.frame_number, [])
            if not candidates:
                fp[index] = 1
                continue
            overlaps = _iou_matrix(
                np.asarray([prediction.xyxy], dtype=np.float64),
                np.asarray(
                    [candidate.xyxy for candidate in candidates],
                    dtype=np.float64,
                ),
            )[0]
            used_in_frame = used.setdefault(
                prediction.frame_number, set()
            )
            available = [
                candidate_index
                for candidate_index in range(len(candidates))
                if candidate_index not in used_in_frame
            ]
            if not available:
                fp[index] = 1
                continue
            best = max(
                available,
                key=lambda candidate_index: float(
                    overlaps[candidate_index]
                ),
            )
            if float(overlaps[best]) >= threshold:
                tp[index] = 1
                used_in_frame.add(best)
            else:
                fp[index] = 1
        cumulative_tp = np.cumsum(tp)
        cumulative_fp = np.cumsum(fp)
        recalls = cumulative_tp / len(gt)
        precisions = cumulative_tp / np.maximum(
            cumulative_tp + cumulative_fp, 1e-12
        )
        recall_axis = np.linspace(0.0, 1.0, 101)
        interpolated = [
            float(precisions[recalls >= point].max())
            if np.any(recalls >= point)
            else 0.0
            for point in recall_axis
        ]
        aps.append(float(np.mean(interpolated)))
        if math.isclose(threshold, 0.5):
            operating_tp = tp
            operating_fp = fp
    if operating_tp is None or operating_fp is None:
        raise ValueError("iou_thresholds must include 0.5")
    tp_count = int(operating_tp.sum())
    fp_count = int(operating_fp.sum())
    fn_count = len(gt) - tp_count
    return {
        "precision": tp_count / max(tp_count + fp_count, 1.0),
        "recall": tp_count / len(gt),
        "AP50": aps[
            next(
                index
                for index, threshold in enumerate(iou_thresholds)
                if math.isclose(threshold, 0.5)
            )
        ],
        "AP50-95": float(np.mean(aps)),
        "ground_truth_boxes": len(gt),
        "predicted_boxes": len(predictions),
        "true_positives": tp_count,
        "false_positives": fp_count,
        "false_negatives": fn_count,
    }


class FrozenStageNTrackerAdapter:
    """Invoke a tracker through the complete Ultralytics ``model.track`` path."""

    def __init__(
        self,
        settings: StageNInferenceSettings,
        *,
        tracker_config: Path,
        model_factory: Any | None = None,
    ) -> None:
        tracker_config = tracker_config.resolve()
        if not tracker_config.is_file():
            raise FileNotFoundError(tracker_config)
        self.settings = settings
        self.tracker_config = tracker_config
        if model_factory is None:
            from ultralytics import YOLO

            model_factory = YOLO
        self.model = model_factory(settings.weights)
        self.frame_number = 0
        self.last_detections: tuple[TrackPrediction, ...] = ()

    def track(self, frame: np.ndarray) -> tuple[TrackPrediction, ...]:
        self.frame_number += 1
        results = self.model.track(
            source=frame,
            persist=True,
            tracker=str(self.tracker_config),
            conf=self.settings.confidence,
            iou=self.settings.nms_iou,
            imgsz=self.settings.image_size,
            agnostic_nms=self.settings.agnostic_nms,
            max_det=self.settings.max_detections,
            classes=[0],
            device=self.settings.device,
            augment=False,
            rect=False,
            half=False,
            verbose=False,
        )
        if not results:
            self.last_detections = ()
            return ()
        boxes = getattr(results[0], "boxes", None)
        if boxes is None or len(boxes) == 0:
            self.last_detections = ()
            return ()
        xyxy = boxes.xyxy.detach().cpu().numpy()
        confidences = boxes.conf.detach().cpu().numpy()
        ids_tensor = getattr(boxes, "id", None)
        self.last_detections = tuple(
            TrackPrediction(
                frame_number=self.frame_number,
                track_id=-(self.frame_number * 100000 + index + 1),
                xyxy=tuple(float(value) for value in bbox),
                confidence=float(confidence),
            )
            for index, (bbox, confidence) in enumerate(
                zip(xyxy, confidences)
            )
        )
        if ids_tensor is None:
            return ()
        ids = ids_tensor.detach().cpu().numpy()
        tracks = tuple(
            TrackPrediction(
                frame_number=self.frame_number,
                track_id=int(track_id),
                xyxy=tuple(float(value) for value in bbox),
                confidence=float(confidence),
            )
            for bbox, confidence, track_id in zip(xyxy, confidences, ids)
        )
        self.last_detections = tuple(
            TrackPrediction(
                frame_number=row.frame_number,
                track_id=row.track_id,
                xyxy=row.xyxy,
                confidence=row.confidence,
            )
            for row in tracks
        )
        return tracks


def _ensure_trackeval_numpy_compatibility() -> None:
    # Official commit 12c879... predates NumPy 2 and uses removed aliases.
    # Aliasing scalar types changes no metric logic and leaves vendor source
    # untouched.
    if not hasattr(np, "float"):
        np.float = float  # type: ignore[attr-defined]
    if not hasattr(np, "int"):
        np.int = int  # type: ignore[attr-defined]


def _metric_input(
    *,
    num_timesteps: int,
    gt: Sequence[LmotAnnotation],
    predictions: Sequence[TrackPrediction],
) -> dict[str, Any]:
    if num_timesteps <= 0:
        raise ValueError("num_timesteps must be positive")
    gt_ids_original = sorted({row.track_id for row in gt})
    pred_ids_original = sorted({row.track_id for row in predictions})
    gt_id_map = {track_id: index for index, track_id in enumerate(gt_ids_original)}
    pred_id_map = {
        track_id: index for index, track_id in enumerate(pred_ids_original)
    }
    gt_by_frame: dict[int, list[LmotAnnotation]] = {}
    pred_by_frame: dict[int, list[TrackPrediction]] = {}
    for row in gt:
        if not 1 <= row.frame_number <= num_timesteps:
            raise StageNDataGateError("GT frame lies outside sequence")
        gt_by_frame.setdefault(row.frame_number, []).append(row)
    for row in predictions:
        if not 1 <= row.frame_number <= num_timesteps:
            raise StageNDataGateError("Prediction frame lies outside sequence")
        pred_by_frame.setdefault(row.frame_number, []).append(row)

    gt_ids: list[np.ndarray] = []
    tracker_ids: list[np.ndarray] = []
    similarities: list[np.ndarray] = []
    for frame_number in range(1, num_timesteps + 1):
        gt_rows = gt_by_frame.get(frame_number, [])
        pred_rows = pred_by_frame.get(frame_number, [])
        gt_ids.append(
            np.asarray(
                [gt_id_map[row.track_id] for row in gt_rows],
                dtype=np.int64,
            )
        )
        tracker_ids.append(
            np.asarray(
                [pred_id_map[row.track_id] for row in pred_rows],
                dtype=np.int64,
            )
        )
        gt_boxes = np.asarray(
            [row.xyxy for row in gt_rows], dtype=np.float64
        ).reshape((-1, 4))
        pred_boxes = np.asarray(
            [row.xyxy for row in pred_rows], dtype=np.float64
        ).reshape((-1, 4))
        similarities.append(_iou_matrix(gt_boxes, pred_boxes))
    return {
        "num_timesteps": num_timesteps,
        "num_gt_ids": len(gt_id_map),
        "num_tracker_ids": len(pred_id_map),
        "num_gt_dets": len(gt),
        "num_tracker_dets": len(predictions),
        "gt_ids": gt_ids,
        "tracker_ids": tracker_ids,
        "similarity_scores": similarities,
    }


class OfficialTrackEvalAdapter:
    """Thin adapter over official HOTA, CLEAR, and Identity implementations."""

    EXPECTED_COMMIT = "12c8791b303e0a0b50f753af204249e622d0281a"

    def __init__(self, *, threshold: float = 0.5) -> None:
        _ensure_trackeval_numpy_compatibility()
        from trackeval.metrics import CLEAR, HOTA, Identity

        self.hota = HOTA()
        self.clear = CLEAR(
            {"THRESHOLD": threshold, "PRINT_CONFIG": False}
        )
        self.identity = Identity(
            {"THRESHOLD": threshold, "PRINT_CONFIG": False}
        )

    @staticmethod
    def runtime_metadata() -> dict[str, Any]:
        return {
            "package": "trackeval",
            "installed_version": importlib.metadata.version("trackeval"),
            "official_commit": OfficialTrackEvalAdapter.EXPECTED_COMMIT,
            "metrics": ["HOTA", "CLEAR", "Identity"],
            "numpy_2_compatibility_shim": ["np.float=float", "np.int=int"],
        }

    @staticmethod
    def _summary(
        hota_result: Mapping[str, Any],
        clear_result: Mapping[str, Any],
        identity_result: Mapping[str, Any],
    ) -> dict[str, float | int]:
        return {
            "HOTA": float(np.mean(hota_result["HOTA"]) * 100.0),
            "DetA": float(np.mean(hota_result["DetA"]) * 100.0),
            "AssA": float(np.mean(hota_result["AssA"]) * 100.0),
            "IDF1": float(identity_result["IDF1"] * 100.0),
            "ID_switches": int(clear_result["IDSW"]),
            "MOTA": float(clear_result["MOTA"] * 100.0),
            "true_positives": int(clear_result["CLR_TP"]),
            "false_positives": int(clear_result["CLR_FP"]),
            "false_negatives": int(clear_result["CLR_FN"]),
        }

    def _raw(
        self,
        *,
        num_timesteps: int,
        gt: Sequence[LmotAnnotation],
        predictions: Sequence[TrackPrediction],
    ) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
        data = _metric_input(
            num_timesteps=num_timesteps,
            gt=gt,
            predictions=predictions,
        )
        return (
            self.hota.eval_sequence(data),
            self.clear.eval_sequence(data),
            self.identity.eval_sequence(data),
        )

    def evaluate_sequence(
        self,
        *,
        num_timesteps: int,
        gt: Sequence[LmotAnnotation],
        predictions: Sequence[TrackPrediction],
    ) -> dict[str, float | int]:
        return self._summary(
            *self._raw(
                num_timesteps=num_timesteps,
                gt=gt,
                predictions=predictions,
            )
        )

    def evaluate_many(
        self,
        sequences: Mapping[
            str,
            tuple[int, Sequence[LmotAnnotation], Sequence[TrackPrediction]],
        ],
    ) -> tuple[
        dict[str, dict[str, float | int]], dict[str, float | int]
    ]:
        raw_hota: dict[str, dict[str, Any]] = {}
        raw_clear: dict[str, dict[str, Any]] = {}
        raw_identity: dict[str, dict[str, Any]] = {}
        summaries: dict[str, dict[str, float | int]] = {}
        for name, (num_timesteps, gt, predictions) in sequences.items():
            hota, clear, identity = self._raw(
                num_timesteps=num_timesteps,
                gt=gt,
                predictions=predictions,
            )
            raw_hota[name] = hota
            raw_clear[name] = clear
            raw_identity[name] = identity
            summaries[name] = self._summary(hota, clear, identity)
        aggregate = self._summary(
            self.hota.combine_sequences(raw_hota),
            self.clear.combine_sequences(raw_clear),
            self.identity.combine_sequences(raw_identity),
        )
        return summaries, aggregate
