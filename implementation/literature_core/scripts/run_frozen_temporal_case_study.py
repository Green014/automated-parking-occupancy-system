"""Run frozen E4/E5 case studies on the verified VIRAT partitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import statistics
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from literature_core.calibration import (  # noqa: E402
    CalibratedFusionModel,
    NonnegativeLogisticFusion,
    PlattCalibrator,
)
from literature_core.classifier import MobileNetSlotClassifier  # noqa: E402
from literature_core.detector import (  # noqa: E402
    ClosedSetYOLODetector,
    YOLOWorldDetector,
)
from literature_core.mapping import map_detections_to_slots  # noqa: E402
from literature_core.metrics import binary_metrics, sequence_temporal_metrics  # noqa: E402
from literature_core.models import FusedEvidence, ParkingSlot  # noqa: E402
from literature_core.patches import extract_slot_patch  # noqa: E402
from literature_core.temporal import TemporalConfig, TemporalFusionFilter  # noqa: E402
from literature_core.temporal_protocol import validate_temporal_protocol  # noqa: E402
from literature_core.temporal_tracking import (  # noqa: E402
    TrackSlotEvidence,
    TrackSlotGate,
    TrackSlotGateConfig,
)


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a mapping")
    return payload


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (ROOT / path).resolve()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_hash(path: Path, expected: str, label: str) -> None:
    actual = _sha256(path)
    if actual != expected.lower():
        raise ValueError(f"{label} SHA-256 mismatch: {actual}")


def _fusion_model(payload: dict[str, Any]) -> CalibratedFusionModel:
    return CalibratedFusionModel(
        PlattCalibrator.from_dict(payload["calibration"]["classifier"]),
        PlattCalibrator.from_dict(payload["calibration"]["detector"]),
        NonnegativeLogisticFusion.from_dict(payload["fusion"]),
    )


def _truth_state(truth: dict[str, Any], slot_id: str, frame: int) -> int:
    slot = next(item for item in truth["slots"] if item["slot_id"] == slot_id)
    for interval in slot["intervals"]:
        if interval["start_frame"] <= frame < interval["end_frame"]:
            return int(interval["state"] == "occupied")
    raise ValueError(f"truth does not cover {slot_id} frame {frame}")


def _mean_timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
        "fps_from_mean": 1000.0 / statistics.fmean(ordered),
    }


def _method_report(
    rows: list[dict[str, Any]],
    method: str,
    fps: float,
) -> dict[str, Any]:
    truth = [int(row["truth"]) for row in rows]
    prediction = [int(row[method]) for row in rows]
    return {
        **binary_metrics(truth, prediction),
        "temporal": sequence_temporal_metrics(
            truth,
            prediction,
            fps,
            stable_frames=3,
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "temporal_e4_e5_frozen.yaml",
    )
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "temporal_protocol_pending.yaml",
    )
    parser.add_argument(
        "--partition",
        choices=("development", "holdout"),
        required=True,
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            "output directory already exists; choose a new path to preserve "
            "all prior results"
        )
    config = _load_yaml(args.config)
    if config.get("status") != "frozen_before_temporal_prediction":
        raise ValueError("temporal case-study configuration is not frozen")
    protocol = _load_yaml(args.protocol)
    validation = validate_temporal_protocol(protocol, project_root=ROOT)
    if not validation["ready_for_experiment"]:
        raise ValueError(f"temporal protocol is not ready: {validation['errors']}")

    selection = protocol["selections"][args.partition]
    configured_partition = config["partitions"][args.partition]
    if selection["scene_id"] != configured_partition["scene_id"]:
        raise ValueError("partition scene does not match the frozen config")
    if (
        selection["source_sha256"].lower()
        != configured_partition["source_sha256"].lower()
    ):
        raise ValueError("partition source hash does not match the frozen config")

    source = _resolve(selection["source_path"])
    truth_path = _resolve(selection["truth_path"])
    truth = _load_yaml(truth_path)
    _verify_hash(source, selection["source_sha256"], "source video")
    if truth["source_sha256"].lower() != selection["source_sha256"].lower():
        raise ValueError("truth/source SHA-256 mismatch")

    e3b = config["e3b"]
    e5 = config["e5"]
    classifier_path = _resolve(e3b["classifier_checkpoint"])
    world_path = _resolve(e3b["world_weights"])
    fusion_path = _resolve(e3b["calibration_config"])
    baseline_path = _resolve(e5["detector_weights"])
    tracker_path = _resolve(e5["tracker_config"])
    for path, digest, label in (
        (classifier_path, e3b["classifier_checkpoint_sha256"], "classifier"),
        (world_path, e3b["world_weights_sha256"], "YOLO-World"),
        (fusion_path, e3b["calibration_config_sha256"], "fusion config"),
        (baseline_path, e5["detector_weights_sha256"], "YOLOv8"),
        (tracker_path, e5["tracker_config_sha256"], "ByteTrack config"),
    ):
        _verify_hash(path, str(digest), label)

    slots = tuple(
        ParkingSlot(
            str(item["slot_id"]),
            tuple((float(x), float(y)) for x, y in item["polygon"]),
        )
        for item in truth["slots"]
    )
    slot_ids = tuple(slot.slot_id for slot in slots)
    classifier = MobileNetSlotClassifier(classifier_path, device=args.device)
    world = YOLOWorldDetector(
        world_path,
        prompts=e3b["prompts"],
        confidence=float(e3b["detector_confidence"]),
        image_size=int(e3b["detector_image_size"]),
        device=args.device,
    )
    baseline = ClosedSetYOLODetector(
        baseline_path,
        confidence=float(e5["detector_confidence"]),
        image_size=int(e5["detector_image_size"]),
        device=args.device,
        use_tracking=True,
        tracker_config=tracker_path,
    )
    fusion = _fusion_model(_load_yaml(fusion_path))
    e4_config = config["e4"]
    e4_filter = TemporalFusionFilter(
        slot_ids,
        TemporalConfig(
            rise_alpha=float(e4_config["rise_alpha"]),
            fall_alpha=float(e4_config["fall_alpha"]),
            occupied_threshold=float(e4_config["occupied_threshold"]),
            vacant_threshold=float(e4_config["vacant_threshold"]),
            raw_threshold=float(e3b["raw_occupied_threshold"]),
        ),
    )
    e5_gate = TrackSlotGate(
        slot_ids,
        TrackSlotGateConfig(
            minimum_coverage=float(e5["minimum_slot_coverage"]),
            maximum_stationary_displacement_px=float(
                e5["maximum_stationary_displacement_px"]
            ),
            occupied_dwell_frames=int(e5["occupied_dwell_frames"]),
            vacant_dwell_frames=int(e5["vacant_dwell_frames"]),
        ),
    )

    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"could not open {source}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count != truth["video"]["frame_count"] or fps <= 0.0:
        capture.release()
        raise ValueError("decoded video metadata does not match truth")

    args.output_dir.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    previous_states = {
        method: {slot_id: False for slot_id in slot_ids}
        for method in ("e0_raw", "e3b_raw", "e4", "e5")
    }
    track_centers: dict[int, tuple[float, float]] = {}
    timing = {"e4": [], "e5": [], "combined": []}
    start_run = time.perf_counter()
    try:
        for frame_index in range(frame_count):
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"could not decode frame {frame_index}")
            frame_start = time.perf_counter()

            e4_start = time.perf_counter()
            patches = [
                extract_slot_patch(frame, slot.points)
                for slot in slots
            ]
            p_cls = classifier.predict_patches(patches)
            world_detections = world.detect(frame)
            world_evidence = map_detections_to_slots(
                world_detections,
                slots,
                minimum_slot_coverage=float(e3b["minimum_slot_coverage"]),
                one_to_one=True,
            )
            p_det = [item.probability for item in world_evidence]
            p_cls_cal, p_det_cal = fusion.predict_branches(p_cls, p_det)
            p_e3b = fusion.fusion.predict(p_cls_cal, p_det_cal)
            e4_states = [
                e4_filter.update(
                    FusedEvidence(
                        slot_id=slot_id,
                        p_cls=cls_score,
                        p_det=det_score,
                        p_track=None,
                        probability=fused_score,
                        effective_weights=(0.0, 0.0, 0.0),
                    )
                )
                for slot_id, cls_score, det_score, fused_score in zip(
                    slot_ids,
                    p_cls,
                    p_det,
                    p_e3b,
                    strict=True,
                )
            ]
            e4_end = time.perf_counter()

            e5_start = e4_end
            baseline_detections = baseline.detect(frame)
            displacement_by_track: dict[int, float] = {}
            for detection in baseline_detections:
                if detection.track_id is None:
                    continue
                x1, y1, x2, y2 = detection.bbox
                center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                previous_center = track_centers.get(detection.track_id)
                displacement_by_track[detection.track_id] = (
                    0.0
                    if previous_center is None
                    else math.dist(previous_center, center)
                )
                track_centers[detection.track_id] = center
            baseline_evidence = map_detections_to_slots(
                baseline_detections,
                slots,
                minimum_slot_coverage=float(e5["minimum_slot_coverage"]),
                one_to_one=True,
            )
            track_evidence = tuple(
                TrackSlotEvidence(
                    slot_id=item.slot_id,
                    track_id=str(item.track_id),
                    coverage=item.coverage,
                    center_displacement_px=displacement_by_track[item.track_id],
                )
                for item in baseline_evidence
                if item.track_id is not None
                and item.track_id in displacement_by_track
            )
            e5_states = e5_gate.update(track_evidence)
            e5_end = time.perf_counter()

            timing["e4"].append((e4_end - e4_start) * 1000.0)
            timing["e5"].append((e5_end - e5_start) * 1000.0)
            timing["combined"].append((e5_end - frame_start) * 1000.0)
            for (
                slot,
                cls_score,
                det_score,
                cls_cal,
                det_cal,
                fused_score,
                world_item,
                baseline_item,
                e4_state,
                e5_state,
            ) in zip(
                slots,
                p_cls,
                p_det,
                p_cls_cal,
                p_det_cal,
                p_e3b,
                world_evidence,
                baseline_evidence,
                e4_states,
                e5_states,
                strict=True,
            ):
                truth_state = _truth_state(truth, slot.slot_id, frame_index)
                method_states = {
                    "e0_raw": int(baseline_item.probability > 0.0),
                    "e3b_raw": int(
                        fused_score >= float(e3b["raw_occupied_threshold"])
                    ),
                    "e4": int(e4_state.occupied),
                    "e5": int(e5_state.occupied),
                }
                row = {
                    "video_id": selection["source_video_id"],
                    "partition": args.partition,
                    "scene_id": selection["scene_id"],
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / fps,
                    "slot_id": slot.slot_id,
                    "truth": truth_state,
                    "p_cls": cls_score,
                    "p_det_evidence": det_score,
                    "p_cls_calibrated": cls_cal,
                    "p_det_calibrated": det_cal,
                    "p_e3b": fused_score,
                    "e0_raw": method_states["e0_raw"],
                    "e3b_raw": method_states["e3b_raw"],
                    "e4": method_states["e4"],
                    "e5": method_states["e5"],
                    "world_coverage": world_item.coverage,
                    "e0_coverage": baseline_item.coverage,
                    "e0_track_id": baseline_item.track_id,
                    "e5_assigned_track_id": e5_state.assigned_track_id,
                    "e5_confirmed_track_id": e5_state.confirmed_track_id,
                    "e5_candidate_dwell_frames": (
                        e5_state.candidate_dwell_frames
                    ),
                    "e5_clear_dwell_frames": e5_state.clear_dwell_frames,
                    "e5_suppressed_moving_track_ids": "|".join(
                        e5_state.suppressed_moving_track_ids
                    ),
                }
                rows.append(row)
                for method, state in method_states.items():
                    previous = previous_states[method][slot.slot_id]
                    if state != previous:
                        event_rows.append(
                            {
                                "video_id": selection["source_video_id"],
                                "partition": args.partition,
                                "frame_index": frame_index,
                                "timestamp_s": frame_index / fps,
                                "slot_id": slot.slot_id,
                                "method": method,
                                "from_state": previous,
                                "to_state": state,
                            }
                        )
                    previous_states[method][slot.slot_id] = bool(state)
    finally:
        capture.release()

    warmup = int(config["evaluation"]["warmup_frames"])
    if frame_count - warmup < 100:
        raise ValueError("fewer than 100 post-warm-up frames")
    evaluated = [row for row in rows if row["frame_index"] >= warmup]
    methods = {
        method: _method_report(evaluated, method, fps)
        for method in ("e0_raw", "e3b_raw", "e4", "e5")
    }
    steady_timing = {
        method: _mean_timing(values[warmup:])
        for method, values in timing.items()
    }
    summary = {
        "schema_version": 1,
        "status": "executed_frozen_case_study",
        "partition": args.partition,
        "scope": config["evaluation"]["reporting_scope"],
        "source_video_id": selection["source_video_id"],
        "source_sha256": selection["source_sha256"],
        "truth_path": str(truth_path),
        "frames": frame_count,
        "post_warmup_frames": frame_count - warmup,
        "slots": len(slots),
        "fps": fps,
        "elapsed_s": time.perf_counter() - start_run,
        "methods": methods,
        "steady_state_timing": steady_timing,
        "confidence_interval": (
            config["evaluation"]["bootstrap_confidence_interval"]
        ),
        "configuration": config,
        "runtime": {
            "classifier": classifier.metadata(),
            "yolo_world": world.metadata(),
            "yolov8_bytetrack": baseline.metadata(),
        },
        "protocol_validation": validation,
        "output_files": ["predictions.csv", "events.csv", "summary.json"],
    }

    with (args.output_dir / "predictions.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (args.output_dir / "events.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(event_rows[0]))
        writer.writeheader()
        writer.writerows(event_rows)
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "partition": args.partition,
                "frames": frame_count,
                "methods": {
                    method: {
                        key: report[key]
                        for key in (
                            "macro_f1",
                            "occupied_recall",
                            "vacant_recall",
                        )
                    }
                    for method, report in methods.items()
                },
                "steady_state_timing": steady_timing,
                "output_dir": str(args.output_dir.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
