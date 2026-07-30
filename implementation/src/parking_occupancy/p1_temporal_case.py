from __future__ import annotations

import csv
import json
import os
import statistics
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

from .detector_comparison import (
    ComparisonDetectorAdapter,
    DetectorSpec,
    sha256_file,
)
from .evaluate import binary_metrics
from .geometry import map_detections_to_slots
from .models import ParkingSlot
from .temporal import HysteresisConfig, TemporalOccupancyFilter
from .visualization import draw_frame


P1_TEMPORAL_PROTOCOL_ID = "P1-B1-VIRAT0502-CONTINUOUS-20260727-01"
P1_TEMPORAL_RECORD_ID = (
    "P1-B1-VIRAT0502-CONTINUOUS-RECORD-20260727-01"
)


class P1TemporalProtocolError(ValueError):
    """Raised when the P1 continuous case differs from its frozen protocol."""


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_p1_temporal_protocol(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != P1_TEMPORAL_PROTOCOL_ID:
        raise P1TemporalProtocolError("Unexpected P1 temporal protocol ID")
    if payload.get("status") != "frozen_before_predictions":
        raise P1TemporalProtocolError("P1 temporal protocol is not frozen")
    scope = payload.get("scope", {})
    if (
        scope.get("data_role") != "consumed_development_case_study"
        or scope.get("untouched_claim") != "prohibited"
        or scope.get("parameter_selection_from_this_run") != "prohibited"
    ):
        raise P1TemporalProtocolError("Invalid P1 temporal scope")
    raw = payload.get("raw_method", {})
    if (
        raw.get("detector_id") != "D1"
        or float(raw.get("confidence", -1)) != 0.30
        or int(raw.get("imgsz", -1)) != 640
        or raw.get("agnostic_nms") is not True
        or int(raw.get("max_detections", -1)) != 300
    ):
        raise P1TemporalProtocolError("Invalid frozen P1 detector settings")
    mapping = raw.get("mapping", {})
    if (
        mapping.get("algorithm") != "slot_polygon_coverage"
        or float(mapping.get("minimum_slot_coverage", -1)) != 0.40
        or mapping.get("one_to_one") is not True
    ):
        raise P1TemporalProtocolError("Invalid frozen B1 mapping")
    hysteresis = payload.get("hysteresis_sensitivity", {})
    if (
        hysteresis.get("enabled") is not True
        or "not_selected_on_VIRAT_or_Stage_K"
        not in str(hysteresis.get("parameter_source"))
    ):
        raise P1TemporalProtocolError(
            "Hysteresis must remain an uncalibrated sensitivity branch"
        )
    if payload.get("tracking_branch", {}).get("enabled") is not False:
        raise P1TemporalProtocolError("Tracking must remain disabled")
    truth_binding = payload["truth"]
    truth_path = _resolve_from_config(config_path, str(truth_binding["path"]))
    if (
        not truth_path.is_file()
        or truth_path.stat().st_size != int(truth_binding["bytes"])
        or sha256_file(truth_path) != str(truth_binding["sha256"])
    ):
        raise P1TemporalProtocolError("P1 temporal truth binding mismatch")
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    if (
        truth["source_sha256"] != payload["source"]["sha256"]
        or truth["video"]["frame_count"]
        != payload["source"]["video"]["frame_count"]
        or truth["slots"][0]["polygon"] != truth_binding["polygon"]
        or truth["slots"][0]["intervals"] != truth_binding["intervals"]
    ):
        raise P1TemporalProtocolError("P1 temporal truth content mismatch")
    return payload


def _truth_state(protocol: dict[str, Any], frame_index: int) -> int:
    for interval in protocol["truth"]["intervals"]:
        if (
            int(interval["start_frame"])
            <= frame_index
            < int(interval["end_frame"])
        ):
            return int(interval["state"] == "occupied")
    raise P1TemporalProtocolError(
        f"Truth does not cover frame {frame_index}"
    )


def p1_temporal_preflight(
    *,
    config_path: Path,
    video_path: Path,
    weights_path: Path,
) -> dict[str, Any]:
    protocol = load_p1_temporal_protocol(config_path)
    video_path = video_path.resolve()
    weights_path = weights_path.resolve()
    if (
        not video_path.is_file()
        or video_path.stat().st_size != int(protocol["source"]["bytes"])
        or sha256_file(video_path) != str(protocol["source"]["sha256"])
    ):
        raise P1TemporalProtocolError("P1 temporal video binding mismatch")
    raw = protocol["raw_method"]
    if (
        not weights_path.is_file()
        or sha256_file(weights_path) != str(raw["weights_sha256"])
    ):
        raise P1TemporalProtocolError("P1 temporal weights mismatch")
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise P1TemporalProtocolError("Could not open P1 temporal video")
    try:
        metadata = {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()
    expected = protocol["source"]["video"]
    if (
        metadata["frame_count"] != int(expected["frame_count"])
        or metadata["width"] != int(expected["width"])
        or metadata["height"] != int(expected["height"])
        or abs(metadata["fps"] - float(expected["fps"])) > 1e-9
    ):
        raise P1TemporalProtocolError("P1 temporal video metadata mismatch")
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "data_role": protocol["scope"]["data_role"],
        "video": {
            **metadata,
            "bytes": video_path.stat().st_size,
            "sha256": sha256_file(video_path),
            "verified": True,
        },
        "truth": {
            "slot_id": protocol["truth"]["slot_id"],
            "transition_frame": protocol["truth"]["transition"][
                "first_vacant_frame"
            ],
            "verified": True,
        },
        "weights": {
            "name": weights_path.name,
            "sha256": sha256_file(weights_path),
            "verified": True,
        },
        "raw_method": raw["method_id"],
        "hysteresis_method": protocol["hysteresis_sensitivity"]["method_id"],
        "tracking_enabled": False,
        "predictions_run": False,
        "parameters_selected": False,
        "execution_gate": "open",
    }


def _timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
        "fps_from_mean": 1000.0 / statistics.fmean(ordered),
    }


def _sequence_metrics(
    truth: list[int],
    prediction: list[int],
    *,
    fps: float,
    stable_frames: int,
    tolerance_frames: int,
) -> dict[str, Any]:
    try:
        from literature_core.metrics import sequence_temporal_metrics
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "literature_core/src must be importable to compute the canonical "
            "signed transition metrics"
        ) from error
    return sequence_temporal_metrics(
        truth,
        prediction,
        fps,
        stable_frames=stable_frames,
        tolerance_frames=tolerance_frames,
    )


def run_p1_temporal_case(
    *,
    config_path: Path,
    video_path: Path,
    weights_path: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite P1 temporal output: {output_root}"
        )
    preflight = p1_temporal_preflight(
        config_path=config_path,
        video_path=video_path,
        weights_path=weights_path,
    )
    protocol = load_p1_temporal_protocol(config_path)
    output_root.mkdir(parents=True)
    ultralytics_config = output_root / "_ultralytics_config"
    ultralytics_config.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config.resolve())
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )

    raw = protocol["raw_method"]
    spec = DetectorSpec(
        method_id="D1",
        name=str(raw["detector_name"]),
        backend=str(raw["backend"]),
        status="ready",
        weights_name=str(raw["weights_name"]),
        weights_sha256=str(raw["weights_sha256"]),
        source_class_ids=tuple(int(x) for x in raw["source_class_ids"]),
        source_class_names=tuple(str(x) for x in raw["source_class_names"]),
        prompts=(),
        project_class_id=0,
        project_class_name="vehicle",
    )
    adapter = ComparisonDetectorAdapter(
        spec=spec,
        weights_path=weights_path,
        common={
            "confidence_floor": float(raw["confidence"]),
            "nms_iou": float(raw["nms_iou"]),
            "imgsz": int(raw["imgsz"]),
            "max_detections": int(raw["max_detections"]),
            "agnostic_nms": bool(raw["agnostic_nms"]),
            "augmentation": bool(raw["augmentation"]),
        },
        device=device,
    )
    slot = ParkingSlot(
        slot_id=str(protocol["truth"]["slot_id"]),
        points=tuple(
            (float(x), float(y)) for x, y in protocol["truth"]["polygon"]
        ),
    )
    hysteresis = protocol["hysteresis_sensitivity"]
    temporal_filter = TemporalOccupancyFilter(
        [slot.slot_id],
        HysteresisConfig(
            rise_alpha=float(hysteresis["rise_alpha"]),
            fall_alpha=float(hysteresis["fall_alpha"]),
            occupied_threshold=float(hysteresis["occupied_threshold"]),
            vacant_threshold=float(hysteresis["vacant_threshold"]),
        ),
    )

    video_path = video_path.resolve()
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = cv2.VideoWriter(
        str(output_root / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create P1 temporal annotated video")

    occupancy_path = output_root / "occupancy.csv"
    events_path = output_root / "events.csv"
    detections_path = output_root / "detections.jsonl"
    rows = []
    event_rows = []
    truth_states: list[int] = []
    raw_states: list[int] = []
    filtered_states: list[int] = []
    frame_times = []
    detector_times = []
    mapping_times = []
    previous = {"truth": None, "raw": None, "hysteresis": None}
    run_start = time.perf_counter()
    try:
        with detections_path.open("x", encoding="utf-8") as detection_handle:
            for frame_index in range(frame_count):
                frame_start = time.perf_counter()
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(f"Could not decode frame {frame_index}")
                detector_start = time.perf_counter()
                detections = adapter.detect(frame)
                detector_end = time.perf_counter()
                evidence_by_slot = map_detections_to_slots(
                    detections=detections,
                    slots=[slot],
                    mode="overlap",
                    overlap_threshold=float(
                        raw["mapping"]["minimum_slot_coverage"]
                    ),
                )
                mapping_end = time.perf_counter()
                filtered = temporal_filter.update(evidence_by_slot)[slot.slot_id]
                evidence = evidence_by_slot[slot.slot_id]
                truth = _truth_state(protocol, frame_index)
                raw_state = int(evidence.occupied)
                filtered_state = int(filtered.occupied)
                truth_states.append(truth)
                raw_states.append(raw_state)
                filtered_states.append(filtered_state)
                row = {
                    "video_id": protocol["source"]["source_video_id"],
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / fps,
                    "slot_id": slot.slot_id,
                    "truth": truth,
                    "raw_state": raw_state,
                    "hysteresis_state": filtered_state,
                    "evidence": evidence.evidence_score,
                    "filtered_score": filtered.filtered_score,
                    "geometric_score": evidence.geometric_score,
                    "detection_confidence": evidence.detection_confidence,
                    "detection_count": len(detections),
                }
                rows.append(row)
                current = {
                    "truth": truth,
                    "raw": raw_state,
                    "hysteresis": filtered_state,
                }
                for method, state in current.items():
                    if (
                        previous[method] is not None
                        and state != previous[method]
                    ):
                        event_rows.append(
                            {
                                "video_id": protocol["source"][
                                    "source_video_id"
                                ],
                                "frame_index": frame_index,
                                "timestamp_s": frame_index / fps,
                                "slot_id": slot.slot_id,
                                "method": method,
                                "from_state": previous[method],
                                "to_state": state,
                                "evidence": evidence.evidence_score,
                                "filtered_score": filtered.filtered_score,
                            }
                        )
                    previous[method] = state
                detection_handle.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "detections": [
                                {
                                    "bbox_xyxy": list(detection.bbox),
                                    "confidence": detection.confidence,
                                    "class_id": detection.class_id,
                                    "class_name": detection.class_name,
                                }
                                for detection in detections
                            ],
                        }
                    )
                    + "\n"
                )
                annotated = draw_frame(
                    frame=frame,
                    detections=detections,
                    slots=[slot],
                    states={slot.slot_id: filtered},
                    experiment="p1_b1_hysteresis_sensitivity",
                    processing_fps=1.0
                    / max(time.perf_counter() - frame_start, 1e-9),
                )
                cv2.rectangle(
                    annotated,
                    (0, 0),
                    (width, 40),
                    (0, 0, 0),
                    -1,
                )
                cv2.putText(
                    annotated,
                    (
                        f"P1+B1 VIRAT 0502 DEVELOPMENT | frame "
                        f"{frame_index:04d} | truth={truth} raw={raw_state} "
                        f"hyst={filtered_state} evidence={evidence.evidence_score:.3f}"
                    ),
                    (10, 27),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.72,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                frame_end = time.perf_counter()
                detector_times.append((detector_end - detector_start) * 1000)
                mapping_times.append((mapping_end - detector_end) * 1000)
                frame_times.append((frame_end - frame_start) * 1000)
    finally:
        capture.release()
        writer.release()

    with occupancy_path.open("x", encoding="utf-8", newline="") as handle:
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        writer_csv.writeheader()
        writer_csv.writerows(rows)
    with events_path.open("x", encoding="utf-8", newline="") as handle:
        event_fields = [
            "video_id",
            "frame_index",
            "timestamp_s",
            "slot_id",
            "method",
            "from_state",
            "to_state",
            "evidence",
            "filtered_score",
        ]
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=event_fields,
            lineterminator="\n",
        )
        writer_csv.writeheader()
        writer_csv.writerows(event_rows)

    evaluation = protocol["evaluation"]
    warmup = int(evaluation["classification_warmup_frames"])
    metric_slices = {
        "raw": raw_states[warmup:],
        "hysteresis": filtered_states[warmup:],
    }
    truth_for_classification = truth_states[warmup:]
    methods = {}
    for method, prediction in metric_slices.items():
        methods[method] = {
            **binary_metrics(truth_for_classification, prediction),
            "temporal": _sequence_metrics(
                truth_states,
                raw_states if method == "raw" else filtered_states,
                fps=fps,
                stable_frames=int(evaluation["stable_frames"]),
                tolerance_frames=int(
                    evaluation["transition_tolerance_frames"]
                ),
            ),
        }
    runtime = {
        "frames": frame_count,
        "elapsed_s": time.perf_counter() - run_start,
        "classification_warmup_frames": warmup,
        "steady_state": {
            "frame": _timing(frame_times[warmup:]),
            "detector": _timing(detector_times[warmup:]),
            "mapping": _timing(mapping_times[warmup:]),
        },
        "detector": adapter.model_metadata(),
    }
    metrics = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "data_role": protocol["scope"]["data_role"],
        "frames": frame_count,
        "post_warmup_classification_frames": frame_count - warmup,
        "slots": 1,
        "methods": methods,
        "tracking_branch_run": False,
        "parameters_selected_from_result": False,
        "claim_boundary": (
            "One consumed-development VIRAT slot with project-created "
            "departure truth; not an untouched benchmark."
        ),
    }
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "executed_frozen_continuous_case",
        "source_video_id": protocol["source"]["source_video_id"],
        "source_sha256": protocol["source"]["sha256"],
        "real_continuous_video": True,
        "static_montage": False,
        "methods": {
            "raw": raw["method_id"],
            "hysteresis": hysteresis["method_id"],
            "tracking": "not_run",
        },
        "metrics": methods,
        "runtime": runtime,
        "outputs": [
            "annotated.mp4",
            "occupancy.csv",
            "events.csv",
            "detections.jsonl",
            "metrics.json",
            "summary.json",
            "runtime_metadata.json",
        ],
        "negative_results_retained": True,
    }
    (output_root / "metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )
    adapter.release()
    return summary


def verify_p1_temporal_record(
    *,
    record_path: Path,
    source_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    if record.get("record_id") != P1_TEMPORAL_RECORD_ID:
        raise P1TemporalProtocolError("Unexpected P1 temporal record ID")
    roots = {
        "source": source_root.resolve(),
        "external": external_root.resolve(),
    }
    checks = []
    for artifact in record["artifacts"]:
        root_name = str(artifact["root"])
        if root_name not in roots:
            raise P1TemporalProtocolError("Unexpected artifact root")
        path = roots[root_name] / str(artifact["path"])
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha256 = sha256_file(path) if path.is_file() else None
        passed = (
            actual_bytes == int(artifact["bytes"])
            and actual_sha256 == str(artifact["sha256"])
        )
        checks.append(
            {
                "role": artifact["role"],
                "path": artifact["path"],
                "expected_bytes": artifact["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": artifact["sha256"],
                "actual_sha256": actual_sha256,
                "passed": passed,
            }
        )
    return {
        "schema_version": 1,
        "record_id": record["record_id"],
        "artifact_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
