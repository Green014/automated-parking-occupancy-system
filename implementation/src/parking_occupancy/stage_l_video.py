from __future__ import annotations

import csv
import json
import math
import statistics
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import yaml

from .detector import UltralyticsDetector
from .detector_comparison import sha256_file
from .geometry import map_detections_to_slots
from .integrated_workflow import (
    UncertaintyGateConfig,
    apply_track_override,
    uncertainty_gated_fusion,
)
from .models import ParkingSlot
from .stage_l_integrated import (
    STAGE_L_PROTOCOL_ID,
    StageLProtocolError,
    _verify_file,
    load_stage_l_protocol,
)
from .temporal import FilteredSlotState
from .visualization import draw_frame


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _truth_state(truth: dict[str, Any], slot_id: str, frame: int) -> int:
    slot = next(item for item in truth["slots"] if item["slot_id"] == slot_id)
    for interval in slot["intervals"]:
        if int(interval["start_frame"]) <= frame < int(interval["end_frame"]):
            return int(interval["state"] == "occupied")
    raise StageLProtocolError(
        f"Truth does not cover slot {slot_id} at frame {frame}"
    )


def stage_l_video_preflight(
    *,
    config_path: Path,
    video_path: Path,
    detector_weights: Path,
    classifier_checkpoint: Path,
    tracker_config: Path,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config_path = config_path.resolve()
    video_path = video_path.resolve()
    detector_weights = detector_weights.resolve()
    classifier_checkpoint = classifier_checkpoint.resolve()
    tracker_config = tracker_config.resolve()
    protocol = load_stage_l_protocol(config_path)
    continuous = protocol["continuous_partition"]
    truth_path = _resolve_from_config(
        config_path,
        str(continuous["truth"]["path"]),
    )
    _verify_file(
        video_path,
        {
            "bytes": continuous["source_bytes"],
            "sha256": continuous["source_sha256"],
        },
        "VIRAT video",
    )
    _verify_file(truth_path, continuous["truth"], "VIRAT occupancy truth")
    for path, expected_hash, label in (
        (
            detector_weights,
            protocol["models"]["D1"]["sha256"],
            "D1 weights",
        ),
        (
            classifier_checkpoint,
            protocol["models"]["E1b"]["sha256"],
            "E1b checkpoint",
        ),
        (
            tracker_config,
            protocol["tracking"]["tracker_config_sha256"],
            "ByteTrack configuration",
        ),
    ):
        if not path.is_file():
            raise StageLProtocolError(f"Missing {label}: {path}")
        if sha256_file(path) != str(expected_hash):
            raise StageLProtocolError(f"{label} SHA-256 mismatch")
    truth = yaml.safe_load(truth_path.read_text(encoding="utf-8"))
    if truth.get("source_sha256") != continuous["source_sha256"]:
        raise StageLProtocolError("Truth/video SHA-256 binding mismatch")
    if len(truth.get("slots", [])) != int(continuous["expected"]["slots"]):
        raise StageLProtocolError("Unexpected continuous slot count")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise StageLProtocolError(f"Could not open video: {video_path}")
    metadata = {
        "frames": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
    }
    capture.release()
    if metadata["frames"] != int(continuous["expected"]["frames"]):
        raise StageLProtocolError("Continuous video frame count mismatch")
    if metadata["fps"] <= 0.0:
        raise StageLProtocolError("Continuous video FPS is invalid")
    report = {
        "schema_version": 1,
        "protocol_id": STAGE_L_PROTOCOL_ID,
        "data_role": continuous["data_role"],
        "source_video_id": continuous["source_video_id"],
        **metadata,
        "slots": len(truth["slots"]),
        "D1_weights_verified": True,
        "E1b_checkpoint_verified": True,
        "ByteTrack_config_verified": True,
        "parameters_selected_from_video": False,
        "execution_gate": "open",
    }
    return report, truth


def _timing(values: list[float], warmup: int) -> dict[str, float]:
    steady = sorted(values[warmup:])
    return {
        "mean_ms": statistics.fmean(steady),
        "p50_ms": statistics.median(steady),
        "p95_ms": steady[round(0.95 * (len(steady) - 1))],
        "fps_from_mean": 1000.0 / statistics.fmean(steady),
    }


def _method_metrics(
    rows: list[dict[str, Any]],
    method: str,
    fps: float,
    stable_frames: int,
    frame_offset: int = 0,
) -> dict[str, Any]:
    from literature_core.metrics import binary_metrics, sequence_temporal_metrics

    truth = [int(row["truth"]) for row in rows]
    predictions = [int(row[method]) for row in rows]
    temporal = sequence_temporal_metrics(
        truth,
        predictions,
        fps,
        stable_frames=stable_frames,
    )
    temporal["evaluation_frame_offset"] = frame_offset
    for event in temporal["transition_events"]:
        event["truth_transition_frame_absolute"] = (
            int(event["truth_transition_frame"]) + frame_offset
        )
        predicted = event["predicted_transition_frame"]
        event["predicted_transition_frame_absolute"] = (
            None if predicted is None else int(predicted) + frame_offset
        )
    return {
        **binary_metrics(truth, predictions),
        "temporal": temporal,
    }


def run_stage_l_video(
    *,
    config_path: Path,
    video_path: Path,
    detector_weights: Path,
    classifier_checkpoint: Path,
    tracker_config: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    """Run the full P3 detector, classifier, temporal, and tracking workflow."""

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Stage L output: {output_root}")
    preflight, truth = stage_l_video_preflight(
        config_path=config_path,
        video_path=video_path,
        detector_weights=detector_weights,
        classifier_checkpoint=classifier_checkpoint,
        tracker_config=tracker_config,
    )
    protocol = load_stage_l_protocol(config_path)
    from literature_core.classifier import MobileNetSlotClassifier
    from literature_core.models import FusedEvidence
    from literature_core.patches import extract_slot_patch
    from literature_core.temporal import TemporalConfig, TemporalFusionFilter
    from literature_core.temporal_tracking import (
        TrackSlotEvidence,
        TrackSlotGate,
        TrackSlotGateConfig,
    )

    continuous = protocol["continuous_partition"]
    d1 = protocol["models"]["D1"]
    classifier_config = protocol["models"]["E1b"]
    detector = UltralyticsDetector(
        weights=str(detector_weights.resolve()),
        confidence=float(d1["confidence"]),
        image_size=int(d1["imgsz"]),
        device=device,
        vehicle_class_ids=tuple(int(value) for value in d1["source_class_ids"]),
        use_tracking=True,
        tracker_config=str(tracker_config.resolve()),
        nms_iou=float(d1["nms_iou"]),
        agnostic_nms=bool(d1["agnostic_nms"]),
        max_detections=int(d1["max_detections"]),
        augmentation=bool(d1["augmentation"]),
        rect=bool(d1["rect"]),
        half=bool(d1["half"]),
    )
    classifier = MobileNetSlotClassifier(
        classifier_checkpoint,
        device=device,
    )
    slots = tuple(
        ParkingSlot(
            slot_id=str(item["slot_id"]),
            points=tuple(
                (float(x), float(y))
                for x, y in item["polygon"]
            ),
        )
        for item in truth["slots"]
    )
    slot_ids = tuple(slot.slot_id for slot in slots)
    gate_config = UncertaintyGateConfig(
        classifier_occupied_threshold=float(
            classifier_config["occupied_threshold"]
        )
    )
    temporal_config = protocol["temporal"]
    temporal_gate = TemporalFusionFilter(
        slot_ids,
        TemporalConfig(
            rise_alpha=float(temporal_config["rise_alpha"]),
            fall_alpha=float(temporal_config["fall_alpha"]),
            occupied_threshold=float(temporal_config["occupied_threshold"]),
            vacant_threshold=float(temporal_config["vacant_threshold"]),
            raw_threshold=float(temporal_config["raw_threshold"]),
        ),
    )
    temporal_full = TemporalFusionFilter(
        slot_ids,
        TemporalConfig(
            rise_alpha=float(temporal_config["rise_alpha"]),
            fall_alpha=float(temporal_config["fall_alpha"]),
            occupied_threshold=float(temporal_config["occupied_threshold"]),
            vacant_threshold=float(temporal_config["vacant_threshold"]),
            raw_threshold=float(temporal_config["raw_threshold"]),
        ),
    )
    tracking_config = protocol["tracking"]
    track_gate = TrackSlotGate(
        slot_ids,
        TrackSlotGateConfig(
            minimum_coverage=float(
                tracking_config["minimum_slot_coverage"]
            ),
            maximum_stationary_displacement_px=float(
                tracking_config["maximum_stationary_displacement_px"]
            ),
            occupied_dwell_frames=int(
                tracking_config["occupied_dwell_frames"]
            ),
            vacant_dwell_frames=int(
                tracking_config["vacant_dwell_frames"]
            ),
        ),
    )

    output_root.mkdir(parents=True)
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )
    capture = cv2.VideoCapture(str(video_path.resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(output_root / "annotated.mp4"),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise RuntimeError("Could not create Stage L annotated video")

    occupancy_path = output_root / "occupancy.csv"
    events_path = output_root / "events.csv"
    detections_path = output_root / "detections.jsonl"
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    previous = {
        method: {slot_id: False for slot_id in slot_ids}
        for method in (
            "p1_raw",
            "p3_gate",
            "p3_temporal",
            "p3_full_tracking_temporal",
        )
    }
    track_centers: dict[int, tuple[float, float]] = {}
    timing = {
        "detector_tracking": [],
        "mapping": [],
        "classifier_gate": [],
        "fusion_temporal": [],
        "frame_total": [],
    }
    frame_index = 0
    run_start = time.perf_counter()
    try:
        with detections_path.open("x", encoding="utf-8") as detection_handle:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                frame_start = time.perf_counter()
                detector_start = frame_start
                detections = detector.detect(frame)
                detector_end = time.perf_counter()
                evidence = map_detections_to_slots(
                    detections,
                    slots,
                    mode="overlap",
                    overlap_threshold=float(
                        protocol["mapping"]["minimum_slot_coverage"]
                    ),
                )
                mapping_end = time.perf_counter()

                classifier_start = mapping_end
                uncertain_slots = [
                    slot
                    for slot in slots
                    if not evidence[slot.slot_id].occupied
                ]
                scores = classifier.predict_patches(
                    [
                        extract_slot_patch(
                            frame,
                            slot.points,
                            output_size=tuple(
                                classifier_config["patch_size"]
                            ),
                            perspective_warp=bool(
                                classifier_config["perspective_warp"]
                            ),
                        )
                        for slot in uncertain_slots
                    ]
                )
                classifier_by_slot = {
                    slot.slot_id: score
                    for slot, score in zip(
                        uncertain_slots,
                        scores,
                        strict=True,
                    )
                }
                decisions = uncertainty_gated_fusion(
                    evidence,
                    classifier_by_slot,
                    gate_config,
                )
                classifier_end = time.perf_counter()

                displacement_by_track: dict[int, float] = {}
                for detection in detections:
                    if detection.track_id is None:
                        continue
                    x1, y1, x2, y2 = detection.bbox
                    center = ((x1 + x2) / 2.0, (y1 + y2) / 2.0)
                    old_center = track_centers.get(detection.track_id)
                    displacement_by_track[detection.track_id] = (
                        0.0
                        if old_center is None
                        else math.dist(old_center, center)
                    )
                    track_centers[detection.track_id] = center
                track_inputs = []
                for slot in slots:
                    item = evidence[slot.slot_id]
                    if (
                        item.detection_index is None
                        or item.track_id is None
                        or item.track_id not in displacement_by_track
                    ):
                        continue
                    track_inputs.append(
                        TrackSlotEvidence(
                            slot_id=slot.slot_id,
                            track_id=str(item.track_id),
                            coverage=item.geometric_score,
                            center_displacement_px=(
                                displacement_by_track[item.track_id]
                            ),
                        )
                    )
                track_states = {
                    state.slot_id: state
                    for state in track_gate.update(track_inputs)
                }

                final_states: dict[str, FilteredSlotState] = {}
                for slot in slots:
                    slot_id = slot.slot_id
                    decision = decisions[slot_id]
                    track_state = track_states[slot_id]
                    tracked_decision = apply_track_override(
                        decision,
                        stationary_track_confirmed=track_state.occupied,
                        moving_track_overlaps=bool(
                            track_state.suppressed_moving_track_ids
                        ),
                    )
                    temporal_state = temporal_gate.update(
                        FusedEvidence(
                            slot_id=slot_id,
                            p_cls=decision.classifier_probability,
                            p_det=decision.detector_score,
                            p_track=None,
                            probability=decision.score,
                            effective_weights=(0.0, 0.0, 0.0),
                        )
                    )
                    full_state = temporal_full.update(
                        FusedEvidence(
                            slot_id=slot_id,
                            p_cls=tracked_decision.classifier_probability,
                            p_det=tracked_decision.detector_score,
                            p_track=float(track_state.occupied),
                            probability=tracked_decision.score,
                            effective_weights=(0.0, 0.0, 0.0),
                        )
                    )
                    method_states = {
                        "p1_raw": int(evidence[slot_id].occupied),
                        "p3_gate": int(decision.occupied),
                        "p3_temporal": int(temporal_state.occupied),
                        "p3_full_tracking_temporal": int(
                            full_state.occupied
                        ),
                    }
                    truth_state = _truth_state(truth, slot_id, frame_index)
                    row = {
                        "video_id": continuous["source_video_id"],
                        "frame_index": frame_index,
                        "timestamp_s": frame_index / fps,
                        "slot_id": slot_id,
                        "truth": truth_state,
                        **method_states,
                        "p3_score": decision.score,
                        "p3_temporal_score": (
                            temporal_state.filtered_probability
                        ),
                        "p3_full_score": tracked_decision.score,
                        "p3_full_temporal_score": (
                            full_state.filtered_probability
                        ),
                        "gate_branch": decision.branch,
                        "full_branch": tracked_decision.branch,
                        "p_cls": decision.classifier_probability,
                        "p_det": decision.detector_score,
                        "mapped_track_id": evidence[slot_id].track_id,
                        "confirmed_track_id": (
                            track_state.confirmed_track_id
                        ),
                        "moving_track_ids": "|".join(
                            track_state.suppressed_moving_track_ids
                        ),
                    }
                    rows.append(row)
                    for method, state in method_states.items():
                        old_state = previous[method][slot_id]
                        if bool(state) != old_state:
                            events.append(
                                {
                                    "video_id": continuous["source_video_id"],
                                    "frame_index": frame_index,
                                    "timestamp_s": frame_index / fps,
                                    "slot_id": slot_id,
                                    "method": method,
                                    "from_state": int(old_state),
                                    "to_state": state,
                                }
                            )
                        previous[method][slot_id] = bool(state)
                    final_states[slot_id] = FilteredSlotState(
                        slot_id=slot_id,
                        occupied=full_state.occupied,
                        filtered_score=full_state.filtered_probability,
                        raw_occupied=tracked_decision.occupied,
                        raw_evidence_score=tracked_decision.score,
                        changed=full_state.changed,
                        track_id=evidence[slot_id].track_id,
                    )
                fusion_end = time.perf_counter()

                detection_handle.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "timestamp_s": frame_index / fps,
                            "detections": [
                                asdict(detection)
                                for detection in detections
                            ],
                        }
                    )
                    + "\n"
                )
                annotated = draw_frame(
                    frame=frame,
                    detections=detections,
                    slots=slots,
                    states=final_states,
                    experiment="p3-full",
                    processing_fps=1.0
                    / max(time.perf_counter() - frame_start, 1e-9),
                )
                writer.write(annotated)
                frame_end = time.perf_counter()
                timing["detector_tracking"].append(
                    (detector_end - detector_start) * 1000.0
                )
                timing["mapping"].append(
                    (mapping_end - detector_end) * 1000.0
                )
                timing["classifier_gate"].append(
                    (classifier_end - classifier_start) * 1000.0
                )
                timing["fusion_temporal"].append(
                    (fusion_end - classifier_end) * 1000.0
                )
                timing["frame_total"].append(
                    (frame_end - frame_start) * 1000.0
                )
                frame_index += 1
    finally:
        capture.release()
        writer.release()

    if frame_index != int(continuous["expected"]["frames"]):
        raise RuntimeError("Stage L video ended before the expected frame count")
    with occupancy_path.open("x", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0]),
            lineterminator="\n",
        )
        csv_writer.writeheader()
        csv_writer.writerows(rows)
    with events_path.open("x", encoding="utf-8", newline="") as handle:
        csv_writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "frame_index",
                "timestamp_s",
                "slot_id",
                "method",
                "from_state",
                "to_state",
            ],
            lineterminator="\n",
        )
        csv_writer.writeheader()
        csv_writer.writerows(events)

    warmup = int(continuous["warmup_frames"])
    evaluated = [row for row in rows if row["frame_index"] >= warmup]
    stable_frames = int(continuous["stable_frames"])
    methods = {
        method: _method_metrics(
            evaluated,
            method,
            fps,
            stable_frames,
            frame_offset=warmup,
        )
        for method in (
            "p1_raw",
            "p3_gate",
            "p3_temporal",
            "p3_full_tracking_temporal",
        )
    }
    runtime = {
        "frames": frame_index,
        "warmup_frames_excluded": warmup,
        "steady_state": {
            component: _timing(values, warmup)
            for component, values in timing.items()
        },
        "detector": detector.metadata(),
        "classifier": classifier.metadata(),
    }
    metrics = {
        "schema_version": 1,
        "protocol_id": STAGE_L_PROTOCOL_ID,
        "data_role": continuous["data_role"],
        "scope": continuous["claims"],
        "frames_evaluated": len(evaluated),
        "methods": methods,
        "parameters_selected_from_video": False,
    }
    summary = {
        "schema_version": 1,
        "protocol_id": STAGE_L_PROTOCOL_ID,
        "status": "executed_integrated_continuous_case_study",
        "source_video_id": continuous["source_video_id"],
        "frames": frame_index,
        "slots": len(slots),
        "fps": fps,
        "elapsed_s": time.perf_counter() - run_start,
        "method": protocol["method"],
        "output_files": [
            "annotated.mp4",
            "occupancy.csv",
            "events.csv",
            "detections.jsonl",
            "metrics.json",
            "summary.json",
            "runtime_metadata.json",
        ],
    }
    for path, payload in (
        (output_root / "metrics.json", metrics),
        (output_root / "summary.json", summary),
        (output_root / "runtime_metadata.json", runtime),
    ):
        path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return metrics
