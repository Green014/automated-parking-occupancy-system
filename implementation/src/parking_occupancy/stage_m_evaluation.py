from __future__ import annotations

import csv
import json
import statistics
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

LITERATURE_CORE_SRC = Path(__file__).resolve().parents[2] / "literature_core" / "src"
if str(LITERATURE_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(LITERATURE_CORE_SRC))

from .geometry import map_detections_to_slots
from .integrated_workflow import (
    UncertaintyGateConfig,
    uncertainty_gated_fusion,
)
from .models import Detection, ParkingSlot
from .stage_m_tracking import (
    STAGE_M_PROTOCOL_ID,
    OS0ParkingAdapter,
    UltralyticsSequenceAdapter,
)
from .temporal import FilteredSlotState
from .visualization import draw_frame


METHODS = ("T0", "T1", "T2", "T3")
EVENT_FIELDS = (
    "source_id",
    "frame_index",
    "timestamp_s",
    "slot_id",
    "method",
    "from_state",
    "to_state",
)


@dataclass(frozen=True, slots=True)
class StageMRunResult:
    rows: tuple[dict[str, Any], ...]
    events: tuple[dict[str, Any], ...]
    detection_records: tuple[dict[str, Any], ...]
    annotated_frames: tuple[np.ndarray, ...]
    metrics: dict[str, Any]
    summary: dict[str, Any]
    runtime_metadata: dict[str, Any]


def _truth_state(
    truth: Mapping[str, Any] | None,
    slot_id: str,
    frame_index: int,
) -> int | None:
    if truth is None:
        return None
    matches = [
        slot
        for slot in truth.get("slots", [])
        if str(slot.get("slot_id")) == slot_id
    ]
    if len(matches) != 1:
        raise ValueError(f"Truth must define slot {slot_id} exactly once")
    for interval in matches[0].get("intervals", []):
        if (
            int(interval["start_frame"])
            <= frame_index
            < int(interval["end_frame"])
        ):
            state = str(interval["state"]).lower()
            if state not in {"occupied", "vacant"}:
                raise ValueError(f"Invalid truth state {state!r}")
            return int(state == "occupied")
    raise ValueError(
        f"Truth does not cover slot {slot_id} at frame {frame_index}"
    )


def _append_events(
    events: list[dict[str, Any]],
    previous: dict[str, dict[str, bool]],
    *,
    source_id: str,
    frame_index: int,
    fps: float,
    slot_id: str,
    states: Mapping[str, bool],
) -> None:
    for method, state in states.items():
        old_state = previous[method][slot_id]
        if bool(state) != old_state:
            events.append(
                {
                    "source_id": source_id,
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / fps,
                    "slot_id": slot_id,
                    "method": method,
                    "from_state": int(old_state),
                    "to_state": int(state),
                }
            )
        previous[method][slot_id] = bool(state)


def _mean_timing(values: Sequence[float]) -> dict[str, float | int]:
    if not values:
        return {"frames": 0, "mean_ms": 0.0, "p50_ms": 0.0, "p95_ms": 0.0}
    ordered = sorted(float(value) for value in values)
    return {
        "frames": len(ordered),
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
        "fps_from_mean": 1000.0 / statistics.fmean(ordered),
    }


def _metrics(
    rows: Sequence[dict[str, Any]],
    methods: Sequence[str],
    *,
    fps: float,
    stable_frames: int,
    claim_scope: str,
) -> dict[str, Any]:
    if not rows or rows[0].get("truth") is None:
        return {
            "schema_version": 1,
            "protocol_id": STAGE_M_PROTOCOL_ID,
            "status": "not_computed_no_truth",
            "claim_scope": claim_scope,
            "methods": {
                method: {"status": "smoke_output_only"}
                for method in methods
            },
        }

    from literature_core.metrics import binary_metrics, sequence_temporal_metrics

    slot_ids = sorted({str(row["slot_id"]) for row in rows})
    payload: dict[str, Any] = {}
    for method in methods:
        truth_values = [int(row["truth"]) for row in rows]
        predictions = [int(row[method]) for row in rows]
        temporal_by_slot: dict[str, Any] = {}
        aggregate = {
            "flicker_count": 0,
            "false_transitions": 0,
            "early_transitions": 0,
            "on_time_transitions": 0,
            "delayed_transitions": 0,
            "missed_transitions": 0,
            "signed_transition_error_values_s": [],
        }
        for slot_id in slot_ids:
            slot_rows = [
                row for row in rows if str(row["slot_id"]) == slot_id
            ]
            temporal = sequence_temporal_metrics(
                [int(row["truth"]) for row in slot_rows],
                [int(row[method]) for row in slot_rows],
                fps,
                stable_frames=stable_frames,
            )
            temporal_by_slot[slot_id] = temporal
            aggregate["flicker_count"] += int(
                temporal["unsupported_flicker_count"]
            )
            aggregate["false_transitions"] += int(
                temporal["unsupported_flicker_count"]
                + temporal["transition_instability_changes"]
            )
            for key in (
                "early_transitions",
                "on_time_transitions",
                "delayed_transitions",
                "missed_transitions",
            ):
                aggregate[key] += int(temporal[key])
            aggregate["signed_transition_error_values_s"].extend(
                temporal["signed_transition_error_values_s"]["entry"]
            )
            aggregate["signed_transition_error_values_s"].extend(
                temporal["signed_transition_error_values_s"]["exit"]
            )
        signed = aggregate["signed_transition_error_values_s"]
        aggregate["signed_transition_error_mean_s"] = (
            statistics.fmean(signed) if signed else None
        )
        payload[method] = {
            **binary_metrics(truth_values, predictions),
            "temporal_aggregate": aggregate,
            "temporal_by_slot": temporal_by_slot,
        }

    return {
        "schema_version": 1,
        "protocol_id": STAGE_M_PROTOCOL_ID,
        "status": "computed_against_frozen_truth",
        "claim_scope": claim_scope,
        "methods": payload,
    }


def run_os0_sequence(
    *,
    frames: Iterable[np.ndarray],
    fps: float,
    source_id: str,
    adapter: OS0ParkingAdapter,
    truth: Mapping[str, Any] | None = None,
    continuous: bool = True,
    claim_scope: str = "smoke_test",
    stable_frames: int = 3,
) -> StageMRunResult:
    """Run official ParkingManagement while logging its exact slot rule."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []
    annotated_frames: list[np.ndarray] = []
    previous = {"OS0": {slot.slot_id: False for slot in adapter.slots}}
    timings: list[float] = []
    started = time.perf_counter()
    if continuous:
        adapter.begin_source(source_id, continuous=True)

    for frame_index, frame in enumerate(frames):
        if not continuous:
            adapter.begin_source(
                f"{source_id}:static:{frame_index}",
                continuous=False,
            )
        frame_start = time.perf_counter()
        result = adapter.process(frame)
        timings.append((time.perf_counter() - frame_start) * 1000.0)
        annotated_frames.append(result.annotated_frame)
        detection_records.append(
            {
                "source_id": source_id,
                "frame_index": frame_index,
                "timestamp_s": frame_index / fps,
                "method": "OS0",
                "detections": [
                    asdict(detection) for detection in result.detections
                ],
            }
        )
        for slot in adapter.slots:
            state = bool(result.slot_states[slot.slot_id])
            truth_state = _truth_state(truth, slot.slot_id, frame_index)
            rows.append(
                {
                    "source_id": source_id,
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / fps,
                    "slot_id": slot.slot_id,
                    "truth": truth_state,
                    "OS0": int(state),
                    "OS0_score": float(state),
                    "logic_provenance": result.logic_provenance,
                }
            )
            _append_events(
                events,
                previous,
                source_id=source_id,
                frame_index=frame_index,
                fps=fps,
                slot_id=slot.slot_id,
                states={"OS0": state},
            )

    metrics = _metrics(
        rows,
        ("OS0",),
        fps=fps,
        stable_frames=stable_frames,
        claim_scope=claim_scope,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "protocol_id": STAGE_M_PROTOCOL_ID,
        "status": "executed" if rows else "empty_input",
        "baseline": "OS0-Controlled",
        "claim_scope": claim_scope,
        "source_id": source_id,
        "continuous": continuous,
        "frames": len(annotated_frames),
        "slots": len(adapter.slots),
        "elapsed_s": elapsed,
        "official_logic": "centre_point_in_polygon",
        "local_adapter_role": "logging_metrics_and_exports_only",
    }
    runtime = {
        "schema_version": 1,
        "protocol_id": STAGE_M_PROTOCOL_ID,
        "frame_processing": _mean_timing(timings),
        "adapter": adapter.metadata(),
    }
    return StageMRunResult(
        rows=tuple(rows),
        events=tuple(events),
        detection_records=tuple(detection_records),
        annotated_frames=tuple(annotated_frames),
        metrics=metrics,
        summary=summary,
        runtime_metadata=runtime,
    )


def run_t0_t3_sequence(
    *,
    frames: Iterable[np.ndarray],
    fps: float,
    source_id: str,
    slots: Sequence[ParkingSlot],
    plain_adapter: UltralyticsSequenceAdapter,
    bytetrack_adapter: UltralyticsSequenceAdapter,
    tracktrack_adapter: UltralyticsSequenceAdapter,
    classifier_scores: Callable[
        [np.ndarray, Sequence[ParkingSlot]], Mapping[str, float]
    ],
    mapping_coverage: float,
    classifier_threshold: float,
    temporal_config: Mapping[str, float],
    truth: Mapping[str, Any] | None = None,
    claim_scope: str = "smoke_test",
    stable_frames: int = 3,
) -> StageMRunResult:
    """Run the frozen T0-T3 ablation with a shared detector/classifier setup."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if not slots:
        raise ValueError("slots must not be empty")
    if not 0.0 <= mapping_coverage <= 1.0:
        raise ValueError("mapping_coverage must be in [0, 1]")

    from literature_core.models import FusedEvidence
    from literature_core.temporal import TemporalConfig, TemporalFusionFilter

    slot_ids = tuple(slot.slot_id for slot in slots)
    filter_config = TemporalConfig(
        rise_alpha=float(temporal_config["rise_alpha"]),
        fall_alpha=float(temporal_config["fall_alpha"]),
        occupied_threshold=float(temporal_config["occupied_threshold"]),
        vacant_threshold=float(temporal_config["vacant_threshold"]),
        raw_threshold=float(temporal_config["raw_threshold"]),
    )
    temporal_filters = {
        method: TemporalFusionFilter(slot_ids, filter_config)
        for method in ("T1", "T2", "T3")
    }
    gate_config = UncertaintyGateConfig(
        classifier_occupied_threshold=classifier_threshold
    )
    for adapter in (plain_adapter, bytetrack_adapter, tracktrack_adapter):
        adapter.begin_source(source_id, continuous=True)

    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    detection_records: list[dict[str, Any]] = []
    annotated_frames: list[np.ndarray] = []
    previous = {
        method: {slot_id: False for slot_id in slot_ids}
        for method in METHODS
    }
    timing = {
        "plain_predict": [],
        "bytetrack": [],
        "tracktrack": [],
        "classifier_and_fusion": [],
        "frame_total": [],
    }
    started = time.perf_counter()

    for frame_index, frame in enumerate(frames):
        frame_started = time.perf_counter()
        start = frame_started
        plain_detections = plain_adapter.detect(frame)
        plain_end = time.perf_counter()
        byte_detections = bytetrack_adapter.detect(frame)
        byte_end = time.perf_counter()
        track_detections = tracktrack_adapter.detect(frame)
        track_end = time.perf_counter()

        evidence = {
            "plain": map_detections_to_slots(
                plain_detections,
                slots,
                mode="overlap",
                overlap_threshold=mapping_coverage,
            ),
            "byte": map_detections_to_slots(
                byte_detections,
                slots,
                mode="overlap",
                overlap_threshold=mapping_coverage,
            ),
            "track": map_detections_to_slots(
                track_detections,
                slots,
                mode="overlap",
                overlap_threshold=mapping_coverage,
            ),
        }
        scores = {
            str(key): float(value)
            for key, value in classifier_scores(frame, slots).items()
        }
        if set(scores) != set(slot_ids):
            raise ValueError(
                "classifier_scores must return every and only frozen slot ID"
            )
        decisions = {
            "T0": uncertainty_gated_fusion(
                evidence["plain"], scores, gate_config
            ),
            "T1": uncertainty_gated_fusion(
                evidence["plain"], scores, gate_config
            ),
            "T2": uncertainty_gated_fusion(
                evidence["byte"], scores, gate_config
            ),
            "T3": uncertainty_gated_fusion(
                evidence["track"], scores, gate_config
            ),
        }

        states_by_method: dict[str, dict[str, bool]] = {"T0": {}}
        score_by_method: dict[str, dict[str, float]] = {"T0": {}}
        for slot_id in slot_ids:
            decision = decisions["T0"][slot_id]
            states_by_method["T0"][slot_id] = decision.occupied
            score_by_method["T0"][slot_id] = decision.score
        for method in ("T1", "T2", "T3"):
            states_by_method[method] = {}
            score_by_method[method] = {}
            for slot_id in slot_ids:
                decision = decisions[method][slot_id]
                filtered = temporal_filters[method].update(
                    FusedEvidence(
                        slot_id=slot_id,
                        p_cls=decision.classifier_probability,
                        p_det=decision.detector_score,
                        p_track=None,
                        probability=decision.score,
                        effective_weights=(0.0, 0.0, 0.0),
                    )
                )
                states_by_method[method][slot_id] = filtered.occupied
                score_by_method[method][slot_id] = (
                    filtered.filtered_probability
                )
        fusion_end = time.perf_counter()

        detection_records.append(
            {
                "source_id": source_id,
                "frame_index": frame_index,
                "timestamp_s": frame_index / fps,
                "detections": {
                    "T0_T1": [
                        asdict(detection)
                        for detection in plain_detections
                    ],
                    "T2": [
                        asdict(detection)
                        for detection in byte_detections
                    ],
                    "T3": [
                        asdict(detection)
                        for detection in track_detections
                    ],
                },
            }
        )

        t3_states: dict[str, FilteredSlotState] = {}
        for slot in slots:
            slot_id = slot.slot_id
            method_states = {
                method: states_by_method[method][slot_id]
                for method in METHODS
            }
            truth_state = _truth_state(truth, slot_id, frame_index)
            rows.append(
                {
                    "source_id": source_id,
                    "frame_index": frame_index,
                    "timestamp_s": frame_index / fps,
                    "slot_id": slot_id,
                    "truth": truth_state,
                    **{
                        method: int(method_states[method])
                        for method in METHODS
                    },
                    **{
                        f"{method}_score": score_by_method[method][slot_id]
                        for method in METHODS
                    },
                    "T0_branch": decisions["T0"][slot_id].branch,
                    "T1_branch": decisions["T1"][slot_id].branch,
                    "T2_branch": decisions["T2"][slot_id].branch,
                    "T3_branch": decisions["T3"][slot_id].branch,
                    "T2_track_id": evidence["byte"][slot_id].track_id,
                    "T3_track_id": evidence["track"][slot_id].track_id,
                }
            )
            _append_events(
                events,
                previous,
                source_id=source_id,
                frame_index=frame_index,
                fps=fps,
                slot_id=slot_id,
                states=method_states,
            )
            t3_decision = decisions["T3"][slot_id]
            t3_states[slot_id] = FilteredSlotState(
                slot_id=slot_id,
                occupied=method_states["T3"],
                filtered_score=score_by_method["T3"][slot_id],
                raw_occupied=t3_decision.occupied,
                raw_evidence_score=t3_decision.score,
                changed=False,
                track_id=evidence["track"][slot_id].track_id,
            )
        annotated_frames.append(
            draw_frame(
                frame=frame,
                detections=track_detections,
                slots=slots,
                states=t3_states,
                experiment="Stage M T3 smoke"
                if claim_scope == "smoke_test"
                else "Stage M T3",
                processing_fps=1.0
                / max(time.perf_counter() - frame_started, 1e-9),
            )
        )
        frame_end = time.perf_counter()
        timing["plain_predict"].append((plain_end - start) * 1000.0)
        timing["bytetrack"].append((byte_end - plain_end) * 1000.0)
        timing["tracktrack"].append((track_end - byte_end) * 1000.0)
        timing["classifier_and_fusion"].append(
            (fusion_end - track_end) * 1000.0
        )
        timing["frame_total"].append(
            (frame_end - frame_started) * 1000.0
        )

    metrics = _metrics(
        rows,
        METHODS,
        fps=fps,
        stable_frames=stable_frames,
        claim_scope=claim_scope,
    )
    elapsed = time.perf_counter() - started
    summary = {
        "schema_version": 1,
        "protocol_id": STAGE_M_PROTOCOL_ID,
        "status": "executed" if rows else "empty_input",
        "claim_scope": claim_scope,
        "source_id": source_id,
        "frames": len(annotated_frames),
        "slots": len(slots),
        "elapsed_s": elapsed,
        "methods": {
            "T0": "P3_gate_without_temporal_or_tracking",
            "T1": "T0_plus_E4",
            "T2": "T1_plus_Ultralytics_ByteTrack",
            "T3": "T1_plus_Ultralytics_TrackTrack",
        },
        "threshold_selection_performed": False,
    }
    runtime = {
        "schema_version": 1,
        "protocol_id": STAGE_M_PROTOCOL_ID,
        "timing": {
            component: _mean_timing(values)
            for component, values in timing.items()
        },
        "plain_adapter": plain_adapter.metadata(),
        "bytetrack_adapter": bytetrack_adapter.metadata(),
        "tracktrack_adapter": tracktrack_adapter.metadata(),
    }
    return StageMRunResult(
        rows=tuple(rows),
        events=tuple(events),
        detection_records=tuple(detection_records),
        annotated_frames=tuple(annotated_frames),
        metrics=metrics,
        summary=summary,
        runtime_metadata=runtime,
    )


def export_stage_m_run(
    result: StageMRunResult,
    *,
    output_root: Path,
    fps: float,
) -> None:
    """Write the common Stage M output contract without overwriting."""

    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Stage M output: {output_root}")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if not result.annotated_frames:
        raise ValueError("Cannot export a run with no annotated frames")
    output_root.mkdir(parents=True)

    occupancy_path = output_root / "occupancy.csv"
    with occupancy_path.open("x", encoding="utf-8", newline="") as handle:
        fieldnames = list(result.rows[0])
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(result.rows)

    with (output_root / "events.csv").open(
        "x", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(EVENT_FIELDS),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result.events)

    with (output_root / "detections.jsonl").open(
        "x", encoding="utf-8"
    ) as handle:
        for record in result.detection_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")

    height, width = result.annotated_frames[0].shape[:2]
    video_path = output_root / "annotated.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create annotated video: {video_path}")
    try:
        for frame in result.annotated_frames:
            if frame.shape[:2] != (height, width):
                raise ValueError("Annotated frames must have one resolution")
            writer.write(frame)
    finally:
        writer.release()

    summary = {
        **result.summary,
        "output_files": [
            "occupancy.csv",
            "events.csv",
            "detections.jsonl",
            "annotated.mp4",
            "metrics.json",
            "summary.json",
            "runtime_metadata.json",
        ],
    }
    for filename, payload in (
        ("metrics.json", result.metrics),
        ("summary.json", summary),
        ("runtime_metadata.json", result.runtime_metadata),
    ):
        (output_root / filename).write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
