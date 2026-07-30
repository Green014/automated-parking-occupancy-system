from __future__ import annotations

import csv
import hashlib
import json
import platform
import statistics
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import cv2

from .detector import Detector
from .geometry import map_detections_to_slots
from .slots import load_slot_map
from .temporal import (
    FilteredSlotState,
    HysteresisConfig,
    TemporalOccupancyFilter,
)
from .visualization import draw_frame

Experiment = Literal["b0", "b1", "proposed", "t0"]


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    experiment: Experiment = "b0"
    method_id: str | None = None
    method_name: str | None = None
    method_registry_path: str | None = None
    data_role: str | None = None
    overlap_threshold: float = 0.30
    max_frames: int | None = None
    write_video: bool = True
    output_codec: str = "mp4v"
    hysteresis: HysteresisConfig = HysteresisConfig()

    def __post_init__(self) -> None:
        if self.experiment not in {"b0", "b1", "proposed", "t0"}:
            raise ValueError(f"Unknown experiment: {self.experiment}")
        if len(self.output_codec) != 4:
            raise ValueError("output_codec must be four characters")
        if self.max_frames is not None and self.max_frames <= 0:
            raise ValueError("max_frames must be positive")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = (len(ordered) - 1) * percentile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    fraction = index - lower
    return ordered[lower] * (1.0 - fraction) + ordered[upper] * fraction


def _raw_states(
    evidence_by_slot: dict[str, Any],
    previous: dict[str, bool],
) -> dict[str, FilteredSlotState]:
    states: dict[str, FilteredSlotState] = {}
    for slot_id, evidence in evidence_by_slot.items():
        occupied = evidence.occupied
        states[slot_id] = FilteredSlotState(
            slot_id=slot_id,
            occupied=occupied,
            filtered_score=evidence.evidence_score,
            raw_occupied=occupied,
            raw_evidence_score=evidence.evidence_score,
            changed=occupied != previous.get(slot_id, False),
            track_id=evidence.track_id,
        )
    return states


def process_video(
    input_path: str | Path,
    slot_map_path: str | Path,
    output_dir: str | Path,
    detector: Detector,
    config: PipelineConfig | None = None,
) -> dict[str, Any]:
    """Run one video and return the same summary saved to disk."""

    config = config or PipelineConfig()
    input_path = Path(input_path).resolve()
    slot_map_path = Path(slot_map_path).resolve()
    output_dir = Path(output_dir).resolve()
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite pipeline output: {output_dir}"
        )
    output_dir.mkdir(parents=True)

    capture = cv2.VideoCapture(str(input_path))
    if not capture.isOpened():
        raise RuntimeError(f"OpenCV could not open video: {input_path}")

    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    source_fps = float(capture.get(cv2.CAP_PROP_FPS))
    if width <= 0 or height <= 0:
        capture.release()
        raise RuntimeError("Input video reports an invalid frame size")
    if source_fps <= 0:
        source_fps = 25.0

    slot_map = load_slot_map(slot_map_path, frame_size=(width, height))
    mode = "center" if config.experiment == "b0" else "overlap"
    temporal_filter = (
        TemporalOccupancyFilter(
            (slot.slot_id for slot in slot_map.slots),
            config.hysteresis,
        )
        if config.experiment == "proposed"
        else None
    )

    writer: cv2.VideoWriter | None = None
    output_video = output_dir / "annotated.mp4"
    if config.write_video:
        fourcc = cv2.VideoWriter_fourcc(*config.output_codec)
        writer = cv2.VideoWriter(
            str(output_video),
            fourcc,
            source_fps,
            (width, height),
        )
        if not writer.isOpened():
            capture.release()
            raise RuntimeError(f"OpenCV could not create video: {output_video}")

    occupancy_path = output_dir / "occupancy.csv"
    event_path = output_dir / "events.csv"
    detections_path = output_dir / "detections.jsonl"
    frame_times_ms: list[float] = []
    detector_times_ms: list[float] = []
    mapping_times_ms: list[float] = []
    render_times_ms: list[float] = []
    frame_count = 0
    event_count = 0
    previous_states = {slot.slot_id: False for slot in slot_map.slots}
    source_id = input_path.stem
    run_start = time.perf_counter()

    try:
        with (
            occupancy_path.open("w", newline="", encoding="utf-8") as occupancy_file,
            event_path.open("w", newline="", encoding="utf-8") as event_file,
            detections_path.open("w", encoding="utf-8") as detections_file,
        ):
            occupancy_writer = csv.DictWriter(
                occupancy_file,
                fieldnames=[
                    "video_id",
                    "frame_index",
                    "timestamp_s",
                    "slot_id",
                    "state",
                    "raw_state",
                    "evidence",
                    "filtered_score",
                    "track_id",
                ],
            )
            event_writer = csv.DictWriter(
                event_file,
                fieldnames=[
                    "video_id",
                    "frame_index",
                    "timestamp_s",
                    "slot_id",
                    "from_state",
                    "to_state",
                    "evidence",
                    "track_id",
                ],
            )
            occupancy_writer.writeheader()
            event_writer.writeheader()

            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if config.max_frames is not None and frame_count >= config.max_frames:
                    break

                frame_start = time.perf_counter()
                detector_start = time.perf_counter()
                detections = detector.detect(frame)
                detector_end = time.perf_counter()
                detections_file.write(
                    json.dumps(
                        {
                            "video_id": source_id,
                            "frame_index": frame_count,
                            "timestamp_s": frame_count / source_fps,
                            "detections": [
                                {
                                    "bbox_xyxy": list(detection.bbox),
                                    "confidence": detection.confidence,
                                    "class_id": detection.class_id,
                                    "class_name": detection.class_name,
                                    "track_id": detection.track_id,
                                }
                                for detection in detections
                            ],
                        }
                    )
                    + "\n"
                )

                mapping_start = detector_end
                evidence_by_slot = map_detections_to_slots(
                    detections=detections,
                    slots=slot_map.slots,
                    mode=mode,
                    overlap_threshold=config.overlap_threshold,
                )
                if temporal_filter is None:
                    states = _raw_states(evidence_by_slot, previous_states)
                else:
                    states = temporal_filter.update(evidence_by_slot)
                mapping_end = time.perf_counter()

                timestamp_s = frame_count / source_fps
                for slot in slot_map.slots:
                    state = states[slot.slot_id]
                    occupancy_writer.writerow(
                        {
                            "video_id": source_id,
                            "frame_index": frame_count,
                            "timestamp_s": f"{timestamp_s:.6f}",
                            "slot_id": slot.slot_id,
                            "state": int(state.occupied),
                            "raw_state": int(state.raw_occupied),
                            "evidence": f"{state.raw_evidence_score:.6f}",
                            "filtered_score": f"{state.filtered_score:.6f}",
                            "track_id": (
                                "" if state.track_id is None else state.track_id
                            ),
                        }
                    )
                    if state.changed:
                        event_count += 1
                        event_writer.writerow(
                            {
                                "video_id": source_id,
                                "frame_index": frame_count,
                                "timestamp_s": f"{timestamp_s:.6f}",
                                "slot_id": slot.slot_id,
                                "from_state": int(previous_states[slot.slot_id]),
                                "to_state": int(state.occupied),
                                "evidence": f"{state.filtered_score:.6f}",
                                "track_id": (
                                    "" if state.track_id is None else state.track_id
                                ),
                            }
                        )
                    previous_states[slot.slot_id] = state.occupied

                render_start = time.perf_counter()
                elapsed = max(time.perf_counter() - frame_start, 1e-9)
                if writer is not None:
                    annotated = draw_frame(
                        frame=frame,
                        detections=detections,
                        slots=slot_map.slots,
                        states=states,
                        experiment=config.experiment,
                        processing_fps=1.0 / elapsed,
                    )
                    writer.write(annotated)
                render_end = time.perf_counter()

                detector_times_ms.append((detector_end - detector_start) * 1000)
                mapping_times_ms.append((mapping_end - mapping_start) * 1000)
                render_times_ms.append((render_end - render_start) * 1000)
                frame_times_ms.append((render_end - frame_start) * 1000)
                frame_count += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed_s = time.perf_counter() - run_start
    if frame_count == 0:
        raise RuntimeError("Input video did not yield any frames")

    def timing(values: list[float]) -> dict[str, float]:
        return {
            "mean": statistics.fmean(values),
            "p50": statistics.median(values),
            "p95": _percentile(values, 0.95),
        }

    warmup_frames = 1 if frame_count > 1 else 0
    steady_frame_times = frame_times_ms[warmup_frames:]
    steady_detector_times = detector_times_ms[warmup_frames:]
    steady_mapping_times = mapping_times_ms[warmup_frames:]
    steady_render_times = render_times_ms[warmup_frames:]
    summary_path = output_dir / "summary.json"
    runtime_metadata_path = output_dir / "runtime_metadata.json"
    metrics_path = output_dir / "metrics.json"
    detector_metadata = detector.metadata()
    runtime_metadata = {
        "detector": detector_metadata,
        "frames_processed": frame_count,
        "elapsed_s": elapsed_s,
        "end_to_end_fps": frame_count / elapsed_s,
        "warmup_frames_excluded": warmup_frames,
        "timing_ms": {
            "frame": timing(frame_times_ms),
            "detector": timing(detector_times_ms),
            "mapping_and_filter": timing(mapping_times_ms),
            "render_and_write": timing(render_times_ms),
        },
    }
    with runtime_metadata_path.open("w", encoding="utf-8") as handle:
        json.dump(runtime_metadata, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    metrics = {
        "evaluation_status": "ground_truth_not_supplied",
        "dataset_role": config.data_role,
        "frames_processed": frame_count,
        "slots": len(slot_map.slots),
        "slot_frame_predictions": frame_count * len(slot_map.slots),
        "events": event_count,
        "end_to_end_fps": frame_count / elapsed_s,
        "claim_boundary": (
            "No slot accuracy metric is reported without matched "
            "ground-truth states."
        ),
    }
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    summary: dict[str, Any] = {
        "experiment": config.experiment,
        "method": {
            "id": config.method_id,
            "canonical_name": config.method_name,
            "registry_path": config.method_registry_path,
            "data_role": config.data_role,
        },
        "input_path": str(input_path),
        "input_sha256": _sha256(input_path),
        "slot_map_path": str(slot_map_path),
        "slot_map_sha256": _sha256(slot_map_path),
        "source_fps": source_fps,
        "frame_size": [width, height],
        "frames_processed": frame_count,
        "slots": len(slot_map.slots),
        "events": event_count,
        "elapsed_s": elapsed_s,
        "end_to_end_fps": frame_count / elapsed_s,
        "warmup_frames_excluded": warmup_frames,
        "steady_state_fps": (
            1000.0 / statistics.fmean(steady_frame_times)
            if steady_frame_times
            else 0.0
        ),
        "timing_ms": {
            "frame": timing(frame_times_ms),
            "detector": timing(detector_times_ms),
            "mapping_and_filter": timing(mapping_times_ms),
            "render_and_write": timing(render_times_ms),
        },
        "steady_state_timing_ms": {
            "frame": timing(steady_frame_times),
            "detector": timing(steady_detector_times),
            "mapping_and_filter": timing(steady_mapping_times),
            "render_and_write": timing(steady_render_times),
        },
        "config": {
            **asdict(config),
            "hysteresis": asdict(config.hysteresis),
        },
        "detector": detector_metadata,
        "mapping": {
            "mode": mode,
            "one_to_one": True,
            "minimum_slot_coverage": (
                config.overlap_threshold if mode == "overlap" else None
            ),
            "evidence": (
                "detector_confidence"
                if mode == "center"
                else "detector_confidence_times_slot_coverage"
            ),
        },
        "python_version": platform.python_version(),
        "outputs": {
            "video": str(output_video) if writer is not None else None,
            "occupancy_csv": str(occupancy_path),
            "events_csv": str(event_path),
            "detections_jsonl": str(detections_path),
            "summary_json": str(summary_path),
            "metrics_json": str(metrics_path),
            "runtime_metadata_json": str(runtime_metadata_path),
        },
    }
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return summary
