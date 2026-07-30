from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml

from .detector_comparison import sha256_file
from .integrated_runner import run_integrated_video
from .stage_t_tracktrack import (
    D1_WEIGHTS_SHA256,
    E1B_CHECKPOINT_SHA256,
    P3_TT_CONFIG_NAME,
    load_p3_tt_config,
    write_tracks_jsonl,
)


GENERIC_P3_TT_RUNTIME_ID = "P3-TT-GENERIC-PORTABLE-RUNTIME-20260730-01"
GENERIC_OUTPUT_FILES = (
    "occupancy.csv",
    "events.csv",
    "detections.jsonl",
    "tracks.jsonl",
    "annotated.mp4",
    "summary.json",
    "runtime_metadata.json",
)
DEFAULT_P3_TT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / P3_TT_CONFIG_NAME
)


class GenericP3TTRuntimeError(ValueError):
    """Raised when a generic P3-TT runtime invariant is violated."""


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def _input_provenance(
    *,
    input_path: Path,
    slots_path: Path,
    detector_weights: Path,
    classifier_checkpoint: Path,
    config_path: Path,
    tracker_config_path: Path,
    truth_path: Path | None,
) -> dict[str, Any]:
    def record(path: Path) -> dict[str, Any]:
        resolved = path.resolve()
        return {
            "path": str(resolved),
            "bytes": resolved.stat().st_size,
            "sha256": sha256_file(resolved),
        }

    return {
        "video": record(input_path),
        "slots": record(slots_path),
        "D1_weights": record(detector_weights),
        "E1b_checkpoint": record(classifier_checkpoint),
        "P3_TT_config": record(config_path),
        "TrackTrack_config": record(tracker_config_path),
        "truth": None if truth_path is None else record(truth_path),
    }


def run_generic_p3_tt(
    *,
    input_path: Path,
    slots_path: Path,
    detector_weights: Path,
    classifier_checkpoint: Path,
    source_id: str,
    output_root: Path,
    truth_path: Path | None = None,
    config_path: Path = DEFAULT_P3_TT_CONFIG,
    device: str = "auto",
    classifier_batch_size: int = 64,
    integrated_runner_fn: Callable[..., dict[str, Any]] | None = None,
    tracks_writer_fn: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Run P3-TT on any local video without the frozen VIRAT experiment gate.

    A call constructs the complete underlying integrated runtime afresh. No
    detector/tracker, temporal, or event object is cached across calls.
    """

    if not source_id.strip():
        raise GenericP3TTRuntimeError("source_id must not be empty")
    if classifier_batch_size <= 0:
        raise GenericP3TTRuntimeError("classifier_batch_size must be positive")

    paths = [
        input_path,
        slots_path,
        detector_weights,
        classifier_checkpoint,
        config_path,
    ]
    if truth_path is not None:
        paths.append(truth_path)
    for path in paths:
        if not path.resolve().is_file():
            raise FileNotFoundError(path)

    config_path = config_path.resolve()
    config = load_p3_tt_config(config_path)
    tracker_config_path = (
        config_path.parent
        / str(config["tracking"]["tracktrack"]["config_path"])
    ).resolve()
    provenance = _input_provenance(
        input_path=input_path,
        slots_path=slots_path,
        detector_weights=detector_weights,
        classifier_checkpoint=classifier_checkpoint,
        config_path=config_path,
        tracker_config_path=tracker_config_path,
        truth_path=truth_path,
    )
    custom_d1 = (
        provenance["D1_weights"]["sha256"] != D1_WEIGHTS_SHA256
    )
    custom_e1b = (
        provenance["E1b_checkpoint"]["sha256"] != E1B_CHECKPOINT_SHA256
    )
    custom_weights = custom_d1 or custom_e1b

    runner = integrated_runner_fn or run_integrated_video
    summary = runner(
        input_path=input_path,
        slots_path=slots_path,
        detector_weights=detector_weights,
        classifier_checkpoint=classifier_checkpoint,
        output_root=output_root,
        config_path=config_path,
        device=device,
        source_id=source_id,
        truth_path=truth_path,
        temporal_enabled=False,
        tracker_backend="tracktrack",
        classifier_batch_size=classifier_batch_size,
    )
    output_root = output_root.resolve()
    track_buffer = int(
        yaml.safe_load(
            tracker_config_path.read_text(encoding="utf-8")
        )["track_buffer"]
    )
    tracks_writer = tracks_writer_fn or write_tracks_jsonl
    track_summary = tracks_writer(
        detections_path=output_root / "detections.jsonl",
        occupancy_path=output_root / "occupancy.csv",
        output_path=output_root / "tracks.jsonl",
        track_buffer=track_buffer,
    )

    summary_path = output_root / "summary.json"
    stored_summary = _read_json(summary_path)
    output_files = list(GENERIC_OUTPUT_FILES)
    if truth_path is not None:
        output_files.insert(5, "metrics.json")
    stored_summary.update(
        {
            "method_id": "P3-TT",
            "variant_id": "P3-TT-GENERIC",
            "runtime_id": GENERIC_P3_TT_RUNTIME_ID,
            "status": "executed_generic_local_video",
            "source_id": source_id,
            "temporal_enabled": False,
            "tracker_backend": "tracktrack",
            "tracker_state_reused": False,
            "custom_weights": custom_weights,
            "custom_D1_weights": custom_d1,
            "custom_E1b_weights": custom_e1b,
            "stage_t_result_comparison_applicable": False,
            "stage_t_comparison_boundary": (
                "Generic runtime inputs are outside the frozen TT0/TT1 "
                "consumed-development protocol."
            ),
            "tracktrack_occupancy_improvement_claimed": False,
            "input_provenance": provenance,
            "track_output": track_summary,
            "output_files": output_files,
        }
    )
    _write_json(summary_path, stored_summary)

    runtime_path = output_root / "runtime_metadata.json"
    runtime = _read_json(runtime_path)
    runtime.update(
        {
            "runtime_id": GENERIC_P3_TT_RUNTIME_ID,
            "variant_id": "P3-TT-GENERIC",
            "source_id": source_id,
            "temporal_enabled": False,
            "tracker_backend": "tracktrack",
            "tracker_state_reused": False,
            "event_state_reused": False,
            "temporal_state_reused": False,
            "custom_weights": custom_weights,
            "custom_D1_weights": custom_d1,
            "custom_E1b_weights": custom_e1b,
            "stage_t_result_comparison_applicable": False,
            "input_provenance": provenance,
            "track_output": track_summary,
        }
    )
    _write_json(runtime_path, runtime)

    metrics_path = output_root / "metrics.json"
    if truth_path is None:
        if metrics_path.exists():
            metrics_path.unlink()
    else:
        metrics = _read_json(metrics_path)
        metrics.update(
            {
                "runtime_id": GENERIC_P3_TT_RUNTIME_ID,
                "claim_class": "user-supplied local truth evaluation",
                "stage_t_result_comparison_applicable": False,
            }
        )
        _write_json(metrics_path, metrics)
    return stored_summary
