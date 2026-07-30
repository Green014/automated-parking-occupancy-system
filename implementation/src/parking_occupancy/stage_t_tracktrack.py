from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .detector_comparison import sha256_file
from .integrated_runner import load_integrated_config, run_integrated_video


STAGE_T_PROTOCOL_ID = "STAGE-T-OPTIONAL-TRACKTRACK-VARIANT-20260729-01"
P3_TT_CONFIG_NAME = "p3_tt_tracktrack_optional_20260729.yaml"
STAGE_S_CONFIG_NAME = "p3_stage_r_recommended_default_20260729.yaml"
TRACKTRACK_CONFIG_NAME = "tracktrack_stage_m_frozen_20260728.yaml"
TRACKTRACK_CONFIG_SHA256 = (
    "54a158728a3dd41b523c7d9054fa0e187548075f563adb71c5db72e797328f37"
)
STAGE_S_CONFIG_SHA256 = (
    "198f627689cd93f66ca0f087af6686d3afc697ff51e2aa77ee56124187b981b0"
)
VIRAT_TRUTH_SHA256 = (
    "325ab6fac1970532cd86d0fc88e2b6acfd66cc8abd5ad8a70b070a318f58c760"
)
VIRAT_VIDEO_SHA256 = (
    "b522ca72eb2244b7b36a955ca6e5ff7d1d4bec39d62a069a584a72d48597a9dd"
)
D1_WEIGHTS_SHA256 = (
    "0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64"
)
E1B_CHECKPOINT_SHA256 = (
    "f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3"
)
TRACK_FIELDS = {
    "track_id",
    "bbox",
    "confidence",
    "class_id",
    "class_name",
    "assigned_slot_ids",
    "observation_index",
    "gap_from_previous_observation",
    "reacquired_after_short_gap",
    "expired_before_observation",
}


class StageTError(ValueError):
    """Raised when a Stage T protocol or output invariant is violated."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def load_p3_tt_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = load_integrated_config(path)
    if payload.get("config_id") != "P3-TT-OPTIONAL-TRACKTRACK-20260729-01":
        raise StageTError("Unexpected P3-TT config ID")
    if payload.get("variant_id") != "P3-TT":
        raise StageTError("P3-TT variant ID is required")
    if payload["temporal"].get("default_enabled") is not False:
        raise StageTError("P3-TT must disable E4 by default")
    if payload["tracking"].get("default_backend") != "tracktrack":
        raise StageTError("P3-TT must explicitly select TrackTrack")
    if payload["claims"].get("replaces_stage_s_default") is not False:
        raise StageTError("P3-TT cannot replace the Stage S default")
    if payload["claims"].get("tracktrack_occupancy_improvement_claimed") is not False:
        raise StageTError("P3-TT config cannot claim occupancy improvement")

    stage_s_path = path.parent / STAGE_S_CONFIG_NAME
    tracktrack_path = path.parent / TRACKTRACK_CONFIG_NAME
    if sha256_file(stage_s_path) != STAGE_S_CONFIG_SHA256:
        raise StageTError("Stage S final config SHA-256 changed")
    if sha256_file(tracktrack_path) != TRACKTRACK_CONFIG_SHA256:
        raise StageTError("Frozen TrackTrack config SHA-256 changed")
    stage_s = load_integrated_config(stage_s_path)
    metadata_only_keys = {
        "detector": {"role"},
        "temporal": {"availability"},
    }
    for section in ("detector", "mapping", "classifier", "fusion", "temporal"):
        ignored = metadata_only_keys.get(section, set())
        stage_t_values = {
            key: value for key, value in payload[section].items() if key not in ignored
        }
        stage_s_values = {
            key: value for key, value in stage_s[section].items() if key not in ignored
        }
        if stage_t_values != stage_s_values:
            raise StageTError(f"P3-TT changed Stage S {section} parameters")
    provenance = payload["parameter_provenance"]
    if provenance.get("source_sha256") != STAGE_S_CONFIG_SHA256:
        raise StageTError("P3-TT Stage S provenance hash mismatch")
    if provenance.get("tracktrack_source_sha256") != TRACKTRACK_CONFIG_SHA256:
        raise StageTError("P3-TT TrackTrack provenance hash mismatch")
    return payload


def _render_json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"


def _render_truth_csv(
    *,
    video_id: str,
    frame_count: int,
    fps: float,
    slots: Sequence[Mapping[str, Any]],
) -> str:
    lines = [
        "video_id,frame_index,timestamp_s,slot_id,state\n",
    ]
    for frame_index in range(frame_count):
        for slot in slots:
            state: int | None = None
            for interval in slot["intervals"]:
                if int(interval["start_frame"]) <= frame_index < int(
                    interval["end_frame"]
                ):
                    state = 1 if str(interval["state"]).lower() == "occupied" else 0
                    break
            if state is None:
                raise StageTError(
                    f"No truth interval for {slot['slot_id']} frame {frame_index}"
                )
            lines.append(
                f"{video_id},{frame_index},{frame_index / fps:.9f},"
                f"{slot['slot_id']},{state}\n"
            )
    return "".join(lines)


def _write_exact_or_validate(path: Path, content: str) -> None:
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise StageTError(f"Refusing to overwrite differing Stage T input: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def prepare_consumed_development_inputs(
    *,
    truth_yaml: Path,
    output_dir: Path,
) -> dict[str, Any]:
    truth_yaml = truth_yaml.resolve()
    if sha256_file(truth_yaml) != VIRAT_TRUTH_SHA256:
        raise StageTError("Consumed VIRAT truth SHA-256 changed")
    source = yaml.safe_load(truth_yaml.read_text(encoding="utf-8"))
    if source.get("review", {}).get("partition_role") != "development":
        raise StageTError("VIRAT 0502 must remain development-only")
    video = source["video"]
    slots = source["slots"]
    slot_payload = {
        "schema_version": 1,
        "source": (
            "project-reviewed VIRAT 0502 consumed-development slot; "
            "derived without threshold selection"
        ),
        "source_width": int(video["width"]),
        "source_height": int(video["height"]),
        "coordinate_system": "pixel",
        "slots": [
            {
                "id": str(slot["slot_id"]),
                "points": slot["polygon"],
            }
            for slot in slots
        ],
    }
    slots_path = output_dir / "virat_0502_slots.json"
    truth_csv_path = output_dir / "virat_0502_slot_truth.csv"
    _write_exact_or_validate(slots_path, _render_json(slot_payload))
    _write_exact_or_validate(
        truth_csv_path,
        _render_truth_csv(
            video_id=str(source["source_video_id"]),
            frame_count=int(video["frame_count"]),
            fps=float(video["fps"]),
            slots=slots,
        ),
    )
    metadata = {
        "schema_version": 1,
        "protocol_id": STAGE_T_PROTOCOL_ID,
        "data_role": "consumed-development diagnostic",
        "untouched_test": False,
        "source_truth_yaml": str(truth_yaml),
        "source_truth_sha256": VIRAT_TRUTH_SHA256,
        "source_video_id": str(source["source_video_id"]),
        "source_video_sha256": str(source["source_sha256"]),
        "frames": int(video["frame_count"]),
        "slots": len(slots),
        "transitions": sum(max(0, len(slot["intervals"]) - 1) for slot in slots),
        "slot_map": {
            "path": str(slots_path),
            "bytes": slots_path.stat().st_size,
            "sha256": sha256_file(slots_path),
        },
        "truth_csv": {
            "path": str(truth_csv_path),
            "bytes": truth_csv_path.stat().st_size,
            "sha256": sha256_file(truth_csv_path),
        },
    }
    metadata_path = output_dir / "STAGE_T_CONSUMED_DEVELOPMENT_INPUTS.json"
    _write_exact_or_validate(metadata_path, _render_json(metadata))
    return metadata


def _occupancy_track_assignments(
    occupancy_path: Path,
) -> dict[tuple[str, int, int], list[str]]:
    assignments: dict[tuple[str, int, int], list[str]] = defaultdict(list)
    with occupancy_path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            track_id = str(row.get("track_id", "")).strip()
            if not track_id:
                continue
            key = (row["video_id"], int(row["frame_index"]), int(track_id))
            assignments[key].append(row["slot_id"])
    return assignments


def build_track_records(
    detection_frames: Iterable[Mapping[str, Any]],
    *,
    assignments: Mapping[tuple[str, int, int], Sequence[str]] | None = None,
    track_buffer: int = 30,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if track_buffer <= 0:
        raise ValueError("track_buffer must be positive")
    assignments = assignments or {}
    last_seen: dict[tuple[str, int], int] = {}
    observations: dict[tuple[str, int], int] = defaultdict(int)
    records: list[dict[str, Any]] = []
    untracked_detections = 0
    tracked_detections = 0
    short_reacquisitions = 0
    expired_reappearances = 0
    maximum_gap = 0
    source_ids: set[str] = set()
    for frame in detection_frames:
        source_id = str(frame["video_id"])
        frame_index = int(frame["frame_index"])
        source_ids.add(source_id)
        tracks: list[dict[str, Any]] = []
        for detection in frame.get("detections", []):
            raw_track_id = detection.get("track_id")
            if raw_track_id is None or str(raw_track_id).strip() == "":
                untracked_detections += 1
                continue
            track_id = int(raw_track_id)
            key = (source_id, track_id)
            previous = last_seen.get(key)
            gap = None if previous is None else frame_index - previous
            if gap is not None and gap <= 0:
                raise StageTError("Track observations must be strictly frame ordered")
            reacquired = gap is not None and 1 < gap <= track_buffer
            expired = gap is not None and gap > track_buffer
            if reacquired:
                short_reacquisitions += 1
            if expired:
                expired_reappearances += 1
            if gap is not None:
                maximum_gap = max(maximum_gap, gap)
            observations[key] += 1
            last_seen[key] = frame_index
            tracked_detections += 1
            tracks.append(
                {
                    "track_id": track_id,
                    "bbox": [float(value) for value in detection["bbox"]],
                    "confidence": float(detection["confidence"]),
                    "class_id": int(detection["class_id"]),
                    "class_name": str(detection["class_name"]),
                    "assigned_slot_ids": sorted(
                        str(value)
                        for value in assignments.get(
                            (source_id, frame_index, track_id), ()
                        )
                    ),
                    "observation_index": observations[key],
                    "gap_from_previous_observation": gap,
                    "reacquired_after_short_gap": reacquired,
                    "expired_before_observation": expired,
                }
            )
        records.append(
            {
                "schema_version": 1,
                "variant_id": "P3-TT",
                "video_id": source_id,
                "frame_index": frame_index,
                "timestamp_s": float(frame["timestamp_s"]),
                "tracks": tracks,
            }
        )
    summary = {
        "frames": len(records),
        "sources": len(source_ids),
        "unique_source_track_ids": len(observations),
        "tracked_detections": tracked_detections,
        "untracked_detections": untracked_detections,
        "short_gap_reacquisitions": short_reacquisitions,
        "expired_id_reappearances": expired_reappearances,
        "maximum_observation_gap_frames": maximum_gap,
        "track_buffer_frames": track_buffer,
        "identity_ground_truth_available": False,
    }
    return records, summary


def validate_tracks_schema(records: Sequence[Mapping[str, Any]]) -> None:
    previous: dict[str, int] = {}
    for frame in records:
        required = {
            "schema_version",
            "variant_id",
            "video_id",
            "frame_index",
            "timestamp_s",
            "tracks",
        }
        if set(frame) != required:
            raise StageTError(f"tracks.jsonl frame schema mismatch: {set(frame)}")
        source_id = str(frame["video_id"])
        frame_index = int(frame["frame_index"])
        if source_id in previous and frame_index <= previous[source_id]:
            raise StageTError("tracks.jsonl frames are not strictly ordered")
        previous[source_id] = frame_index
        seen: set[int] = set()
        for track in frame["tracks"]:
            if set(track) != TRACK_FIELDS:
                raise StageTError("tracks.jsonl track schema mismatch")
            track_id = int(track["track_id"])
            if track_id in seen:
                raise StageTError("Duplicate track ID within one frame")
            seen.add(track_id)
            if len(track["bbox"]) != 4:
                raise StageTError("Track bbox must contain four coordinates")
            if len(track["assigned_slot_ids"]) > 1:
                raise StageTError("B1 one-to-one mapping assigned one track to many slots")


def write_tracks_jsonl(
    *,
    detections_path: Path,
    occupancy_path: Path,
    output_path: Path,
    track_buffer: int,
) -> dict[str, Any]:
    detection_frames = [
        json.loads(line)
        for line in detections_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assignments = _occupancy_track_assignments(occupancy_path)
    records, summary = build_track_records(
        detection_frames,
        assignments=assignments,
        track_buffer=track_buffer,
    )
    validate_tracks_schema(records)
    with output_path.open("x", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary["path"] = str(output_path)
    summary["bytes"] = output_path.stat().st_size
    summary["sha256"] = sha256_file(output_path)
    return summary


def run_stage_t_variant(
    *,
    variant_id: str,
    tracker_backend: str,
    input_path: Path,
    slots_path: Path,
    detector_weights: Path,
    classifier_checkpoint: Path,
    output_root: Path,
    config_path: Path,
    truth_path: Path | None,
    source_id: str,
    device: str = "auto",
    classifier_batch_size: int = 64,
) -> dict[str, Any]:
    if variant_id not in {"TT0", "TT1"}:
        raise StageTError("variant_id must be TT0 or TT1")
    expected_backend = "none" if variant_id == "TT0" else "tracktrack"
    if tracker_backend != expected_backend:
        raise StageTError(f"{variant_id} requires tracker backend {expected_backend}")
    config = load_p3_tt_config(config_path)
    for path, expected, label in (
        (input_path, VIRAT_VIDEO_SHA256, "VIRAT consumed-development video"),
        (detector_weights, D1_WEIGHTS_SHA256, "D1 weights"),
        (classifier_checkpoint, E1B_CHECKPOINT_SHA256, "E1b checkpoint"),
    ):
        if sha256_file(path.resolve()) != expected:
            raise StageTError(f"{label} SHA-256 changed")

    summary = run_integrated_video(
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
        tracker_backend=tracker_backend,
        classifier_batch_size=classifier_batch_size,
    )
    track_buffer = int(
        yaml.safe_load(
            (config_path.parent / TRACKTRACK_CONFIG_NAME).read_text(encoding="utf-8")
        )["track_buffer"]
    )
    track_summary = write_tracks_jsonl(
        detections_path=output_root / "detections.jsonl",
        occupancy_path=output_root / "occupancy.csv",
        output_path=output_root / "tracks.jsonl",
        track_buffer=track_buffer,
    )

    summary_path = output_root / "summary.json"
    summary = _read_json(summary_path)
    summary.update(
        {
            "method_id": "P3" if variant_id == "TT0" else "P3-TT",
            "method_name": (
                "D1 + B1 + F2 consumed-development baseline"
                if variant_id == "TT0"
                else "D1 + TrackTrack + B1 + F2 optional identity-enhanced variant"
            ),
            "variant_id": variant_id,
            "stage_t_protocol_id": STAGE_T_PROTOCOL_ID,
            "status": "executed_consumed_development_diagnostic",
            "data_role": "consumed-development diagnostic",
            "untouched_test": False,
            "temporal_enabled": False,
            "tracktrack_occupancy_improvement_claimed": False,
            "formal_occupancy_improvement_conclusion": "blocked",
            "output_files": [
                "occupancy.csv",
                "events.csv",
                "detections.jsonl",
                "tracks.jsonl",
                "annotated.mp4",
                "metrics.json",
                "summary.json",
                "runtime_metadata.json",
            ],
            "track_output": track_summary,
        }
    )
    _write_json(summary_path, summary)

    runtime_path = output_root / "runtime_metadata.json"
    runtime = _read_json(runtime_path)
    runtime.update(
        {
            "stage_t_protocol_id": STAGE_T_PROTOCOL_ID,
            "variant_id": variant_id,
            "temporal_enabled": False,
            "tracker_backend": tracker_backend,
            "tracktrack_frozen_config_sha256": (
                TRACKTRACK_CONFIG_SHA256 if tracker_backend == "tracktrack" else None
            ),
            "track_output": track_summary,
        }
    )
    _write_json(runtime_path, runtime)

    metrics_path = output_root / "metrics.json"
    metrics = _read_json(metrics_path)
    metrics.update(
        {
            "variant_id": variant_id,
            "claim_class": "consumed-development diagnostic",
            "untouched_test": False,
            "formal_occupancy_improvement_conclusion": "blocked",
        }
    )
    _write_json(metrics_path, metrics)
    return summary


def _binary_metrics(
    truth: Sequence[int],
    prediction: Sequence[int],
) -> dict[str, Any]:
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must be equally sized and non-empty")
    tp = sum(t == 1 and p == 1 for t, p in zip(truth, prediction, strict=True))
    tn = sum(t == 0 and p == 0 for t, p in zip(truth, prediction, strict=True))
    fp = sum(t == 0 and p == 1 for t, p in zip(truth, prediction, strict=True))
    fn = sum(t == 1 and p == 0 for t, p in zip(truth, prediction, strict=True))

    def ratio(numerator: int, denominator: int) -> float:
        return numerator / denominator if denominator else 0.0

    occupied_precision = ratio(tp, tp + fp)
    occupied_recall = ratio(tp, tp + fn)
    vacant_precision = ratio(tn, tn + fn)
    vacant_recall = ratio(tn, tn + fp)
    occupied_f1 = ratio(
        2 * occupied_precision * occupied_recall,
        occupied_precision + occupied_recall,
    )
    vacant_f1 = ratio(
        2 * vacant_precision * vacant_recall,
        vacant_precision + vacant_recall,
    )
    return {
        "samples": len(truth),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": ratio(tp + tn, len(truth)),
        "balanced_accuracy": (occupied_recall + vacant_recall) / 2.0,
        "macro_f1": (occupied_f1 + vacant_f1) / 2.0,
        "occupied_precision": occupied_precision,
        "occupied_recall": occupied_recall,
        "occupied_f1": occupied_f1,
        "vacant_precision": vacant_precision,
        "vacant_recall": vacant_recall,
        "vacant_f1": vacant_f1,
        "false_free_rate": ratio(fn, tp + fn),
        "false_occupied_rate": ratio(fp, tn + fp),
    }


def _read_state_rows(path: Path) -> dict[tuple[str, int, str], int]:
    rows: dict[tuple[str, int, str], int] = {}
    with path.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["video_id"], int(row["frame_index"]), row["slot_id"])
            if key in rows:
                raise StageTError(f"Duplicate state key: {key}")
            rows[key] = int(row["state"])
    return rows


def _state_change_frames(values: Sequence[int]) -> list[int]:
    return [
        index
        for index in range(1, len(values))
        if values[index] != values[index - 1]
    ]


def analyze_stage_t_outputs(
    *,
    truth_path: Path,
    tt0_root: Path,
    tt1_root: Path,
) -> dict[str, Any]:
    truth_rows = _read_state_rows(truth_path)
    ordered_keys = sorted(truth_rows, key=lambda key: (key[0], key[2], key[1]))
    truth = [truth_rows[key] for key in ordered_keys]
    variants: dict[str, Any] = {}
    predictions_by_variant: dict[str, list[int]] = {}
    for variant_id, root in (("TT0", tt0_root), ("TT1", tt1_root)):
        rows = _read_state_rows(root / "occupancy.csv")
        if set(rows) != set(truth_rows):
            missing = len(set(truth_rows) - set(rows))
            extra = len(set(rows) - set(truth_rows))
            raise StageTError(
                f"{variant_id} truth/prediction key mismatch missing={missing} extra={extra}"
            )
        prediction = [rows[key] for key in ordered_keys]
        predictions_by_variant[variant_id] = prediction
        metrics = _binary_metrics(truth, prediction)
        changes = _state_change_frames(prediction)
        truth_changes = _state_change_frames(truth)
        events = list(
            csv.DictReader(
                (root / "events.csv").open(encoding="utf-8", newline="")
            )
        )
        runtime = _read_json(root / "runtime_metadata.json")
        timing = runtime["timing"]["end_to_end"]
        p50_ms = float(timing["p50_ms"])
        variants[variant_id] = {
            "occupancy": metrics,
            "state_jitter": {
                "prediction_state_changes": len(changes),
                "prediction_change_frames": changes,
                "truth_state_changes": len(truth_changes),
                "truth_change_frames": truth_changes,
                "extra_state_changes_vs_truth_count": max(
                    0, len(changes) - len(truth_changes)
                ),
                "changes_per_100_frames": len(changes) / len(prediction) * 100.0,
            },
            "events": {
                "rows": len(events),
                "entry_events": sum(row["to_state"] == "1" for row in events),
                "exit_events": sum(row["to_state"] == "0" for row in events),
                "frames": [int(row["frame_index"]) for row in events],
            },
            "runtime": {
                "end_to_end_mean_ms": float(timing["mean_ms"]),
                "end_to_end_p50_ms": p50_ms,
                "end_to_end_p95_ms": float(timing["p95_ms"]),
                "mean_fps": float(timing["fps_from_mean"]),
                "steady_state_fps_p50_proxy": 1000.0 / p50_ms,
                "steady_state_definition": (
                    "inverse median per-frame end-to-end time; descriptive proxy"
                ),
            },
            "tracking": runtime["track_output"],
        }
    metric_names = (
        "macro_f1",
        "occupied_recall",
        "vacant_recall",
        "false_free_rate",
        "false_occupied_rate",
        "accuracy",
        "balanced_accuracy",
    )
    delta = {
        name: (
            variants["TT1"]["occupancy"][name]
            - variants["TT0"]["occupancy"][name]
        )
        for name in metric_names
    }
    delta.update(
        {
            "prediction_state_changes": (
                variants["TT1"]["state_jitter"]["prediction_state_changes"]
                - variants["TT0"]["state_jitter"]["prediction_state_changes"]
            ),
            "event_rows": (
                variants["TT1"]["events"]["rows"]
                - variants["TT0"]["events"]["rows"]
            ),
            "steady_state_fps_p50_proxy": (
                variants["TT1"]["runtime"]["steady_state_fps_p50_proxy"]
                - variants["TT0"]["runtime"]["steady_state_fps_p50_proxy"]
            ),
        }
    )
    same_predictions = (
        predictions_by_variant["TT0"] == predictions_by_variant["TT1"]
    )
    return {
        "schema_version": 1,
        "protocol_id": STAGE_T_PROTOCOL_ID,
        "status": "CONSUMED_DEVELOPMENT_DIAGNOSTIC_COMPLETE",
        "claim_class": "consumed-development diagnostic",
        "untouched_test": False,
        "comparison": "TT0 D1+B1+F2 versus TT1 D1+TrackTrack+B1+F2",
        "E4_included": False,
        "same_D1_B1_F2_parameters": True,
        "variants": variants,
        "TT1_minus_TT0": delta,
        "occupancy_predictions_identical": same_predictions,
        "formal_occupancy_improvement_conclusion": "blocked",
        "blocked_reason": (
            "No new continuous parking video with independent per-slot truth is "
            "available; VIRAT 0502 is already-consumed development data."
        ),
        "identity_metric_boundary": (
            "No identity ground truth is available for VIRAT 0502; TrackTrack IDs "
            "are logged descriptively, not scored as HOTA/IDF1."
        ),
    }


def write_stage_t_comparison(
    payload: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    _write_json(json_path, payload)
    fields = [
        "variant",
        "tracker",
        "macro_f1",
        "occupied_recall",
        "vacant_recall",
        "false_free_rate",
        "false_occupied_rate",
        "accuracy",
        "balanced_accuracy",
        "state_changes",
        "events",
        "steady_state_fps_p50_proxy",
        "claim_class",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for variant_id, tracker in (("TT0", "none"), ("TT1", "tracktrack")):
            row = payload["variants"][variant_id]
            writer.writerow(
                {
                    "variant": variant_id,
                    "tracker": tracker,
                    "macro_f1": f"{row['occupancy']['macro_f1']:.9f}",
                    "occupied_recall": f"{row['occupancy']['occupied_recall']:.9f}",
                    "vacant_recall": f"{row['occupancy']['vacant_recall']:.9f}",
                    "false_free_rate": f"{row['occupancy']['false_free_rate']:.9f}",
                    "false_occupied_rate": (
                        f"{row['occupancy']['false_occupied_rate']:.9f}"
                    ),
                    "accuracy": f"{row['occupancy']['accuracy']:.9f}",
                    "balanced_accuracy": (
                        f"{row['occupancy']['balanced_accuracy']:.9f}"
                    ),
                    "state_changes": row["state_jitter"]["prediction_state_changes"],
                    "events": row["events"]["rows"],
                    "steady_state_fps_p50_proxy": (
                        f"{row['runtime']['steady_state_fps_p50_proxy']:.6f}"
                    ),
                    "claim_class": payload["claim_class"],
                }
            )


def _artifact_record(
    path: Path,
    *,
    repository_root: Path,
    role: str,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(repository_root.resolve()).as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_stage_t_registry(
    registry_path: Path,
    *,
    repository_root: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("protocol_id") != STAGE_T_PROTOCOL_ID:
        errors.append("protocol_id")
    records = payload.get("artifacts", [])
    if int(payload.get("artifact_count", -1)) != len(records):
        errors.append("artifact_count")
    for record in records:
        path = repository_root / str(record["path"])
        if not path.is_file():
            errors.append(f"missing:{record['path']}")
        elif path.stat().st_size != int(record["bytes"]):
            errors.append(f"bytes:{record['path']}")
        elif sha256_file(path) != str(record["sha256"]):
            errors.append(f"sha256:{record['path']}")
    return {
        "protocol_id": STAGE_T_PROTOCOL_ID,
        "artifact_count": len(records),
        "verified": not errors,
        "errors": errors,
        "registry_bytes": registry_path.stat().st_size,
        "registry_sha256": sha256_file(registry_path),
    }


def write_stage_t_registry(
    registry_path: Path,
    *,
    repository_root: Path,
    artifacts: Iterable[tuple[Path, str]],
) -> dict[str, Any]:
    records = [
        _artifact_record(
            path,
            repository_root=repository_root,
            role=role,
        )
        for path, role in artifacts
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": STAGE_T_PROTOCOL_ID,
        "registry_id": "STAGE-T-ARTIFACT-REGISTRY-20260729-01",
        "status": "OPTIONAL_TRACKTRACK_VARIANT_COMPLETE_HASH_VERIFIED",
        "created_on": "2026-07-29",
        "artifact_count": len(records),
        "registry_self_hash_included": False,
        "model_inference_run": True,
        "inference_scope": "TT0/TT1 consumed-development diagnostic only",
        "training_or_threshold_tuning_run": False,
        "stage_s_default_modified": False,
        "formal_occupancy_improvement_conclusion": "blocked",
        "artifacts": records,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )
    result = verify_stage_t_registry(
        registry_path,
        repository_root=repository_root,
    )
    if not result["verified"]:
        raise StageTError(f"Stage T registry verification failed: {result}")
    return result
