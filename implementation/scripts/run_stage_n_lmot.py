from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
import time
from pathlib import Path
from typing import Any

import cv2
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import (
    FrozenStageNTrackerAdapter,
    OfficialTrackEvalAdapter,
    StageNInferenceSettings,
    VerifiedLmotClassMap,
    audit_lmot_sequence,
    evaluate_motor_vehicle_detections,
    load_stage_n_protocol,
    parse_lmot_gt,
    read_image,
    sha256_file,
    split_motor_vehicle_truth,
    suppress_predictions_on_excluded_truth,
    write_image,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_n_lmot_tracking_diagnostic_frozen_20260728.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen Stage N LMOT L0-L3 diagnostics"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def _load_class_map(path: Path) -> tuple[VerifiedLmotClassMap, frozenset[int]]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = VerifiedLmotClassMap(
        id_to_name={
            int(key): value for key, value in payload["id_to_name"].items()
        },
        verification_status=payload["verification_status"],
        evidence=payload["evidence"],
        evidence_sha256=payload.get("evidence_sha256"),
    )
    if mapping.verification_status != "official_verified":
        raise ValueError("LMOT run requires an official verified class map")
    return mapping, frozenset(
        int(value) for value in payload["evaluated_ignore_values"]
    )


def _frame_paths(path: Path) -> list[Path]:
    accepted = {".jpg", ".jpeg", ".png", ".bmp"}
    return sorted(
        (row for row in path.iterdir() if row.suffix.lower() in accepted),
        key=lambda row: int(row.stem),
    )


def _row_payload(row) -> dict[str, Any]:
    return {
        "frame": row.frame_number,
        "track_id": row.track_id,
        "bbox_xyxy": list(row.xyxy),
        "confidence": row.confidence,
        "class_id": 0,
        "class_name": "motor_vehicle",
    }


def _run_one(
    *,
    method_id: str,
    sequence_root: Path,
    image_directory: str,
    tracker_path: Path,
    settings: StageNInferenceSettings,
    evaluated_gt,
    suppression_gt,
    output_root: Path,
) -> tuple[dict[str, Any], list, list]:
    adapter = FrozenStageNTrackerAdapter(
        settings, tracker_config=tracker_path
    )
    detections = []
    tracks = []
    latencies_ms: list[float] = []
    first_frame = None
    for image_path in _frame_paths(sequence_root / image_directory):
        frame = read_image(image_path)
        if frame is None:
            raise RuntimeError(f"Could not decode {image_path}")
        if first_frame is None:
            first_frame = frame.copy()
        started = time.perf_counter()
        frame_tracks = adapter.track(frame)
        latencies_ms.append((time.perf_counter() - started) * 1000.0)
        detections.extend(adapter.last_detections)
        tracks.extend(frame_tracks)
    kept_detections, suppressed_detections = (
        suppress_predictions_on_excluded_truth(
            detections, evaluated_gt, suppression_gt
        )
    )
    kept_tracks, suppressed_tracks = suppress_predictions_on_excluded_truth(
        tracks, evaluated_gt, suppression_gt
    )
    detection_metrics = evaluate_motor_vehicle_detections(
        gt=evaluated_gt, predictions=kept_detections
    )
    tracking_metrics = OfficialTrackEvalAdapter().evaluate_sequence(
        num_timesteps=len(latencies_ms),
        gt=evaluated_gt,
        predictions=kept_tracks,
    )
    sequence_name = sequence_root.name
    detection_path = (
        output_root / "detections" / method_id / f"{sequence_name}.jsonl"
    )
    track_path = (
        output_root / "tracks" / method_id / f"{sequence_name}.jsonl"
    )
    detection_path.parent.mkdir(parents=True, exist_ok=True)
    track_path.parent.mkdir(parents=True, exist_ok=True)
    detection_path.write_text(
        "\n".join(json.dumps(_row_payload(row)) for row in kept_detections)
        + ("\n" if kept_detections else ""),
        encoding="utf-8",
    )
    track_path.write_text(
        "\n".join(json.dumps(_row_payload(row)) for row in kept_tracks)
        + ("\n" if kept_tracks else ""),
        encoding="utf-8",
    )
    if first_frame is not None:
        for row in kept_tracks:
            if row.frame_number != 1:
                continue
            x1, y1, x2, y2 = (int(value) for value in row.xyxy)
            cv2.rectangle(first_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                first_frame,
                str(row.track_id),
                (x1, max(10, y1 - 3)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        qualitative = (
            output_root
            / "qualitative_frames"
            / method_id
            / f"{sequence_name}_f000001.jpg"
        )
        qualitative.parent.mkdir(parents=True, exist_ok=True)
        write_image(qualitative, first_frame)
    steady = latencies_ms[min(5, len(latencies_ms)) :]
    if not steady:
        steady = latencies_ms
    runtime = {
        "frames": len(latencies_ms),
        "latency_ms_all_mean": statistics.fmean(latencies_ms),
        "steady_state_latency_ms": statistics.fmean(steady),
        "steady_state_fps": 1000.0 / statistics.fmean(steady),
        "warmup_frames_excluded": min(5, len(latencies_ms)),
    }
    metrics = {
        "method_id": method_id,
        "sequence": sequence_name,
        "illumination": (
            "well_lit_RGB"
            if image_directory == "img_light_rgb"
            else "low_light_RGB"
        ),
        "detection": detection_metrics,
        "tracking": tracking_metrics,
        "runtime": runtime,
        "suppressed_predictions": {
            "detections": suppressed_detections,
            "tracks": suppressed_tracks,
        },
    }
    return metrics, kept_tracks, kept_detections


def _aggregate(
    sequence_metrics: dict[str, dict[str, dict[str, Any]]],
    official_tracking_aggregates: dict[str, dict[str, float | int]],
) -> dict[str, Any]:
    output: dict[str, Any] = {"methods": {}}
    for method_id, sequences in sequence_metrics.items():
        rows = list(sequences.values())
        detection_count_keys = (
            "ground_truth_boxes",
            "predicted_boxes",
            "true_positives",
            "false_positives",
            "false_negatives",
        )
        detection_counts = {
            key: sum(int(row["detection"].get(key, 0)) for row in rows)
            for key in detection_count_keys
        }
        detection_micro = {
            **detection_counts,
            "precision": detection_counts["true_positives"]
            / max(
                detection_counts["true_positives"]
                + detection_counts["false_positives"],
                1,
            ),
            "recall": detection_counts["true_positives"]
            / max(detection_counts["ground_truth_boxes"], 1),
        }
        detection_macro = {
            key: statistics.fmean(
                float(row["detection"][key]) for row in rows
            )
            for key in ("precision", "recall", "AP50", "AP50-95")
        }
        runtime = {
            key: statistics.fmean(
                float(row["runtime"][key]) for row in rows
            )
            for key in rows[0]["runtime"]
            if isinstance(rows[0]["runtime"][key], (int, float))
            and key not in {"frames", "warmup_frames_excluded"}
        }
        runtime["frames"] = sum(
            int(row["runtime"]["frames"]) for row in rows
        )
        runtime["warmup_frames_excluded"] = sum(
            int(row["runtime"]["warmup_frames_excluded"]) for row in rows
        )
        output["methods"][method_id] = {
            "detection": detection_micro,
            "detection_per_sequence_macro": detection_macro,
            "detection_aggregation": {
                "primary": "micro_from_summed_iou_0.50_counts",
                "counts": "sum_across_sequences",
                "ap_note": (
                    "AP requires pooled confidence ordering and is therefore "
                    "reported only in detection_per_sequence_macro here"
                ),
            },
            "runtime": runtime,
        }
        output["methods"][method_id]["sequence_count"] = len(rows)
        output["methods"][method_id]["tracking"] = (
            official_tracking_aggregates[method_id]
        )
    deltas: dict[str, Any] = {}
    for tracker, light, dark in (
        ("ByteTrack", "L0", "L2"),
        ("TrackTrack", "L1", "L3"),
    ):
        deltas[f"{tracker}_dark_minus_light"] = {
            "tracking": {
                metric: (
                    output["methods"][dark]["tracking"][metric]
                    - output["methods"][light]["tracking"][metric]
                )
                for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
            },
            "detection": {
                metric: (
                    output["methods"][dark][section][metric]
                    - output["methods"][light][section][metric]
                )
                for section, metrics in (
                    ("detection", ("precision", "recall")),
                    (
                        "detection_per_sequence_macro",
                        ("AP50", "AP50-95"),
                    ),
                )
                for metric in metrics
            },
        }
    for light, byte, track in (
        ("well_lit_RGB", "L0", "L1"),
        ("low_light_RGB", "L2", "L3"),
    ):
        deltas[f"TrackTrack_minus_ByteTrack_{light}"] = {
            metric: (
                output["methods"][track]["tracking"][metric]
                - output["methods"][byte]["tracking"][metric]
            )
            for metric in ("HOTA", "DetA", "AssA", "IDF1", "MOTA")
        }
    output["paired_deltas"] = deltas
    output["comparison_label"] = (
        "controlled_end_to_end_tracker_backend_comparison"
    )
    return output


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    protocol = load_stage_n_protocol(config_path, verify_files=True)
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    mapping, evaluated_values = _load_class_map(args.class_map.resolve())
    shared = protocol["shared_inference"]
    weights = Path(shared["weights_path"]).resolve()
    if sha256_file(weights) != shared["weights_sha256"]:
        raise ValueError("D1 weights SHA-256 mismatch")
    settings = StageNInferenceSettings(
        weights=str(weights),
        device=args.device,
    )
    tracker_paths = {
        "ByteTrack": (config_path.parent / protocol["trackers"]["bytetrack"]["config_path"]).resolve(),
        "TrackTrack": (config_path.parent / protocol["trackers"]["tracktrack"]["config_path"]).resolve(),
    }
    for name, path in tracker_paths.items():
        if sha256_file(path) != protocol["trackers"][name.lower()]["config_sha256"]:
            raise ValueError(f"{name} config SHA-256 mismatch")
    sequences = sorted(
        path
        for path in args.validation_root.resolve().iterdir()
        if path.is_dir()
    )
    if not sequences:
        raise ValueError("No validation sequences found")
    for sequence in sequences:
        audit = audit_lmot_sequence(sequence)
        if not audit["passed"]:
            raise ValueError(f"Sequence audit failed: {sequence.name}")
    output_root.mkdir(parents=True)
    shutil.copyfile(config_path, output_root / "configuration_snapshot.yaml")
    sequence_metrics: dict[str, dict[str, dict[str, Any]]] = {
        method_id: {} for method_id in protocol["methods"]
    }
    trackeval_inputs: dict[str, dict[str, tuple[int, Any, Any]]] = {
        method_id: {} for method_id in protocol["methods"]
    }
    for sequence in sequences:
        annotations = parse_lmot_gt(sequence / "gt" / "gt.txt")
        evaluated_gt, suppression_gt = split_motor_vehicle_truth(
            annotations,
            class_map=mapping,
            evaluated_ignore_values=evaluated_values,
        )
        for method_id, method in protocol["methods"].items():
            metrics, tracks, _detections = _run_one(
                method_id=method_id,
                sequence_root=sequence,
                image_directory=method["image_directory"],
                tracker_path=tracker_paths[method["tracker"]],
                settings=settings,
                evaluated_gt=evaluated_gt,
                suppression_gt=suppression_gt,
                output_root=output_root,
            )
            sequence_metrics[method_id][sequence.name] = metrics
            trackeval_inputs[method_id][sequence.name] = (
                int(metrics["runtime"]["frames"]),
                evaluated_gt,
                tracks,
            )
    official_tracking_aggregates = {}
    for method_id, method_inputs in trackeval_inputs.items():
        _per_sequence, official_tracking_aggregates[method_id] = (
            OfficialTrackEvalAdapter().evaluate_many(method_inputs)
        )
    aggregate = _aggregate(
        sequence_metrics, official_tracking_aggregates
    )
    (output_root / "sequence_metrics.json").write_text(
        json.dumps(sequence_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    runtime = {
        "claim_scope": "LMOT validation上的低光多目标跟踪诊断",
        "python": platform.python_version(),
        "trackeval": OfficialTrackEvalAdapter.runtime_metadata(),
        "config_sha256": sha256_file(config_path),
        "class_map_sha256": sha256_file(args.class_map),
        "D1_weights_sha256": sha256_file(weights),
        "actual_lmot_run": True,
        "parameter_tuning_from_results": False,
    }
    (output_root / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
