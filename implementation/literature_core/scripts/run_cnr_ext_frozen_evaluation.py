"""Run the once-only frozen CNR-EXT external slot-occupancy evaluation."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.calibration import (  # noqa: E402
    CalibratedFusionModel,
    NonnegativeLogisticFusion,
    PlattCalibrator,
    calibration_metrics,
    reliability_bins,
)
from literature_core.classifier import MobileNetSlotClassifier  # noqa: E402
from literature_core.cnrpark import (  # noqa: E402
    CNRSlotLabel,
    load_cnr_ext_boxes,
    load_cnr_ext_metadata,
)
from literature_core.config import load_yaml  # noqa: E402
from literature_core.data import read_image  # noqa: E402
from literature_core.detector import (  # noqa: E402
    ClosedSetYOLODetector,
    YOLOWorldDetector,
)
from literature_core.mapping import map_detections_to_slots  # noqa: E402
from literature_core.metrics import (  # noqa: E402
    evaluate_probabilities,
    grouped_bootstrap_binary_metrics,
)
from literature_core.patches import extract_slot_patch  # noqa: E402


def resolve_project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def frozen_fusion_model(payload: dict[str, Any]) -> CalibratedFusionModel:
    return CalibratedFusionModel(
        PlattCalibrator.from_dict(payload["calibration"]["classifier"]),
        PlattCalibrator.from_dict(payload["calibration"]["detector"]),
        NonnegativeLogisticFusion.from_dict(payload["fusion"]),
    )


def frame_path(dataset_root: Path, labels: tuple[CNRSlotLabel, ...]) -> Path:
    paths = {label.relative_frame_path for label in labels}
    if len(paths) != 1:
        raise ValueError("One CNR group resolved to multiple full-frame paths")
    return dataset_root / next(iter(paths))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def method_metrics(
    records: list[dict[str, Any]],
    *,
    probability_key: str,
    threshold: float,
    iterations: int,
    seed: int,
) -> dict[str, Any]:
    truth = [int(record["truth"]) for record in records]
    probabilities = [float(record[probability_key]) for record in records]
    predictions = [int(value >= threshold) for value in probabilities]
    point = evaluate_probabilities(truth, probabilities, threshold)
    confidence_intervals = grouped_bootstrap_binary_metrics(
        truth,
        predictions,
        [str(record["group_id"]) for record in records],
        iterations=iterations,
        seed=seed,
    )
    return {
        **point,
        "bootstrap_unit": "complete camera_datetime image",
        "bootstrap_iterations": iterations,
        "confidence_intervals_95": confidence_intervals,
    }


def error_overlap(
    records: list[dict[str, Any]],
    thresholds: dict[str, float],
) -> dict[str, int]:
    counts = defaultdict(int)
    for record in records:
        truth = int(record["truth"])
        classifier_correct = (
            int(record["p_e1a"] >= thresholds["E1a"]) == truth
        )
        detector_correct = int(record["p_e2"] >= thresholds["E2"]) == truth
        e3a_correct = int(record["p_e3a"] >= thresholds["E3a"]) == truth
        e3b_correct = int(record["p_e3b"] >= thresholds["E3b"]) == truth
        category = (
            "both_branches_correct"
            if classifier_correct and detector_correct
            else "classifier_only_correct"
            if classifier_correct
            else "detector_only_correct"
            if detector_correct
            else "both_branches_wrong"
        )
        counts[category] += 1
        if e3a_correct and not (classifier_correct and detector_correct):
            counts["e3a_rescues_branch_error"] += 1
        if e3b_correct and not (classifier_correct and detector_correct):
            counts["e3b_rescues_branch_error"] += 1
        if not e3a_correct and classifier_correct and detector_correct:
            counts["e3a_wrong_when_both_branches_correct"] += 1
        if not e3b_correct and classifier_correct and detector_correct:
            counts["e3b_wrong_when_both_branches_correct"] += 1
    return dict(counts)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "external_holdout_frozen.yaml",
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup-frames", type=int, default=8)
    args = parser.parse_args()

    if args.output_dir.exists():
        raise FileExistsError(
            "External evaluation output already exists; choose a new directory "
            "so no prior result is overwritten."
        )
    if args.warmup_frames < 1:
        raise ValueError("At least one complete frame must be excluded as warm-up")
    config = load_yaml(args.config)
    if config.get("status") != "frozen_before_external_prediction":
        raise ValueError("External protocol is not marked frozen")
    args.output_dir.mkdir(parents=True)

    dataset_config = config["dataset"]
    metadata = load_cnr_ext_metadata(
        resolve_project_path(dataset_config["metadata_file"]),
        cameras=dataset_config["include_cameras"],
    )
    boxes = load_cnr_ext_boxes(args.dataset_root)
    if len(metadata) < 100:
        raise ValueError("External evaluation requires at least 100 image groups")

    e1a_config = config["classifiers"]["e1a"]
    e1b_config = config["classifiers"]["e1b"]
    e1a = MobileNetSlotClassifier(
        resolve_project_path(e1a_config["checkpoint"]),
        device=args.device,
    )
    e1b = MobileNetSlotClassifier(
        resolve_project_path(e1b_config["checkpoint"]),
        device=args.device,
    )
    detector_config = config["detectors"]
    world = YOLOWorldDetector(
        resolve_project_path(detector_config["e2"]["weights"]),
        prompts=detector_config["e2"]["prompts"],
        confidence=float(detector_config["confidence"]),
        image_size=int(detector_config["image_size"]),
        device=args.device,
    )
    baseline = ClosedSetYOLODetector(
        resolve_project_path(detector_config["e0"]["weights"]),
        class_ids=detector_config["class_ids"],
        confidence=float(detector_config["confidence"]),
        image_size=int(detector_config["image_size"]),
        device=args.device,
    )
    proposed_payload = load_yaml(
        resolve_project_path(config["fusion"]["e3b"]["config"])
    )
    proposed = frozen_fusion_model(proposed_payload)

    e3a_config = config["fusion"]["e3a"]
    thresholds = {
        "E0": 1e-12,
        "E1a": float(e1a_config["occupied_threshold"]),
        "E1b": float(e1b_config["occupied_threshold"]),
        "E2": float(detector_config["e2"]["occupied_evidence_threshold"]),
        "E3a": float(e3a_config["occupied_threshold"]),
        "E3b": float(proposed_payload["fusion"]["occupied_threshold"]),
    }
    records: list[dict[str, Any]] = []
    timing_totals = defaultdict(float)
    timed_frames = 0
    missing_slots: list[dict[str, str]] = []
    raw_path = args.output_dir / "frame_results.jsonl"

    with raw_path.open("x", encoding="utf-8") as raw_log:
        for frame_index, (group_id, labels) in enumerate(metadata.items()):
            path = frame_path(args.dataset_root, labels)
            image = read_image(path)
            camera = labels[0].camera
            camera_slots = boxes.get(camera)
            if camera_slots is None:
                raise ValueError(f"No official geometry for camera {camera}")
            slot_pairs = []
            for label in labels:
                slot = camera_slots.get(label.slot_id)
                if slot is None:
                    missing_slots.append(
                        {
                            "group_id": group_id,
                            "camera": camera,
                            "slot_id": label.slot_id,
                        }
                    )
                    continue
                slot_pairs.append((label, slot))
            if len(slot_pairs) != len(labels):
                continue
            slots = tuple(slot for _, slot in slot_pairs)

            start_frame = time.perf_counter()
            patches = [
                extract_slot_patch(
                    image,
                    slot.points,
                    perspective_warp=False,
                )
                for slot in slots
            ]
            start = time.perf_counter()
            p_e1a = e1a.predict_patches(patches)
            classifier_e1a_s = time.perf_counter() - start
            start = time.perf_counter()
            p_e1b = e1b.predict_patches(patches)
            classifier_e1b_s = time.perf_counter() - start

            start = time.perf_counter()
            world_detections = world.detect(image)
            world_s = time.perf_counter() - start
            world_evidence = map_detections_to_slots(
                world_detections,
                slots,
                minimum_slot_coverage=float(
                    detector_config["e2"]["minimum_mapping_coverage"]
                ),
                one_to_one=True,
            )
            start = time.perf_counter()
            baseline_detections = baseline.detect(image)
            baseline_s = time.perf_counter() - start
            baseline_evidence = map_detections_to_slots(
                baseline_detections,
                slots,
                minimum_slot_coverage=float(
                    detector_config["e0"]["minimum_slot_coverage"]
                ),
                one_to_one=True,
            )
            p_e0 = [evidence.probability for evidence in baseline_evidence]
            p_e2 = [evidence.probability for evidence in world_evidence]
            p_e3a = [
                float(e3a_config["classifier_weight"]) * classifier_score
                + float(e3a_config["detector_weight"]) * detector_score
                for classifier_score, detector_score in zip(
                    p_e1a,
                    p_e2,
                    strict=True,
                )
            ]
            p_cls_calibrated, p_det_calibrated = proposed.predict_branches(
                p_e1a,
                p_e2,
            )
            p_e3b = proposed.fusion.predict(
                p_cls_calibrated,
                p_det_calibrated,
            )
            frame_s = time.perf_counter() - start_frame
            timed = frame_index >= args.warmup_frames
            if timed:
                timed_frames += 1
                timing_totals["end_to_end_s"] += frame_s
                timing_totals["classifier_e1a_s"] += classifier_e1a_s
                timing_totals["classifier_e1b_s"] += classifier_e1b_s
                timing_totals["yolo_world_s"] += world_s
                timing_totals["yolov8_s"] += baseline_s

            slot_rows = []
            for (
                label,
                slot,
            ), e0_score, e1a_score, e1b_score, e2_score, cls_cal, det_cal, e3a_score, e3b_score in zip(
                slot_pairs,
                p_e0,
                p_e1a,
                p_e1b,
                p_e2,
                p_cls_calibrated,
                p_det_calibrated,
                p_e3a,
                p_e3b,
                strict=True,
            ):
                row = {
                    "group_id": group_id,
                    "camera": camera,
                    "datetime": label.datetime,
                    "weather": label.weather,
                    "image_path": str(path),
                    "slot_id": label.slot_id,
                    "truth": label.occupancy,
                    "slot_points": slot.points,
                    "p_e0": e0_score,
                    "p_e1a": e1a_score,
                    "p_e1b": e1b_score,
                    "p_e2": e2_score,
                    "p_cls_calibrated": cls_cal,
                    "p_det_calibrated": det_cal,
                    "p_e3a": e3a_score,
                    "p_e3b": e3b_score,
                }
                records.append(row)
                slot_rows.append(row)
            raw_log.write(
                json.dumps(
                    {
                        "frame_index": frame_index,
                        "group_id": group_id,
                        "timed_after_warmup": timed,
                        "timing_s": {
                            "end_to_end": frame_s,
                            "classifier_e1a": classifier_e1a_s,
                            "classifier_e1b": classifier_e1b_s,
                            "yolo_world": world_s,
                            "yolov8": baseline_s,
                        },
                        "yolo_world": [
                            asdict(detection) for detection in world_detections
                        ],
                        "yolov8": [
                            asdict(detection) for detection in baseline_detections
                        ],
                        "slots": slot_rows,
                    }
                )
                + "\n"
            )
            raw_log.flush()
            if (frame_index + 1) % 100 == 0:
                print(
                    json.dumps(
                        {
                            "processed_frames": frame_index + 1,
                            "total_frames": len(metadata),
                            "slot_records": len(records),
                        }
                    ),
                    flush=True,
                )

    if missing_slots:
        write_csv(args.output_dir / "missing_geometry.csv", missing_slots)
        raise ValueError(
            f"{len(missing_slots)} labels had no official camera geometry; "
            "external metrics were not generated"
        )
    if timed_frames < 100:
        raise ValueError("Post-warm-up timing did not cover at least 100 frames")

    iterations = int(dataset_config["bootstrap_resamples"])
    seed = int(dataset_config["bootstrap_seed"])
    probability_keys = {
        "E0": "p_e0",
        "E1a": "p_e1a",
        "E1b": "p_e1b",
        "E2": "p_e2",
        "E3a": "p_e3a",
        "E3b": "p_e3b",
    }
    overall = {
        method: method_metrics(
            records,
            probability_key=probability_key,
            threshold=thresholds[method],
            iterations=iterations,
            seed=seed,
        )
        for method, probability_key in probability_keys.items()
    }
    by_camera = {}
    for camera in sorted({str(record["camera"]) for record in records}):
        camera_records = [
            record for record in records if record["camera"] == camera
        ]
        by_camera[camera] = {
            method: evaluate_probabilities(
                [int(record["truth"]) for record in camera_records],
                [
                    float(record[probability_key])
                    for record in camera_records
                ],
                thresholds[method],
            )
            for method, probability_key in probability_keys.items()
        }

    truth = [int(record["truth"]) for record in records]
    calibration = {
        "p_cls_raw": calibration_metrics(
            truth,
            [float(record["p_e1a"]) for record in records],
        ),
        "p_cls_calibrated": calibration_metrics(
            truth,
            [float(record["p_cls_calibrated"]) for record in records],
        ),
        "p_det_raw_evidence_as_score": calibration_metrics(
            truth,
            [float(record["p_e2"]) for record in records],
        ),
        "p_det_calibrated": calibration_metrics(
            truth,
            [float(record["p_det_calibrated"]) for record in records],
        ),
        "p_e3a_raw_weighted": calibration_metrics(
            truth,
            [float(record["p_e3a"]) for record in records],
        ),
        "p_e3b_calibrated_fusion": calibration_metrics(
            truth,
            [float(record["p_e3b"]) for record in records],
        ),
    }
    reliability_rows = []
    for label, key in (
        ("p_cls_raw", "p_e1a"),
        ("p_cls_calibrated", "p_cls_calibrated"),
        ("p_det_raw_evidence_as_score", "p_e2"),
        ("p_det_calibrated", "p_det_calibrated"),
        ("p_e3a_raw_weighted", "p_e3a"),
        ("p_e3b_calibrated_fusion", "p_e3b"),
    ):
        for row in reliability_bins(
            truth,
            [float(record[key]) for record in records],
        ):
            reliability_rows.append({"score": label, **row})

    timing = {
        "warmup_frames_excluded": args.warmup_frames,
        "post_warmup_timed_frames": timed_frames,
        **dict(timing_totals),
        "end_to_end_frames_per_s": (
            timed_frames / timing_totals["end_to_end_s"]
        ),
        "end_to_end_ms_per_frame": (
            timing_totals["end_to_end_s"] / timed_frames * 1000
        ),
    }
    report = {
        "protocol": {
            "role": "external_holdout_once_only",
            "config": str(args.config.resolve()),
            "configuration_status": config["status"],
            "external_data_used_for_selection": False,
            "slot_level_random_bootstrap": False,
            "geometry": "official axis-aligned boxes scaled to released frames",
            "geometry_is_precise_polygon": False,
            "threshold_sensitivity_computed_on_external": False,
        },
        "integrity": {
            "frames": len(metadata),
            "slot_records": len(records),
            "cameras": sorted({record["camera"] for record in records}),
            "missing_geometry": 0,
        },
        "thresholds": thresholds,
        "overall": overall,
        "by_camera": by_camera,
        "calibration": calibration,
        "error_overlap": error_overlap(records, thresholds),
        "timing": timing,
        "classifier_e1a": e1a.metadata(),
        "classifier_e1b": e1b.metadata(),
        "yolo_world": world.metadata(),
        "baseline_yolo": baseline.metadata(),
    }
    write_csv(args.output_dir / "predictions.csv", records)
    write_csv(args.output_dir / "reliability_curves.csv", reliability_rows)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
