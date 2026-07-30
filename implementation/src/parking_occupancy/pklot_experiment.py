from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector import UltralyticsDetector
from .evaluate import (
    _plot_confusion,
    _plot_pr,
    binary_metrics,
    precision_recall_curve,
)
from .geometry import (
    map_detections_to_slots,
    point_in_slot,
    slot_overlap_score,
)
from .image_io import read_image, write_image
from .models import Detection, ParkingSlot


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _slots_and_truth(
    record: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[ParkingSlot], dict[str, int | None]]:
    slots: list[ParkingSlot] = []
    truth: dict[str, int | None] = {}
    for polygon in record["sample"]["parking_spaces"]["polylines"]:
        slot_id = f"slot_{int(polygon['space_id']):03d}"
        points = tuple(
            (float(point[0]) * width, float(point[1]) * height)
            for point in polygon["points"][0]
        )
        slots.append(ParkingSlot(slot_id=slot_id, points=points))
        status = str(polygon.get("occupancy_status", "unknown"))
        truth[slot_id] = (
            1 if status == "occupied" else 0 if status == "not occupied" else None
        )
    return slots, truth


def _cache_detections(
    records: list[dict[str, Any]],
    project_root: Path,
    cache_path: Path,
    detector: UltralyticsDetector,
) -> tuple[dict[str, list[Detection]], dict[str, Any]]:
    detections_by_sample: dict[str, list[Detection]] = {}
    sample_rows: list[dict[str, Any]] = []
    inference_times_ms: list[float] = []
    for record in records:
        image_path = project_root / record["local_path"]
        frame = read_image(image_path)
        start = time.perf_counter()
        detections = detector.detect(frame)
        inference_times_ms.append((time.perf_counter() - start) * 1000.0)
        detections_by_sample[record["sample_id"]] = detections
        sample_rows.append(
            {
                "type": "sample",
                "sample_id": record["sample_id"],
                "image_path": str(image_path.resolve()),
                "image_sha256": _sha256(image_path),
                "detections": [asdict(detection) for detection in detections],
            }
        )

    metadata = detector.metadata()
    metadata.update(
        {
            "samples": len(records),
            "total_detections": sum(map(len, detections_by_sample.values())),
            "inference_timing_ms": {
                "mean": statistics.fmean(inference_times_ms),
                "p50": statistics.median(inference_times_ms),
                "steady_state_fps": (
                    1000.0 / statistics.fmean(inference_times_ms[1:])
                    if len(inference_times_ms) > 1
                    else 0.0
                ),
            },
        }
    )
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps({"type": "metadata", **metadata}) + "\n")
        for row in sample_rows:
            handle.write(json.dumps(row) + "\n")
    return detections_by_sample, metadata


def _load_detection_cache(
    cache_path: Path,
) -> tuple[dict[str, list[Detection]], dict[str, Any]]:
    rows = _read_jsonl(cache_path)
    if not rows or rows[0].get("type") != "metadata":
        raise ValueError("Detection cache has no metadata header")
    metadata = {key: value for key, value in rows[0].items() if key != "type"}
    detections_by_sample = {
        row["sample_id"]: [
            Detection(
                bbox=tuple(item["bbox"]),
                confidence=float(item["confidence"]),
                class_id=int(item["class_id"]),
                class_name=str(item["class_name"]),
                track_id=item.get("track_id"),
            )
            for item in row["detections"]
        ]
        for row in rows[1:]
        if row.get("type") == "sample"
    }
    return detections_by_sample, metadata


def _subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    y_true = [int(row["truth"]) for row in rows]
    y_pred = [int(row["prediction"]) for row in rows]
    evidence = [float(row["evidence"]) for row in rows]
    metrics = binary_metrics(y_true, y_pred)
    _precision, _recall, _thresholds, ap = precision_recall_curve(
        y_true,
        evidence,
    )
    metrics["slot_average_precision"] = ap
    return metrics


def _stratified_metrics(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[field])].append(row)
    return {
        group: _subset_metrics(group_rows)
        for group, group_rows in sorted(groups.items())
    }


def _error_cause(
    experiment: str,
    truth: int,
    prediction: int,
    any_detection_center: bool,
    max_slot_overlap: float,
    overlap_threshold: float,
) -> str:
    if truth == prediction:
        return ""
    if prediction == 1:
        return "mapping_false_occupied"
    if max_slot_overlap < 0.10:
        return "detector_miss_or_severe_localization"
    if experiment == "b0" and not any_detection_center:
        return "centre_mapping_failure"
    if experiment == "b1" and max_slot_overlap < overlap_threshold:
        return "overlap_threshold_failure"
    return "one_to_one_assignment_conflict"


def _render_failures(
    records: list[dict[str, Any]],
    project_root: Path,
    detections_by_sample: dict[str, list[Detection]],
    rows_by_experiment: dict[str, list[dict[str, Any]]],
    output_dir: Path,
    limit: int,
) -> None:
    records_by_id = {record["sample_id"]: record for record in records}
    for experiment, rows in rows_by_experiment.items():
        errors_per_sample: dict[str, int] = defaultdict(int)
        for row in rows:
            if row["truth"] != row["prediction"]:
                errors_per_sample[row["sample_id"]] += 1
        selected = sorted(
            errors_per_sample,
            key=lambda sample_id: (-errors_per_sample[sample_id], sample_id),
        )[:limit]
        target_dir = output_dir / "failure_cases" / experiment
        target_dir.mkdir(parents=True, exist_ok=True)
        rows_by_sample: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
        for row in rows:
            rows_by_sample[row["sample_id"]][row["slot_id"]] = row

        for sample_id in selected:
            record = records_by_id[sample_id]
            image = read_image(project_root / record["local_path"])
            height, width = image.shape[:2]
            slots, _truth = _slots_and_truth(record, width, height)
            overlay = image.copy()
            for slot in slots:
                row = rows_by_sample[sample_id].get(slot.slot_id)
                if row is None:
                    continue
                expected = int(row["truth"])
                predicted = int(row["prediction"])
                if expected == predicted:
                    color = (0, 150, 0) if expected == 0 else (180, 120, 0)
                else:
                    color = (0, 0, 255) if expected == 1 else (255, 0, 255)
                contour = np.asarray(slot.points, dtype=np.int32)
                cv2.polylines(overlay, [contour], True, color, 2)
            for detection in detections_by_sample[sample_id]:
                x1, y1, x2, y2 = (int(value) for value in detection.bbox)
                cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 220, 255), 1)
            title = (
                f"{experiment.upper()} {sample_id} "
                f"errors={errors_per_sample[sample_id]} "
                f"{record['source']}/{record['weather']}"
            )
            cv2.rectangle(overlay, (0, 0), (width, 30), (0, 0, 0), -1)
            cv2.putText(
                overlay,
                title,
                (8, 21),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            write_image(target_dir / f"{sample_id}.jpg", overlay)


def run_experiment(
    annotations_path: Path,
    output_dir: Path,
    weights: str,
    confidence: float,
    image_size: int,
    device: str,
    overlap_threshold: float,
    reuse_cache: bool,
    failure_limit: int,
    detection_cache: Path | None = None,
) -> dict[str, Any]:
    annotations_path = annotations_path.resolve()
    project_root = annotations_path.parents[2]
    records = _read_jsonl(annotations_path)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        detection_cache.resolve()
        if detection_cache is not None
        else output_dir / "detections.jsonl"
    )
    if reuse_cache and cache_path.is_file():
        detections_by_sample, detector_metadata = _load_detection_cache(cache_path)
        expected = {
            "weights": weights,
            "confidence": confidence,
            "image_size": image_size,
        }
        mismatch = {
            key: (detector_metadata.get(key), value)
            for key, value in expected.items()
            if detector_metadata.get(key) != value
        }
        if mismatch:
            raise ValueError(f"Detection cache configuration mismatch: {mismatch}")
    else:
        detector = UltralyticsDetector(
            weights=weights,
            confidence=confidence,
            image_size=image_size,
            device=device,
        )
        detections_by_sample, detector_metadata = _cache_detections(
            records,
            project_root,
            cache_path,
            detector,
        )

    rows_by_experiment: dict[str, list[dict[str, Any]]] = {
        "b0": [],
        "b1": [],
    }
    failure_rows: list[dict[str, Any]] = []
    for record in records:
        image = read_image(project_root / record["local_path"])
        height, width = image.shape[:2]
        slots, truth = _slots_and_truth(record, width, height)
        detections = detections_by_sample[record["sample_id"]]
        diagnostics = {
            slot.slot_id: {
                "any_detection_center": any(
                    point_in_slot(detection.center, slot)
                    for detection in detections
                ),
                "max_slot_overlap": max(
                    (
                        slot_overlap_score(detection, slot)
                        for detection in detections
                    ),
                    default=0.0,
                ),
            }
            for slot in slots
        }
        for experiment, mode in (("b0", "center"), ("b1", "overlap")):
            evidence = map_detections_to_slots(
                detections,
                slots,
                mode=mode,
                overlap_threshold=overlap_threshold,
            )
            for slot in slots:
                expected = truth[slot.slot_id]
                if expected is None:
                    continue
                slot_evidence = evidence[slot.slot_id]
                predicted = int(slot_evidence.occupied)
                diagnostic = diagnostics[slot.slot_id]
                row = {
                    "experiment": experiment,
                    "sample_id": record["sample_id"],
                    "source": record["source"],
                    "weather": record["weather"],
                    "date": record["date"],
                    "group_id": f"{record['source']}/{record['date']}",
                    "slot_id": slot.slot_id,
                    "truth": expected,
                    "prediction": predicted,
                    "evidence": slot_evidence.evidence_score,
                    "geometric_score": slot_evidence.geometric_score,
                    "track_id": slot_evidence.track_id,
                    "any_detection_center": int(
                        diagnostic["any_detection_center"]
                    ),
                    "max_slot_overlap": diagnostic["max_slot_overlap"],
                }
                rows_by_experiment[experiment].append(row)
                if expected != predicted:
                    cause = _error_cause(
                        experiment=experiment,
                        truth=expected,
                        prediction=predicted,
                        any_detection_center=diagnostic["any_detection_center"],
                        max_slot_overlap=diagnostic["max_slot_overlap"],
                        overlap_threshold=overlap_threshold,
                    )
                    failure_rows.append(
                        {
                            **row,
                            "error_type": (
                                "false_occupied" if predicted else "false_free"
                            ),
                            "cause": cause,
                        }
                    )

    fieldnames = list(rows_by_experiment["b0"][0])
    for experiment, rows in rows_by_experiment.items():
        with (output_dir / f"{experiment}_predictions.csv").open(
            "w",
            newline="",
            encoding="utf-8",
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    with (output_dir / "failure_cases.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[*fieldnames, "error_type", "cause"],
        )
        writer.writeheader()
        writer.writerows(failure_rows)

    report: dict[str, Any] = {
        "dataset": "PKLot development subset",
        "annotations": str(annotations_path),
        "samples": len(records),
        "unknown_slots_excluded": sum(
            1
            for record in records
            for polygon in record["sample"]["parking_spaces"]["polylines"]
            if polygon.get("occupancy_status") == "unknown"
        ),
        "shared_detector": detector_metadata,
        "overlap_threshold": overlap_threshold,
        "error_causes": {
            experiment: dict(
                sorted(
                    (
                        cause,
                        sum(
                            row["experiment"] == experiment
                            and row["cause"] == cause
                            for row in failure_rows
                        ),
                    )
                    for cause in sorted(
                        {
                            row["cause"]
                            for row in failure_rows
                            if row["experiment"] == experiment
                        }
                    )
                )
            )
            for experiment in ("b0", "b1")
        },
        "experiments": {},
    }
    for experiment, rows in rows_by_experiment.items():
        y_true = [int(row["truth"]) for row in rows]
        evidence = [float(row["evidence"]) for row in rows]
        precision, recall, _thresholds, ap = precision_recall_curve(
            y_true,
            evidence,
        )
        overall = _subset_metrics(rows)
        _plot_confusion(
            overall,
            output_dir / f"{experiment}_confusion_matrix.png",
        )
        _plot_pr(
            precision,
            recall,
            ap,
            output_dir / f"{experiment}_pr_curve.png",
        )
        report["experiments"][experiment] = {
            "overall": overall,
            "by_source": _stratified_metrics(rows, "source"),
            "by_weather": _stratified_metrics(rows, "weather"),
            "by_group": _stratified_metrics(rows, "group_id"),
        }

    _render_failures(
        records,
        project_root,
        detections_by_sample,
        rows_by_experiment,
        output_dir,
        failure_limit,
    )
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate B0/B1 on a PKLot image subset with shared boxes"
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--conf", type=float, default=0.20)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overlap-threshold", type=float, default=0.30)
    parser.add_argument("--reuse-cache", action="store_true")
    parser.add_argument(
        "--detection-cache",
        help="Optional shared detections.jsonl path",
    )
    parser.add_argument("--failure-limit", type=int, default=6)
    args = parser.parse_args()
    report = run_experiment(
        annotations_path=Path(args.annotations),
        output_dir=Path(args.output_dir),
        weights=args.weights,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
        overlap_threshold=args.overlap_threshold,
        reuse_cache=args.reuse_cache,
        failure_limit=args.failure_limit,
        detection_cache=(
            Path(args.detection_cache) if args.detection_cache else None
        ),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
