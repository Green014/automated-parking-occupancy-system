from __future__ import annotations

import csv
import json
import os
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import yaml

from .detector_comparison import (
    ComparisonDetectorAdapter,
    DetectorSpec,
    sha256_file,
)
from .evaluate import (
    _plot_confusion,
    _plot_pr,
    binary_metrics,
    precision_recall_curve,
)
from .geometry import map_detections_to_slots
from .image_io import read_image
from .models import ParkingSlot
from .temporal import FilteredSlotState
from .visualization import draw_frame


STAGE_J_PROTOCOL_ID = "P-COMP-PKLOT-DEV-STAGEJ-20260727-01"
STAGE_J_RECORD_ID = "P-COMP-PKLOT-DEV-STAGEJ-RECORD-20260727-01"
PIPELINE_METHOD_IDS = ("P0", "P1", "P2")


class StageJProtocolError(ValueError):
    """Raised when Stage J inputs differ from the frozen protocol."""


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _verified_artifact(
    *,
    config_path: Path,
    binding: dict[str, Any],
    label: str,
) -> Path:
    path = _resolve_from_config(config_path, str(binding["path"]))
    if not path.is_file():
        raise StageJProtocolError(f"Missing {label}: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise StageJProtocolError(f"{label} byte size mismatch")
    if sha256_file(path) != str(binding["sha256"]):
        raise StageJProtocolError(f"{label} SHA-256 mismatch")
    return path


def load_stage_j_protocol(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_J_PROTOCOL_ID:
        raise StageJProtocolError("Unexpected Stage J protocol ID")
    if payload.get("status") != "frozen_before_predictions":
        raise StageJProtocolError("Stage J protocol is not frozen")
    if payload.get("scope", {}).get("data_role") != "consumed_development":
        raise StageJProtocolError("Stage J data must remain development")
    if tuple(payload.get("methods", {})) != PIPELINE_METHOD_IDS:
        raise StageJProtocolError("Stage J methods must be P0/P1/P2")
    if [
        payload["methods"][method_id]["detector_id"]
        for method_id in PIPELINE_METHOD_IDS
    ] != ["D0", "D1", "D2"]:
        raise StageJProtocolError("P0/P1/P2 detector mapping is invalid")
    common = payload.get("common_inference", {})
    if (
        common.get("agnostic_nms") is not True
        or int(common.get("max_detections", -1)) != 300
        or int(common.get("imgsz", -1)) != 640
    ):
        raise StageJProtocolError("Stage J detector settings are not corrected")
    mapping = payload.get("common_mapping", {})
    if (
        mapping.get("algorithm") != "slot_polygon_coverage"
        or mapping.get("one_to_one") is not True
        or float(mapping.get("minimum_slot_coverage", -1.0)) != 0.40
        or mapping.get("temporal_stabilization") is not False
    ):
        raise StageJProtocolError("Stage J B1 mapping is not frozen")
    _verified_artifact(
        config_path=config_path,
        binding=payload["data"]["annotations"],
        label="Stage J annotations",
    )
    _verified_artifact(
        config_path=config_path,
        binding=payload["data"]["membership_manifest"],
        label="Stage J membership manifest",
    )
    return payload


def _detector_specs(protocol: dict[str, Any]) -> dict[str, DetectorSpec]:
    specs = {}
    for pipeline_id in PIPELINE_METHOD_IDS:
        item = protocol["methods"][pipeline_id]
        specs[pipeline_id] = DetectorSpec(
            method_id=str(item["detector_id"]),
            name=str(item["name"]),
            backend=str(item["backend"]),
            status="ready",
            weights_name=str(item["weights_name"]),
            weights_sha256=str(item["weights_sha256"]),
            source_class_ids=tuple(
                int(value) for value in item["source_class_ids"]
            ),
            source_class_names=tuple(
                str(value) for value in item["source_class_names"]
            ),
            prompts=tuple(str(value) for value in item.get("prompts", [])),
            project_class_id=0,
            project_class_name="vehicle",
        )
    return specs


def _load_annotations(path: Path) -> list[dict[str, Any]]:
    records = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        record = json.loads(line)
        if not isinstance(record, dict):
            raise StageJProtocolError(
                f"Expected object in annotations line {line_number}"
            )
        records.append(record)
    return records


def _slots_and_truth(
    record: dict[str, Any],
    width: int,
    height: int,
) -> tuple[list[ParkingSlot], dict[str, int | None]]:
    slots = []
    truth = {}
    for polygon in record["sample"]["parking_spaces"]["polylines"]:
        slot_id = f"slot_{int(polygon['space_id']):03d}"
        points = tuple(
            (float(point[0]) * width, float(point[1]) * height)
            for point in polygon["points"][0]
        )
        slots.append(ParkingSlot(slot_id=slot_id, points=points))
        status = str(polygon.get("occupancy_status", "unknown"))
        truth[slot_id] = (
            1
            if status == "occupied"
            else 0
            if status == "not occupied"
            else None
        )
    return slots, truth


def stage_j_preflight(
    *,
    config_path: Path,
    source_root: Path,
    weight_paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, DetectorSpec]]:
    config_path = config_path.resolve()
    source_root = source_root.resolve()
    protocol = load_stage_j_protocol(config_path)
    annotations_path = _resolve_from_config(
        config_path,
        str(protocol["data"]["annotations"]["path"]),
    )
    manifest_path = _resolve_from_config(
        config_path,
        str(protocol["data"]["membership_manifest"]["path"]),
    )
    records = _load_annotations(annotations_path)
    with manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        manifest = list(csv.DictReader(handle))
    expected = protocol["data"]["expected"]
    if len(records) != int(expected["images"]) or len(manifest) != len(records):
        raise StageJProtocolError("Stage J membership count mismatch")
    if [row["sample_id"] for row in manifest] != [
        str(record["sample_id"]) for record in records
    ]:
        raise StageJProtocolError("Stage J annotation/manifest order mismatch")

    known = occupied = vacant = unknown = 0
    image_checks = []
    for row, record in zip(manifest, records, strict=True):
        for key in ("source", "date", "weather", "local_path"):
            if str(row[key]) != str(record[key]):
                raise StageJProtocolError(
                    f"Stage J manifest mismatch for {row['sample_id']}:{key}"
                )
        if row["role"] != "consumed_development":
            raise StageJProtocolError("Stage J row is not development")
        image_path = source_root / row["local_path"]
        if not image_path.is_file():
            raise StageJProtocolError(f"Missing Stage J image: {image_path}")
        actual_hash = sha256_file(image_path)
        if (
            image_path.stat().st_size != int(row["image_bytes"])
            or actual_hash != row["image_sha256"]
        ):
            raise StageJProtocolError(
                f"Stage J image binding mismatch: {row['sample_id']}"
            )
        image = read_image(image_path)
        if image.shape[:2] != (720, 1280):
            raise StageJProtocolError(
                f"Unexpected Stage J image size: {row['sample_id']}"
            )
        known += int(row["known_slots"])
        occupied += int(row["occupied"])
        vacant += int(row["vacant"])
        unknown += int(row["unknown"])
        image_checks.append(
            {
                "sample_id": row["sample_id"],
                "path": row["local_path"],
                "sha256": actual_hash,
                "verified": True,
            }
        )
    if {
        "slot_labels_known": known,
        "occupied": occupied,
        "vacant": vacant,
        "unknown_excluded": unknown,
    } != {
        key: int(expected[key])
        for key in (
            "slot_labels_known",
            "occupied",
            "vacant",
            "unknown_excluded",
        )
    }:
        raise StageJProtocolError("Stage J slot-label totals mismatch")

    specs = _detector_specs(protocol)
    model_checks = {}
    for pipeline_id in PIPELINE_METHOD_IDS:
        path = weight_paths[pipeline_id].resolve()
        expected_hash = specs[pipeline_id].weights_sha256
        actual_hash = sha256_file(path) if path.is_file() else None
        if actual_hash != expected_hash:
            raise StageJProtocolError(
                f"{pipeline_id} weights SHA-256 mismatch"
            )
        model_checks[pipeline_id] = {
            "detector_id": specs[pipeline_id].method_id,
            "weights_name": path.name,
            "weights_sha256": actual_hash,
            "ready": True,
        }
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "data_role": "consumed_development",
        "images": len(records),
        "slot_labels_known": known,
        "occupied": occupied,
        "vacant": vacant,
        "unknown_excluded": unknown,
        "models": model_checks,
        "image_bindings_verified": len(image_checks),
        "parameters_selected_from_slot_truth": False,
        "predictions_run": False,
        "execution_gate": "open",
    }
    return report, records, specs


def _subset_metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    truth = [int(row["truth"]) for row in rows]
    prediction = [int(row["prediction"]) for row in rows]
    evidence = [float(row["evidence"]) for row in rows]
    metrics = binary_metrics(truth, prediction)
    _precision, _recall, _thresholds, average_precision = (
        precision_recall_curve(truth, evidence)
    )
    metrics["slot_average_precision"] = average_precision
    metrics["confusion_matrix"] = {
        "vacant": {
            "predicted_vacant": metrics["tn"],
            "predicted_occupied": metrics["fp"],
        },
        "occupied": {
            "predicted_vacant": metrics["fn"],
            "predicted_occupied": metrics["tp"],
        },
    }
    return metrics


def _stratified(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        key: _subset_metrics(group)
        for key, group in sorted(grouped.items())
    }


def _timing(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.fmean(values),
        "p50": statistics.median(values),
        "p95": sorted(values)[round((len(values) - 1) * 0.95)],
    }


def _run_method(
    *,
    pipeline_id: str,
    protocol: dict[str, Any],
    records: list[dict[str, Any]],
    source_root: Path,
    output_root: Path,
    spec: DetectorSpec,
    weights_path: Path,
    device: str,
    dataset_role: str = "consumed_development",
    video_id: str = "pklot_stage_j_development",
    montage_label: str = "NON-CONTIGUOUS DEVELOPMENT MONTAGE",
    claim_boundary: str = (
        "This is consumed development evaluation, not a new untouched "
        "slot-occupancy test."
    ),
) -> dict[str, Any]:
    method = protocol["methods"][pipeline_id]
    common = {
        **protocol["common_inference"],
        "confidence_floor": float(method["confidence"]),
    }
    adapter = ComparisonDetectorAdapter(
        spec=spec,
        weights_path=weights_path,
        common=common,
        device=device,
    )
    method_root = output_root / pipeline_id
    method_root.mkdir()
    video_path = method_root / "annotated.mp4"
    occupancy_path = method_root / "occupancy.csv"
    events_path = method_root / "events.csv"
    detections_path = method_root / "detections.jsonl"
    metrics_path = method_root / "metrics.json"
    summary_path = method_root / "summary.json"
    runtime_path = method_root / "runtime_metadata.json"
    errors_path = method_root / "errors.csv"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (1280, 720),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create Stage J video: {video_path}")

    slot_rows = []
    error_rows = []
    detector_times = []
    mapping_times = []
    frame_times = []
    try:
        with (
            occupancy_path.open(
                "w",
                newline="",
                encoding="utf-8",
            ) as occupancy_handle,
            events_path.open("w", newline="", encoding="utf-8") as event_handle,
            detections_path.open("w", encoding="utf-8") as detection_handle,
        ):
            occupancy_fields = [
                "video_id",
                "frame_index",
                "timestamp_s",
                "slot_id",
                "state",
                "raw_state",
                "evidence",
                "filtered_score",
                "track_id",
                "sample_id",
                "camera",
                "date",
                "weather",
                "truth",
                "geometric_score",
            ]
            occupancy_writer = csv.DictWriter(
                occupancy_handle,
                fieldnames=occupancy_fields,
                lineterminator="\n",
            )
            occupancy_writer.writeheader()
            event_writer = csv.DictWriter(
                event_handle,
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
                lineterminator="\n",
            )
            event_writer.writeheader()

            for frame_index, record in enumerate(records):
                frame_start = time.perf_counter()
                image_path = source_root / record["local_path"]
                frame = read_image(image_path)
                height, width = frame.shape[:2]
                slots, truth = _slots_and_truth(record, width, height)

                detector_start = time.perf_counter()
                detections = adapter.detect(frame)
                detector_end = time.perf_counter()
                evidence_by_slot = map_detections_to_slots(
                    detections=detections,
                    slots=slots,
                    mode="overlap",
                    overlap_threshold=float(
                        protocol["common_mapping"][
                            "minimum_slot_coverage"
                        ]
                    ),
                )
                mapping_end = time.perf_counter()
                states = {
                    slot.slot_id: FilteredSlotState(
                        slot_id=slot.slot_id,
                        occupied=evidence_by_slot[slot.slot_id].occupied,
                        filtered_score=evidence_by_slot[
                            slot.slot_id
                        ].evidence_score,
                        raw_occupied=evidence_by_slot[
                            slot.slot_id
                        ].occupied,
                        raw_evidence_score=evidence_by_slot[
                            slot.slot_id
                        ].evidence_score,
                        changed=False,
                        track_id=None,
                    )
                    for slot in slots
                }
                for slot in slots:
                    evidence = evidence_by_slot[slot.slot_id]
                    expected = truth[slot.slot_id]
                    row = {
                        "video_id": video_id,
                        "frame_index": frame_index,
                        "timestamp_s": f"{frame_index / 2.0:.6f}",
                        "slot_id": slot.slot_id,
                        "state": int(evidence.occupied),
                        "raw_state": int(evidence.occupied),
                        "evidence": f"{evidence.evidence_score:.6f}",
                        "filtered_score": f"{evidence.evidence_score:.6f}",
                        "track_id": "",
                        "sample_id": record["sample_id"],
                        "camera": record["source"],
                        "date": record["date"],
                        "weather": record["weather"],
                        "truth": "" if expected is None else expected,
                        "geometric_score": f"{evidence.geometric_score:.6f}",
                    }
                    occupancy_writer.writerow(row)
                    if expected is not None:
                        metric_row = {
                            "sample_id": record["sample_id"],
                            "camera": record["source"],
                            "date": record["date"],
                            "weather": record["weather"],
                            "slot_id": slot.slot_id,
                            "truth": expected,
                            "prediction": int(evidence.occupied),
                            "evidence": evidence.evidence_score,
                        }
                        slot_rows.append(metric_row)
                        if expected != int(evidence.occupied):
                            error_rows.append(
                                {
                                    **metric_row,
                                    "error_type": (
                                        "false_occupied"
                                        if evidence.occupied
                                        else "false_free"
                                    ),
                                }
                            )

                detection_handle.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "sample_id": record["sample_id"],
                            "camera": record["source"],
                            "date": record["date"],
                            "weather": record["weather"],
                            "image_path": record["local_path"],
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
                    slots=slots,
                    states=states,
                    experiment="b1",
                    processing_fps=1.0
                    / max(time.perf_counter() - frame_start, 1e-9),
                )
                cv2.rectangle(annotated, (0, 0), (850, 34), (0, 0, 0), -1)
                cv2.putText(
                    annotated,
                    (
                        f"{pipeline_id} | {record['sample_id']} | "
                        f"{record['source']} {record['weather']} | "
                        + montage_label
                    ),
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                frame_end = time.perf_counter()
                detector_times.append((detector_end - detector_start) * 1000)
                mapping_times.append((mapping_end - detector_end) * 1000)
                frame_times.append((frame_end - frame_start) * 1000)
    finally:
        writer.release()

    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        fields = [
            "sample_id",
            "camera",
            "date",
            "weather",
            "slot_id",
            "truth",
            "prediction",
            "evidence",
            "error_type",
        ]
        writer_csv = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer_csv.writeheader()
        writer_csv.writerows(error_rows)

    overall = _subset_metrics(slot_rows)
    precision, recall, _thresholds, average_precision = precision_recall_curve(
        [int(row["truth"]) for row in slot_rows],
        [float(row["evidence"]) for row in slot_rows],
    )
    _plot_confusion(overall, method_root / "confusion_matrix.png")
    _plot_pr(
        precision,
        recall,
        average_precision,
        method_root / "pr_curve.png",
    )
    runtime = {
        "frames": len(records),
        "end_to_end_fps": 1000.0 / statistics.fmean(frame_times),
        "timing_ms": {
            "frame": _timing(frame_times),
            "detector": _timing(detector_times),
            "mapping": _timing(mapping_times),
        },
        "detector": adapter.model_metadata(),
    }
    metrics = {
        "pipeline_id": pipeline_id,
        "detector_id": method["detector_id"],
        "dataset": "PKLot",
        "dataset_role": dataset_role,
        "images": len(records),
        "known_slot_labels": len(slot_rows),
        "unknown_slot_labels_excluded": (
            int(protocol["data"]["expected"]["unknown_excluded"])
        ),
        "overall": overall,
        "by_camera": _stratified(slot_rows, "camera"),
        "by_weather": _stratified(slot_rows, "weather"),
        "end_to_end_fps": runtime["end_to_end_fps"],
        "errors": len(error_rows),
        "parameters_selected_from_this_result": False,
        "negative_results_retained": True,
        "claim_boundary": claim_boundary,
    }
    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "pipeline_id": pipeline_id,
        "name": method["name"],
        "detector_id": method["detector_id"],
        "dataset_role": dataset_role,
        "frames_processed": len(records),
        "montage_fps": 2.0,
        "temporal_interpretation": False,
        "events_policy": "header_only_non_contiguous_images",
        "detector_confidence": method["confidence"],
        "common_inference": protocol["common_inference"],
        "common_mapping": protocol["common_mapping"],
        "outputs": {
            "annotated_video": str(video_path),
            "occupancy_csv": str(occupancy_path),
            "events_csv": str(events_path),
            "detections_jsonl": str(detections_path),
            "summary_json": str(summary_path),
            "metrics_json": str(metrics_path),
            "runtime_metadata_json": str(runtime_path),
            "errors_csv": str(errors_path),
        },
    }
    for path, payload in (
        (runtime_path, runtime),
        (metrics_path, metrics),
        (summary_path, summary),
    ):
        path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    adapter.release()
    return metrics


def run_stage_j_comparison(
    *,
    config_path: Path,
    source_root: Path,
    output_root: Path,
    weight_paths: dict[str, Path],
    device: str,
) -> dict[str, Any]:
    """Run P0/P1/P2 with identical geometry on consumed PKLot development."""

    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage J output: {output_root}"
        )
    preflight, records, specs = stage_j_preflight(
        config_path=config_path,
        source_root=source_root,
        weight_paths=weight_paths,
    )
    protocol = load_stage_j_protocol(config_path)
    output_root.mkdir(parents=True)
    ultralytics_config = output_root / "_ultralytics_config"
    ultralytics_config.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config.resolve())
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )
    reports = {}
    for pipeline_id in PIPELINE_METHOD_IDS:
        reports[pipeline_id] = _run_method(
            pipeline_id=pipeline_id,
            protocol=protocol,
            records=records,
            source_root=source_root.resolve(),
            output_root=output_root,
            spec=specs[pipeline_id],
            weights_path=weight_paths[pipeline_id],
            device=device,
        )
    comparison = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "dataset_role": "consumed_development",
        "common_mapping": protocol["common_mapping"],
        "methods": reports,
        "selected_detector_before_slot_evaluation": "D1",
        "pipeline_reselection_from_this_result": False,
        "negative_results_retained": True,
        "stage_k_untouched_test_available": False,
    }
    (output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison


def verify_stage_j_record(
    *,
    record_path: Path,
    source_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    """Verify frozen Stage J source and generated artifacts across two roots."""

    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    if record.get("record_id") != STAGE_J_RECORD_ID:
        raise StageJProtocolError("Unexpected Stage J result record ID")
    roots = {
        "source": source_root.resolve(),
        "external": external_root.resolve(),
    }
    checks = []
    for artifact in record["artifacts"]:
        root_name = str(artifact["root"])
        if root_name not in roots:
            raise StageJProtocolError(
                f"Unexpected Stage J artifact root: {root_name}"
            )
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
                "root": root_name,
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
