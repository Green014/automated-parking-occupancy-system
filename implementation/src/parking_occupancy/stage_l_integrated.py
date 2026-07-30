from __future__ import annotations

import csv
import json
import statistics
import time
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import yaml

from .detector_comparison import sha256_file
from .geometry import map_detections_to_slots
from .image_io import read_image
from .integrated_workflow import (
    UncertaintyGateConfig,
    uncertainty_gated_fusion,
)
from .models import Detection
from .stage_j_occupancy import (
    _load_annotations,
    _slots_and_truth,
    _stratified,
    _subset_metrics,
)
from .temporal import FilteredSlotState
from .visualization import draw_frame


STAGE_L_PROTOCOL_ID = "P3-INTEGRATED-LITERATURE-WORKFLOW-20260728-01"
STATIC_PARTITIONS = ("development", "retrospective")


class StageLProtocolError(ValueError):
    """Raised when an integrated-workflow input violates the frozen protocol."""


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _verify_file(
    path: Path,
    binding: dict[str, Any],
    label: str,
) -> None:
    if not path.is_file():
        raise StageLProtocolError(f"Missing {label}: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise StageLProtocolError(f"{label} byte size mismatch")
    if sha256_file(path) != str(binding["sha256"]):
        raise StageLProtocolError(f"{label} SHA-256 mismatch")


def load_stage_l_protocol(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_L_PROTOCOL_ID:
        raise StageLProtocolError("Unexpected Stage L protocol ID")
    if payload.get("status") != "frozen_before_P3_predictions":
        raise StageLProtocolError("Stage L protocol is not frozen")
    if tuple(payload.get("static_partitions", {})) != STATIC_PARTITIONS:
        raise StageLProtocolError("Stage L static partitions changed")
    if (
        float(payload["models"]["E1b"]["occupied_threshold"]) != 0.76
        or float(payload["mapping"]["minimum_slot_coverage"]) != 0.40
        or payload["mapping"]["one_to_one"] is not True
    ):
        raise StageLProtocolError("Stage L gate or B1 mapping changed")
    if payload["scope"]["parameter_selection_from_stage_k"] != "prohibited":
        raise StageLProtocolError("Stage K parameter selection must be prohibited")
    return payload


def _load_detection_records(path: Path) -> list[dict[str, Any]]:
    rows = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            continue
        row = json.loads(line)
        if not isinstance(row, dict):
            raise StageLProtocolError(
                f"Detection line {line_number} is not an object"
            )
        rows.append(row)
    return rows


def _detections(record: dict[str, Any]) -> list[Detection]:
    detections = []
    for item in record.get("detections", []):
        bbox = item.get("bbox_xyxy", item.get("bbox"))
        if bbox is None:
            raise StageLProtocolError("Cached detection has no bounding box")
        detections.append(
            Detection(
                bbox=tuple(float(value) for value in bbox),
                confidence=float(item["confidence"]),
                class_id=int(item.get("class_id", 0)),
                class_name=str(item.get("class_name", "vehicle")),
            )
        )
    return detections


def stage_l_static_preflight(
    *,
    config_path: Path,
    partition: str,
    source_root: Path,
    detections_path: Path,
    classifier_checkpoint: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    if partition not in STATIC_PARTITIONS:
        raise StageLProtocolError(f"Unknown static partition: {partition}")
    config_path = config_path.resolve()
    source_root = source_root.resolve()
    detections_path = detections_path.resolve()
    classifier_checkpoint = classifier_checkpoint.resolve()
    protocol = load_stage_l_protocol(config_path)
    partition_config = protocol["static_partitions"][partition]
    annotations_path = _resolve_from_config(
        config_path,
        str(partition_config["annotations"]["path"]),
    )
    _verify_file(
        annotations_path,
        partition_config["annotations"],
        f"{partition} annotations",
    )
    _verify_file(
        detections_path,
        partition_config["cached_D1_detections"],
        f"{partition} cached D1 detections",
    )
    classifier_binding = {
        "bytes": classifier_checkpoint.stat().st_size
        if classifier_checkpoint.is_file()
        else -1,
        "sha256": protocol["models"]["E1b"]["sha256"],
    }
    _verify_file(
        classifier_checkpoint,
        classifier_binding,
        "E1b checkpoint",
    )
    records = _load_annotations(annotations_path)
    detection_records = _load_detection_records(detections_path)
    expected = partition_config["expected"]
    if (
        len(records) != int(expected["images"])
        or len(detection_records) != len(records)
    ):
        raise StageLProtocolError("Static record count mismatch")

    known_slots = 0
    for annotation, detection_record in zip(
        records,
        detection_records,
        strict=True,
    ):
        if annotation["sample_id"] != detection_record["sample_id"]:
            raise StageLProtocolError("Annotation/detection order mismatch")
        image_path = source_root / str(annotation["local_path"])
        if not image_path.is_file():
            raise StageLProtocolError(f"Missing source image: {image_path}")
        known_slots += sum(
            str(item.get("occupancy_status", "unknown")) != "unknown"
            for item in annotation["sample"]["parking_spaces"]["polylines"]
        )
    if known_slots != int(expected["known_slot_labels"]):
        raise StageLProtocolError("Known slot-label count mismatch")

    report = {
        "schema_version": 1,
        "protocol_id": STAGE_L_PROTOCOL_ID,
        "partition": partition,
        "data_role": partition_config["data_role"],
        "images": len(records),
        "known_slot_labels": known_slots,
        "cached_D1_detections_verified": True,
        "E1b_checkpoint_verified": True,
        "detector_inference_rerun": False,
        "parameters_selected_from_this_partition": False,
        "execution_gate": "open",
    }
    return report, records, detection_records


def _mean_timing(values: list[float]) -> dict[str, float]:
    ordered = sorted(values)
    return {
        "mean_ms": statistics.fmean(ordered),
        "p50_ms": statistics.median(ordered),
        "p95_ms": ordered[round(0.95 * (len(ordered) - 1))],
    }


def run_stage_l_static(
    *,
    config_path: Path,
    partition: str,
    source_root: Path,
    detections_path: Path,
    classifier_checkpoint: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    """Run P3 static fusion while reusing frozen P1 detector outputs."""

    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite Stage L output: {output_root}")
    preflight, records, detection_records = stage_l_static_preflight(
        config_path=config_path,
        partition=partition,
        source_root=source_root,
        detections_path=detections_path,
        classifier_checkpoint=classifier_checkpoint,
    )
    protocol = load_stage_l_protocol(config_path)
    from literature_core.classifier import MobileNetSlotClassifier
    from literature_core.patches import extract_slot_patch

    classifier = MobileNetSlotClassifier(
        classifier_checkpoint,
        device=device,
    )
    gate_config = UncertaintyGateConfig(
        classifier_occupied_threshold=float(
            protocol["models"]["E1b"]["occupied_threshold"]
        )
    )
    output_root.mkdir(parents=True)
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )
    video_path = output_root / "annotated.mp4"
    predictions_path = output_root / "predictions.csv"
    events_path = output_root / "events.csv"
    metrics_path = output_root / "metrics.json"
    summary_path = output_root / "summary.json"
    runtime_path = output_root / "runtime_metadata.json"

    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        2.0,
        (1280, 720),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create Stage L video: {video_path}")

    prediction_rows: list[dict[str, Any]] = []
    classifier_times: list[float] = []
    frame_times: list[float] = []
    branch_counts: Counter[str] = Counter()
    fields = [
        "sample_id",
        "camera",
        "date",
        "weather",
        "slot_id",
        "truth",
        "p1_prediction",
        "p3_prediction",
        "p3_score",
        "branch",
        "detector_score",
        "classifier_probability",
    ]
    try:
        with predictions_path.open(
            "x",
            encoding="utf-8",
            newline="",
        ) as handle:
            csv_writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                lineterminator="\n",
            )
            csv_writer.writeheader()
            for annotation, detection_record in zip(
                records,
                detection_records,
                strict=True,
            ):
                frame_start = time.perf_counter()
                image = read_image(source_root / annotation["local_path"])
                height, width = image.shape[:2]
                slots, truth = _slots_and_truth(
                    annotation,
                    width,
                    height,
                )
                detections = _detections(detection_record)
                evidence = map_detections_to_slots(
                    detections,
                    slots,
                    mode="overlap",
                    overlap_threshold=float(
                        protocol["mapping"]["minimum_slot_coverage"]
                    ),
                )
                uncertain_slots = [
                    slot
                    for slot in slots
                    if not evidence[slot.slot_id].occupied
                ]
                classifier_start = time.perf_counter()
                classifier_scores = classifier.predict_patches(
                    [
                        extract_slot_patch(
                            image,
                            slot.points,
                            output_size=tuple(
                                protocol["models"]["E1b"]["patch_size"]
                            ),
                            perspective_warp=bool(
                                protocol["models"]["E1b"][
                                    "perspective_warp"
                                ]
                            ),
                        )
                        for slot in uncertain_slots
                    ]
                )
                classifier_times.append(
                    (time.perf_counter() - classifier_start) * 1000.0
                )
                classifier_by_slot = {
                    slot.slot_id: score
                    for slot, score in zip(
                        uncertain_slots,
                        classifier_scores,
                        strict=True,
                    )
                }
                decisions = uncertainty_gated_fusion(
                    evidence,
                    classifier_by_slot,
                    gate_config,
                )
                states = {}
                for slot in slots:
                    slot_id = slot.slot_id
                    decision = decisions[slot_id]
                    expected = truth[slot_id]
                    branch_counts[decision.branch] += 1
                    states[slot_id] = FilteredSlotState(
                        slot_id=slot_id,
                        occupied=decision.occupied,
                        filtered_score=decision.score,
                        raw_occupied=decision.occupied,
                        raw_evidence_score=decision.score,
                        changed=False,
                        track_id=decision.track_id,
                    )
                    if expected is None:
                        continue
                    row = {
                        "sample_id": annotation["sample_id"],
                        "camera": annotation["source"],
                        "date": annotation["date"],
                        "weather": annotation["weather"],
                        "slot_id": slot_id,
                        "truth": expected,
                        "p1_prediction": int(evidence[slot_id].occupied),
                        "p3_prediction": int(decision.occupied),
                        "p3_score": f"{decision.score:.8f}",
                        "branch": decision.branch,
                        "detector_score": (
                            f"{decision.detector_score:.8f}"
                        ),
                        "classifier_probability": (
                            ""
                            if decision.classifier_probability is None
                            else f"{decision.classifier_probability:.8f}"
                        ),
                    }
                    csv_writer.writerow(row)
                    prediction_rows.append(row)

                annotated = draw_frame(
                    frame=image,
                    detections=detections,
                    slots=slots,
                    states=states,
                    experiment="p3",
                    processing_fps=1.0
                    / max(time.perf_counter() - frame_start, 1e-9),
                )
                cv2.rectangle(annotated, (0, 0), (1100, 34), (0, 0, 0), -1)
                cv2.putText(
                    annotated,
                    (
                        f"P3 | {annotation['sample_id']} | "
                        f"{annotation['source']} {annotation['weather']} | "
                        "NON-CONTIGUOUS STATIC MONTAGE"
                    ),
                    (8, 24),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.58,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(annotated)
                frame_times.append(
                    (time.perf_counter() - frame_start) * 1000.0
                )
    finally:
        writer.release()

    with events_path.open("x", encoding="utf-8", newline="") as handle:
        csv.DictWriter(
            handle,
            fieldnames=[
                "sample_id",
                "frame_index",
                "slot_id",
                "from_state",
                "to_state",
            ],
            lineterminator="\n",
        ).writeheader()

    def method_rows(field: str) -> list[dict[str, Any]]:
        return [
            {
                **row,
                "prediction": int(row[field]),
                "evidence": float(row["p3_score"])
                if field == "p3_prediction"
                else float(row["detector_score"]),
            }
            for row in prediction_rows
        ]

    p1_rows = method_rows("p1_prediction")
    p3_rows = method_rows("p3_prediction")
    p1_metrics = _subset_metrics(p1_rows)
    p3_metrics = _subset_metrics(p3_rows)
    recovered_true_occupied = sum(
        int(row["truth"]) == 1
        and int(row["p1_prediction"]) == 0
        and int(row["p3_prediction"]) == 1
        for row in prediction_rows
    )
    introduced_false_occupied = sum(
        int(row["truth"]) == 0
        and int(row["p1_prediction"]) == 0
        and int(row["p3_prediction"]) == 1
        for row in prediction_rows
    )
    metrics = {
        "schema_version": 1,
        "protocol_id": STAGE_L_PROTOCOL_ID,
        "partition": partition,
        "data_role": protocol["static_partitions"][partition]["data_role"],
        "methods": {
            "P1_D1_B1": {
                "overall": p1_metrics,
                "by_camera": _stratified(p1_rows, "camera"),
                "by_weather": _stratified(p1_rows, "weather"),
            },
            "P3_static_gate": {
                "overall": p3_metrics,
                "by_camera": _stratified(p3_rows, "camera"),
                "by_weather": _stratified(p3_rows, "weather"),
            },
        },
        "P3_minus_P1_macro_f1": (
            p3_metrics["macro_f1"] - p1_metrics["macro_f1"]
        ),
        "gate_effect": {
            "recovered_true_occupied": recovered_true_occupied,
            "introduced_false_occupied": introduced_false_occupied,
            "net_corrected_slots": (
                recovered_true_occupied - introduced_false_occupied
            ),
            "branch_counts": dict(sorted(branch_counts.items())),
        },
        "parameters_selected_from_this_partition": False,
        "claim_boundary": protocol["static_partitions"][partition]["claims"],
    }
    runtime = {
        "frames": len(records),
        "classifier_incremental": _mean_timing(classifier_times),
        "cached_detector_latency_included": False,
        "observed_frame_pipeline": _mean_timing(frame_times),
        "classifier": classifier.metadata(),
    }
    summary = {
        "schema_version": 1,
        "protocol_id": STAGE_L_PROTOCOL_ID,
        "partition": partition,
        "frames": len(records),
        "slot_predictions": len(prediction_rows),
        "method": protocol["method"],
        "detector_inference_rerun": False,
        "cached_detection_source": str(detections_path.resolve()),
        "outputs": [
            path.name
            for path in (
                video_path,
                predictions_path,
                events_path,
                metrics_path,
                summary_path,
                runtime_path,
            )
        ],
    }
    for path, payload in (
        (metrics_path, metrics),
        (summary_path, summary),
        (runtime_path, runtime),
    ):
        path.write_text(
            json.dumps(payload, indent=2) + "\n",
            encoding="utf-8",
        )
    return metrics
