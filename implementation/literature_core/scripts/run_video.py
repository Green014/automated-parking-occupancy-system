from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.classifier import MobileNetSlotClassifier  # noqa: E402
from literature_core.config import load_yaml  # noqa: E402
from literature_core.data import load_slot_map  # noqa: E402
from literature_core.detector import YOLOWorldDetector  # noqa: E402
from literature_core.fusion import FusionConfig  # noqa: E402
from literature_core.pipeline import (  # noqa: E402
    LiteratureCorePipeline,
    PipelineConfig,
)
from literature_core.temporal import TemporalConfig  # noqa: E402


def draw(
    frame: np.ndarray,
    slots,
    result,
) -> np.ndarray:
    canvas = frame.copy()
    decision_by_slot = {
        decision.slot_id: decision for decision in result.decisions
    }
    overlay = canvas.copy()
    for slot in slots:
        decision = decision_by_slot[slot.slot_id]
        color = (40, 40, 230) if decision.occupied else (40, 190, 40)
        contour = np.rint(slot.points).astype(np.int32).reshape((-1, 1, 2))
        cv2.fillPoly(overlay, [contour], color)
        cv2.polylines(canvas, [contour], True, color, 2)
    cv2.addWeighted(overlay, 0.18, canvas, 0.82, 0, canvas)
    for detection in result.detections:
        x1, y1, x2, y2 = (round(value) for value in detection.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 255), 2)
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(description="Run literature-core video pipeline")
    parser.add_argument("--input", required=True)
    parser.add_argument("--slots", required=True)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--world-weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-video", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    capture = cv2.VideoCapture(str(Path(args.input).resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {args.input}")
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 25.0
    slots = load_slot_map(args.slots, (width, height))
    classifier = MobileNetSlotClassifier(
        args.classifier_checkpoint,
        device=args.device,
    )
    world_config = config["yolo_world"]
    detector = YOLOWorldDetector(
        args.world_weights,
        prompts=world_config["prompts"],
        confidence=float(world_config["confidence"]),
        image_size=int(world_config["image_size"]),
        device=args.device,
    )
    fusion_config = config["fusion"]
    temporal_config = config["temporal"]
    patch_config = config["patch"]
    mapping_config = config["mapping"]
    pipeline = LiteratureCorePipeline(
        slots,
        classifier,
        detector,
        PipelineConfig(
            patch_size=(int(patch_config["width"]), int(patch_config["height"])),
            perspective_warp=bool(patch_config["perspective_warp"]),
            minimum_slot_coverage=float(
                mapping_config["minimum_slot_coverage"]
            ),
            one_to_one=bool(mapping_config["one_to_one"]),
            decision_threshold=float(fusion_config["occupied_threshold"]),
            fusion=FusionConfig(
                classifier_weight=float(fusion_config["classifier_weight"]),
                detector_weight=float(fusion_config["detector_weight"]),
                track_weight=float(fusion_config["track_weight"]),
            ),
            temporal=TemporalConfig(
                rise_alpha=float(temporal_config["rise_alpha"]),
                fall_alpha=float(temporal_config["fall_alpha"]),
                occupied_threshold=float(
                    temporal_config["occupied_threshold"]
                ),
                vacant_threshold=float(temporal_config["vacant_threshold"]),
                raw_threshold=float(fusion_config["occupied_threshold"]),
            ),
            use_temporal=bool(temporal_config["enabled"]),
        ),
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = None
    if not args.no_video:
        writer = cv2.VideoWriter(
            str(output_dir / "annotated.mp4"),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("Could not create annotated video")

    occupancy_fields = [
        "video_id",
        "frame_index",
        "timestamp_s",
        "slot_id",
        "state",
        "raw_state",
        "p_cls",
        "p_det",
        "p_track",
        "p_occ",
        "p_occ_filtered",
    ]
    event_fields = [
        "video_id",
        "frame_index",
        "timestamp_s",
        "slot_id",
        "to_state",
        "p_occ",
        "p_occ_filtered",
    ]
    frame_index = 0
    input_id = Path(args.input).stem
    start_time = time.perf_counter()
    try:
        with (
            (output_dir / "occupancy.csv").open(
                "w", newline="", encoding="utf-8"
            ) as occupancy_file,
            (output_dir / "events.csv").open(
                "w", newline="", encoding="utf-8"
            ) as event_file,
            (output_dir / "detections.jsonl").open(
                "w", encoding="utf-8"
            ) as detection_file,
        ):
            occupancy_writer = csv.DictWriter(
                occupancy_file, fieldnames=occupancy_fields
            )
            event_writer = csv.DictWriter(event_file, fieldnames=event_fields)
            occupancy_writer.writeheader()
            event_writer.writeheader()
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                if args.max_frames is not None and frame_index >= args.max_frames:
                    break
                timestamp_s = frame_index / fps
                result = pipeline.process_frame(frame, frame_index, timestamp_s)
                detection_file.write(
                    json.dumps(
                        {
                            "frame_index": frame_index,
                            "timestamp_s": timestamp_s,
                            "detections": [
                                asdict(detection)
                                for detection in result.detections
                            ],
                            "mapping": [
                                asdict(evidence)
                                for evidence in result.detector_evidence
                            ],
                        }
                    )
                    + "\n"
                )
                for decision in result.decisions:
                    occupancy_writer.writerow(
                        {
                            "video_id": input_id,
                            "frame_index": frame_index,
                            "timestamp_s": f"{timestamp_s:.6f}",
                            "slot_id": decision.slot_id,
                            "state": int(decision.occupied),
                            "raw_state": int(decision.raw_occupied),
                            "p_cls": decision.p_cls,
                            "p_det": decision.p_det,
                            "p_track": decision.p_track,
                            "p_occ": decision.probability,
                            "p_occ_filtered": decision.filtered_probability,
                        }
                    )
                    if decision.changed:
                        event_writer.writerow(
                            {
                                "video_id": input_id,
                                "frame_index": frame_index,
                                "timestamp_s": f"{timestamp_s:.6f}",
                                "slot_id": decision.slot_id,
                                "to_state": int(decision.occupied),
                                "p_occ": decision.probability,
                                "p_occ_filtered": decision.filtered_probability,
                            }
                        )
                if writer is not None:
                    writer.write(draw(frame, slots, result))
                frame_index += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()

    elapsed_s = time.perf_counter() - start_time
    summary = {
        "input": str(Path(args.input).resolve()),
        "slots": str(Path(args.slots).resolve()),
        "frames": frame_index,
        "source_fps": fps,
        "elapsed_s": elapsed_s,
        "processing_fps": frame_index / elapsed_s if elapsed_s else 0.0,
        "classifier": classifier.metadata(),
        "detector": detector.metadata(),
        "config": config,
        "intermediate_outputs_retained": [
            "p_cls",
            "raw detections/confidences/boxes",
            "p_det and mapping",
            "p_occ and p_occ_filtered",
            "raw/final states",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
