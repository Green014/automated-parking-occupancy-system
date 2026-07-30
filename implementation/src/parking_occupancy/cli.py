from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detector import UltralyticsDetector
from .method_registry import (
    DEFAULT_REGISTRY_PATH,
    RUNNABLE_METHOD_IDS,
    resolve_runnable_method,
)
from .pipeline import PipelineConfig, process_video
from .temporal import HysteresisConfig

DEFAULT_TRACKER_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / "bytetrack_parking.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run vehicle detection and convert detections to parking-slot states."
        )
    )
    parser.add_argument("--input", required=True, help="Input video path")
    parser.add_argument("--slots", required=True, help="Parking-slot JSON path")
    parser.add_argument("--output-dir", required=True, help="Run output directory")
    method_group = parser.add_mutually_exclusive_group()
    method_group.add_argument(
        "--method",
        choices=tuple(sorted(RUNNABLE_METHOD_IDS)),
        help="Canonical closed baseline ID loaded from --registry",
    )
    method_group.add_argument(
        "--experiment",
        choices=("b0", "b1", "proposed", "t0"),
        help="Legacy/custom experiment selector (prefer --method for baselines)",
    )
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY_PATH))
    parser.add_argument("--weights")
    parser.add_argument("--tracker-config", default=str(DEFAULT_TRACKER_CONFIG))
    parser.add_argument("--conf", type=float)
    parser.add_argument("--imgsz", type=int)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overlap-threshold", type=float)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--rise-alpha", type=float, default=0.60)
    parser.add_argument("--fall-alpha", type=float, default=0.15)
    parser.add_argument("--occupied-threshold", type=float, default=0.18)
    parser.add_argument("--vacant-threshold", type=float, default=0.06)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    method_id = None
    method_name = None
    registry_path = None
    data_role = None
    if args.method:
        forbidden_overrides = [
            flag
            for flag, value in (
                ("--weights", args.weights),
                ("--conf", args.conf),
                ("--imgsz", args.imgsz),
                ("--overlap-threshold", args.overlap_threshold),
            )
            if value is not None
        ]
        if forbidden_overrides:
            parser.error(
                "--method uses the registered canonical configuration; "
                f"remove overrides: {', '.join(forbidden_overrides)}"
            )
        registered = resolve_runnable_method(args.method, args.registry)
        experiment = registered.pipeline_experiment
        weights = registered.weights
        confidence = registered.confidence
        image_size = registered.image_size
        overlap_threshold = registered.minimum_slot_coverage or 0.30
        class_ids = registered.class_ids
        method_id = registered.method_id
        method_name = registered.canonical_name
        registry_path = str(Path(args.registry).resolve())
        data_role = registered.data_role
    else:
        experiment = args.experiment or "b0"
        weights = args.weights or "yolov8n.pt"
        confidence = 0.25 if args.conf is None else args.conf
        image_size = 640 if args.imgsz is None else args.imgsz
        overlap_threshold = (
            0.30
            if args.overlap_threshold is None
            else args.overlap_threshold
        )
        class_ids = (2, 3, 5, 7)

    use_tracking = experiment == "proposed"
    detector = UltralyticsDetector(
        weights=weights,
        confidence=confidence,
        image_size=image_size,
        device=args.device,
        vehicle_class_ids=class_ids,
        use_tracking=use_tracking,
        tracker_config=args.tracker_config,
    )
    config = PipelineConfig(
        experiment=experiment,
        method_id=method_id,
        method_name=method_name,
        method_registry_path=registry_path,
        data_role=data_role,
        overlap_threshold=overlap_threshold,
        max_frames=args.max_frames,
        write_video=not args.no_video,
        hysteresis=HysteresisConfig(
            rise_alpha=args.rise_alpha,
            fall_alpha=args.fall_alpha,
            occupied_threshold=args.occupied_threshold,
            vacant_threshold=args.vacant_threshold,
        ),
    )
    summary = process_video(
        input_path=Path(args.input),
        slot_map_path=Path(args.slots),
        output_dir=Path(args.output_dir),
        detector=detector,
        config=config,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
