from __future__ import annotations

import argparse
import json
from pathlib import Path

from .detector import UltralyticsDetector
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
    parser.add_argument(
        "--experiment",
        choices=("b0", "b1", "proposed"),
        default="b0",
    )
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--tracker-config", default=str(DEFAULT_TRACKER_CONFIG))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overlap-threshold", type=float, default=0.30)
    parser.add_argument("--max-frames", type=int)
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--rise-alpha", type=float, default=0.60)
    parser.add_argument("--fall-alpha", type=float, default=0.15)
    parser.add_argument("--occupied-threshold", type=float, default=0.18)
    parser.add_argument("--vacant-threshold", type=float, default=0.06)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    use_tracking = args.experiment == "proposed"
    detector = UltralyticsDetector(
        weights=args.weights,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
        use_tracking=use_tracking,
        tracker_config=args.tracker_config,
    )
    config = PipelineConfig(
        experiment=args.experiment,
        overlap_threshold=args.overlap_threshold,
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
