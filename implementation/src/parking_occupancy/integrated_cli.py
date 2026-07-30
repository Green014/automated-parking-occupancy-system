from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .integrated_runner import (
    run_integrated_video,
)

DEFAULT_FINAL_INTEGRATED_CONFIG = (
    Path(__file__).resolve().parents[2]
    / "configs"
    / "p3_stage_r_recommended_default_20260729.yaml"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parking-run-final",
        description=(
            "Run final default P3: D1 -> B1 -> E1b/F2 -> occupancy; "
            "E4 and ByteTrack/TrackTrack require explicit opt-in"
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--d1-weights", type=Path, required=True)
    parser.add_argument("--e1b-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_FINAL_INTEGRATED_CONFIG,
        help="Explicit P3 runtime config; defaults to the Stage R recommendation",
    )
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--source-id")
    parser.add_argument("--device", default="auto")
    parser.add_argument(
        "--temporal",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Enable/disable E4; omitted uses the config default",
    )
    parser.add_argument(
        "--tracker",
        choices=("none", "bytetrack", "tracktrack"),
        default=None,
        help="Optional tracker; omitted uses the config default",
    )
    parser.add_argument("--tracker-config", type=Path)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_integrated_video(
        input_path=args.input,
        slots_path=args.slots,
        detector_weights=args.d1_weights,
        classifier_checkpoint=args.e1b_checkpoint,
        output_root=args.output_dir,
        config_path=args.config,
        device=args.device,
        source_id=args.source_id,
        truth_path=args.truth,
        temporal_enabled=args.temporal,
        tracker_backend=args.tracker,
        tracker_config_override=args.tracker_config,
        classifier_batch_size=args.classifier_batch_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
