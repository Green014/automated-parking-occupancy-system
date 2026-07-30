from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

from .stage_t_tracktrack import (
    P3_TT_CONFIG_NAME,
    run_stage_t_variant,
)


DEFAULT_P3_TT_CONFIG = (
    Path(__file__).resolve().parents[2] / "configs" / P3_TT_CONFIG_NAME
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="parking-run-p3-tt",
        description=(
            "Run the explicit optional P3-TT variant: "
            "D1 -> TrackTrack -> B1 -> E1b/F2; E4 is fixed off"
        ),
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--d1-weights", type=Path, required=True)
    parser.add_argument("--e1b-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_P3_TT_CONFIG,
        help="Explicit Stage T P3-TT config",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    summary = run_stage_t_variant(
        variant_id="TT1",
        tracker_backend="tracktrack",
        input_path=args.input,
        slots_path=args.slots,
        detector_weights=args.d1_weights,
        classifier_checkpoint=args.e1b_checkpoint,
        output_root=args.output_dir,
        config_path=args.config,
        truth_path=args.truth,
        source_id=args.source_id,
        device=args.device,
        classifier_batch_size=args.classifier_batch_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
