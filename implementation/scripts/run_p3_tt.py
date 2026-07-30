from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOTS = (
    IMPLEMENTATION_ROOT / "src",
    IMPLEMENTATION_ROOT / "literature_core" / "src",
)
for source_root in reversed(SOURCE_ROOTS):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from parking_occupancy.p3_tt_runtime import (
    DEFAULT_P3_TT_CONFIG,
    run_generic_p3_tt,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run generic P3-TT on a local video: "
            "D1 -> TrackTrack -> B1 -> E1b/F2; E4 is always off."
        )
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--d1-weights", type=Path, required=True)
    parser.add_argument("--e1b-checkpoint", type=Path, required=True)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--config", type=Path, default=DEFAULT_P3_TT_CONFIG)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(IMPLEMENTATION_ROOT / "outputs" / ".ultralytics_config"),
    )
    summary = run_generic_p3_tt(
        input_path=args.input,
        slots_path=args.slots,
        detector_weights=args.d1_weights,
        classifier_checkpoint=args.e1b_checkpoint,
        source_id=args.source_id,
        output_root=args.output_dir,
        truth_path=args.truth,
        config_path=args.config,
        device=args.device,
        classifier_batch_size=args.classifier_batch_size,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
