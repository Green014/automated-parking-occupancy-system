from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LITERATURE_CORE_SRC = (
    Path(__file__).resolve().parents[1] / "literature_core" / "src"
)
if str(LITERATURE_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(LITERATURE_CORE_SRC))

from parking_occupancy.stage_l_video import (
    run_stage_l_video,
    stage_l_video_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Stage L P3 continuous D1+B1+E1b+E4+ByteTrack "
            "case study."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--detector-weights", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--tracker-config", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.execute and args.output_dir is None:
        parser.error("--output-dir is required with --execute")

    if args.execute:
        report = run_stage_l_video(
            config_path=args.config,
            video_path=args.video,
            detector_weights=args.detector_weights,
            classifier_checkpoint=args.classifier_checkpoint,
            tracker_config=args.tracker_config,
            output_root=args.output_dir,
            device=args.device,
        )
    else:
        report, _truth = stage_l_video_preflight(
            config_path=args.config,
            video_path=args.video,
            detector_weights=args.detector_weights,
            classifier_checkpoint=args.classifier_checkpoint,
            tracker_config=args.tracker_config,
        )
        if args.preflight_output is not None:
            if args.preflight_output.exists():
                parser.error(
                    f"refusing to overwrite preflight: {args.preflight_output}"
                )
            args.preflight_output.parent.mkdir(parents=True, exist_ok=True)
            args.preflight_output.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
