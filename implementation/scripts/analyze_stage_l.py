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

from parking_occupancy.stage_l_analysis import (
    analyze_static_predictions,
    analyze_video_predictions,
    write_new_json,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run read-only Stage L paired or temporal analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    static = subparsers.add_parser("static")
    static.add_argument("--predictions", type=Path, required=True)
    static.add_argument("--output", type=Path, required=True)
    static.add_argument("--seed", type=int, default=20260728)
    static.add_argument("--resamples", type=int, default=2000)
    video = subparsers.add_parser("video")
    video.add_argument("--occupancy", type=Path, required=True)
    video.add_argument("--output", type=Path, required=True)
    video.add_argument("--fps", type=float, required=True)
    video.add_argument("--warmup-frames", type=int, default=30)
    video.add_argument("--stable-frames", type=int, default=3)
    args = parser.parse_args()

    if args.command == "static":
        payload = analyze_static_predictions(
            args.predictions,
            seed=args.seed,
            resamples=args.resamples,
        )
    else:
        payload = analyze_video_predictions(
            args.occupancy,
            fps=args.fps,
            warmup_frames=args.warmup_frames,
            stable_frames=args.stable_frames,
        )
    write_new_json(args.output, payload)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
