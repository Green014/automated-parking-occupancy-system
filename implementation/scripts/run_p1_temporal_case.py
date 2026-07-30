from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.p1_temporal_case import (
    p1_temporal_preflight,
    run_p1_temporal_case,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute the frozen P1+B1 VIRAT 0502 continuous "
            "development case. The default performs no prediction."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.execute and args.output_dir is None:
        parser.error("--output-dir is required with --execute")
    if args.execute:
        report = run_p1_temporal_case(
            config_path=args.config,
            video_path=args.video,
            weights_path=args.weights,
            output_root=args.output_dir,
            device=args.device,
        )
    else:
        report = p1_temporal_preflight(
            config_path=args.config,
            video_path=args.video,
            weights_path=args.weights,
        )
        if args.preflight_output:
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
