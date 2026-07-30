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

from parking_occupancy.stage_l_integrated import (
    run_stage_l_static,
    stage_l_static_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen Stage L P3 static detector-classifier gate while "
            "reusing an existing P1 detections artifact."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument(
        "--partition",
        choices=("development", "retrospective"),
        required=True,
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--detections", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--preflight-output", type=Path)
    args = parser.parse_args()
    if args.execute and args.output_dir is None:
        parser.error("--output-dir is required with --execute")

    if args.execute:
        report = run_stage_l_static(
            config_path=args.config,
            partition=args.partition,
            source_root=args.source_root,
            detections_path=args.detections,
            classifier_checkpoint=args.classifier_checkpoint,
            output_root=args.output_dir,
            device=args.device,
        )
    else:
        report, _records, _detections = stage_l_static_preflight(
            config_path=args.config,
            partition=args.partition,
            source_root=args.source_root,
            detections_path=args.detections,
            classifier_checkpoint=args.classifier_checkpoint,
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
