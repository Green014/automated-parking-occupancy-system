from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.stage_j_posthoc_analysis import (
    run_stage_j_posthoc_analysis,
    stage_j_posthoc_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute the frozen read-only Stage J layered and "
            "paired analysis. This command never loads a model."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--stage-j-root", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight-output")
    args = parser.parse_args()

    if args.execute:
        if not args.output_dir:
            parser.error("--output-dir is required with --execute")
        report = run_stage_j_posthoc_analysis(
            config_path=Path(args.config),
            stage_j_root=Path(args.stage_j_root),
            output_root=Path(args.output_dir),
        )
    else:
        report, _rows = stage_j_posthoc_preflight(
            config_path=Path(args.config),
            stage_j_root=Path(args.stage_j_root),
        )
        if args.preflight_output:
            output = Path(args.preflight_output)
            if output.exists():
                parser.error(f"refusing to overwrite preflight: {output}")
            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
