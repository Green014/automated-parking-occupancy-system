from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.stage_k_occupancy import (
    run_stage_k_comparison,
    stage_k_preflight,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute the frozen Stage K P0/P1/P2 PKLot test. "
            "The default performs no prediction."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--p0-weights", required=True)
    parser.add_argument("--p1-weights", required=True)
    parser.add_argument("--p2-weights", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument("--preflight-output")
    args = parser.parse_args()
    if args.execute and not args.output_dir:
        parser.error("--output-dir is required with --execute")
    weights = {
        "P0": Path(args.p0_weights),
        "P1": Path(args.p1_weights),
        "P2": Path(args.p2_weights),
    }
    if args.execute:
        report = run_stage_k_comparison(
            config_path=Path(args.config),
            source_root=Path(args.source_root),
            output_root=Path(args.output_dir),
            weight_paths=weights,
            device=args.device,
        )
    else:
        report, _records, _specs = stage_k_preflight(
            config_path=Path(args.config),
            source_root=Path(args.source_root),
            weight_paths=weights,
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
