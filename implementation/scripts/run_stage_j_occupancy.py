from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from parking_occupancy.stage_j_occupancy import (
    run_stage_j_comparison,
    stage_j_preflight,
)


def _default_source_root() -> str | None:
    data_root = os.environ.get("PARKING_DATA_ROOT")
    if not data_root:
        return None
    candidate = Path(data_root)
    implementation_root = candidate / "implementation"
    return str(
        implementation_root if implementation_root.is_dir() else candidate
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute the frozen Stage J P0/P1/P2 comparison. "
            "The default is preflight-only and performs no prediction."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument(
        "--source-root",
        default=_default_source_root(),
        help=(
            "Root used to resolve local_path entries. Defaults to "
            "PARKING_DATA_ROOT (or its implementation subdirectory)."
        ),
    )
    parser.add_argument("--p0-weights", required=True)
    parser.add_argument("--p1-weights", required=True)
    parser.add_argument("--p2-weights", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--preflight-output",
        help=(
            "Optional new JSON path for the preflight-only record. Existing "
            "files are never overwritten."
        ),
    )
    args = parser.parse_args()
    if not args.source_root:
        parser.error(
            "--source-root is required when PARKING_DATA_ROOT is not set"
        )
    if args.execute and not args.output_dir:
        parser.error("--output-dir is required with --execute")

    weights = {
        "P0": Path(args.p0_weights),
        "P1": Path(args.p1_weights),
        "P2": Path(args.p2_weights),
    }
    if args.execute:
        report = run_stage_j_comparison(
            config_path=Path(args.config),
            source_root=Path(args.source_root),
            output_root=Path(args.output_dir),
            weight_paths=weights,
            device=args.device,
        )
    else:
        report, _records, _specs = stage_j_preflight(
            config_path=Path(args.config),
            source_root=Path(args.source_root),
            weight_paths=weights,
        )
        if args.preflight_output:
            output_path = Path(args.preflight_output)
            if output_path.exists():
                parser.error(
                    f"refusing to overwrite preflight output: {output_path}"
                )
            output_path.parent.mkdir(parents=True, exist_ok=True)
            output_path.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
