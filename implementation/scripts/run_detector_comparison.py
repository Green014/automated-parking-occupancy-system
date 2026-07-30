from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.detector_comparison import (
    comparison_preflight,
    run_comparison,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute the frozen D0/D1/D2 detector comparison. "
            "The default is preflight-only and runs no prediction."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--d0-weights", required=True)
    parser.add_argument("--d1-weights")
    parser.add_argument("--d1-sha256")
    parser.add_argument("--d2-weights", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--output-dir")
    parser.add_argument(
        "--preflight-output",
        help=(
            "Optional new JSON path for a preflight-only record. "
            "Existing files are never overwritten."
        ),
    )
    args = parser.parse_args()

    weight_paths = {
        "D0": Path(args.d0_weights),
        "D1": Path(args.d1_weights) if args.d1_weights else None,
        "D2": Path(args.d2_weights),
    }
    runtime_hashes = {"D1": args.d1_sha256}
    if args.execute:
        if not args.output_dir:
            parser.error("--output-dir is required with --execute")
        report = run_comparison(
            config_path=Path(args.config),
            data_yaml=Path(args.data),
            output_root=Path(args.output_dir),
            weight_paths=weight_paths,
            runtime_weight_hashes=runtime_hashes,
            device=args.device,
        )
    else:
        report, _ = comparison_preflight(
            config_path=Path(args.config),
            data_yaml=Path(args.data),
            weight_paths=weight_paths,
            runtime_weight_hashes=runtime_hashes,
        )
        if args.preflight_output:
            preflight_output = Path(args.preflight_output)
            if preflight_output.exists():
                parser.error(
                    f"refusing to overwrite preflight output: "
                    f"{preflight_output}"
                )
            preflight_output.parent.mkdir(parents=True, exist_ok=True)
            preflight_output.write_text(
                json.dumps(report, indent=2) + "\n",
                encoding="utf-8",
            )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
