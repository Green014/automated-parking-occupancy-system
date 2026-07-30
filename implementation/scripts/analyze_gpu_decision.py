from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

from parking_occupancy.gpu_decision import (
    build_gpu_decision,
    load_smoke_summary,
    write_gpu_decision,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the Stage G local-GPU gate from a completed Stage F "
            "smoke summary. This command does not train or predict."
        )
    )
    parser.add_argument(
        "--smoke-summary",
        type=Path,
        required=True,
        help="Completed, ignored Stage F smoke_summary.json.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="New JSON path; existing files are never overwritten.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite Stage G output: {output}")
    source = args.smoke_summary.resolve()
    smoke = load_smoke_summary(source)
    decision = build_gpu_decision(smoke)
    decision["source"]["smoke_summary_path"] = str(source)
    decision["source"]["smoke_summary_sha256"] = hashlib.sha256(
        source.read_bytes()
    ).hexdigest()
    write_gpu_decision(output, decision)
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
