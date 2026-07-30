from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.stage_j_posthoc_analysis import (
    verify_stage_j_posthoc_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen Stage J post-hoc source and output artifacts."
        )
    )
    parser.add_argument("--record", required=True)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--external-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite verification output: {output}"
        )
    report = verify_stage_j_posthoc_record(
        record_path=Path(args.record),
        source_root=Path(args.source_root),
        external_root=Path(args.external_root),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
