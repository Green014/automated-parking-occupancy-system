from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.stage_k_data_gate import (
    verify_stage_k_gate_v2_record,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify the Stage K data-gate v2 evidence record."
    )
    parser.add_argument("--record", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
    result = verify_stage_k_gate_v2_record(
        record_path=args.record,
        source_root=args.source_root,
        external_root=args.external_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
