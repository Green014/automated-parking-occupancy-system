from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.stage_k_data_gate import render_truth_contact_sheet


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a no-prediction PKLot Stage K truth overlay."
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = render_truth_contact_sheet(
        annotations_path=args.annotations.resolve(),
        source_root=args.source_root.resolve(),
        output_path=args.output.resolve(),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
