from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_t_demo import render_stage_t_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the optional Stage T demo from completed TT1 output."
    )
    parser.add_argument(
        "--tt1-root",
        type=Path,
        default=(
            IMPLEMENTATION_ROOT
            / "outputs"
            / "stage_t_tracktrack_consumed_dev_20260729"
            / "tt1"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IMPLEMENTATION_ROOT / "data" / "stage_t" / "demo",
    )
    args = parser.parse_args()
    metadata = render_stage_t_demo(
        tt1_root=args.tt1_root,
        output_dir=args.output_dir,
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
