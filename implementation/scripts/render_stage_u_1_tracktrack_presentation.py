from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_u_1_presentation import (
    render_stage_u_1_presentation_copy,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Create a post-hoc explanatory copy of the frozen Stage T demo. "
            "No model inference is run and the frozen source is never overwritten."
        )
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=(
            IMPLEMENTATION_ROOT
            / "data"
            / "stage_t"
            / "demo"
            / "demo_tracktrack_optional.mp4"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IMPLEMENTATION_ROOT / "data" / "stage_u_1" / "demo",
    )
    args = parser.parse_args()
    result = render_stage_u_1_presentation_copy(
        frozen_demo_path=args.source,
        output_dir=args.output_dir,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
