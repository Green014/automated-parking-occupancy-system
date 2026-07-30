from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_s_demo import render_stage_s_demo


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render the Stage S demo from frozen Stage Q-v2 artifacts."
    )
    parser.add_argument(
        "--implementation-root",
        type=Path,
        default=IMPLEMENTATION_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IMPLEMENTATION_ROOT / "data" / "stage_s" / "demo",
    )
    args = parser.parse_args()
    metadata = render_stage_s_demo(
        args.implementation_root.resolve(),
        args.output_dir.resolve(),
    )
    print(json.dumps(metadata["video_validation"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
