from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_u_1_release import build_submission_zip


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build the exact Stage U.1 submission candidate set as an external ZIP."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build_submission_zip(
        repository_root=REPOSITORY_ROOT,
        output_zip=args.output,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
