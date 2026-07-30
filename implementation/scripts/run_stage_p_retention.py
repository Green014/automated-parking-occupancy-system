from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_p_retention import run_stage_p_retention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the frozen, non-overwriting Stage P NDISPark D1 versus "
            "D1-LL retrospective detector/count diagnostic."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--train-manifest", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument("--test-manifest", type=Path, required=True)
    parser.add_argument("--d1-weights", type=Path, required=True)
    parser.add_argument("--d1-ll-weights", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_stage_p_retention(
        protocol_path=args.protocol,
        prepared_root=args.prepared_root,
        manifest_paths={
            "train": args.train_manifest,
            "validation": args.validation_manifest,
            "test": args.test_manifest,
        },
        model_paths={
            "D1": args.d1_weights,
            "D1_LL": args.d1_ll_weights,
        },
        output_root=args.output_root,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "status": "ok",
                "protocol_id": result["protocol_id"],
                "output_root": str(args.output_root.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
