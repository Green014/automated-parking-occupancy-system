from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_v3_correction import (
    recompute_emitted_box_metrics,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Offline Stage N-v3 emitted-box correction from saved Stage N-v2 "
            "detection JSONL and LMOT GT; never loads or runs a model"
        )
    )
    parser.add_argument("--v2-output-root", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument(
        "--class-map",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "stage_n_v2"
            / "LMOT_CLASS_MAP_FROZEN_20260729.yaml"
        ),
    )
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics = recompute_emitted_box_metrics(
        v2_output_root=args.v2_output_root,
        validation_root=args.validation_root,
        class_map_path=args.class_map,
        output_root=args.output_root,
    )
    summary = {
        method: values["aggregate"]
        for method, values in metrics["methods"].items()
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
