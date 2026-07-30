from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parking_occupancy.stage_o_training import prepare_stage_o_training_data


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract only frozen LMOT-train paired sRGB samples and build "
            "the Stage O LMOT+NDISPark YOLO dataset without modifying sources."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--light-parts-dir", type=Path, required=True)
    parser.add_argument("--dark-parts-dir", type=Path, required=True)
    parser.add_argument("--ndispark-prepared-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    result = prepare_stage_o_training_data(
        protocol_path=args.config,
        annotations_tar=args.annotations,
        light_parts_dir=args.light_parts_dir,
        dark_parts_dir=args.dark_parts_dir,
        ndispark_prepared_root=args.ndispark_prepared_root,
        output_root=args.output_root,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
