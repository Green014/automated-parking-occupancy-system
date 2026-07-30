from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parking_occupancy.stage_o_training import run_stage_o_training


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Stage O O3 smoke or the one frozen formal fine-tune."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--training-manifest", type=Path, required=True)
    parser.add_argument("--initial-weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    phase = parser.add_mutually_exclusive_group(required=True)
    phase.add_argument("--smoke", action="store_true")
    phase.add_argument("--formal", action="store_true")
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    result = run_stage_o_training(
        protocol_path=args.config,
        data_yaml=args.data,
        training_manifest=args.training_manifest,
        initial_weights=args.initial_weights,
        output_dir=args.output_dir,
        smoke=args.smoke,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
