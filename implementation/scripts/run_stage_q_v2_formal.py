from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_q_v2_evaluation import run_stage_q_v2_formal


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the two frozen Stage Q-v2 UPM-GTI occupancy methods exactly "
            "once after the human polygon-confirmation gate."
        )
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "stage_q_v2_external_night_occupancy_frozen_20260729_v2.yaml"
        ),
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_stage_q_v2_formal(
        config_path=args.config,
        device=args.device,
        classifier_batch_size=args.classifier_batch_size,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
