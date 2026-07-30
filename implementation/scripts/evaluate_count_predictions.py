from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.count_metrics import evaluate_count_files


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate one prediction count per frozen image using MAE/RMSE. "
            "This command does not compute detector mAP."
        )
    )
    parser.add_argument("--truth-manifest", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    report = evaluate_count_files(
        truth_manifest=Path(args.truth_manifest),
        predictions_csv=Path(args.predictions),
        output_path=Path(args.output),
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
