from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parking_occupancy.training_smoke import run_smoke, smoke_preflight


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute the frozen local NDISPark D1 smoke run. "
            "The default is preflight-only."
        )
    )
    parser.add_argument("--dataset-protocol", required=True)
    parser.add_argument("--comparison-protocol", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--initial-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="0")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    kwargs = {
        "dataset_protocol_path": Path(args.dataset_protocol),
        "comparison_protocol_path": Path(args.comparison_protocol),
        "data_yaml": Path(args.data),
        "initial_weights": Path(args.initial_weights),
        "output_dir": Path(args.output_dir),
        "device": args.device,
        "workers": args.workers,
    }
    if args.execute:
        report = run_smoke(**kwargs)
    else:
        report, _ = smoke_preflight(**kwargs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
