from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parking_occupancy.formal_training import (
    formal_preflight,
    run_formal_training,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Preflight or execute the one frozen local D1 formal run. "
            "The default is preflight-only."
        )
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--freeze-registry", required=True)
    parser.add_argument("--dataset-protocol", required=True)
    parser.add_argument("--comparison-protocol", required=True)
    parser.add_argument("--data", required=True)
    parser.add_argument("--initial-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    kwargs = {
        "config_path": Path(args.config),
        "freeze_registry_path": Path(args.freeze_registry),
        "dataset_protocol_path": Path(args.dataset_protocol),
        "comparison_protocol_path": Path(args.comparison_protocol),
        "data_yaml": Path(args.data),
        "initial_weights": Path(args.initial_weights),
        "output_dir": Path(args.output_dir),
    }
    if args.execute:
        report = run_formal_training(**kwargs)
    else:
        report, _ = formal_preflight(**kwargs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
