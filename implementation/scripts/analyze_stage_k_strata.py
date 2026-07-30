from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.stage_k_stratified_analysis import run_analysis


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage-k-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    report = run_analysis(
        config_path=args.config,
        stage_k_root=args.stage_k_root,
        output_root=args.output_dir,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
