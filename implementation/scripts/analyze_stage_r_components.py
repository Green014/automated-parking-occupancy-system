from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_r_component_attribution import (
    build_stage_r_analysis,
    write_stage_r_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compute Stage R post-hoc component attribution from frozen "
            "Stage Q-v2 occupancy outputs without model inference."
        )
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "data" / "stage_r",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    output_dir = args.output_dir.resolve()
    expected_parent = (project_root / "data").resolve()
    if expected_parent not in output_dir.parents:
        raise ValueError("Stage R outputs must stay under implementation/data")
    analysis = build_stage_r_analysis(project_root)
    outputs = write_stage_r_outputs(analysis, output_dir)
    overall = [
        row
        for row in analysis["comparison"]
        if row["scope_type"] == "overall"
    ]
    print(
        json.dumps(
            {
                "status": analysis["status"],
                "model_inference_run": analysis["model_inference_run"],
                "outputs": [str(path) for path in outputs],
                "overall": overall,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
