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
    write_stage_r_registry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create and verify the additive Stage R artifact registry."
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=PROJECT_ROOT,
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "stage_r"
            / "STAGE_R_ARTIFACT_REGISTRY_20260729.yaml"
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    project_root = args.project_root.resolve()
    data_root = project_root / "data"
    stage_r_root = data_root / "stage_r"
    artifacts = [
        (
            project_root
            / "src"
            / "parking_occupancy"
            / "stage_r_component_attribution.py",
            "analysis_implementation",
        ),
        (
            project_root / "scripts" / "analyze_stage_r_components.py",
            "analysis_entrypoint",
        ),
        (
            project_root / "scripts" / "freeze_stage_r_artifacts.py",
            "registry_entrypoint",
        ),
        (
            project_root / "tests" / "test_stage_r_component_attribution.py",
            "verification_test",
        ),
        (
            data_root / "STAGE_R_COMPONENT_ATTRIBUTION_REPORT.md",
            "final_report",
        ),
        (
            data_root / "FINAL_RESULTS_INDEX.md",
            "final_results_index",
        ),
        *[
            (path, "generated_stage_r_artifact")
            for path in sorted(stage_r_root.iterdir())
            if path.is_file()
            and path.name != "STAGE_R_ARTIFACT_REGISTRY_20260729.yaml"
        ],
    ]
    result = write_stage_r_registry(
        args.registry.resolve(),
        project_root=project_root,
        artifacts=artifacts,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
