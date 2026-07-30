from __future__ import annotations

import json
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_s_release import verify_stage_s_registry
from parking_occupancy.stage_t_demo import verify_stage_t_demo
from parking_occupancy.stage_t_tracktrack import (
    P3_TT_CONFIG_NAME,
    analyze_stage_t_outputs,
    load_p3_tt_config,
    verify_stage_t_registry,
    write_stage_t_registry,
)


def main() -> None:
    load_p3_tt_config(IMPLEMENTATION_ROOT / "configs" / P3_TT_CONFIG_NAME)
    stage_s = verify_stage_s_registry(
        IMPLEMENTATION_ROOT
        / "data"
        / "stage_s"
        / "STAGE_S_ARTIFACT_REGISTRY_20260729.yaml",
        implementation_root=REPOSITORY_ROOT,
    )
    if not stage_s["verified"]:
        raise RuntimeError(f"Stage S registry no longer verifies: {stage_s}")

    runtime_root = (
        IMPLEMENTATION_ROOT
        / "outputs"
        / "stage_t_tracktrack_consumed_dev_20260729"
    )
    data_root = IMPLEMENTATION_ROOT / "data" / "stage_t"
    comparison = analyze_stage_t_outputs(
        truth_path=data_root / "virat_0502_slot_truth.csv",
        tt0_root=runtime_root / "tt0",
        tt1_root=runtime_root / "tt1",
    )
    if comparison["formal_occupancy_improvement_conclusion"] != "blocked":
        raise RuntimeError("Stage T formal occupancy claim boundary changed")
    demo = verify_stage_t_demo(
        data_root / "demo" / "demo_tracktrack_optional.mp4"
    )

    artifacts: list[tuple[Path, str]] = [
        (
            IMPLEMENTATION_ROOT / "configs" / P3_TT_CONFIG_NAME,
            "explicit optional P3-TT configuration",
        ),
        (
            IMPLEMENTATION_ROOT
            / "configs"
            / "tracktrack_stage_m_frozen_20260728.yaml",
            "reused frozen TrackTrack parameters",
        ),
        (
            IMPLEMENTATION_ROOT
            / "configs"
            / "p3_stage_r_recommended_default_20260729.yaml",
            "unchanged Stage S default configuration dependency",
        ),
        (
            IMPLEMENTATION_ROOT
            / "src"
            / "parking_occupancy"
            / "stage_t_tracktrack.py",
            "Stage T wrapper, track schema, and analysis",
        ),
        (
            IMPLEMENTATION_ROOT
            / "src"
            / "parking_occupancy"
            / "stage_t_cli.py",
            "explicit P3-TT command module",
        ),
        (
            IMPLEMENTATION_ROOT
            / "src"
            / "parking_occupancy"
            / "stage_t_demo.py",
            "optional TrackTrack demo renderer",
        ),
        (
            IMPLEMENTATION_ROOT
            / "scripts"
            / "run_stage_t_consumed_development.py",
            "controlled TT0/TT1 execution command",
        ),
        (
            IMPLEMENTATION_ROOT / "scripts" / "render_stage_t_demo.py",
            "Stage T demo command",
        ),
        (
            IMPLEMENTATION_ROOT / "scripts" / "finalize_stage_t.py",
            "Stage T registry finalization command",
        ),
        (
            IMPLEMENTATION_ROOT / "tests" / "test_stage_t_tracktrack.py",
            "Stage T tracking and output tests",
        ),
        (
            IMPLEMENTATION_ROOT / "tests" / "test_stage_t_demo.py",
            "Stage T demo tests",
        ),
        (
            IMPLEMENTATION_ROOT
            / "data"
            / "STAGE_T_TRACKTRACK_ENHANCED_VARIANT_REPORT.md",
            "Stage T final report",
        ),
        (
            IMPLEMENTATION_ROOT / "data" / "STAGE_T_RELEASE_INDEX.md",
            "Stage T additive release index",
        ),
    ]
    for path in sorted(data_root.rglob("*")):
        if path.is_file() and path.name != "STAGE_T_ARTIFACT_REGISTRY_20260729.yaml":
            artifacts.append((path, "Stage T compact evidence or presentation artifact"))
    for path in sorted(runtime_root.rglob("*")):
        if path.is_file():
            variant = "TT0" if "\\tt0\\" in str(path).lower() else "TT1"
            artifacts.append(
                (path, f"{variant} consumed-development runtime output")
            )

    registry_path = data_root / "STAGE_T_ARTIFACT_REGISTRY_20260729.yaml"
    result = write_stage_t_registry(
        registry_path,
        repository_root=REPOSITORY_ROOT,
        artifacts=artifacts,
    )
    verification = verify_stage_t_registry(
        registry_path,
        repository_root=REPOSITORY_ROOT,
    )
    print(
        json.dumps(
            {
                "stage_s_registry": stage_s,
                "comparison_status": comparison["status"],
                "occupancy_predictions_identical": comparison[
                    "occupancy_predictions_identical"
                ],
                "formal_occupancy_improvement_conclusion": comparison[
                    "formal_occupancy_improvement_conclusion"
                ],
                "demo": demo,
                "stage_t_registry": result,
                "stage_t_registry_recheck": verification,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
