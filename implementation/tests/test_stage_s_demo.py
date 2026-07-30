from __future__ import annotations

import json
from pathlib import Path

import pytest

from parking_occupancy.stage_s_demo import (
    CONTINUOUS_SEGMENTS,
    build_demo_plan,
    load_demo_assets,
    verify_demo_video,
)


def test_demo_plan_uses_required_frozen_consecutive_segments() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = load_demo_assets(root)
    plan = build_demo_plan(assets)
    assert CONTINUOUS_SEGMENTS == (
        ("gopro1", 92, 116),
        ("gopro4", 95, 123),
        ("gopro26", 109, 136),
    )
    assert len(plan["continuous"]) == 82
    assert len(plan["comparison"]) == 13
    assert len(plan["recoveries"]) == 10
    assert len(plan["failures"]) == 7
    assert plan["timeline"] == {
        "continuous": [0, 200],
        "detector_comparison": [200, 330],
        "F2_recovery": [330, 430],
        "failure_cases": [430, 500],
    }


def test_demo_recoveries_and_failures_follow_frozen_fields() -> None:
    root = Path(__file__).resolve().parents[1]
    assets = load_demo_assets(root)
    plan = build_demo_plan(assets)
    for case in plan["recoveries"]:
        key = (
            case["sequence_id"],
            case["frame_index"],
            case["slot_id"],
        )
        row = assets["predictions"]["D1"][key]
        assert assets["truth_values"][key] == 1
        assert int(row["detector_occupied"]) == 0
        assert int(row["raw_state"]) == 1
    for case in plan["failures"]:
        key = (
            case["sequence_id"],
            case["frame_index"],
            case["slot_id"],
        )
        row = assets["predictions"]["D1"][key]
        assert assets["truth_values"][key] == 0
        assert int(row["detector_occupied"]) == 1
        assert int(row["raw_state"]) == 1


def test_repository_demo_contract_when_rendered() -> None:
    root = Path(__file__).resolve().parents[1]
    demo_root = root / "data" / "stage_s" / "demo"
    metadata_path = demo_root / "STAGE_S_DEMO_METADATA.json"
    if not metadata_path.exists():
        pytest.skip("Stage S demo is rendered after frozen-plan tests")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    validation = verify_demo_video(demo_root / "demo_main.mp4")
    assert metadata["model_inference_run"] is False
    assert metadata["main_state_field"] == "raw_state"
    assert metadata["E4_state_used_for_main_visualization"] is False
    assert metadata["tracker_used"] is False
    assert validation["frames"] == 500
    assert validation["duration_seconds"] == pytest.approx(50.0)
