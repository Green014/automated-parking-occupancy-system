from __future__ import annotations

import json
from pathlib import Path

from parking_occupancy.stage_t_demo import (
    DEMO_FRAME_COUNT,
    DEMO_START_FRAME,
    verify_stage_t_demo,
)


def test_stage_t_demo_selection_covers_departure_boundary() -> None:
    assert DEMO_START_FRAME == 1450
    assert DEMO_FRAME_COUNT == 450
    assert DEMO_START_FRAME <= 1660 < DEMO_START_FRAME + DEMO_FRAME_COUNT


def test_stage_t_demo_contract_when_rendered() -> None:
    root = Path(__file__).resolve().parents[1]
    demo_root = root / "data" / "stage_t" / "demo"
    metadata_path = demo_root / "STAGE_T_DEMO_METADATA.json"
    if not metadata_path.exists():
        return
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    video = verify_stage_t_demo(demo_root / "demo_tracktrack_optional.mp4")
    assert metadata["title"] == "Optional TrackTrack-enhanced variant"
    assert metadata["source_claim"] == "consumed-development diagnostic"
    assert metadata["model_inference_run_for_demo_render"] is False
    assert metadata["stage_s_default_demo"] is False
    assert metadata["temporal_enabled"] is False
    assert metadata["tracktrack_occupancy_improvement_claimed"] is False
    assert video["frames"] == 450
    assert 10.0 <= video["duration_seconds"] <= 20.0
