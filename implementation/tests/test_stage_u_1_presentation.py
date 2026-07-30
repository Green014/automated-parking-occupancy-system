from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from parking_occupancy.stage_u_1_presentation import (
    FROZEN_STAGE_T_DEMO_SHA256,
    KNOWN_FAILURE_START_FRAME,
    PRESENTATION_FILENAME,
    PRESENTATION_METADATA_FILENAME,
    overlay_presentation_labels,
)
from parking_occupancy.stage_u_portable_release import sha256_file


def _implementation_root() -> Path:
    return Path(__file__).resolve().parents[1]


def test_overlay_adds_distinct_failure_banner_after_frame_1660() -> None:
    frame = np.full((300, 640, 3), 128, dtype=np.uint8)
    before = overlay_presentation_labels(
        frame,
        source_frame=KNOWN_FAILURE_START_FRAME - 1,
    )
    failure = overlay_presentation_labels(
        frame,
        source_frame=KNOWN_FAILURE_START_FRAME,
    )
    assert not np.array_equal(before, frame)
    assert not np.array_equal(failure, before)
    assert float(failure[-80:, :, 2].mean()) > float(before[-80:, :, 2].mean())


def test_frozen_stage_t_demo_hash_remains_unchanged() -> None:
    source = (
        _implementation_root()
        / "data"
        / "stage_t"
        / "demo"
        / "demo_tracktrack_optional.mp4"
    )
    assert sha256_file(source) == FROZEN_STAGE_T_DEMO_SHA256


def test_generated_presentation_metadata_has_required_claim_boundaries() -> None:
    output = _implementation_root() / "data" / "stage_u_1" / "demo"
    metadata_path = output / PRESENTATION_METADATA_FILENAME
    video_path = output / PRESENTATION_FILENAME
    if not metadata_path.exists():
        return
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert video_path.is_file()
    assert payload["claim_class"] == "TrackTrack identity-output diagnostic"
    assert payload["evaluated_slots"] == 1
    assert payload["new_experiment"] is False
    assert payload["model_inference_run"] is False
    assert payload["source_demo_modified"] is False
    assert payload["known_failure"] == {
        "from_source_frame_inclusive": 1660,
        "truth": "vacant",
        "prediction": "occupied",
        "description": "known false-occupied failure",
    }
    assert "Other visible parking positions are not evaluated" in payload["legend"]
