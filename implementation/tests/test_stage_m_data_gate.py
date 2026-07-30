from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from parking_occupancy.stage_m_data_gate import (
    load_stage_m_gate_audit,
    validate_formal_parking_gate,
)
from parking_occupancy.stage_m_tracking import StageMProtocolError


def _artifact(path: Path) -> dict[str, object]:
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
    }


def test_checked_data_gate_audit_separates_task_claims() -> None:
    root = Path(__file__).resolve().parents[1]
    decisions = {
        decision.gate_id: decision
        for decision in load_stage_m_gate_audit(
            root
            / "data"
            / "stage_m"
            / "STAGE_M_DATA_GATES_20260728.yaml"
        )
    }

    assert decisions["aodraw_detector_diagnostic"].status == "blocked"
    assert decisions["aodraw_detector_diagnostic"].allowed_claim == "audit_only"
    assert (
        decisions["lmot_tracking_diagnostic"].allowed_claim
        == "validation_tracking_diagnostic_only"
    )
    assert decisions["low_light_slot_occupancy"].status == "blocked"
    assert (
        decisions["ndispark_low_light_detector_count"].allowed_claim
        == "detector_count_only_supporting_experiment"
    )


def test_formal_gate_requires_distinct_reviewed_hash_bound_scenes(
    tmp_path: Path,
) -> None:
    dev_video = tmp_path / "dev.mp4"
    test_video = tmp_path / "test.mp4"
    dev_polygons = tmp_path / "dev-regions.json"
    test_polygons = tmp_path / "test-regions.json"
    dev_truth = tmp_path / "dev-truth.yaml"
    test_truth = tmp_path / "test-truth.yaml"
    dev_video.write_bytes(b"development-video")
    test_video.write_bytes(b"test-video")
    dev_polygons.write_text('[{"slot_id":"dev"}]', encoding="utf-8")
    test_polygons.write_text('[{"slot_id":"test"}]', encoding="utf-8")
    dev_truth.write_text("scene: development\n", encoding="utf-8")
    test_truth.write_text("scene: test\n", encoding="utf-8")
    payload = {
        "gate_id": "formal-gate",
        "license_status": "verified",
        "camera_type": "fixed",
        "annotation_review": "human_reviewed",
        "threshold_selection_after_freeze": False,
        "scenes": {
            "development": {
                "physical_scene_id": "lot-a",
                "event_types": ["entry"],
                "video": _artifact(dev_video),
                "polygons": _artifact(dev_polygons),
                "truth": _artifact(dev_truth),
            },
            "test": {
                "physical_scene_id": "lot-b",
                "event_types": ["departure", "occlusion"],
                "video": _artifact(test_video),
                "polygons": _artifact(test_polygons),
                "truth": _artifact(test_truth),
            },
        },
        "freeze": {
            "truth_reviewed_at": "2026-07-28T10:00:00+08:00",
            "frozen_at": "2026-07-28T11:00:00+08:00",
            "test_runs_after_freeze": 0,
        },
    }

    decision = validate_formal_parking_gate(
        payload, base_dir=tmp_path, verify_files=True
    )
    assert decision.status == "eligible"
    assert decision.allowed_claim == "formal_continuous_slot_occupancy_test"

    payload["scenes"]["test"]["physical_scene_id"] = "lot-a"
    blocked = validate_formal_parking_gate(
        payload, base_dir=tmp_path, verify_files=True
    )
    assert blocked.status == "blocked"
    assert "test_scene_must_be_physically_distinct" in blocked.reasons

    payload["scenes"]["test"]["physical_scene_id"] = "lot-b"
    payload["freeze"]["test_runs_after_freeze"] = 1
    blocked = validate_formal_parking_gate(
        payload, base_dir=tmp_path, verify_files=True
    )
    assert blocked.status == "blocked"
    assert (
        "formal_test_requires_zero_prior_test_runs_after_freeze"
        in blocked.reasons
    )


def test_aodraw_gate_cannot_be_relabelled_as_occupancy(tmp_path: Path) -> None:
    audit = {
        "schema_version": 1,
        "gates": [
            {
                "gate_id": "aodraw_detector_diagnostic",
                "status": "eligible",
                "allowed_claim": "parking_slot_occupancy",
                "reasons": [],
            }
        ],
    }
    path = tmp_path / "invalid.yaml"
    path.write_text(yaml.safe_dump(audit), encoding="utf-8")

    with pytest.raises(StageMProtocolError, match="AODRaw"):
        load_stage_m_gate_audit(path)
