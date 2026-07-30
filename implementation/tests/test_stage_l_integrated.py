from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from parking_occupancy.stage_l_integrated import (
    STAGE_L_PROTOCOL_ID,
    StageLProtocolError,
    load_stage_l_protocol,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_l_integrated_workflow_frozen_20260728.yaml"
)


def test_stage_l_protocol_freezes_part1_aligned_components() -> None:
    protocol = load_stage_l_protocol(CONFIG)

    assert protocol["protocol_id"] == STAGE_L_PROTOCOL_ID
    assert protocol["method"]["steps"] == [
        "D1_NDISPark_finetuned_YOLOv8n",
        "B1_polygon_coverage_one_to_one_mapping",
        "E1b_MobileNetV3_CBAM_detector_negative_review",
        "F2_asymmetric_uncertainty_gate",
        "E4_asymmetric_EMA_hysteresis",
        "E5_optional_ByteTrack_moving_vehicle_suppression",
    ]
    assert protocol["models"]["E1b"]["occupied_threshold"] == 0.76
    assert protocol["mapping"]["minimum_slot_coverage"] == 0.40
    assert protocol["scope"]["parameter_selection_from_stage_k"] == (
        "prohibited"
    )
    assert protocol["continuous_partition"]["claims"] == (
        "single_slot_single_departure_case_study_only"
    )


def test_stage_l_protocol_rejects_posthoc_stage_k_tuning(tmp_path: Path) -> None:
    payload = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    payload["scope"]["parameter_selection_from_stage_k"] = "allowed"
    candidate = tmp_path / "invalid.yaml"
    candidate.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(StageLProtocolError, match="must be prohibited"):
        load_stage_l_protocol(candidate)
