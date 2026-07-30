from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from parking_occupancy.stage_q_external import (
    FORMAL_METHOD_OUTPUTS,
    SOURCE_SLOT_IDS,
    ManifestRecord,
    StageQDataGateError,
    execute_formal_run_if_authorized,
    load_candidate_gate,
    manifest_fingerprint,
    parse_groundtruth_text,
    truth_to_long_form,
    validate_frozen_comparison,
    validate_image_truth_bijection,
    validate_method_output_contract,
    validate_slot_map_contract,
    validate_temporal_metric_request,
    verify_manifest_records,
)


def _vector(values: str = "10") -> str:
    return (values * 11)[:21]


def _slot_payload(*, out_of_range: bool = False) -> dict:
    slots = []
    for index, slot_id in enumerate(SOURCE_SLOT_IDS):
        x = float((index % 7) * 12)
        y = float((index // 7) * 20)
        if out_of_range and index == 20:
            x = 100.0
        slots.append(
            {
                "id": slot_id,
                "points": [
                    [x, y],
                    [x + 8.0, y],
                    [x + 8.0, y + 12.0],
                    [x, y + 12.0],
                ],
            }
        )
    return {
        "schema_version": 1,
        "source": "human_annotation_before_inference",
        "source_width": 100,
        "source_height": 80,
        "coordinate_system": "pixel",
        "slots": slots,
    }


def _gate() -> dict:
    return {
        "status": "BLOCKED_BEFORE_DOWNLOAD",
        "formal_inference_authorized": False,
        "comparison_roles": {
            "primary": "P3-D1",
            "secondary": "P3-D1-LL",
            "default_change_allowed": False,
            "purpose": (
                "independent_external_occupancy_evidence_not_model_selection"
            ),
        },
        "shared_inference": {
            "imgsz": 640,
            "confidence": 0.30,
            "nms_iou": 0.70,
            "agnostic_nms": True,
            "max_detections": 300,
            "classes": [0],
        },
    }


def _p3_defaults() -> dict:
    return {
        "detector": {
            "id": "D1",
            "confidence": 0.30,
            "image_size": 640,
            "class_ids": [0],
            "nms_iou": 0.70,
            "agnostic_nms": True,
            "max_detections": 300,
        }
    }


def test_parses_contiguous_21_value_occupancy_vector() -> None:
    vector = _vector()
    records = parse_groundtruth_text(f"night_0001.jpg {vector}\n")
    assert records[0].source_values == tuple(int(value) for value in vector)
    assert len(records[0].project_states) == 21


def test_parses_separated_values_and_maps_source_semantics() -> None:
    values = ["1", "0"] + ["1"] * 19
    record = parse_groundtruth_text(
        "night_0001.jpg, [" + ", ".join(values) + "]\n"
    )[0]
    assert record.source_values[:2] == (1, 0)
    assert record.project_states[:2] == (0, 1)


@pytest.mark.parametrize("length", [0, 20, 22])
def test_non_21_value_vector_fails_safely(length: int) -> None:
    values = " ".join(["1"] * length)
    with pytest.raises(StageQDataGateError, match="truth line|expected 21"):
        parse_groundtruth_text(f"night.jpg {values}\n")


def test_image_truth_bijection_rejects_missing_or_extra_membership() -> None:
    records = parse_groundtruth_text(f"a.jpg {_vector()}\n")
    assert validate_image_truth_bijection(["a.jpg"], records)["one_to_one"]
    with pytest.raises(StageQDataGateError, match="membership mismatch"):
        validate_image_truth_bijection(["a.jpg", "b.jpg"], records)


def test_unknown_truth_is_excluded_from_long_form() -> None:
    values = ["?"] + ["1"] * 20
    records = parse_groundtruth_text("a.jpg " + " ".join(values))
    result = truth_to_long_form(records, video_id="night_test")
    assert result["unknown_excluded"] == 1
    assert len(result["rows"]) == 20
    assert all(row["slot_id"] != "slot_01" for row in result["rows"])
    assert all(row["timestamp_s"] == "" for row in result["rows"])


def test_polygon_ids_and_truth_slot_ids_must_match() -> None:
    slot_map = validate_slot_map_contract(_slot_payload())
    assert tuple(slot.slot_id for slot in slot_map.slots) == SOURCE_SLOT_IDS
    payload = _slot_payload()
    payload["slots"][0]["id"] = "wrong"
    with pytest.raises(StageQDataGateError, match="do not match"):
        validate_slot_map_contract(payload)


def test_polygon_coordinates_must_stay_inside_source_image() -> None:
    with pytest.raises(StageQDataGateError, match="out of range"):
        validate_slot_map_contract(_slot_payload(out_of_range=True))


def test_scene_manifest_hash_and_file_integrity(tmp_path: Path) -> None:
    image = tmp_path / "night.jpg"
    image.write_bytes(b"not-a-real-image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    records = [
        ManifestRecord(
            relative_path="night.jpg",
            bytes=image.stat().st_size,
            sha256=digest,
        )
    ]
    fingerprint = manifest_fingerprint(records)
    result = verify_manifest_records(
        tmp_path,
        records,
        expected_manifest_sha256=fingerprint,
    )
    assert result["verified"] is True
    with pytest.raises(StageQDataGateError, match="fingerprint"):
        verify_manifest_records(
            tmp_path,
            records,
            expected_manifest_sha256="0" * 64,
        )


def test_d1_and_d1_ll_share_exact_frozen_settings_and_roles() -> None:
    result = validate_frozen_comparison(_gate(), _p3_defaults())
    assert result["same_inference_settings"] is True
    assert result["primary"] == "P3-D1"
    assert result["secondary"] == "P3-D1-LL"
    assert result["default_detector"] == "D1"
    assert result["default_change_allowed"] is False


def test_comparison_cannot_change_default_or_favor_one_model() -> None:
    gate = _gate()
    gate["shared_inference"]["confidence"] = 0.31
    with pytest.raises(StageQDataGateError, match="setting mismatch"):
        validate_frozen_comparison(gate, _p3_defaults())
    defaults = _p3_defaults()
    defaults["detector"]["id"] = "D1-LL"
    with pytest.raises(StageQDataGateError, match="remain D1"):
        validate_frozen_comparison(_gate(), defaults)


def test_low_rate_sequence_bans_seconds_latency_but_allows_frame_index() -> None:
    with pytest.raises(StageQDataGateError, match="seconds-level"):
        validate_temporal_metric_request(
            media_type="low_frame_rate_image_sequence",
            requested_metrics=["signed_transition_error_seconds"],
        )
    result = validate_temporal_metric_request(
        media_type="low_frame_rate_image_sequence",
        requested_metrics=[
            "state_change_agreement",
            "frame_index_transition_difference",
        ],
    )
    assert result["seconds_level_latency_supported"] is False
    assert result["interpretation"] == "sequence_index_only_not_realtime_latency"


def test_formal_output_schema_handles_ordered_and_unordered_inputs(
    tmp_path: Path,
) -> None:
    for relative in FORMAL_METHOD_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (tmp_path / "annotated_frames").mkdir()
    assert validate_method_output_contract(
        tmp_path, orderable_sequence=False
    )["verified"]
    ordered = validate_method_output_contract(
        tmp_path, orderable_sequence=True
    )
    assert ordered["verified"] is False
    assert ordered["missing"] == ["annotated.mp4"]


def test_blocked_gate_stops_before_model_callback() -> None:
    calls = 0

    def model_callback() -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(StageQDataGateError, match="not authorized"):
        execute_formal_run_if_authorized(_gate(), model_callback)
    assert calls == 0


def test_repository_gate_preserves_stage_p_fail_and_d1_default() -> None:
    project_root = Path(__file__).resolve().parents[1]
    gate_path = (
        project_root
        / "data/stage_q/STAGE_Q_CANDIDATE_GATE_20260729.yaml"
    )
    gate = load_candidate_gate(gate_path)
    defaults = yaml.safe_load(
        (
            project_root
            / "configs/p3_integrated_runtime_defaults_20260729.yaml"
        ).read_text(encoding="utf-8")
    )
    result = validate_frozen_comparison(gate, defaults)
    assert gate["frozen_prior_conclusions"]["stage_p2_status"] == "FAIL"
    assert result["default_detector"] == "D1"
