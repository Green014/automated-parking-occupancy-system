from __future__ import annotations

from pathlib import Path

import pytest

from parking_occupancy.stage_r_component_attribution import (
    binary_metrics,
    build_stage_r_analysis,
    component_comparison,
    occupied_count_metrics,
    temporal_validity_audit,
    verify_stage_r_registry,
)


def test_binary_metrics_are_class_aware() -> None:
    metrics = binary_metrics(
        [1, 1, 0, 0, 0],
        [1, 0, 1, 0, 0],
    )
    assert (metrics["tp"], metrics["tn"], metrics["fp"], metrics["fn"]) == (
        1,
        2,
        1,
        1,
    )
    assert metrics["occupied_precision"] == pytest.approx(0.5)
    assert metrics["occupied_recall"] == pytest.approx(0.5)
    assert metrics["vacant_precision"] == pytest.approx(2 / 3)
    assert metrics["vacant_recall"] == pytest.approx(2 / 3)
    assert metrics["balanced_accuracy"] == pytest.approx(7 / 12)
    assert metrics["macro_f1"] == pytest.approx(7 / 12)


def test_occupied_count_metrics_group_by_source_frame() -> None:
    records = [
        {
            "video_id": "s",
            "frame_index": 1,
            "truth": 1,
            "prediction": 1,
        },
        {
            "video_id": "s",
            "frame_index": 1,
            "truth": 0,
            "prediction": 1,
        },
        {
            "video_id": "s",
            "frame_index": 2,
            "truth": 1,
            "prediction": 0,
        },
        {
            "video_id": "s",
            "frame_index": 2,
            "truth": 0,
            "prediction": 0,
        },
    ]
    metrics = occupied_count_metrics(records)
    assert metrics["frames"] == 2
    assert metrics["occupied_count_mae"] == pytest.approx(1.0)
    assert metrics["occupied_count_rmse"] == pytest.approx(1.0)
    assert metrics["occupied_count_mean_signed_error"] == pytest.approx(0.0)


def test_component_comparison_uses_three_frozen_fields() -> None:
    keys = [
        ("s", 1, "slot_00"),
        ("s", 2, "slot_00"),
    ]
    truth = {
        keys[0]: {"state": "1"},
        keys[1]: {"state": "0"},
    }
    detector_rows = {
        keys[0]: {
            "detector_occupied": "0",
            "raw_state": "1",
            "state": "1",
        },
        keys[1]: {
            "detector_occupied": "0",
            "raw_state": "0",
            "state": "1",
        },
    }
    results = component_comparison(
        truth,
        {"D1": detector_rows, "D1-LL": detector_rows},
    )
    overall = {
        (row["detector"], row["component"]): row
        for row in results
        if row["scope_type"] == "overall"
    }
    assert overall[("D1", "R0")]["fn"] == 1
    assert overall[("D1", "R1")]["macro_f1"] == pytest.approx(1.0)
    assert overall[("D1", "R2")]["fp"] == 1


def test_temporal_audit_counts_sparse_boundaries_and_e4_changes() -> None:
    frames = [1, 2, 5, 6, 7]
    manifest = {("s", frame): {} for frame in frames}
    states = [0, 0, 1, 1, 0]
    raw_states = [0, 0, 0, 1, 0]
    rows = {
        ("s", frame, "slot_00"): {
            "raw_state": str(raw),
            "state": str(state),
        }
        for frame, raw, state in zip(
            frames,
            raw_states,
            states,
            strict=True,
        )
    }
    audit = temporal_validity_audit(
        manifest,
        {"D1": rows, "D1-LL": rows},
    )
    assert audit["gap_distribution"] == {"1": 3, "3": 1}
    assert audit["gap_gt_1_boundaries"] == 1
    assert audit["maximum_gap_frames"] == 3
    assert audit["continuous_segment_lengths_frames"] == [2, 3]
    assert (
        audit["detector_state_audit"]["D1"][
            "E4_state_changes_on_gap_gt_1_boundaries"
        ]
        == 1
    )


def test_repository_stage_r_matches_independent_sanity_values() -> None:
    root = Path(__file__).resolve().parents[1]
    analysis = build_stage_r_analysis(root)
    overall = {
        (row["detector"], row["component"]): row
        for row in analysis["comparison"]
        if row["scope_type"] == "overall"
    }
    expected = {
        ("D1", "R0"): 0.613207,
        ("D1", "R1"): 0.706681,
        ("D1", "R2"): 0.664318,
        ("D1-LL", "R0"): 0.597168,
        ("D1-LL", "R1"): 0.666978,
        ("D1-LL", "R2"): 0.617484,
    }
    for key, value in expected.items():
        assert overall[key]["macro_f1"] == pytest.approx(value, abs=5e-7)
    distribution = analysis["truth_class_distribution"]
    assert distribution["occupied"] == 798
    assert distribution["vacant"] == 7098
    assert analysis["model_inference_run"] is False


def test_repository_stage_r_registry_verifies_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = (
        root / "data" / "stage_r" / "STAGE_R_ARTIFACT_REGISTRY_20260729.yaml"
    )
    if not registry.exists():
        pytest.skip("Stage R registry is created after report finalization")
    result = verify_stage_r_registry(registry, project_root=root)
    assert result["verified"] is True
