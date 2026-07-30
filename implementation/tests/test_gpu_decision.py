from __future__ import annotations

import json
from pathlib import Path

import pytest

from parking_occupancy.gpu_decision import (
    GpuDecisionError,
    build_gpu_decision,
    load_smoke_summary,
    projected_reserved_bytes,
    write_gpu_decision,
)


def _smoke() -> dict:
    return {
        "experiment_id": "D1-NDISPARK-SMOKE-20260727-01",
        "run_id": "d1_ndispark_smoke_20260727_v3",
        "status": "complete",
        "duration_seconds": 21.368030499997985,
        "training": {
            "arguments": {"imgsz": 640, "batch": 4},
            "epochs_completed": 3,
            "epoch_time_seconds": [
                5.003259897232056,
                2.8234293460845947,
                2.5707907676696777,
            ],
            "oom_detected": False,
            "nan_detected": False,
            "batch_auto_reduced": False,
            "validation_inference_succeeded": True,
            "weights_updated": True,
        },
        "resources": {
            "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
            "gpu_total_memory_bytes": 6_441_926_656,
            "cuda_free_before_training_bytes": 5_379_194_880,
            "cuda_peak_allocated_bytes": 635_307_520,
            "cuda_peak_reserved_bytes": 767_557_632,
        },
        "runtime": {
            "torch": "2.13.0+cu130",
            "cuda": "13.0",
            "ultralytics": "8.4.104",
        },
    }


def test_projection_scales_with_pixels_batch_and_margin() -> None:
    projected = projected_reserved_bytes(
        measured_peak_bytes=100,
        measured_batch=4,
        measured_imgsz=640,
        target_batch=8,
        target_imgsz=1280,
        safety_factor=1.25,
    )
    assert projected == 1000


def test_decision_selects_largest_directly_measured_batch() -> None:
    decision = build_gpu_decision(_smoke())

    selected = decision["selected_formal_configuration"]
    assert selected["physical_batch"] == 4
    assert selected["post_warmup_accumulation_steps"] == 16
    assert selected["nominal_effective_batch"] == 64
    assert decision["gpu_capacity_decision"][
        "six_gib_can_complete_selected_run"
    ]
    assert not decision["gpu_capacity_decision"]["rent_gpu_worthwhile"]
    assert not decision["gpu_capacity_decision"]["a100_needed"]
    assert decision["gate"]["stage_H_local_formal_training_allowed"]


def test_runtime_stress_bound_is_below_two_hours() -> None:
    runtime = build_gpu_decision(_smoke())[
        "formal_runtime_estimate_at_640_batch4"
    ]
    assert runtime["central_minutes"] == pytest.approx(2.4304, abs=0.001)
    assert runtime["stress_upper_minutes"] == pytest.approx(
        8.5216,
        abs=0.001,
    )
    assert runtime["passes_two_hour_gate"]


def test_resolution_projections_distinguish_estimate_from_execution() -> None:
    cases = build_gpu_decision(_smoke())["memory_projection"]["cases"]
    measured = next(
        item
        for item in cases
        if item["imgsz"] == 640 and item["batch"] == 4
    )
    high = next(
        item
        for item in cases
        if item["imgsz"] == 1280 and item["batch"] == 8
    )
    assert measured["validated_by_execution"]
    assert not high["validated_by_execution"]
    assert not high["analytical_fit_against_smoke_free_memory"]


def test_invalid_or_changed_smoke_is_rejected(tmp_path: Path) -> None:
    smoke = _smoke()
    smoke["training"]["arguments"]["batch"] = 8
    path = tmp_path / "smoke.json"
    path.write_text(json.dumps(smoke), encoding="utf-8")

    with pytest.raises(GpuDecisionError, match="640/batch-4"):
        load_smoke_summary(path)


def test_writer_creates_machine_readable_json(tmp_path: Path) -> None:
    path = tmp_path / "decision.json"
    decision = build_gpu_decision(_smoke())
    write_gpu_decision(path, decision)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["decision_id"] == "GPU-GATE-NDISPARK-D1-20260727-01"
    assert loaded["scope"]["training_run"] is False
