from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


GIB = 1024**3
EXPECTED_SMOKE_EXPERIMENT = "D1-NDISPARK-SMOKE-20260727-01"
FORMAL_EXPERIMENT = "D1-NDISPARK-FT-20260727-01"


class GpuDecisionError(ValueError):
    """Raised when Stage F evidence cannot support the Stage G decision."""


def load_smoke_summary(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    validate_smoke_summary(payload)
    return payload


def validate_smoke_summary(payload: dict[str, Any]) -> None:
    if payload.get("experiment_id") != EXPECTED_SMOKE_EXPERIMENT:
        raise GpuDecisionError("Unexpected Stage F smoke experiment")
    if payload.get("status") != "complete":
        raise GpuDecisionError("Stage F smoke run is not complete")

    training = payload.get("training", {})
    arguments = training.get("arguments", {})
    if training.get("epochs_completed") != 3:
        raise GpuDecisionError("Stage F must contain three completed epochs")
    if arguments.get("imgsz") != 640 or arguments.get("batch") != 4:
        raise GpuDecisionError("Stage F must be the frozen 640/batch-4 run")
    if training.get("oom_detected") or training.get("nan_detected"):
        raise GpuDecisionError("Stage F contains an OOM or NaN failure")
    if training.get("batch_auto_reduced"):
        raise GpuDecisionError("Stage F silently changed the batch size")
    if not training.get("validation_inference_succeeded"):
        raise GpuDecisionError("Stage F validation did not complete")
    if not training.get("weights_updated"):
        raise GpuDecisionError("Stage F weights did not update")

    resources = payload.get("resources", {})
    required_positive = (
        "gpu_total_memory_bytes",
        "cuda_free_before_training_bytes",
        "cuda_peak_reserved_bytes",
    )
    for key in required_positive:
        value = resources.get(key)
        if not isinstance(value, int) or value <= 0:
            raise GpuDecisionError(f"Invalid Stage F resource field: {key}")

    epoch_times = training.get("epoch_time_seconds")
    if not isinstance(epoch_times, list) or len(epoch_times) != 3:
        raise GpuDecisionError("Stage F epoch timings are incomplete")
    if not all(
        isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > 0
        for value in epoch_times
    ):
        raise GpuDecisionError("Stage F epoch timings are invalid")


def projected_reserved_bytes(
    *,
    measured_peak_bytes: int,
    measured_batch: int,
    measured_imgsz: int,
    target_batch: int,
    target_imgsz: int,
    safety_factor: float,
) -> int:
    if min(
        measured_peak_bytes,
        measured_batch,
        measured_imgsz,
        target_batch,
        target_imgsz,
    ) <= 0:
        raise GpuDecisionError("Projection inputs must be positive")
    if not math.isfinite(safety_factor) or safety_factor < 1.0:
        raise GpuDecisionError("Safety factor must be finite and >= 1")
    scale = (
        target_batch
        / measured_batch
        * (target_imgsz / measured_imgsz) ** 2
    )
    return math.ceil(measured_peak_bytes * scale * safety_factor)


def _runtime_estimate(
    *,
    wall_seconds: float,
    epoch_times: list[float],
    formal_epochs: int,
) -> dict[str, Any]:
    if formal_epochs <= 0:
        raise GpuDecisionError("Formal epoch count must be positive")
    fixed_overhead = max(0.0, wall_seconds - sum(epoch_times))
    steady_epoch_seconds = sum(epoch_times[1:]) / len(epoch_times[1:])
    maximum_smoke_epoch_seconds = max(epoch_times)
    central_seconds = fixed_overhead + formal_epochs * steady_epoch_seconds
    conservative_seconds = (
        fixed_overhead + formal_epochs * maximum_smoke_epoch_seconds
    )
    stress_upper_seconds = (
        fixed_overhead + formal_epochs * maximum_smoke_epoch_seconds * 2.0
    )
    return {
        "method": (
            "fixed smoke overhead plus per-epoch extrapolation; the central "
            "estimate uses epochs 2-3, the conservative estimate uses the "
            "slowest smoke epoch, and the stress bound doubles that epoch"
        ),
        "fixed_overhead_seconds": fixed_overhead,
        "steady_epoch_seconds": steady_epoch_seconds,
        "maximum_smoke_epoch_seconds": maximum_smoke_epoch_seconds,
        "max_epochs": formal_epochs,
        "central_seconds": central_seconds,
        "central_minutes": central_seconds / 60.0,
        "conservative_seconds": conservative_seconds,
        "conservative_minutes": conservative_seconds / 60.0,
        "stress_upper_seconds": stress_upper_seconds,
        "stress_upper_minutes": stress_upper_seconds / 60.0,
        "two_hour_gate_seconds": 2 * 60 * 60,
        "passes_two_hour_gate": stress_upper_seconds <= 2 * 60 * 60,
        "limitation": (
            "An extrapolation from a three-epoch smoke run, not a measured "
            "50-epoch duration; early stopping may shorten the actual run."
        ),
    }


def build_gpu_decision(
    smoke: dict[str, Any],
    *,
    formal_epochs: int = 50,
    patience: int = 10,
    selected_batch: int = 4,
    selected_imgsz: int = 640,
    nominal_batch: int = 64,
    safety_factor: float = 1.25,
) -> dict[str, Any]:
    validate_smoke_summary(smoke)
    if selected_batch != 4 or selected_imgsz != 640:
        raise GpuDecisionError(
            "Stage G freezes the largest directly measured configuration: "
            "640/batch 4"
        )
    if patience <= 0 or nominal_batch <= 0:
        raise GpuDecisionError("Patience and nominal batch must be positive")

    training = smoke["training"]
    arguments = training["arguments"]
    resources = smoke["resources"]
    epoch_times = [
        float(value) for value in training["epoch_time_seconds"]
    ]
    wall_seconds = float(smoke["duration_seconds"])
    peak_reserved = int(resources["cuda_peak_reserved_bytes"])
    free_before = int(resources["cuda_free_before_training_bytes"])
    total_memory = int(resources["gpu_total_memory_bytes"])

    target_cases = (
        (640, 4),
        (640, 8),
        (960, 4),
        (960, 8),
        (1280, 4),
        (1280, 8),
    )
    projections = []
    for imgsz, batch in target_cases:
        estimate = projected_reserved_bytes(
            measured_peak_bytes=peak_reserved,
            measured_batch=int(arguments["batch"]),
            measured_imgsz=int(arguments["imgsz"]),
            target_batch=batch,
            target_imgsz=imgsz,
            safety_factor=safety_factor,
        )
        is_measured_geometry = imgsz == 640 and batch == 4
        projections.append(
            {
                "imgsz": imgsz,
                "batch": batch,
                "basis": (
                    "measured_configuration_with_25pct_margin"
                    if is_measured_geometry
                    else "analytical_pixel_and_batch_scaling_only"
                ),
                "estimated_reserved_bytes_with_margin": estimate,
                "estimated_reserved_gib_with_margin": estimate / GIB,
                "cuda_free_before_smoke_bytes": free_before,
                "estimated_margin_bytes": free_before - estimate,
                "analytical_fit_against_smoke_free_memory": (
                    estimate <= free_before
                ),
                "validated_by_execution": is_measured_geometry,
            }
        )

    selected_projection = next(
        item
        for item in projections
        if item["imgsz"] == selected_imgsz
        and item["batch"] == selected_batch
    )
    runtime = _runtime_estimate(
        wall_seconds=wall_seconds,
        epoch_times=epoch_times,
        formal_epochs=formal_epochs,
    )
    accumulate = max(round(nominal_batch / selected_batch), 1)
    local_feasible = bool(
        selected_projection["analytical_fit_against_smoke_free_memory"]
        and runtime["passes_two_hour_gate"]
    )

    return {
        "schema_version": 1,
        "decision_id": "GPU-GATE-NDISPARK-D1-20260727-01",
        "status": "complete",
        "source": {
            "smoke_experiment_id": smoke["experiment_id"],
            "smoke_run_id": smoke["run_id"],
            "smoke_summary_sha256": (
                "recorded_separately_at_generation_time"
            ),
            "gpu": resources["gpu_name"],
            "runtime": smoke["runtime"],
        },
        "measured": {
            "imgsz": int(arguments["imgsz"]),
            "physical_batch": int(arguments["batch"]),
            "epochs": int(training["epochs_completed"]),
            "gpu_total_memory_bytes": total_memory,
            "gpu_total_memory_gib": total_memory / GIB,
            "cuda_free_before_training_bytes": free_before,
            "cuda_free_before_training_gib": free_before / GIB,
            "cuda_peak_allocated_bytes": int(
                resources["cuda_peak_allocated_bytes"]
            ),
            "cuda_peak_reserved_bytes": peak_reserved,
            "cuda_peak_reserved_gib": peak_reserved / GIB,
            "cuda_peak_reserved_fraction_of_total": (
                peak_reserved / total_memory
            ),
            "model_train_wall_seconds": wall_seconds,
            "epoch_time_seconds": epoch_times,
            "oom_detected": False,
            "batch_auto_reduced": False,
        },
        "formal_runtime_estimate_at_640_batch4": runtime,
        "memory_projection": {
            "formula": (
                "measured_peak_reserved * target_batch/measured_batch * "
                "(target_imgsz/measured_imgsz)^2 * safety_factor"
            ),
            "safety_factor": safety_factor,
            "warning": (
                "Torch reserved-memory scaling is a conservative planning "
                "heuristic, not a substitute for an executed dry run."
            ),
            "cases": projections,
        },
        "selected_formal_configuration": {
            "experiment_id": FORMAL_EXPERIMENT,
            "initialization": "fresh_COCO_pretrained_yolov8n_not_smoke",
            "max_epochs": formal_epochs,
            "early_stopping_patience": patience,
            "imgsz": selected_imgsz,
            "physical_batch": selected_batch,
            "nominal_batch": nominal_batch,
            "post_warmup_accumulation_steps": accumulate,
            "nominal_effective_batch": selected_batch * accumulate,
            "selection_basis": (
                "batch 4 is the largest allowed batch directly measured by "
                "the frozen smoke run; batch 8 remains an unexecuted estimate"
            ),
        },
        "resolution_feasibility": {
            "imgsz_640": (
                "executed at batch 4; selected for formal training"
            ),
            "imgsz_960": (
                "analytical batch-4 estimate fits, but is unmeasured and "
                "outside the frozen formal comparison"
            ),
            "imgsz_1280": (
                "analytical batch-4 estimate fits with less margin, but is "
                "unmeasured and outside the frozen formal comparison; "
                "batch 8 does not fit the conservative estimate"
            ),
        },
        "gpu_capacity_decision": {
            "six_gib_can_complete_selected_run": local_feasible,
            "operational_minimum_vram_gib": 4,
            "recommended_minimum_vram_gib": 6,
            "local_device_selected": True,
            "rent_gpu_worthwhile": False,
            "rental_duration_hours": 0,
            "local_vs_rental": (
                "No remote benchmark was executed, so no fabricated speedup "
                "is reported. The local stress-bound estimate is under nine "
                "minutes; remote setup and transfer would dominate."
            ),
            "a100_needed": False,
            "yolo_world_fine_tuning_in_scope": False,
        },
        "scope": {
            "training_run": False,
            "prediction_run": False,
            "test_data_accessed": False,
            "cnr_ext_accessed": False,
            "pklot_accessed": False,
            "virat_accessed": False,
            "remote_or_paid_gpu_used": False,
        },
        "gate": {
            "stage_G_passed": local_feasible,
            "stage_H_local_formal_training_allowed": local_feasible,
            "paid_or_remote_gpu_allowed": False,
            "formal_output_must_be_new": True,
            "formal_checkpoint_may_initialize_from_smoke": False,
        },
    }


def write_gpu_decision(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
