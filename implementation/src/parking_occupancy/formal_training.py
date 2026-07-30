from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
import re
import time
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .detector_comparison import (
    load_comparison_protocol,
    sha256_file,
)
from .training_smoke import (
    TrainingResourceProbe,
    analyze_results_rows,
    smoke_preflight,
)


FORMAL_EXPERIMENT_ID = "D1-NDISPARK-FT-20260727-01"
GPU_DECISION_ID = "GPU-GATE-NDISPARK-D1-20260727-01"
FREEZE_REGISTRY_ID = "GPU-GATE-NDISPARK-D1-FREEZE-20260727-01"


class FormalTrainingProtocolError(ValueError):
    """Raised when a formal D1 input differs from the frozen protocol."""


@dataclass(frozen=True)
class FormalTrainingSettings:
    experiment_id: str
    max_epochs: int
    patience: int
    imgsz: int
    batch: int
    nbs: int
    expected_accumulation: int
    seed: int
    deterministic: bool
    amp: bool
    device: str
    workers: int
    optimizer: str
    lr0: float
    lrf: float
    momentum: float
    weight_decay: float
    cos_lr: bool
    warmup_epochs: float
    warmup_momentum: float
    warmup_bias_lr: float
    close_mosaic: int
    save_period: int
    initial_weights_sha256: str
    augmentation: dict[str, float | bool | str]


def _project_root(config_path: Path) -> Path:
    if config_path.parent.name != "configs":
        raise FormalTrainingProtocolError(
            "Formal config must reside in implementation/configs"
        )
    return config_path.parent.parent


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise FormalTrainingProtocolError(f"YAML root is not a mapping: {path}")
    return payload


def verify_formal_freeze(
    *,
    config_path: Path,
    freeze_registry_path: Path,
) -> dict[str, Any]:
    config_path = config_path.resolve()
    freeze_registry_path = freeze_registry_path.resolve()
    registry = _load_yaml(freeze_registry_path)
    if registry.get("registry_id") != FREEZE_REGISTRY_ID:
        raise FormalTrainingProtocolError("Unexpected Stage G freeze registry")
    root = _project_root(config_path)
    verified: list[dict[str, Any]] = []
    config_seen = False
    for artifact in registry.get("artifacts", []):
        path = (root / str(artifact["path"])).resolve()
        if path == config_path:
            config_seen = True
        if not path.is_file():
            raise FormalTrainingProtocolError(
                f"Frozen Stage G artifact is missing: {path}"
            )
        actual_bytes = path.stat().st_size
        actual_sha256 = sha256_file(path)
        if actual_bytes != int(artifact["bytes"]):
            raise FormalTrainingProtocolError(
                f"Frozen Stage G artifact byte mismatch: {path}"
            )
        if actual_sha256 != str(artifact["sha256"]):
            raise FormalTrainingProtocolError(
                f"Frozen Stage G artifact SHA-256 mismatch: {path}"
            )
        verified.append(
            {
                "path": str(path),
                "bytes": actual_bytes,
                "sha256": actual_sha256,
            }
        )
    if not config_seen:
        raise FormalTrainingProtocolError(
            "Formal configuration is absent from the freeze registry"
        )
    return {
        "registry_id": registry["registry_id"],
        "registry_path": str(freeze_registry_path),
        "artifacts": verified,
    }


def load_formal_settings(
    *,
    config_path: Path,
    freeze_registry_path: Path,
) -> tuple[dict[str, Any], FormalTrainingSettings, dict[str, Any]]:
    config_path = config_path.resolve()
    freeze = verify_formal_freeze(
        config_path=config_path,
        freeze_registry_path=freeze_registry_path,
    )
    payload = _load_yaml(config_path)
    if payload.get("experiment_id") != FORMAL_EXPERIMENT_ID:
        raise FormalTrainingProtocolError("Unexpected formal experiment ID")
    if payload.get("status") != "frozen_not_executed":
        raise FormalTrainingProtocolError("Formal configuration is not frozen")
    if payload.get("protocol", {}).get("gpu_decision_id") != GPU_DECISION_ID:
        raise FormalTrainingProtocolError("Unexpected GPU decision ID")
    if not payload.get("execution_gate", {}).get(
        "stage_H_local_execution_allowed"
    ):
        raise FormalTrainingProtocolError("Stage H execution gate is closed")
    if payload["resource_gate"].get("paid_or_remote_gpu_allowed"):
        raise FormalTrainingProtocolError("Paid/remote GPU must remain disabled")
    if payload["model"].get("smoke_checkpoint_initialization") != "prohibited":
        raise FormalTrainingProtocolError(
            "Smoke-checkpoint initialization must be prohibited"
        )

    training = payload["training"]
    expected = {
        "max_epochs": 50,
        "early_stopping_patience": 10,
        "imgsz": 640,
        "physical_batch": 4,
        "nominal_batch": 64,
        "post_warmup_accumulation_steps": 16,
        "seed": 20260727,
        "deterministic": True,
        "amp": True,
        "device": "0",
        "workers": 2,
        "cache": False,
        "pretrained": True,
        "resume": False,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "lrf": 0.01,
        "momentum": 0.937,
        "weight_decay": 0.0005,
        "cos_lr": False,
        "warmup_epochs": 3.0,
        "warmup_momentum": 0.8,
        "warmup_bias_lr": 0.1,
        "close_mosaic": 10,
        "validation_each_epoch": True,
        "validation_split": "val",
        "plots": True,
        "save": True,
        "save_period": 10,
        "save_best_and_last": True,
    }
    for key, expected_value in expected.items():
        if training.get(key) != expected_value:
            raise FormalTrainingProtocolError(
                f"Frozen formal setting {key} must be {expected_value!r}"
            )

    augmentation = payload["augmentation"]
    expected_augmentation = {
        "hsv_h": 0.015,
        "hsv_s": 0.7,
        "hsv_v": 0.4,
        "degrees": 0.0,
        "translate": 0.1,
        "scale": 0.5,
        "shear": 0.0,
        "perspective": 0.0,
        "flipud": 0.0,
        "fliplr": 0.5,
        "bgr": 0.0,
        "mosaic": 1.0,
        "mixup": 0.0,
        "cutmix": 0.0,
        "copy_paste": 0.0,
        "rect": False,
        "multi_scale": 0.0,
    }
    for key, expected_value in expected_augmentation.items():
        if augmentation.get(key) != expected_value:
            raise FormalTrainingProtocolError(
                f"Frozen augmentation {key} must be {expected_value!r}"
            )

    initial_hash = str(payload["paths"]["initial_weights"]["sha256"])
    settings = FormalTrainingSettings(
        experiment_id=FORMAL_EXPERIMENT_ID,
        max_epochs=int(training["max_epochs"]),
        patience=int(training["early_stopping_patience"]),
        imgsz=int(training["imgsz"]),
        batch=int(training["physical_batch"]),
        nbs=int(training["nominal_batch"]),
        expected_accumulation=int(
            training["post_warmup_accumulation_steps"]
        ),
        seed=int(training["seed"]),
        deterministic=bool(training["deterministic"]),
        amp=bool(training["amp"]),
        device=str(training["device"]),
        workers=int(training["workers"]),
        optimizer=str(training["optimizer"]),
        lr0=float(training["lr0"]),
        lrf=float(training["lrf"]),
        momentum=float(training["momentum"]),
        weight_decay=float(training["weight_decay"]),
        cos_lr=bool(training["cos_lr"]),
        warmup_epochs=float(training["warmup_epochs"]),
        warmup_momentum=float(training["warmup_momentum"]),
        warmup_bias_lr=float(training["warmup_bias_lr"]),
        close_mosaic=int(training["close_mosaic"]),
        save_period=int(training["save_period"]),
        initial_weights_sha256=initial_hash,
        augmentation={
            key: augmentation[key] for key in expected_augmentation
        },
    )
    return payload, settings, freeze


def formal_training_arguments(
    *,
    settings: FormalTrainingSettings,
    data_yaml: Path,
    output_dir: Path,
) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "data": str(data_yaml.resolve()),
        "epochs": settings.max_epochs,
        "patience": settings.patience,
        "imgsz": settings.imgsz,
        "batch": settings.batch,
        "nbs": settings.nbs,
        "seed": settings.seed,
        "deterministic": settings.deterministic,
        "amp": settings.amp,
        "optimizer": settings.optimizer,
        "lr0": settings.lr0,
        "lrf": settings.lrf,
        "momentum": settings.momentum,
        "weight_decay": settings.weight_decay,
        "cos_lr": settings.cos_lr,
        "warmup_epochs": settings.warmup_epochs,
        "warmup_momentum": settings.warmup_momentum,
        "warmup_bias_lr": settings.warmup_bias_lr,
        "close_mosaic": settings.close_mosaic,
        "device": settings.device,
        "workers": settings.workers,
        "cache": False,
        "pretrained": True,
        "val": True,
        "split": "val",
        "plots": True,
        "save": True,
        "save_period": settings.save_period,
        "project": str(output_dir.resolve().parent),
        "name": output_dir.name,
        "exist_ok": True,
        "resume": False,
        "verbose": True,
    }
    arguments.update(settings.augmentation)
    return arguments


def formal_preflight(
    *,
    config_path: Path,
    freeze_registry_path: Path,
    dataset_protocol_path: Path,
    comparison_protocol_path: Path,
    data_yaml: Path,
    initial_weights: Path,
    output_dir: Path,
) -> tuple[dict[str, Any], FormalTrainingSettings]:
    config_path = config_path.resolve()
    dataset_protocol_path = dataset_protocol_path.resolve()
    comparison_protocol_path = comparison_protocol_path.resolve()
    data_yaml = data_yaml.resolve()
    initial_weights = initial_weights.resolve()
    output_dir = output_dir.resolve()
    config, settings, freeze = load_formal_settings(
        config_path=config_path,
        freeze_registry_path=freeze_registry_path,
    )
    smoke_report, _ = smoke_preflight(
        dataset_protocol_path=dataset_protocol_path,
        comparison_protocol_path=comparison_protocol_path,
        data_yaml=data_yaml,
        initial_weights=initial_weights,
        output_dir=output_dir,
        device=settings.device,
        workers=settings.workers,
    )
    comparison, _ = load_comparison_protocol(comparison_protocol_path)
    if config["protocol"]["dataset_protocol_id"] != (
        "DPROTO-NDISPARK-ONLY-20260727-01"
    ):
        raise FormalTrainingProtocolError("Dataset protocol ID changed")
    if config["protocol"]["comparison_protocol_id"] != (
        comparison["comparison_protocol_id"]
    ):
        raise FormalTrainingProtocolError("Comparison protocol ID changed")
    if sha256_file(initial_weights) != settings.initial_weights_sha256:
        raise FormalTrainingProtocolError(
            "Formal initialization is not the frozen D0 weight"
        )

    return (
        {
            "schema_version": 1,
            "experiment_id": settings.experiment_id,
            "run_id": output_dir.name,
            "status": "ready",
            "training_run": False,
            "prediction_run": False,
            "formal_config": {
                "path": str(config_path),
                "bytes": config_path.stat().st_size,
                "sha256": sha256_file(config_path),
            },
            "freeze": freeze,
            "data": smoke_report["data"],
            "initialization": smoke_report["initialization"],
            "settings": asdict(settings),
            "output_dir": str(output_dir),
            "scope": {
                "count_test_accessed": False,
                "cnr_ext_accessed": False,
                "pklot_accessed": False,
                "virat_accessed": False,
                "remote_or_paid_gpu_used": False,
            },
            "gate": "open_for_one_local_formal_run",
        },
        settings,
    )


class FormalTrainingResourceProbe(TrainingResourceProbe):
    """Adds optimizer accumulation to the Stage F resource measurements."""

    def __init__(self) -> None:
        super().__init__()
        self._training_epoch_active = False

    def on_train_epoch_start(self, trainer: Any) -> None:
        super().on_train_epoch_start(trainer)
        self._training_epoch_active = True

    def on_fit_epoch_end(self, trainer: Any) -> None:
        if not self._training_epoch_active:
            return
        before = len(self.epochs)
        super().on_fit_epoch_end(trainer)
        self._training_epoch_active = False
        if len(self.epochs) > before:
            self.epochs[-1]["optimizer_accumulation_steps"] = int(
                trainer.accumulate
            )


def _read_results(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _numeric_row(row: dict[str, str]) -> dict[str, float]:
    return {str(key).strip(): float(value) for key, value in row.items()}


def best_epoch_indices(
    reported_epoch: int,
    completed_epochs: int,
) -> tuple[int, int]:
    """Normalize Ultralytics' one-based EarlyStopping epoch."""
    one_based = int(reported_epoch)
    zero_based = one_based - 1
    if not 1 <= one_based <= completed_epochs:
        raise RuntimeError("Best epoch is outside the executed run")
    return zero_based, one_based


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def _artifact(path: Path) -> dict[str, Any]:
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_formal_training(
    *,
    config_path: Path,
    freeze_registry_path: Path,
    dataset_protocol_path: Path,
    comparison_protocol_path: Path,
    data_yaml: Path,
    initial_weights: Path,
    output_dir: Path,
) -> dict[str, Any]:
    preflight, settings = formal_preflight(
        config_path=config_path,
        freeze_registry_path=freeze_registry_path,
        dataset_protocol_path=dataset_protocol_path,
        comparison_protocol_path=comparison_protocol_path,
        data_yaml=data_yaml,
        initial_weights=initial_weights,
        output_dir=output_dir,
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "preflight.json", preflight)
    ultralytics_config = output_dir / "_ultralytics_config"
    matplotlib_config = output_dir / "_matplotlib_config"
    ultralytics_config.mkdir()
    matplotlib_config.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config.resolve())
    os.environ["MPLCONFIGDIR"] = str(matplotlib_config.resolve())
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ.setdefault("WANDB_DISABLED", "true")

    log_handler: logging.Handler | None = None
    started_at = datetime.now().astimezone()
    try:
        import cv2
        import torch
        import ultralytics
        from ultralytics import YOLO
        from ultralytics.utils import LOGGER

        log_handler = logging.FileHandler(
            output_dir / "training_console.log",
            encoding="utf-8",
        )
        log_handler.setFormatter(
            logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        )
        LOGGER.addHandler(log_handler)

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable for formal D1 training")
        gpu_name = torch.cuda.get_device_name(0)
        if "RTX 3060 Laptop GPU" not in gpu_name:
            raise RuntimeError(f"Unexpected formal-training GPU: {gpu_name}")
        properties = torch.cuda.get_device_properties(0)
        torch.cuda.empty_cache()
        free_before, cuda_total = torch.cuda.mem_get_info(0)
        torch.cuda.reset_peak_memory_stats(0)

        model = YOLO(str(initial_weights.resolve()))
        probe = FormalTrainingResourceProbe()
        model.add_callback("on_train_epoch_start", probe.on_train_epoch_start)
        model.add_callback("on_train_batch_start", probe.on_train_batch_start)
        model.add_callback("on_train_batch_end", probe.on_train_batch_end)
        model.add_callback("on_fit_epoch_end", probe.on_fit_epoch_end)
        training_args = formal_training_arguments(
            settings=settings,
            data_yaml=data_yaml,
            output_dir=output_dir,
        )
        wall_start = time.perf_counter()
        model.train(**training_args)
        wall_seconds = time.perf_counter() - wall_start
        trainer = model.trainer
        if trainer is None:
            raise RuntimeError("Ultralytics did not expose a trainer")

        required = {
            "args": output_dir / "args.yaml",
            "results_csv": output_dir / "results.csv",
            "training_curves": output_dir / "results.png",
            "best": output_dir / "weights" / "best.pt",
            "last": output_dir / "weights" / "last.pt",
        }
        missing = [
            str(path) for path in required.values() if not path.is_file()
        ]
        if missing:
            raise RuntimeError(f"Formal training artifacts missing: {missing}")

        rows = _read_results(required["results_csv"])
        completed_epochs = len(rows)
        if not 1 <= completed_epochs <= settings.max_epochs:
            raise RuntimeError(
                f"Invalid completed epoch count: {completed_epochs}"
            )
        analysis = analyze_results_rows(
            rows,
            expected_epochs=completed_epochs,
        )
        if not analysis["any_loss_changed"]:
            raise RuntimeError("Formal training losses did not change")
        best_epoch_zero_based, best_epoch_one_based = best_epoch_indices(
            int(trainer.stopper.best_epoch),
            completed_epochs,
        )
        best_row = _numeric_row(rows[best_epoch_zero_based])
        best_metrics = {
            key: value
            for key, value in best_row.items()
            if key.startswith("metrics/")
        }
        if not best_metrics or not all(
            math.isfinite(value) for value in best_metrics.values()
        ):
            raise RuntimeError("Best-epoch validation metrics are invalid")

        validator_seen = int(
            getattr(getattr(trainer, "validator", None), "seen", -1)
        )
        expected_seen = int(
            preflight["data"]["prepared_counts"]["val"]["images"]
        )
        if validator_seen != expected_seen:
            raise RuntimeError(
                f"Validation saw {validator_seen}, expected {expected_seen}"
            )

        checkpoints = {
            "initial": _artifact(initial_weights),
            "best": _artifact(required["best"]),
            "last": _artifact(required["last"]),
        }
        if any(
            checkpoints[name]["sha256"]
            == checkpoints["initial"]["sha256"]
            for name in ("best", "last")
        ):
            raise RuntimeError("Formal checkpoint matches initialization")

        peak_reserved = max(
            [
                int(item.get("cuda_peak_reserved_bytes", 0))
                for item in probe.epochs
            ]
            + [int(torch.cuda.max_memory_reserved(0))]
        )
        peak_allocated = max(
            [
                int(item.get("cuda_peak_allocated_bytes", 0))
                for item in probe.epochs
            ]
            + [int(torch.cuda.max_memory_allocated(0))]
        )
        epoch_times = [
            float(item["epoch_time_seconds"]) for item in probe.epochs
        ]
        if len(epoch_times) != completed_epochs:
            raise RuntimeError("Resource probe epoch count mismatch")
        accumulation = sorted(
            {
                int(item["optimizer_accumulation_steps"])
                for item in probe.epochs
            }
        )
        if accumulation[-1] != settings.expected_accumulation:
            raise RuntimeError(
                "Post-warm-up optimizer accumulation differs from freeze"
            )

        ended_at = datetime.now().astimezone()
        early_stopped = completed_epochs < settings.max_epochs
        report = {
            "schema_version": 1,
            "experiment_id": settings.experiment_id,
            "run_id": output_dir.name,
            "status": "complete",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "model_train_wall_seconds": wall_seconds,
            "protocol": {
                "formal_config": preflight["formal_config"],
                "freeze": preflight["freeze"],
                "dataset": str(dataset_protocol_path.resolve()),
                "comparison": str(comparison_protocol_path.resolve()),
            },
            "data": preflight["data"],
            "training": {
                "arguments": training_args,
                "max_epochs": settings.max_epochs,
                "epochs_completed": completed_epochs,
                "early_stopped": early_stopped,
                "stop_reason": (
                    "early_stopping_patience"
                    if early_stopped
                    else "maximum_epochs"
                ),
                "patience": settings.patience,
                "best_epoch_zero_based": best_epoch_zero_based,
                "best_epoch_one_based": best_epoch_one_based,
                "best_fitness": float(trainer.stopper.best_fitness),
                "best_development_validation_metrics": best_metrics,
                "final_development_validation_metrics": analysis[
                    "final_validation_metrics"
                ],
                "results_analysis": analysis,
                "validation_inference_succeeded": True,
                "validation_images_seen": validator_seen,
                "weights_updated": True,
                "nan_detected": False,
                "oom_detected": False,
                "batch_auto_reduced": False,
            },
            "resources": {
                "gpu_name": gpu_name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "cuda_mem_get_info_total_bytes": int(cuda_total),
                "cuda_free_before_training_bytes": int(free_before),
                "cuda_peak_allocated_bytes": peak_allocated,
                "cuda_peak_reserved_bytes": peak_reserved,
                "cuda_peak_reserved_fraction": (
                    peak_reserved / int(properties.total_memory)
                ),
                "epoch_time_seconds": epoch_times,
                "mean_epoch_time_seconds": float(np.mean(epoch_times)),
                "min_epoch_time_seconds": min(epoch_times),
                "max_epoch_time_seconds": max(epoch_times),
                "observed_accumulation_steps": accumulation,
                "per_epoch": probe.epochs,
                "dataloader": probe.dataloader_assessment(),
            },
            "checkpoints": checkpoints,
            "artifacts": {
                name: _artifact(path)
                for name, path in required.items()
                if name not in {"best", "last"}
            },
            "runtime": {
                "python": platform.python_version(),
                "torch": torch.__version__,
                "cuda": torch.version.cuda,
                "ultralytics": ultralytics.__version__,
                "opencv": cv2.__version__,
                "platform": platform.platform(),
            },
            "scope": {
                "count_test_accessed": False,
                "cnr_ext_accessed": False,
                "pklot_accessed": False,
                "virat_accessed": False,
                "remote_or_paid_gpu_used": False,
                "formal_training_only": True,
            },
            "gate": {
                "stage_H_formal_training_passed": True,
                "stage_I_detector_evaluation_may_start": True,
                "count_test_allowed_now": False,
                "slot_occupancy_prediction_allowed_now": False,
            },
        }
        _write_json(output_dir / "resource_metrics.json", report["resources"])
        _write_json(output_dir / "formal_training_summary.json", report)
        return report
    except Exception as exc:
        failure = {
            "schema_version": 1,
            "experiment_id": settings.experiment_id,
            "run_id": output_dir.name,
            "status": "failed",
            "started_at": started_at.isoformat(),
            "failed_at": datetime.now().astimezone().isoformat(),
            "error_type": type(exc).__name__,
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "negative_result_retained": True,
            "scope": {
                "count_test_accessed": False,
                "remote_or_paid_gpu_used": False,
            },
        }
        _write_json(output_dir / "formal_training_failure.json", failure)
        raise
    finally:
        if log_handler is not None:
            from ultralytics.utils import LOGGER

            LOGGER.removeHandler(log_handler)
            log_handler.close()


def _validated_executed_args(
    *,
    args_path: Path,
    settings: FormalTrainingSettings,
) -> dict[str, Any]:
    args = _load_yaml(args_path)
    expected = {
        "epochs": settings.max_epochs,
        "patience": settings.patience,
        "imgsz": settings.imgsz,
        "batch": settings.batch,
        "nbs": settings.nbs,
        "seed": settings.seed,
        "deterministic": settings.deterministic,
        "amp": settings.amp,
        "optimizer": settings.optimizer,
        "lr0": settings.lr0,
        "lrf": settings.lrf,
        "momentum": settings.momentum,
        "weight_decay": settings.weight_decay,
        "cos_lr": settings.cos_lr,
        "warmup_epochs": settings.warmup_epochs,
        "warmup_momentum": settings.warmup_momentum,
        "warmup_bias_lr": settings.warmup_bias_lr,
        "close_mosaic": settings.close_mosaic,
        "device": settings.device,
        "workers": settings.workers,
        "cache": False,
        "pretrained": True,
        "val": True,
        "split": "val",
        "plots": True,
        "save": True,
        "save_period": settings.save_period,
        "resume": False,
    }
    expected.update(settings.augmentation)
    for key, expected_value in expected.items():
        if args.get(key) != expected_value:
            raise RuntimeError(
                f"Executed argument {key} differs from the freeze"
            )
    return args


def _derived_best_epoch(rows: list[dict[str, str]]) -> tuple[int, float]:
    scores = []
    for row in rows:
        numeric = _numeric_row(row)
        score = (
            0.1 * numeric["metrics/mAP50(B)"]
            + 0.9 * numeric["metrics/mAP50-95(B)"]
        )
        scores.append((int(numeric["epoch"]), score))
    return max(scores, key=lambda item: item[1])


def finalize_existing_formal_run(
    *,
    config_path: Path,
    freeze_registry_path: Path,
    output_dir: Path,
    launcher_stdout: Path,
    launcher_stderr: Path,
) -> dict[str, Any]:
    """Recover the completed v1 run after its post-run callback audit failed.

    This reads existing artifacts only and writes new audit JSON files. It
    never loads a model, trains, validates, or predicts.
    """
    config_path = config_path.resolve()
    output_dir = output_dir.resolve()
    launcher_stdout = launcher_stdout.resolve()
    launcher_stderr = launcher_stderr.resolve()
    _, settings, freeze = load_formal_settings(
        config_path=config_path,
        freeze_registry_path=freeze_registry_path,
    )
    if not output_dir.is_dir():
        raise FileNotFoundError(f"Formal run directory is missing: {output_dir}")
    summary_path = output_dir / "formal_training_summary.json"
    resources_path = output_dir / "resource_metrics.json"
    if summary_path.exists() or resources_path.exists():
        raise FileExistsError("Refusing to overwrite recovered formal audit")

    required = {
        "preflight": output_dir / "preflight.json",
        "args": output_dir / "args.yaml",
        "results_csv": output_dir / "results.csv",
        "training_curves": output_dir / "results.png",
        "best": output_dir / "weights" / "best.pt",
        "last": output_dir / "weights" / "last.pt",
        "console": output_dir / "training_console.log",
        "retained_failure": output_dir / "formal_training_failure.json",
        "launcher_stdout": launcher_stdout,
        "launcher_stderr": launcher_stderr,
    }
    missing = [str(path) for path in required.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Formal recovery artifacts missing: {missing}")

    preflight = json.loads(
        required["preflight"].read_text(encoding="utf-8")
    )
    failure = json.loads(
        required["retained_failure"].read_text(encoding="utf-8")
    )
    if preflight.get("experiment_id") != settings.experiment_id:
        raise RuntimeError("Preflight experiment ID mismatch")
    if preflight.get("status") != "ready":
        raise RuntimeError("Formal preflight was not ready")
    if preflight["formal_config"]["sha256"] != sha256_file(config_path):
        raise RuntimeError("Preflight formal-config hash mismatch")
    if failure.get("error") != "Resource probe epoch count mismatch":
        raise RuntimeError("Recovery is limited to the known v1 audit failure")

    initial_weights = Path(preflight["initialization"]["weights"])
    data_yaml = Path(preflight["data"]["dataset_yaml"])
    if sha256_file(initial_weights) != settings.initial_weights_sha256:
        raise RuntimeError("Initial D0 weight changed after preflight")
    if sha256_file(data_yaml) != preflight["data"]["dataset_yaml_sha256"]:
        raise RuntimeError("Prepared dataset YAML changed after preflight")
    executed_args = _validated_executed_args(
        args_path=required["args"],
        settings=settings,
    )

    rows = _read_results(required["results_csv"])
    completed_epochs = len(rows)
    analysis = analyze_results_rows(
        rows,
        expected_epochs=completed_epochs,
    )
    if not 1 <= completed_epochs < settings.max_epochs:
        raise RuntimeError("Expected an early-stopped formal run")
    if not analysis["any_loss_changed"]:
        raise RuntimeError("Formal losses did not change")

    stdout_text = launcher_stdout.read_text(
        encoding="utf-8",
        errors="replace",
    )
    console_text = required["console"].read_text(
        encoding="utf-8",
        errors="replace",
    )
    combined_log = stdout_text + "\n" + console_text
    best_matches = [
        int(value)
        for value in re.findall(
            r"Best results observed at epoch (\d+)",
            combined_log,
        )
    ]
    if not best_matches:
        raise RuntimeError("Best epoch is absent from retained logs")
    reported_best_epoch = best_matches[-1]
    derived_best_epoch, derived_best_fitness = _derived_best_epoch(rows)
    if reported_best_epoch != derived_best_epoch:
        raise RuntimeError("Logged and result-derived best epochs disagree")
    best_zero, best_one = best_epoch_indices(
        reported_best_epoch,
        completed_epochs,
    )
    best_row = _numeric_row(rows[best_zero])
    best_metrics = {
        key: value
        for key, value in best_row.items()
        if key.startswith("metrics/")
    }

    gpu_gib_values = [
        float(value)
        for value in re.findall(r"(?<![\d.])(\d+(?:\.\d+)?)G\b", stdout_text)
    ]
    if not gpu_gib_values:
        raise RuntimeError("GPU memory readings are absent from stdout")
    peak_reserved_gib = max(gpu_gib_values)
    cumulative = [
        float(_numeric_row(row)["time"]) for row in rows
    ]
    epoch_times = [
        cumulative[0],
        *[
            cumulative[index] - cumulative[index - 1]
            for index in range(1, len(cumulative))
        ],
    ]
    if not all(math.isfinite(value) and value > 0 for value in epoch_times):
        raise RuntimeError("Recovered epoch timings are invalid")
    hours_matches = re.findall(
        r"(\d+) epochs completed in ([\d.]+) hours",
        combined_log,
    )
    if not hours_matches:
        raise RuntimeError("Ultralytics duration is absent from logs")
    logged_epochs, logged_hours = hours_matches[-1]
    if int(logged_epochs) != completed_epochs:
        raise RuntimeError("Logged epoch count differs from results.csv")

    lower_log = combined_log.lower()
    if "out of memory" in lower_log or "nan recovery" in lower_log:
        raise RuntimeError("OOM or NaN recovery appears in formal logs")
    checkpoints = {
        "initial": _artifact(initial_weights),
        "best": _artifact(required["best"]),
        "last": _artifact(required["last"]),
    }
    if any(
        checkpoints[name]["sha256"] == checkpoints["initial"]["sha256"]
        for name in ("best", "last")
    ):
        raise RuntimeError("Recovered formal checkpoint matches D0")

    start = datetime.fromisoformat(failure["started_at"])
    failed = datetime.fromisoformat(failure["failed_at"])
    runner_elapsed = (failed - start).total_seconds()
    resources = {
        "gpu_name": "NVIDIA GeForce RTX 3060 Laptop GPU",
        "gpu_total_memory_bytes": 6_441_926_656,
        "peak_reserved_gib_from_ultralytics_progress": peak_reserved_gib,
        "peak_reserved_bytes_approximate": round(
            peak_reserved_gib * 1024**3
        ),
        "peak_measurement_precision": (
            "Ultralytics progress display rounded to three significant "
            "digits; exact callback bytes were lost in the retained audit "
            "failure"
        ),
        "epoch_time_seconds_from_results_csv": epoch_times,
        "mean_epoch_time_seconds": float(np.mean(epoch_times)),
        "min_epoch_time_seconds": min(epoch_times),
        "max_epoch_time_seconds": max(epoch_times),
        "results_cumulative_time_seconds": cumulative[-1],
        "ultralytics_reported_training_hours": float(logged_hours),
        "runner_start_to_audit_failure_seconds": runner_elapsed,
        "configured_post_warmup_accumulation_steps": (
            settings.expected_accumulation
        ),
        "runtime_accumulation_probe_available": False,
        "dataloader_callback_metrics_available": False,
    }
    report = {
        "schema_version": 1,
        "experiment_id": settings.experiment_id,
        "run_id": output_dir.name,
        "status": "complete_with_retained_postrun_audit_failure",
        "recovered_without_retraining": True,
        "recovery_scope": {
            "model_loaded": False,
            "training_run": False,
            "validation_run": False,
            "prediction_run": False,
            "new_checkpoint_written": False,
        },
        "execution": {
            "started_at": failure["started_at"],
            "postrun_audit_failed_at": failure["failed_at"],
            "runner_start_to_audit_failure_seconds": runner_elapsed,
            "completed_epochs": completed_epochs,
            "early_stopped": True,
            "stop_reason": "early_stopping_patience",
            "patience": settings.patience,
            "best_epoch_one_based": best_one,
            "best_epoch_zero_based": best_zero,
            "best_fitness_derived": derived_best_fitness,
            "best_development_validation_metrics": best_metrics,
            "final_development_validation_metrics": analysis[
                "final_validation_metrics"
            ],
            "validation_images": int(
                preflight["data"]["prepared_counts"]["val"]["images"]
            ),
            "loss_analysis": analysis,
            "weights_updated": True,
            "nan_or_oom_observed": False,
            "batch_auto_reduced": False,
        },
        "protocol": {
            "formal_config": preflight["formal_config"],
            "freeze": freeze,
            "executed_arguments": executed_args,
        },
        "data": preflight["data"],
        "resources": resources,
        "checkpoints": checkpoints,
        "artifacts": {
            name: _artifact(path)
            for name, path in required.items()
            if name not in {"best", "last"}
        },
        "retained_engineering_failure": {
            "artifact": _artifact(required["retained_failure"]),
            "cause": (
                "The Stage F callback guard mistook Ultralytics' final "
                "best-checkpoint evaluation for a training epoch after early "
                "stopping, creating an extra resource record."
            ),
            "scientific_training_invalidated": False,
            "rerun_performed": False,
        },
        "scope": {
            "count_test_accessed": False,
            "cnr_ext_accessed": False,
            "pklot_accessed": False,
            "virat_accessed": False,
            "remote_or_paid_gpu_used": False,
        },
        "gate": {
            "stage_H_formal_training_passed": True,
            "stage_I_development_detector_comparison_may_start": True,
            "count_test_allowed_now": False,
            "slot_occupancy_prediction_allowed_now": False,
        },
    }
    _write_json(resources_path, resources)
    _write_json(summary_path, report)
    return report


def verify_formal_training_record(
    *,
    record_path: Path,
    implementation_root: Path,
) -> dict[str, Any]:
    record_path = record_path.resolve()
    implementation_root = implementation_root.resolve()
    record = _load_yaml(record_path)
    if record.get("record_id") != (
        "D1-NDISPARK-FORMAL-RECORD-20260727-01"
    ):
        raise FormalTrainingProtocolError(
            "Unexpected formal-training record ID"
        )
    if record.get("experiment_id") != FORMAL_EXPERIMENT_ID:
        raise FormalTrainingProtocolError(
            "Formal-training record experiment mismatch"
        )
    artifacts = record.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise FormalTrainingProtocolError(
            "Formal-training artifact list is empty"
        )

    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    for artifact in artifacts:
        role = str(artifact["role"])
        if role in seen:
            raise FormalTrainingProtocolError(
                f"Duplicate formal artifact role: {role}"
            )
        seen.add(role)
        path = (implementation_root / str(artifact["path"])).resolve()
        try:
            path.relative_to(implementation_root)
        except ValueError as exc:
            raise FormalTrainingProtocolError(
                f"Formal artifact escapes implementation root: {path}"
            ) from exc
        exists = path.is_file()
        actual_bytes = path.stat().st_size if exists else None
        actual_sha256 = sha256_file(path) if exists else None
        expected_bytes = int(artifact["bytes"])
        expected_sha256 = str(artifact["sha256"])
        results.append(
            {
                "role": role,
                "path": str(path),
                "exists": exists,
                "expected_bytes": expected_bytes,
                "actual_bytes": actual_bytes,
                "bytes_match": actual_bytes == expected_bytes,
                "expected_sha256": expected_sha256,
                "actual_sha256": actual_sha256,
                "sha256_match": actual_sha256 == expected_sha256,
            }
        )
    passed = all(
        item["exists"]
        and item["bytes_match"]
        and item["sha256_match"]
        for item in results
    )
    return {
        "schema_version": 1,
        "record_id": record["record_id"],
        "experiment_id": record["experiment_id"],
        "record_path": str(record_path),
        "implementation_root": str(implementation_root),
        "artifact_count": len(results),
        "artifacts": results,
        "passed": passed,
    }
