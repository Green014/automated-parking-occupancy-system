from __future__ import annotations

import csv
import json
import logging
import math
import os
import platform
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
    validate_prepared_dataset,
)


class SmokeProtocolError(ValueError):
    """Raised when a smoke-run input conflicts with the frozen protocol."""


@dataclass(frozen=True)
class SmokeSettings:
    experiment_id: str
    epochs: int
    batch: int
    imgsz: int
    seed: int
    amp: bool
    deterministic: bool
    optimizer: str
    lr0: float
    weight_decay: float
    initial_weights_sha256: str


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def load_smoke_settings(config_path: Path) -> tuple[dict[str, Any], SmokeSettings]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != "DPROTO-NDISPARK-ONLY-20260727-01":
        raise SmokeProtocolError("Unexpected dataset protocol ID")
    if payload.get("status") != "frozen":
        raise SmokeProtocolError("Dataset protocol is not frozen")

    fixed = payload["training_configuration_space"]["fixed"]
    smoke = payload["training_configuration_space"]["smoke"]
    expected = {
        "imgsz": 640,
        "seed": 20260727,
        "amp": True,
        "deterministic": True,
        "optimizer": "AdamW",
        "lr0": 0.001,
        "weight_decay": 0.0005,
    }
    for key, value in expected.items():
        if fixed.get(key) != value:
            raise SmokeProtocolError(
                f"Frozen smoke setting {key} must be {value!r}"
            )
    if smoke.get("experiment_id") != "D1-NDISPARK-SMOKE-20260727-01":
        raise SmokeProtocolError("Unexpected smoke experiment ID")
    if smoke.get("epochs") != 3 or smoke.get("batch") != 4:
        raise SmokeProtocolError("Smoke run must use 3 epochs and batch 4")

    detector = payload["detector_comparison"]["models"]["D0"]
    settings = SmokeSettings(
        experiment_id=str(smoke["experiment_id"]),
        epochs=int(smoke["epochs"]),
        batch=int(smoke["batch"]),
        imgsz=int(fixed["imgsz"]),
        seed=int(fixed["seed"]),
        amp=bool(fixed["amp"]),
        deterministic=bool(fixed["deterministic"]),
        optimizer=str(fixed["optimizer"]),
        lr0=float(fixed["lr0"]),
        weight_decay=float(fixed["weight_decay"]),
        initial_weights_sha256=str(detector["weights_sha256"]),
    )
    return payload, settings


def _prepared_label_summary(
    data_yaml: Path,
    split: str,
) -> dict[str, int]:
    data = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(str(data["path"]))
    if not root.is_absolute():
        root = (data_yaml.parent / root).resolve()
    image_root = Path(str(data[split]))
    if not image_root.is_absolute():
        image_root = root / image_root
    relative = image_root.resolve().relative_to(root.resolve())
    parts = list(relative.parts)
    if not parts or parts[0] != "images":
        raise SmokeProtocolError(f"{split} does not resolve below images/")
    parts[0] = "labels"
    label_root = root.joinpath(*parts)
    label_paths = sorted(label_root.glob("*.txt"))
    empty = 0
    nonempty = 0
    for path in label_paths:
        if path.read_text(encoding="utf-8").strip():
            nonempty += 1
        else:
            empty += 1
    return {
        "label_files": len(label_paths),
        "nonempty_label_files": nonempty,
        "background_images": empty,
    }


def smoke_preflight(
    *,
    dataset_protocol_path: Path,
    comparison_protocol_path: Path,
    data_yaml: Path,
    initial_weights: Path,
    output_dir: Path,
    device: str,
    workers: int,
) -> tuple[dict[str, Any], SmokeSettings]:
    dataset_protocol_path = dataset_protocol_path.resolve()
    comparison_protocol_path = comparison_protocol_path.resolve()
    data_yaml = data_yaml.resolve()
    initial_weights = initial_weights.resolve()
    output_dir = output_dir.resolve()
    data_protocol, settings = load_smoke_settings(dataset_protocol_path)
    comparison_protocol, _ = load_comparison_protocol(
        comparison_protocol_path
    )
    if comparison_protocol["data"]["dataset_protocol_id"] != (
        data_protocol["protocol_id"]
    ):
        raise SmokeProtocolError("Dataset/comparison protocols disagree")

    prepared = validate_prepared_dataset(
        data_yaml=data_yaml,
        protocol=comparison_protocol,
    )
    preparation_record = _resolve_from_config(
        comparison_protocol_path,
        comparison_protocol["data"]["preparation_record"]["path"],
    )
    preparation = yaml.safe_load(
        preparation_record.read_text(encoding="utf-8")
    )
    expected_yaml = preparation["generated_artifacts"]["dataset_yaml"]
    if data_yaml.stat().st_size != int(expected_yaml["bytes"]):
        raise SmokeProtocolError("Prepared dataset YAML byte size mismatch")
    data_yaml_sha256 = sha256_file(data_yaml)
    if data_yaml_sha256 != str(expected_yaml["sha256"]):
        raise SmokeProtocolError("Prepared dataset YAML SHA-256 mismatch")

    if not initial_weights.is_file():
        raise SmokeProtocolError("Initial D0 weight file is missing")
    initial_hash = sha256_file(initial_weights)
    if initial_hash != settings.initial_weights_sha256:
        raise SmokeProtocolError("Initial D0 weight SHA-256 mismatch")
    if output_dir.exists():
        raise FileExistsError(
            f"Refusing to overwrite smoke output: {output_dir}"
        )
    if workers < 0:
        raise SmokeProtocolError("workers must be non-negative")
    if device not in {"0", "cuda:0"}:
        raise SmokeProtocolError(
            "Frozen Stage F run must use local CUDA device 0"
        )

    labels = {
        "train": _prepared_label_summary(data_yaml, "train"),
        "val": _prepared_label_summary(data_yaml, "val"),
    }
    if labels["train"]["background_images"] != 1:
        raise SmokeProtocolError(
            "Expected one legitimate background training image"
        )
    if labels["val"]["background_images"] != 0:
        raise SmokeProtocolError("Validation labels unexpectedly empty")

    report = {
        "schema_version": 1,
        "experiment_id": settings.experiment_id,
        "run_id": output_dir.name,
        "status": "ready",
        "predictions_run": False,
        "training_run": False,
        "data": {
            "dataset_yaml": str(data_yaml),
            "dataset_yaml_bytes": data_yaml.stat().st_size,
            "dataset_yaml_sha256": data_yaml_sha256,
            "prepared_counts": prepared,
            "label_summary": labels,
            "count_test_accessed": False,
        },
        "initialization": {
            "name": "COCO-pretrained YOLOv8n",
            "weights": str(initial_weights),
            "bytes": initial_weights.stat().st_size,
            "sha256": initial_hash,
        },
        "settings": asdict(settings),
        "runtime_only": {
            "device": device,
            "workers": workers,
            "output_dir": str(output_dir),
        },
        "gate": "open_for_local_smoke_only",
    }
    return report, settings


def analyze_results_rows(
    rows: list[dict[str, str]],
    *,
    expected_epochs: int,
) -> dict[str, Any]:
    if len(rows) != expected_epochs:
        raise RuntimeError(
            f"Expected {expected_epochs} result rows, found {len(rows)}"
        )
    normalized = [
        {
            str(key).strip(): str(value).strip()
            for key, value in row.items()
            if key is not None
        }
        for row in rows
    ]
    numeric: list[dict[str, float]] = []
    nonfinite: list[dict[str, Any]] = []
    for row_index, row in enumerate(normalized):
        parsed: dict[str, float] = {}
        for key, value in row.items():
            number = float(value)
            parsed[key] = number
            if not math.isfinite(number):
                nonfinite.append(
                    {"row": row_index + 1, "column": key, "value": value}
                )
        numeric.append(parsed)
    if nonfinite:
        raise RuntimeError(f"Non-finite training values: {nonfinite}")

    loss_columns = sorted(
        key
        for key in numeric[0]
        if key.startswith("train/") and key.endswith("_loss")
    )
    if not loss_columns:
        raise RuntimeError("No training loss columns found")
    loss_changes = {
        key: {
            "first": numeric[0][key],
            "last": numeric[-1][key],
            "delta_last_minus_first": numeric[-1][key] - numeric[0][key],
            "changed": not math.isclose(
                numeric[-1][key],
                numeric[0][key],
                rel_tol=0.0,
                abs_tol=1e-12,
            ),
        }
        for key in loss_columns
    }
    metric_columns = sorted(
        key for key in numeric[0] if key.startswith("metrics/")
    )
    if not metric_columns:
        raise RuntimeError("No validation metric columns found")
    return {
        "epochs_recorded": len(numeric),
        "all_numeric_values_finite": True,
        "loss_columns": loss_columns,
        "loss_changes": loss_changes,
        "any_loss_changed": any(
            item["changed"] for item in loss_changes.values()
        ),
        "validation_metric_columns": metric_columns,
        "final_validation_metrics": {
            key: numeric[-1][key] for key in metric_columns
        },
        "cumulative_time_seconds": [row["time"] for row in numeric],
    }


def _tensor_items(value: Any) -> dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result = {}
    for key, item in value.items():
        if hasattr(item, "detach"):
            item = item.detach().cpu().item()
        result[str(key)] = float(item)
    return result


class TrainingResourceProbe:
    """Ultralytics callback probe for epoch, batch, and CUDA measurements."""

    def __init__(self) -> None:
        self.epochs: list[dict[str, Any]] = []
        self._epoch_start = 0.0
        self._wait_anchor = 0.0
        self._batch_start = 0.0
        self._wait_seconds: list[float] = []
        self._compute_seconds: list[float] = []

    def on_train_epoch_start(self, trainer: Any) -> None:
        import torch

        self._epoch_start = time.perf_counter()
        self._wait_anchor = self._epoch_start
        self._wait_seconds = []
        self._compute_seconds = []
        if trainer.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(trainer.device)

    def on_train_batch_start(self, trainer: Any) -> None:
        now = time.perf_counter()
        self._wait_seconds.append(now - self._wait_anchor)
        self._batch_start = now

    def on_train_batch_end(self, trainer: Any) -> None:
        now = time.perf_counter()
        self._compute_seconds.append(now - self._batch_start)
        self._wait_anchor = now

    def on_fit_epoch_end(self, trainer: Any) -> None:
        import torch

        if int(trainer.epoch) >= int(trainer.epochs):
            return
        wait_total = float(sum(self._wait_seconds))
        compute_total = float(sum(self._compute_seconds))
        observed = wait_total + compute_total
        record = {
            "epoch": int(trainer.epoch) + 1,
            "epoch_time_seconds": float(
                trainer.epoch_time
                if trainer.epoch_time is not None
                else time.perf_counter() - self._epoch_start
            ),
            "batches": len(self._compute_seconds),
            "train_batch_wait_seconds_total": wait_total,
            "train_batch_compute_seconds_total": compute_total,
            "train_batch_wait_fraction": (
                wait_total / observed if observed > 0 else None
            ),
            "train_batch_wait_mean_seconds": (
                float(np.mean(self._wait_seconds))
                if self._wait_seconds
                else None
            ),
            "train_batch_wait_p95_seconds": (
                float(np.percentile(self._wait_seconds, 95))
                if self._wait_seconds
                else None
            ),
            "train_batch_compute_mean_seconds": (
                float(np.mean(self._compute_seconds))
                if self._compute_seconds
                else None
            ),
            "train_losses": _tensor_items(trainer.tloss),
            "validation_metrics": {
                str(key): float(value)
                for key, value in (trainer.metrics or {}).items()
                if isinstance(value, (int, float))
            },
            "actual_batch": int(trainer.batch_size),
        }
        if trainer.device.type == "cuda":
            record.update(
                {
                    "cuda_peak_allocated_bytes": int(
                        torch.cuda.max_memory_allocated(trainer.device)
                    ),
                    "cuda_peak_reserved_bytes": int(
                        torch.cuda.max_memory_reserved(trainer.device)
                    ),
                }
            )
        self.epochs.append(record)

    def dataloader_assessment(self) -> dict[str, Any]:
        fractions = [
            float(item["train_batch_wait_fraction"])
            for item in self.epochs
            if item["train_batch_wait_fraction"] is not None
        ]
        mean_fraction = float(np.mean(fractions)) if fractions else None
        if mean_fraction is None:
            finding = "not_measured"
        elif mean_fraction >= 0.30:
            finding = "likely_material_wait_or_loader_overhead"
        elif mean_fraction >= 0.15:
            finding = "moderate_wait_or_loader_overhead"
        else:
            finding = "no_material_loader_bottleneck_observed"
        return {
            "definition": (
                "Time from the previous training-batch callback to the next "
                "batch-start callback divided by observed train batch time."
            ),
            "mean_wait_fraction": mean_fraction,
            "heuristic": "material>=0.30; moderate>=0.15",
            "finding": finding,
        }


def _read_results_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )


def run_smoke(
    *,
    dataset_protocol_path: Path,
    comparison_protocol_path: Path,
    data_yaml: Path,
    initial_weights: Path,
    output_dir: Path,
    device: str,
    workers: int,
) -> dict[str, Any]:
    preflight, settings = smoke_preflight(
        dataset_protocol_path=dataset_protocol_path,
        comparison_protocol_path=comparison_protocol_path,
        data_yaml=data_yaml,
        initial_weights=initial_weights,
        output_dir=output_dir,
        device=device,
        workers=workers,
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
            raise RuntimeError("CUDA is unavailable for the frozen smoke run")
        cuda_index = 0
        gpu_name = torch.cuda.get_device_name(cuda_index)
        if "RTX 3060 Laptop GPU" not in gpu_name:
            raise RuntimeError(
                f"Unexpected GPU for frozen local smoke run: {gpu_name}"
            )
        properties = torch.cuda.get_device_properties(cuda_index)
        free_before, total_memory = torch.cuda.mem_get_info(cuda_index)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(cuda_index)

        model = YOLO(str(initial_weights.resolve()))
        probe = TrainingResourceProbe()
        model.add_callback(
            "on_train_epoch_start",
            probe.on_train_epoch_start,
        )
        model.add_callback(
            "on_train_batch_start",
            probe.on_train_batch_start,
        )
        model.add_callback(
            "on_train_batch_end",
            probe.on_train_batch_end,
        )
        model.add_callback("on_fit_epoch_end", probe.on_fit_epoch_end)

        training_args = {
            "data": str(data_yaml.resolve()),
            "epochs": settings.epochs,
            "imgsz": settings.imgsz,
            "batch": settings.batch,
            "seed": settings.seed,
            "deterministic": settings.deterministic,
            "amp": settings.amp,
            "optimizer": settings.optimizer,
            "lr0": settings.lr0,
            "weight_decay": settings.weight_decay,
            "device": device,
            "workers": workers,
            "cache": False,
            "val": True,
            "plots": True,
            "save": True,
            "save_period": 1,
            "project": str(output_dir.parent),
            "name": output_dir.name,
            "exist_ok": True,
            "resume": False,
            "verbose": True,
        }
        wall_start = time.perf_counter()
        model.train(**training_args)
        wall_seconds = time.perf_counter() - wall_start
        trainer = model.trainer
        if trainer is None:
            raise RuntimeError("Ultralytics did not expose a trainer")

        results_csv = output_dir / "results.csv"
        best = output_dir / "weights" / "best.pt"
        last = output_dir / "weights" / "last.pt"
        required = [results_csv, best, last, output_dir / "args.yaml"]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            raise RuntimeError(f"Smoke artifacts missing: {missing}")
        analysis = analyze_results_rows(
            _read_results_csv(results_csv),
            expected_epochs=settings.epochs,
        )
        if not analysis["any_loss_changed"]:
            raise RuntimeError("Training losses did not change")

        peak_reserved_candidates = [
            int(item.get("cuda_peak_reserved_bytes", 0))
            for item in probe.epochs
        ]
        peak_allocated_candidates = [
            int(item.get("cuda_peak_allocated_bytes", 0))
            for item in probe.epochs
        ]
        peak_reserved_candidates.append(
            int(torch.cuda.max_memory_reserved(cuda_index))
        )
        peak_allocated_candidates.append(
            int(torch.cuda.max_memory_allocated(cuda_index))
        )
        peak_reserved = max(peak_reserved_candidates)
        peak_allocated = max(peak_allocated_candidates)
        actual_batches = sorted(
            {int(item["actual_batch"]) for item in probe.epochs}
        )
        if actual_batches != [settings.batch]:
            raise RuntimeError(
                f"Configured batch changed during smoke run: {actual_batches}"
            )
        validator_seen = int(
            getattr(getattr(trainer, "validator", None), "seen", 0)
        )
        expected_val_images = int(
            preflight["data"]["prepared_counts"]["val"]["images"]
        )
        if validator_seen != expected_val_images:
            raise RuntimeError(
                f"Validation saw {validator_seen}, expected "
                f"{expected_val_images} images"
            )

        checkpoints = {
            "initial": {
                "path": str(initial_weights.resolve()),
                "bytes": initial_weights.stat().st_size,
                "sha256": sha256_file(initial_weights),
            },
            "best": {
                "path": str(best),
                "bytes": best.stat().st_size,
                "sha256": sha256_file(best),
            },
            "last": {
                "path": str(last),
                "bytes": last.stat().st_size,
                "sha256": sha256_file(last),
            },
        }
        weights_updated = all(
            checkpoints[name]["sha256"]
            != checkpoints["initial"]["sha256"]
            for name in ("best", "last")
        )
        if not weights_updated:
            raise RuntimeError("Fine-tuned checkpoints match initialization")

        epoch_times = [
            float(item["epoch_time_seconds"]) for item in probe.epochs
        ]
        if len(epoch_times) != settings.epochs:
            raise RuntimeError(
                f"Resource probe recorded {len(epoch_times)} epochs"
            )
        ended_at = datetime.now().astimezone()
        report = {
            "schema_version": 1,
            "experiment_id": settings.experiment_id,
            "run_id": output_dir.name,
            "status": "complete",
            "started_at": started_at.isoformat(),
            "ended_at": ended_at.isoformat(),
            "duration_seconds": wall_seconds,
            "protocol": {
                "dataset": str(dataset_protocol_path.resolve()),
                "comparison": str(comparison_protocol_path.resolve()),
            },
            "data": preflight["data"],
            "training": {
                "arguments": training_args,
                "epochs_completed": len(epoch_times),
                "epoch_time_seconds": epoch_times,
                "mean_epoch_time_seconds": float(np.mean(epoch_times)),
                "min_epoch_time_seconds": min(epoch_times),
                "max_epoch_time_seconds": max(epoch_times),
                "results_analysis": analysis,
                "nan_detected": False,
                "oom_detected": False,
                "batch_auto_reduced": False,
                "validation_inference_succeeded": True,
                "validation_images_seen": validator_seen,
                "weights_updated": weights_updated,
            },
            "resources": {
                "gpu_name": gpu_name,
                "gpu_total_memory_bytes": int(properties.total_memory),
                "cuda_mem_get_info_total_bytes": int(total_memory),
                "cuda_free_before_training_bytes": int(free_before),
                "cuda_peak_allocated_bytes": peak_allocated,
                "cuda_peak_reserved_bytes": peak_reserved,
                "cuda_peak_reserved_fraction": (
                    peak_reserved / int(properties.total_memory)
                ),
                "per_epoch": probe.epochs,
                "dataloader": probe.dataloader_assessment(),
            },
            "checkpoints": checkpoints,
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
            },
            "gate": {
                "stage_F_smoke_passed": True,
                "stage_G_gpu_decision_may_start": True,
                "formal_training_allowed_now": False,
            },
        }
        _write_json(output_dir / "resource_metrics.json", report["resources"])
        _write_json(output_dir / "smoke_summary.json", report)
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
        _write_json(output_dir / "smoke_failure.json", failure)
        raise
    finally:
        if log_handler is not None:
            from ultralytics.utils import LOGGER

            LOGGER.removeHandler(log_handler)
            log_handler.close()
