from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from .evaluate import binary_metrics
from .stage_j_posthoc_analysis import paired_bootstrap_mean_difference
from .stage_l_video import _method_metrics


STATIC_METHOD_FIELDS = {
    "P1_D1_B1": "p1_prediction",
    "P3_static_gate": "p3_prediction",
}
VIDEO_METHODS = (
    "p1_raw",
    "p3_gate",
    "p3_temporal",
    "p3_full_tracking_temporal",
)


def _load_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def analyze_static_predictions(
    predictions_path: Path,
    *,
    seed: int = 20260728,
    resamples: int = 2000,
) -> dict[str, Any]:
    rows = _load_csv(predictions_path)
    if not rows:
        raise ValueError("Static predictions are empty")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    cameras: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["sample_id"]].append(row)
        cameras[row["camera"]].append(row)

    sample_metrics: dict[str, dict[str, float]] = {}
    differences = {}
    outcomes = Counter()
    for sample_id, sample_rows in sorted(grouped.items()):
        truth = [int(row["truth"]) for row in sample_rows]
        scores = {
            method: float(
                binary_metrics(
                    truth,
                    [int(row[field]) for row in sample_rows],
                )["macro_f1"]
            )
            for method, field in STATIC_METHOD_FIELDS.items()
        }
        difference = scores["P3_static_gate"] - scores["P1_D1_B1"]
        outcome = (
            "win"
            if difference > 1e-12
            else "loss"
            if difference < -1e-12
            else "tie"
        )
        outcomes[outcome] += 1
        differences[sample_id] = difference
        sample_metrics[sample_id] = {
            **scores,
            "P3_minus_P1": difference,
        }

    camera_metrics = {}
    for camera, camera_rows in sorted(cameras.items()):
        truth = [int(row["truth"]) for row in camera_rows]
        camera_metrics[camera] = {
            method: float(
                binary_metrics(
                    truth,
                    [int(row[field]) for row in camera_rows],
                )["macro_f1"]
            )
            for method, field in STATIC_METHOD_FIELDS.items()
        }
    return {
        "schema_version": 1,
        "analysis": "read_only_stage_l_static_paired_analysis",
        "source": str(predictions_path.resolve()),
        "images": len(grouped),
        "slot_rows": len(rows),
        "outcomes": {
            key: int(outcomes.get(key, 0))
            for key in ("win", "tie", "loss")
        },
        "paired_bootstrap": paired_bootstrap_mean_difference(
            differences,
            seed=seed,
            resamples=resamples,
            confidence_level=0.95,
        ),
        "camera_macro_f1": {
            method: statistics.fmean(
                metrics[method] for metrics in camera_metrics.values()
            )
            for method in STATIC_METHOD_FIELDS
        },
        "by_camera": camera_metrics,
        "per_sample": sample_metrics,
        "model_prediction_rerun": False,
        "parameter_selection_performed": False,
    }


def analyze_video_predictions(
    occupancy_path: Path,
    *,
    fps: float,
    warmup_frames: int,
    stable_frames: int,
) -> dict[str, Any]:
    rows = _load_csv(occupancy_path)
    if not rows:
        raise ValueError("Video occupancy rows are empty")
    evaluated = [
        row
        for row in rows
        if int(row["frame_index"]) >= warmup_frames
    ]
    methods = {
        method: _method_metrics(
            evaluated,
            method,
            fps,
            stable_frames,
            frame_offset=warmup_frames,
        )
        for method in VIDEO_METHODS
    }
    transition_frame = min(
        int(row["frame_index"])
        for row in rows
        if int(row["truth"]) == 0
    )
    after = [
        row
        for row in rows
        if int(row["frame_index"]) >= transition_frame
    ]
    return {
        "schema_version": 2,
        "analysis": "read_only_stage_l_video_absolute_frame_analysis",
        "source": str(occupancy_path.resolve()),
        "frames": len(rows),
        "warmup_frames": warmup_frames,
        "truth_transition_frame_absolute": transition_frame,
        "methods": methods,
        "diagnostics": {
            "gate_branch_counts": dict(
                sorted(Counter(row["gate_branch"] for row in rows).items())
            ),
            "full_branch_counts": dict(
                sorted(Counter(row["full_branch"] for row in rows).items())
            ),
            "post_transition_detector_positive_frames": sum(
                int(row["p1_raw"]) == 1 for row in after
            ),
            "post_transition_frames": len(after),
            "post_transition_classifier_calls": sum(
                str(row["p_cls"]).strip() != "" for row in after
            ),
            "interpretation": (
                "D1+B1 remained detector-positive after departure, so E1b "
                "was never called; the stationary-track branch confirmed the "
                "wrong geometric association and temporal filtering retained it."
            ),
        },
        "model_prediction_rerun": False,
        "parameter_selection_performed": False,
    }


def write_new_json(path: Path, payload: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite analysis: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
