from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file
from parking_occupancy.stage_n_lmot_v2 import EXPECTED_VALIDATION_SEQUENCES


EXPECTED_METHODS = ("L0", "L1", "L2", "L3")
EXPECTED_MOTOR_VEHICLE_GT = 68887


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify a completed frozen Stage N-v2 output"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "stage_n_v2_lmot_tracking_diagnostic_frozen_20260729.yaml"
        ),
    )
    parser.add_argument(
        "--class-map",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "stage_n_v2"
            / "LMOT_CLASS_MAP_FROZEN_20260729.yaml"
        ),
    )
    return parser.parse_args()


def _finite_numbers(value: Any) -> bool:
    if isinstance(value, dict):
        return all(_finite_numbers(row) for row in value.values())
    if isinstance(value, list):
        return all(_finite_numbers(row) for row in value)
    if isinstance(value, float):
        return math.isfinite(value)
    return True


def main() -> None:
    args = parse_args()
    root = args.output_root.resolve()
    required_files = {
        "aggregate_metrics.json",
        "sequence_metrics.json",
        "sequence_audits.json",
        "runtime_metadata.json",
        "configuration_snapshot.yaml",
        "class_map_snapshot.yaml",
    }
    observed_files = {
        path.name for path in root.iterdir() if path.is_file()
    }
    missing = sorted(required_files - observed_files)
    if missing:
        raise ValueError(f"Missing Stage N-v2 output files: {missing}")
    if sha256_file(root / "configuration_snapshot.yaml") != sha256_file(
        args.config.resolve()
    ):
        raise ValueError("Configuration snapshot differs from frozen config")
    if sha256_file(root / "class_map_snapshot.yaml") != sha256_file(
        args.class_map.resolve()
    ):
        raise ValueError("Class-map snapshot differs from frozen class map")

    aggregate = json.loads(
        (root / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    sequence_metrics = json.loads(
        (root / "sequence_metrics.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (root / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    if not _finite_numbers(aggregate) or not _finite_numbers(sequence_metrics):
        raise ValueError("Metrics contain non-finite values")
    if tuple(aggregate["methods"]) != EXPECTED_METHODS:
        raise ValueError("Unexpected aggregate method order")
    if tuple(sequence_metrics) != EXPECTED_METHODS:
        raise ValueError("Unexpected sequence-metric method order")
    for method in EXPECTED_METHODS:
        if tuple(sequence_metrics[method]) != EXPECTED_VALIDATION_SEQUENCES:
            raise ValueError(f"Unexpected sequences for {method}")
        tracking = aggregate["methods"][method]["tracking"]
        if (
            int(tracking["true_positives"])
            + int(tracking["false_negatives"])
            != EXPECTED_MOTOR_VEHICLE_GT
        ):
            raise ValueError(f"GT accounting mismatch for {method}")
        for sequence in EXPECTED_VALIDATION_SEQUENCES:
            if sequence_metrics[method][sequence]["runtime"]["frames"] != 1210:
                raise ValueError(f"Frame count mismatch for {method}/{sequence}")

    for directory in ("detections", "tracks", "qualitative_frames"):
        paths = [
            path for path in (root / directory).rglob("*") if path.is_file()
        ]
        if len(paths) != 16:
            raise ValueError(
                f"{directory} contains {len(paths)} files instead of 16"
            )
    if runtime["processed_frames"] != 19360:
        raise ValueError("Runtime processed-frame total mismatch")
    if runtime["parameter_tuning_from_results"] is not False:
        raise ValueError("LMOT result-driven tuning flag must be false")
    if (
        runtime["trackeval"]["official_commit"]
        != "12c8791b303e0a0b50f753af204249e622d0281a"
    ):
        raise ValueError("Unexpected TrackEval commit")
    print(
        json.dumps(
            {
                "status": "passed",
                "methods": list(EXPECTED_METHODS),
                "sequences": list(EXPECTED_VALIDATION_SEQUENCES),
                "processed_frames": runtime["processed_frames"],
                "motor_vehicle_gt": EXPECTED_MOTOR_VEHICLE_GT,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
