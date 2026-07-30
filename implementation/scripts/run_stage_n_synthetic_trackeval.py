from __future__ import annotations

import argparse
import json
import platform
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import (
    LmotAnnotation,
    OfficialTrackEvalAdapter,
    TrackPrediction,
    sha256_file,
    write_image,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_n_lmot_tracking_diagnostic_frozen_20260728.yaml"
)


def _gt(frame: int) -> LmotAnnotation:
    return LmotAnnotation(frame, 1, 10, 10, 20, 20, 0, 1, 1.0)


def _prediction(frame: int, track_id: int, x: float = 10) -> TrackPrediction:
    return TrackPrediction(frame, track_id, (x, 10, x + 20, 30), 0.9)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create an official-TrackEval synthetic Stage N artifact"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    gt = [_gt(frame) for frame in range(1, 5)]
    scenarios = {
        "perfect": (4, gt, [_prediction(frame, 11) for frame in range(1, 5)]),
        "id_switch": (
            4,
            gt,
            [
                _prediction(1, 11),
                _prediction(2, 11),
                _prediction(3, 12),
                _prediction(4, 12),
            ],
        ),
        "missed_detection": (
            4,
            gt,
            [_prediction(frame, 11) for frame in range(1, 4)],
        ),
        "false_positive": (
            4,
            gt,
            [_prediction(frame, 11) for frame in range(1, 5)]
            + [_prediction(2, 99, x=100)],
        ),
    }
    adapter = OfficialTrackEvalAdapter()
    sequence_metrics, aggregate = adapter.evaluate_many(scenarios)
    output_root.mkdir(parents=True)
    for directory in ("detections", "tracks", "qualitative_frames"):
        (output_root / directory).mkdir()
    for name, (_frames, _gt_rows, predictions) in scenarios.items():
        lines = [
            json.dumps(
                {
                    "frame": row.frame_number,
                    "track_id": row.track_id,
                    "bbox_xyxy": list(row.xyxy),
                    "confidence": row.confidence,
                }
            )
            for row in predictions
        ]
        content = "\n".join(lines) + "\n"
        (output_root / "detections" / f"{name}.jsonl").write_text(
            content, encoding="utf-8"
        )
        (output_root / "tracks" / f"{name}.jsonl").write_text(
            content, encoding="utf-8"
        )
    image = np.zeros((80, 160, 3), dtype=np.uint8)
    cv2.putText(
        image,
        "SYNTHETIC ONLY",
        (8, 45),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
    )
    write_image(
        output_root / "qualitative_frames" / "synthetic_only.jpg",
        image,
    )
    (output_root / "sequence_metrics.json").write_text(
        json.dumps(sequence_metrics, indent=2) + "\n", encoding="utf-8"
    )
    (output_root / "aggregate_metrics.json").write_text(
        json.dumps(
            {
                "status": "synthetic_adapter_verification_only",
                "not_an_lmot_result": True,
                "metrics": aggregate,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    config_path = args.config.resolve()
    shutil.copyfile(config_path, output_root / "configuration_snapshot.yaml")
    runtime = {
        "status": "synthetic_adapter_verification_only",
        "actual_lmot_frames": 0,
        "actual_download_bytes": 0,
        "python": platform.python_version(),
        "trackeval": adapter.runtime_metadata(),
        "config_sha256": sha256_file(config_path),
    }
    (output_root / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"sequence_metrics": sequence_metrics, "aggregate": aggregate}, indent=2))


if __name__ == "__main__":
    main()
