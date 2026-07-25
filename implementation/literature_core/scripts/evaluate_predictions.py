from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.metrics import (  # noqa: E402
    binary_metrics,
    sequence_temporal_metrics,
)

Key = tuple[str, int, str]


def read_states(path: str | Path) -> dict[Key, dict[str, float | int]]:
    rows = {}
    with Path(path).open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["video_id"], int(row["frame_index"]), row["slot_id"])
            rows[key] = {
                "state": int(row["state"]),
                "timestamp_s": float(row.get("timestamp_s", 0.0)),
            }
    return rows


def summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    values.sort()
    return {
        "count": len(values),
        "median": statistics.median(values),
        "p90": values[round((len(values) - 1) * 0.9)],
        "maximum": values[-1],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate slot classification and corrected temporal metrics"
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--tolerance-frames", type=int, default=0)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    truth = read_states(args.ground_truth)
    prediction = read_states(args.predictions)
    if set(truth) != set(prediction):
        raise ValueError(
            f"Truth/prediction key mismatch: "
            f"missing={len(set(truth) - set(prediction))}, "
            f"extra={len(set(prediction) - set(truth))}"
        )
    keys = sorted(truth)
    classification = binary_metrics(
        [int(truth[key]["state"]) for key in keys],
        [int(prediction[key]["state"]) for key in keys],
    )
    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for video_id, frame_index, slot_id in keys:
        grouped[(video_id, slot_id)].append(frame_index)

    per_sequence: dict[str, Any] = {}
    total_flicker = 0
    total_instability = 0
    total_frames = 0
    entry_latency: list[float] = []
    exit_latency: list[float] = []
    for (video_id, slot_id), frames in sorted(grouped.items()):
        frames.sort()
        y_true = [
            int(truth[(video_id, frame, slot_id)]["state"]) for frame in frames
        ]
        y_pred = [
            int(prediction[(video_id, frame, slot_id)]["state"])
            for frame in frames
        ]
        metrics = sequence_temporal_metrics(
            y_true,
            y_pred,
            args.fps,
            stable_frames=args.stable_frames,
            tolerance_frames=args.tolerance_frames,
        )
        per_sequence[f"{video_id}/{slot_id}"] = metrics
        total_flicker += metrics["unsupported_flicker_count"]
        total_instability += metrics["transition_instability_changes"]
        total_frames += len(frames)
        entry_latency.extend(
            metrics["transition_latency_values_s"]["entry"]
        )
        exit_latency.extend(metrics["transition_latency_values_s"]["exit"])

    slot_minutes = total_frames / args.fps / 60.0
    report = {
        "classification": classification,
        "temporal": {
            "sequences": len(per_sequence),
            "unsupported_flicker_count": total_flicker,
            "flicker_rate_per_slot_minute": (
                total_flicker / slot_minutes if slot_minutes else 0.0
            ),
            "transition_instability_changes": total_instability,
            "transition_latency_s": {
                "all": summary(entry_latency + exit_latency),
                "entry": summary(entry_latency),
                "exit": summary(exit_latency),
            },
            "stable_frames": args.stable_frames,
            "tolerance_frames": args.tolerance_frames,
            "per_video_slot": per_sequence,
        },
        "ground_truth": str(Path(args.ground_truth).resolve()),
        "predictions": str(Path(args.predictions).resolve()),
        "fps": args.fps,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

