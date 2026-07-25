from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

from parking_occupancy.evaluate import binary_metrics


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Sweep EMA hysteresis thresholds over cached raw evidence"
    )
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--occupied-thresholds", default="0.02,0.04,0.06,0.08,0.10")
    parser.add_argument("--vacant-ratio", type=float, default=0.25)
    parser.add_argument("--rise-alpha", type=float, default=0.60)
    parser.add_argument("--fall-alpha", type=float, default=0.15)
    parser.add_argument("--warmup-frames", type=int, default=6)
    parser.add_argument("--fps", type=float, required=True)
    args = parser.parse_args()

    with Path(args.predictions).open(newline="", encoding="utf-8") as handle:
        prediction_rows = list(csv.DictReader(handle))
    with Path(args.ground_truth).open(newline="", encoding="utf-8") as handle:
        truth_rows = list(csv.DictReader(handle))
    truth = {
        (row["video_id"], int(row["frame_index"]), row["slot_id"]): int(
            row["state"]
        )
        for row in truth_rows
    }
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in prediction_rows:
        grouped[(row["video_id"], row["slot_id"])].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row: int(row["frame_index"]))

    results: list[dict[str, float | int]] = []
    thresholds = [
        float(value)
        for value in args.occupied_thresholds.split(",")
        if value.strip()
    ]
    for occupied_threshold in thresholds:
        vacant_threshold = occupied_threshold * args.vacant_ratio
        predicted: dict[tuple[str, int, str], int] = {}
        unsupported_changes = 0
        for (video_id, slot_id), rows in grouped.items():
            score = 0.0
            occupied = False
            previous = False
            for row in rows:
                frame_index = int(row["frame_index"])
                evidence = float(row["evidence"])
                alpha = (
                    args.rise_alpha
                    if evidence >= score
                    else args.fall_alpha
                )
                score = alpha * evidence + (1.0 - alpha) * score
                if not occupied and score >= occupied_threshold:
                    occupied = True
                elif occupied and score <= vacant_threshold:
                    occupied = False
                predicted[(video_id, frame_index, slot_id)] = int(occupied)
                if (
                    frame_index >= args.warmup_frames
                    and occupied != previous
                    and truth[(video_id, frame_index, slot_id)]
                    == truth.get(
                        (video_id, frame_index - 1, slot_id),
                        truth[(video_id, frame_index, slot_id)],
                    )
                ):
                    unsupported_changes += 1
                previous = occupied

        keys = sorted(truth)
        metrics = binary_metrics(
            [truth[key] for key in keys],
            [predicted[key] for key in keys],
        )
        total_slot_minutes = (
            sum(
                int(key[1] >= args.warmup_frames)
                for key in keys
            )
            / args.fps
            / 60.0
        )
        results.append(
            {
                "occupied_threshold": occupied_threshold,
                "vacant_threshold": vacant_threshold,
                "precision": float(metrics["precision"]),
                "recall": float(metrics["recall"]),
                "f1": float(metrics["f1"]),
                "false_free_rate": float(metrics["false_free_rate"]),
                "unsupported_flicker_count": unsupported_changes,
                "flicker_rate_per_slot_minute": (
                    unsupported_changes / total_slot_minutes
                    if total_slot_minutes
                    else 0.0
                ),
            }
        )

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    for result in results:
        print(result)


if __name__ == "__main__":
    main()
