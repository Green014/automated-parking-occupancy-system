from __future__ import annotations

import argparse
import csv
from pathlib import Path

from parking_occupancy.sequence_io import evenly_spaced_indices


def _select(
    manifest_path: Path,
    split: str,
    count: int,
) -> list[dict[str, str]]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    selected = [rows[index] for index in evenly_spaced_indices(len(rows), count)]
    for row in selected:
        row["split"] = split
        row["annotation_status"] = "needs_manual_review"
        row["preannotation_source"] = "Grand Bassin machine-generated COCO"
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a video-level custom-label preparation manifest"
    )
    parser.add_argument("--train-manifest", required=True)
    parser.add_argument("--validation-manifest", required=True)
    parser.add_argument("--holdout-manifest", required=True)
    parser.add_argument("--frames-per-split", type=int, default=24)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selected = []
    selected.extend(
        _select(Path(args.train_manifest), "train", args.frames_per_split)
    )
    selected.extend(
        _select(
            Path(args.validation_manifest),
            "validation",
            args.frames_per_split,
        )
    )
    selected.extend(
        _select(
            Path(args.holdout_manifest),
            "holdout",
            args.frames_per_split,
        )
    )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(selected[0])
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(selected)
    print(
        f"Selected {len(selected)} frames across three disjoint video sequences"
    )


if __name__ == "__main__":
    main()
