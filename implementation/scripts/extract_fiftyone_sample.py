from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def first_sample_from_partial(path: Path) -> dict:
    """Parse the first sample from a byte-range prefix of FiftyOne samples.json."""

    text = path.read_text(encoding="utf-8")
    array_start = text.index("[") + 1
    sample, _end = json.JSONDecoder().raw_decode(text, array_start)
    return sample


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="samples.json prefix")
    parser.add_argument("--slots-output", required=True)
    parser.add_argument("--truth-output", required=True)
    parser.add_argument("--width", type=int, required=True)
    parser.add_argument("--height", type=int, required=True)
    parser.add_argument(
        "--video-id",
        help="Identifier used in truth rows; defaults to the source image stem",
    )
    parser.add_argument(
        "--frames",
        type=int,
        default=1,
        help="Repeat the static labels for this many smoke-test frames",
    )
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()
    if args.frames < 1:
        parser.error("--frames must be at least 1")
    if args.fps <= 0:
        parser.error("--fps must be positive")

    sample = first_sample_from_partial(Path(args.input))
    source_path = sample["filepath"]
    slot_items = []
    slot_truth = []
    for polyline in sample["parking_spaces"]["polylines"]:
        slot_id = f"slot_{int(polyline['space_id']):03d}"
        points = [
            [point[0] * args.width, point[1] * args.height]
            for point in polyline["points"][0]
        ]
        slot_items.append({"id": slot_id, "points": points})
        slot_truth.append(
            (slot_id, int(polyline["occupancy_status"] == "occupied"))
        )

    video_id = args.video_id or Path(source_path).stem
    truth_rows = [
        {
            "video_id": video_id,
            "frame_index": frame_index,
            "timestamp_s": f"{frame_index / args.fps:.6f}",
            "slot_id": slot_id,
            "state": state,
        }
        for frame_index in range(args.frames)
        for slot_id, state in slot_truth
    ]

    slots_path = Path(args.slots_output)
    slots_path.parent.mkdir(parents=True, exist_ok=True)
    with slots_path.open("w", encoding="utf-8") as handle:
        json.dump(
            {
                "schema_version": 1,
                "source": source_path,
                "source_width": args.width,
                "source_height": args.height,
                "coordinate_system": "pixel",
                "slots": slot_items,
            },
            handle,
            indent=2,
        )
        handle.write("\n")

    truth_path = Path(args.truth_output)
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    with truth_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "frame_index",
                "timestamp_s",
                "slot_id",
                "state",
            ],
        )
        writer.writeheader()
        writer.writerows(truth_rows)
    print(
        f"Extracted {len(slot_items)} slots and {len(truth_rows)} truth rows "
        f"from {source_path}"
    )


if __name__ == "__main__":
    main()
