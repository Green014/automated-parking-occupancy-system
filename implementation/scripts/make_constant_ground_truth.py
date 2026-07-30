from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create a full-frame constant-state ground-truth CSV"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--slots", required=True)
    parser.add_argument("--video-id", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--state", choices=("occupied", "vacant"), required=True)
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    slot_map = json.loads(Path(args.slots).read_text(encoding="utf-8"))
    slot_ids = [str(slot["id"]) for slot in slot_map["slots"]]
    state = int(args.state == "occupied")

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "frame_index",
                "timestamp_s",
                "slot_id",
                "state",
                "evidence",
            ],
        )
        writer.writeheader()
        for frame_index, _row in enumerate(manifest):
            for slot_id in slot_ids:
                writer.writerow(
                    {
                        "video_id": args.video_id,
                        "frame_index": frame_index,
                        "timestamp_s": f"{frame_index / args.fps:.3f}",
                        "slot_id": slot_id,
                        "state": state,
                        "evidence": state,
                    }
                )
    print(
        f"Wrote {len(manifest) * len(slot_ids)} labels "
        f"for {len(slot_ids)} slots and {len(manifest)} frames"
    )


if __name__ == "__main__":
    main()
