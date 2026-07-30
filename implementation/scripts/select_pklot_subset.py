from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any
from urllib.parse import quote

from pklot_metadata import iter_fiftyone_samples, label_value, scalar_date

BASE_URL = "https://huggingface.co/datasets/Voxel51/PKLot/resolve/main"
TARGET_OCCUPANCY = (0.20, 0.50, 0.80)


def summarize(sample: dict[str, Any]) -> dict[str, Any]:
    polygons = sample["parking_spaces"]["polylines"]
    states = [str(item.get("occupancy_status", "unknown")) for item in polygons]
    occupied = states.count("occupied")
    vacant = states.count("not occupied")
    unknown = len(states) - occupied - vacant
    known = occupied + vacant
    source = str(sample["source"])
    weather = label_value(sample["weather"])
    date = scalar_date(sample["date"])[:10]
    timestamp = scalar_date(sample["parking_timestamp"])
    remote_path = str(sample["filepath"]).replace("\\", "/")
    filename = Path(remote_path).name
    metadata = sample["metadata"]
    return {
        "sample": sample,
        "source": source,
        "weather": weather,
        "date": date,
        "timestamp": timestamp,
        "group_id": f"{source}/{date}",
        "remote_path": remote_path,
        "download_url": f"{BASE_URL}/{quote(remote_path, safe='/')}?download=true",
        "local_path": f"data/raw/pklot/images/{source}/{filename}",
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "known_slots": known,
        "occupied": occupied,
        "vacant": vacant,
        "unknown": unknown,
        "occupancy_ratio": occupied / known if known else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select a deterministic balanced PKLot development subset"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--annotations-output", required=True)
    args = parser.parse_args()

    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for sample in iter_fiftyone_samples(Path(args.input)):
        record = summarize(sample)
        if record["known_slots"]:
            buckets[(record["source"], record["weather"])].append(record)

    selected: list[dict[str, Any]] = []
    for bucket_key in sorted(buckets):
        candidates = buckets[bucket_key]
        used_dates: set[str] = set()
        for target in TARGET_OCCUPANCY:
            eligible = [
                item for item in candidates if item["date"] not in used_dates
            ]
            if not eligible:
                raise RuntimeError(f"Not enough unique dates for {bucket_key}")
            choice = min(
                eligible,
                key=lambda item: (
                    abs(item["occupancy_ratio"] - target),
                    item["timestamp"],
                ),
            )
            choice["selection_target_occupancy"] = target
            used_dates.add(choice["date"])
            selected.append(choice)

    selected.sort(
        key=lambda item: (
            item["source"],
            item["weather"],
            item["selection_target_occupancy"],
        )
    )
    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sample_id",
        "split",
        "source",
        "weather",
        "date",
        "timestamp",
        "group_id",
        "selection_target_occupancy",
        "occupancy_ratio",
        "known_slots",
        "occupied",
        "vacant",
        "unknown",
        "width",
        "height",
        "remote_path",
        "download_url",
        "local_path",
    ]
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for index, item in enumerate(selected, start=1):
            row = {key: item[key] for key in fieldnames if key in item}
            row["sample_id"] = f"pklot_dev_{index:03d}"
            row["split"] = "development"
            row["occupancy_ratio"] = f"{item['occupancy_ratio']:.6f}"
            writer.writerow(row)

    annotations_path = Path(args.annotations_output)
    annotations_path.parent.mkdir(parents=True, exist_ok=True)
    with annotations_path.open("w", encoding="utf-8") as handle:
        for index, item in enumerate(selected, start=1):
            payload = {
                "sample_id": f"pklot_dev_{index:03d}",
                "source": item["source"],
                "weather": item["weather"],
                "date": item["date"],
                "timestamp": item["timestamp"],
                "local_path": item["local_path"],
                "sample": item["sample"],
            }
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")

    print(
        f"Selected {len(selected)} samples from {len(buckets)} "
        f"source/weather buckets"
    )


if __name__ == "__main__":
    main()
