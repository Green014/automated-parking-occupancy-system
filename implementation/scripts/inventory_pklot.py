from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

from pklot_metadata import iter_fiftyone_samples, label_value, scalar_date


def main() -> None:
    parser = argparse.ArgumentParser(description="Inventory PKLot metadata")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    source_counts: Counter[str] = Counter()
    weather_counts: Counter[str] = Counter()
    source_weather_counts: Counter[str] = Counter()
    group_counts: Counter[str] = Counter()
    dates_by_source: dict[str, set[str]] = defaultdict(set)
    slot_states: Counter[str] = Counter()
    sample_count = 0

    for sample in iter_fiftyone_samples(Path(args.input)):
        sample_count += 1
        source = str(sample.get("source", "unknown"))
        weather = label_value(sample.get("weather")) or "unknown"
        date = scalar_date(sample.get("date"))[:10]
        source_counts[source] += 1
        weather_counts[weather] += 1
        source_weather_counts[f"{source}/{weather}"] += 1
        group_counts[f"{source}/{date}"] += 1
        dates_by_source[source].add(date)
        for polygon in sample["parking_spaces"]["polylines"]:
            slot_states[str(polygon.get("occupancy_status", "unknown"))] += 1

    report = {
        "samples": sample_count,
        "sources": dict(sorted(source_counts.items())),
        "weather": dict(sorted(weather_counts.items())),
        "source_weather": dict(sorted(source_weather_counts.items())),
        "camera_date_groups": len(group_counts),
        "dates_by_source": {
            source: sorted(dates)
            for source, dates in sorted(dates_by_source.items())
        },
        "group_sizes": dict(sorted(group_counts.items())),
        "slot_states": dict(sorted(slot_states.items())),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
