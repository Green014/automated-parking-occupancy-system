from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from parking_occupancy.dataset_preparation import prepare_ndispark_dataset


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the frozen NDISPark-only train/validation/count-test "
            "protocol as a one-class Ultralytics dataset."
        )
    )
    parser.add_argument("--protocol", required=True)
    parser.add_argument(
        "--source-root",
        help=(
            "Extracted NDISPark root containing train/, validation/, and test/. "
            "Defaults to PARKING_DATA_ROOT."
        ),
    )
    parser.add_argument("--output-root", required=True)
    args = parser.parse_args()

    source_value = args.source_root or os.environ.get("PARKING_DATA_ROOT")
    if not source_value:
        parser.error("--source-root or PARKING_DATA_ROOT is required")
    summary = prepare_ndispark_dataset(
        protocol_path=Path(args.protocol),
        source_root=Path(source_value),
        output_root=Path(args.output_root),
    )
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
