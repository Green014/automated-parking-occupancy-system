"""Catalog and acquire a byte-bounded VIRAT screening subset.

The script uses only the official Kitware release. It never replaces a target
file and it keeps the default screening download below 100 MiB.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from literature_core.virat_access import (
    VIDEOS_ORIGINAL_FOLDER_ID,
    ViratItem,
    download_item,
    fetch_folder_items,
    parse_video_item,
    select_cross_scene_candidates,
    select_named_items,
)

DEFAULT_BUDGET_BYTES = 100 * 1024 * 1024
DEFAULT_MAX_ITEM_BYTES = 25 * 1024 * 1024


def _catalog() -> list[ViratItem]:
    return [
        item
        for raw in fetch_folder_items(VIDEOS_ORIGINAL_FOLDER_ID)
        if (item := parse_video_item(raw)) is not None
    ]


def _record(item: ViratItem) -> dict[str, object]:
    return {
        "item_id": item.item_id,
        "name": item.name,
        "size": item.size,
        "scene_id": item.scene_id,
        "sequence_id": item.sequence_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("datasets/virat/screening/videos"),
    )
    parser.add_argument("--max-candidates", type=int, default=8)
    parser.add_argument("--budget-bytes", type=int, default=DEFAULT_BUDGET_BYTES)
    parser.add_argument("--max-item-bytes", type=int, default=DEFAULT_MAX_ITEM_BYTES)
    parser.add_argument(
        "--names",
        nargs="+",
        help="exact official MP4 names; byte limits still apply",
    )
    parser.add_argument(
        "--download",
        action="store_true",
        help="download the selected subset; otherwise print the catalog selection",
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=Path("configs/temporal_protocol_pending.yaml"),
        help="tracked protocol containing the individual acceptance record",
    )
    args = parser.parse_args()

    catalog = _catalog()
    if args.names:
        selected = select_named_items(
            catalog,
            args.names,
            max_total_bytes=args.budget_bytes,
            max_item_bytes=args.max_item_bytes,
        )
    else:
        selected = select_cross_scene_candidates(
            catalog,
            max_candidates=args.max_candidates,
            max_total_bytes=args.budget_bytes,
            max_item_bytes=args.max_item_bytes,
        )
    declared_total = sum(item.size for item in selected)
    payload: dict[str, object] = {
        "official_folder_id": VIDEOS_ORIGINAL_FOLDER_ID,
        "budget_bytes": args.budget_bytes,
        "declared_total_bytes": declared_total,
        "items": [_record(item) for item in selected],
    }
    if args.download:
        protocol = yaml.safe_load(args.protocol_config.read_text(encoding="utf-8"))
        acceptance = (
            protocol.get("dataset", {}).get("access", {}).get(
                "user_acceptance_recorded"
            )
            if isinstance(protocol, dict)
            else None
        )
        if acceptance is not True:
            parser.error(
                "download is blocked until protocol-config records individual "
                "license acceptance"
            )
        payload["acceptance_record"] = str(args.protocol_config)
        payload["downloads"] = [
            {"item": _record(item), **download_item(item, args.output_dir)}
            for item in selected
        ]
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
