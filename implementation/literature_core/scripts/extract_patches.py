from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.data import (  # noqa: E402
    load_pklot_slot_samples,
    read_image,
    write_image,
)
from literature_core.patches import extract_slot_patch  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract a small PKLot patch audit")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit-per-split", type=int, default=24)
    args = parser.parse_args()

    samples = load_pklot_slot_samples(
        args.annotations,
        args.project_root,
        args.split_config,
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    counts: defaultdict[str, int] = defaultdict(int)
    cache = {}
    rows = []
    for sample in samples:
        if counts[sample.split] >= args.limit_per_split:
            continue
        if sample.image_path not in cache:
            cache[sample.image_path] = read_image(sample.image_path)
        patch = extract_slot_patch(cache[sample.image_path], sample.points)
        filename = (
            f"{sample.split}_{sample.sample_id}_slot{sample.slot_id}_"
            f"label{sample.label}.jpg"
        )
        relative_path = Path(sample.split) / filename
        write_image(output_dir / relative_path, patch)
        rows.append(
            {
                "split": sample.split,
                "sample_id": sample.sample_id,
                "source": sample.source,
                "group_id": sample.group_id,
                "slot_id": sample.slot_id,
                "label": sample.label,
                "patch_path": relative_path.as_posix(),
            }
        )
        counts[sample.split] += 1

    with (output_dir / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(dict(counts))


if __name__ == "__main__":
    main()

