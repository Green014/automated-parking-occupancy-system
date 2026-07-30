from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file


REQUIRED_FILES = {
    "sequence_metrics.json",
    "aggregate_metrics.json",
    "runtime_metadata.json",
    "configuration_snapshot.yaml",
}
REQUIRED_DIRECTORIES = {"detections", "tracks", "qualitative_frames"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify one Stage N output")
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    args = parser.parse_args()
    root = args.output_root.resolve()
    missing = [
        name for name in REQUIRED_FILES if not (root / name).is_file()
    ] + [
        name for name in REQUIRED_DIRECTORIES if not (root / name).is_dir()
    ]
    if missing:
        raise FileNotFoundError(f"Missing Stage N outputs: {missing}")
    empty = [
        name
        for name in REQUIRED_DIRECTORIES
        if not any(path.is_file() for path in (root / name).rglob("*"))
    ]
    if empty:
        raise FileNotFoundError(
            f"Stage N output directories contain no artifacts: {empty}"
        )
    for name in ("sequence_metrics.json", "aggregate_metrics.json", "runtime_metadata.json"):
        json.loads((root / name).read_text(encoding="utf-8"))
    records = [
        {
            "path": path.relative_to(root).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
        for path in sorted(root.rglob("*"))
        if path.is_file()
    ]
    payload = {
        "schema_version": 1,
        "output_root": str(root),
        "files": len(records),
        "bytes": sum(record["bytes"] for record in records),
        "artifacts": records,
    }
    if args.manifest:
        if args.manifest.exists():
            raise FileExistsError(f"Refusing to overwrite {args.manifest}")
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
