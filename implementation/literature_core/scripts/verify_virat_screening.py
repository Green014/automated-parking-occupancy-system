"""Verify every local VIRAT screening video against the tracked manifest."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

import yaml

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from literature_core.virat_access import verify_screening_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/manifests/virat_screening_20260726.yaml"),
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("datasets/virat/screening"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")

    payload = yaml.safe_load(args.manifest.read_text(encoding="utf-8"))
    report = verify_screening_manifest(payload, args.data_root)
    report["schema_version"] = 1
    report["verified_at"] = datetime.now().astimezone().isoformat()
    report["manifest"] = str(args.manifest.resolve())
    report["data_root"] = str(args.data_root.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
