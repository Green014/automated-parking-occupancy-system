"""Validate a temporal development/holdout protocol before experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.temporal_protocol import validate_temporal_protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--protocol",
        type=Path,
        default=ROOT / "configs" / "temporal_protocol_pending.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            "Validation output already exists; choose a new path so prior "
            "audit evidence is not overwritten."
        )
    payload = yaml.safe_load(args.protocol.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("protocol root must be a mapping")

    report = validate_temporal_protocol(payload, project_root=ROOT)
    report["validated_at"] = datetime.now().astimezone().isoformat()
    report["protocol"] = str(args.protocol.resolve())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["schema_valid"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
