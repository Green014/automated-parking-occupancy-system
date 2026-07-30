from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import yaml

from parking_occupancy.stage_m_data_gate import (
    load_stage_m_gate_audit,
    validate_formal_parking_gate,
)


DEFAULT_AUDIT = (
    PROJECT_ROOT
    / "data"
    / "stage_m"
    / "STAGE_M_DATA_GATES_20260728.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check Stage M data gates")
    parser.add_argument("--audit", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--formal-gate", type=Path)
    parser.add_argument("--skip-file-verification", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = {
        "audit": [
            asdict(decision)
            for decision in load_stage_m_gate_audit(args.audit.resolve())
        ]
    }
    if args.formal_gate is not None:
        payload = yaml.safe_load(
            args.formal_gate.read_text(encoding="utf-8")
        )
        report["formal_gate"] = asdict(
            validate_formal_parking_gate(
                payload,
                base_dir=args.formal_gate.resolve().parent,
                verify_files=not args.skip_file_verification,
            )
        )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
