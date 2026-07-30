from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parking_occupancy.formal_training import verify_formal_training_record


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify every frozen Stage H artifact by size and SHA-256."
    )
    parser.add_argument("--record", required=True)
    parser.add_argument("--implementation-root", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite verification output: {output}"
        )
    report = verify_formal_training_record(
        record_path=Path(args.record),
        implementation_root=Path(args.implementation_root),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
