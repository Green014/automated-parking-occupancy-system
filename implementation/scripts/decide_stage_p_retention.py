from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file
from parking_occupancy.stage_p_retention import (
    STAGE_P_PROTOCOL_ID,
    decide_parking_retention,
    load_stage_p_protocol,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply the pre-frozen Stage P2 retention decision rule."
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--comparison-metrics", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    protocol = load_stage_p_protocol(args.protocol.resolve())
    comparison_path = args.comparison_metrics.resolve()
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    decision = decide_parking_retention(
        comparison, protocol["retention_decision_rule"]
    )
    payload = {
        "schema_version": 1,
        "decision_id": "STAGE-P-PARKING-RETENTION-DECISION-20260729-01",
        "protocol_id": STAGE_P_PROTOCOL_ID,
        **decision,
        "comparison_metrics": {
            "path": str(comparison_path),
            "bytes": comparison_path.stat().st_size,
            "sha256": sha256_file(comparison_path),
        },
        "selection_rule_frozen_before_predictions": True,
        "thresholds_or_parameters_reselected": False,
        "P4_executed": False,
        "final_night_parking_data_gate": "BLOCKED",
        "P3_LL_defaults_created": False,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
