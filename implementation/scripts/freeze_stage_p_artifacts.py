from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_p_artifacts import (
    REQUIRED_FORMAL_OUTPUTS,
    artifact_record,
    verify_artifact_records,
    verify_formal_output,
)
from parking_occupancy.stage_p_retention import STAGE_P_PROTOCOL_ID


SOURCE_FILES = (
    "src/parking_occupancy/stage_p_retention.py",
    "src/parking_occupancy/stage_p_artifacts.py",
    "scripts/run_stage_p_retention.py",
    "scripts/decide_stage_p_retention.py",
    "scripts/freeze_stage_p_artifacts.py",
    "tests/test_stage_p_retention.py",
    "tests/test_stage_p_artifacts.py",
)
CONTROL_FILES = (
    "configs/stage_p_parking_domain_retention_frozen_20260729.yaml",
    "data/stage_p/STAGE_P_NDISPARK_DATA_AUDIT_20260729.md",
    "data/stage_p/STAGE_P_NDISPARK_DATA_GATE_20260729.yaml",
    "data/stage_p/STAGE_P_FINAL_NIGHT_PARKING_DATA_GATE_20260729.yaml",
    "data/stage_p/STAGE_P_PARKING_RETENTION_DECISION_20260729.json",
    "data/STAGE_P_PARKING_DOMAIN_VALIDATION_REPORT.md",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the non-overwriting Stage P artifact registry."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--formal-output-root", type=Path, required=True)
    parser.add_argument("--d1-weights", type=Path, required=True)
    parser.add_argument("--d1-ll-weights", type=Path, required=True)
    parser.add_argument(
        "--input", type=Path, action="append", default=[], dest="inputs"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    formal_root = args.formal_output_root.resolve()
    formal_verification = verify_formal_output(formal_root)
    if not formal_verification["verified"]:
        raise RuntimeError(formal_verification)

    records: list[dict] = []
    seen: set[Path] = set()

    def append(label: str, path: Path, role: str) -> None:
        resolved = path.resolve()
        if resolved in seen:
            return
        records.append(
            artifact_record(label=label, path=resolved, role=role)
        )
        seen.add(resolved)

    for relative in SOURCE_FILES:
        append(relative, PROJECT_ROOT / relative, "implementation")
    for relative in CONTROL_FILES:
        append(relative, PROJECT_ROOT / relative, "protocol_gate_or_report")
    for relative in REQUIRED_FORMAL_OUTPUTS:
        append(
            f"formal_output:{relative}",
            formal_root / relative,
            "consumed_development_retrospective_output",
        )
    append("D1_best", args.d1_weights, "frozen_model_input")
    append("D1_LL_best", args.d1_ll_weights, "frozen_model_input")
    for index, path in enumerate(args.inputs, start=1):
        append(f"bound_input:{index}:{path.name}", path, "bound_input")

    verification = verify_artifact_records(records)
    if not verification["verified"]:
        raise RuntimeError(verification)
    payload = {
        "schema_version": 1,
        "registry_id": "STAGE-P-ARTIFACT-REGISTRY-20260729-01",
        "protocol_id": STAGE_P_PROTOCOL_ID,
        "status": "complete_additive_stage_p_record_with_blocked_P4",
        "consumed_development_diagnostic": True,
        "interface_only_evidence": False,
        "P4_executed": False,
        "final_occupancy_evaluation": "BLOCKED",
        "P3_LL_defaults_created": False,
        "historical_stage_l_m_n_o_modified": False,
        "artifact_count": len(records),
        "artifacts": records,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        yaml.safe_dump(
            {
                "status": "ok",
                "artifact_count": len(records),
                "output": str(output),
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )


if __name__ == "__main__":
    main()
