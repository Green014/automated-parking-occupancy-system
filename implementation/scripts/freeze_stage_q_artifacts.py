from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_q_artifacts import (
    artifact_record,
    verify_artifact_records,
)
from parking_occupancy.stage_q_external import STAGE_Q_PROTOCOL_ID


SOURCE_FILES = (
    "src/parking_occupancy/stage_q_external.py",
    "src/parking_occupancy/stage_q_artifacts.py",
    "scripts/verify_stage_q_gate.py",
    "scripts/freeze_stage_q_artifacts.py",
    "tests/test_stage_q_external.py",
    "tests/test_stage_q_artifacts.py",
)
CONTROL_FILES = (
    "data/stage_q/STAGE_Q_ONLINE_DATASET_AUDIT_20260729.md",
    "data/stage_q/STAGE_Q_CANDIDATE_GATE_20260729.yaml",
    "data/STAGE_Q_EXTERNAL_NIGHT_OCCUPANCY_REPORT.md",
)
FROZEN_BOUNDARY_FILES = (
    "configs/p3_integrated_runtime_defaults_20260729.yaml",
    "configs/stage_p_parking_domain_retention_frozen_20260729.yaml",
    "data/stage_p/STAGE_P_PARKING_RETENTION_DECISION_20260729.json",
    "data/stage_p/STAGE_P_FINAL_NIGHT_PARKING_DATA_GATE_20260729.yaml",
    "data/stage_p/STAGE_P_ARTIFACT_REGISTRY_20260729.yaml",
    "configs/stage_o_low_light_adaptation_frozen_20260729.yaml",
    "data/stage_o/STAGE_O_SELECTION_20260729.json",
    "data/stage_o/STAGE_O_ARTIFACT_REGISTRY_20260729.yaml",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create the additive, non-overwriting Stage Q blocked registry."
        )
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--audited-existing-input",
        type=Path,
        action="append",
        default=[],
        help=(
            "Optional pre-existing, non-formal input observed during the "
            "audit (for example the already-consumed CNR-EXT archive)."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    records: list[dict] = []
    for relative in SOURCE_FILES:
        records.append(
            artifact_record(
                label=relative,
                path=PROJECT_ROOT / relative,
                role="implementation",
            )
        )
    for relative in CONTROL_FILES:
        records.append(
            artifact_record(
                label=relative,
                path=PROJECT_ROOT / relative,
                role="stage_q_gate_or_report",
            )
        )
    for relative in FROZEN_BOUNDARY_FILES:
        records.append(
            artifact_record(
                label=f"frozen_boundary:{relative}",
                path=PROJECT_ROOT / relative,
                role="pre_existing_frozen_boundary",
            )
        )
    for index, path in enumerate(args.audited_existing_input, start=1):
        records.append(
            artifact_record(
                label=f"audited_existing_input:{index}:{path.name}",
                path=path,
                role="pre_existing_consumed_audit_evidence_not_formal_input",
            )
        )
    verification = verify_artifact_records(records)
    if not verification["verified"]:
        raise RuntimeError(verification)
    payload = {
        "schema_version": 1,
        "registry_id": "STAGE-Q-ARTIFACT-REGISTRY-20260729-01",
        "protocol_id": STAGE_Q_PROTOCOL_ID,
        "status": "blocked_before_download_no_formal_inference",
        "candidate_gate_status": "BLOCKED_BEFORE_DOWNLOAD",
        "download_performed": False,
        "formal_source_archive_present": False,
        "audited_existing_consumed_archive_present": bool(
            args.audited_existing_input
        ),
        "formal_inference_executed": False,
        "model_loaded": False,
        "quantitative_occupancy_results_created": False,
        "primary_method": "P3-D1",
        "secondary_method": "P3-D1-LL",
        "default_detector_after_stage_q": "D1",
        "historical_stage_l_m_n_o_p_modified": False,
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
