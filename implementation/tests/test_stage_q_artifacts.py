from __future__ import annotations

from pathlib import Path

import yaml

from parking_occupancy.stage_q_artifacts import (
    artifact_record,
    verify_artifact_records,
    verify_stage_q_registry,
)
from parking_occupancy.stage_q_external import STAGE_Q_PROTOCOL_ID


def test_stage_q_artifact_hash_and_size_verification(tmp_path: Path) -> None:
    artifact = tmp_path / "audit.md"
    artifact.write_text("blocked\n", encoding="utf-8")
    record = artifact_record(
        label="audit",
        path=artifact,
        role="stage_q_gate_or_report",
    )
    assert verify_artifact_records([record])["verified"] is True
    artifact.write_text("changed\n", encoding="utf-8")
    result = verify_artifact_records([record])
    assert result["verified"] is False
    assert result["errors"] == ["sha256:audit"]


def test_blocked_registry_schema_and_hash_verification(tmp_path: Path) -> None:
    artifact = tmp_path / "gate.yaml"
    artifact.write_text("status: BLOCKED\n", encoding="utf-8")
    record = artifact_record(label="gate", path=artifact, role="gate")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "protocol_id": STAGE_Q_PROTOCOL_ID,
                "status": "blocked_before_download_no_formal_inference",
                "formal_inference_executed": False,
                "artifact_count": 1,
                "artifacts": [record],
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    result = verify_stage_q_registry(registry)
    assert result["verified"] is True
    assert len(result["registry_sha256"]) == 64
