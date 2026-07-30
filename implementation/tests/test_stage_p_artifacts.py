from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from parking_occupancy.stage_p_artifacts import (
    REQUIRED_FORMAL_OUTPUTS,
    artifact_record,
    authorize_p3_ll_defaults,
    verify_artifact_records,
    verify_formal_output,
    verify_stage_p_registry,
)
from parking_occupancy.stage_p_retention import STAGE_P_PROTOCOL_ID


def test_artifact_hash_and_size_verification(tmp_path: Path) -> None:
    path = tmp_path / "artifact.txt"
    path.write_text("frozen\n", encoding="utf-8")
    record = artifact_record(label="fixture", path=path, role="test")
    result = verify_artifact_records([record])
    assert result["verified"] is True
    path.write_text("mutate\n", encoding="utf-8")
    result = verify_artifact_records([record])
    assert result["verified"] is False
    assert result["errors"] == ["sha256:fixture"]


def test_formal_output_schema(tmp_path: Path) -> None:
    for relative in REQUIRED_FORMAL_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    result = verify_formal_output(tmp_path)
    assert result["verified"] is True
    assert result["artifact_count"] == len(REQUIRED_FORMAL_OUTPUTS)
    (tmp_path / REQUIRED_FORMAL_OUTPUTS[0]).unlink()
    assert verify_formal_output(tmp_path)["verified"] is False


def test_stage_p_registry_schema_and_verification(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    record = artifact_record(label="artifact", path=artifact, role="test")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "protocol_id": STAGE_P_PROTOCOL_ID,
                "artifact_count": 1,
                "artifacts": [record],
            }
        ),
        encoding="utf-8",
    )
    result = verify_stage_p_registry(registry)
    assert result["verified"] is True
    assert len(result["registry_sha256"]) == 64


def test_blocked_p4_cannot_create_or_overwrite_p3_ll_defaults(
    tmp_path: Path,
) -> None:
    target = tmp_path / "p3_ll_integrated_runtime_defaults_20260729.yaml"
    assert (
        authorize_p3_ll_defaults(
            retention_status="PASS",
            final_night_gate_status="BLOCKED",
            real_occupancy_evidence=False,
            target_path=target,
        )
        is False
    )
    assert not target.exists()
    target.write_text("frozen: true\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="overwrite"):
        authorize_p3_ll_defaults(
            retention_status="PASS",
            final_night_gate_status="PASS",
            real_occupancy_evidence=True,
            target_path=target,
        )
