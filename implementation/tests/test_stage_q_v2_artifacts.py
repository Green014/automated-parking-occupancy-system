from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from parking_occupancy.stage_q_v2_artifacts import (
    artifact_record,
    validate_annotation_freeze,
    validate_source_archive_audit,
    verify_artifact_records,
    verify_stage_q_v2_registry,
)
from parking_occupancy.stage_q_v2_upm import STAGE_Q_V2_PROTOCOL_ID


def test_artifact_hash_and_size_validation(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.txt"
    artifact.write_text("frozen\n", encoding="utf-8")
    record = artifact_record(label="artifact", path=artifact, role="test")
    assert verify_artifact_records([record])["verified"] is True
    artifact.write_text("mutate\n", encoding="utf-8")
    assert verify_artifact_records([record])["errors"] == [
        "sha256:artifact"
    ]


def test_source_archive_audit_requires_exact_license_boundary(
    tmp_path: Path,
) -> None:
    payload = {
        "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
        "status": "ARCHIVE_VALID",
        "official_public_download": True,
        "explicit_dataset_license_found": False,
        "use_scope": "local_noncommercial_course_research",
        "redistribution": "prohibited_by_project_policy",
        "attribution_required": True,
        "legal_interpretation_not_claimed": True,
        "archive": {
            "archive_bytes": 250698837,
            "archive_sha256": (
                "92d61d8f87fe3e7068d8c42ce8dc2c415c08071c92eeddfd4d47260e8922efdc"
            ),
            "zip_readable": True,
            "crc_verified": True,
            "path_traversal_safe": True,
        },
    }
    path = tmp_path / "audit.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert validate_source_archive_audit(path)["status"] == "ARCHIVE_VALID"
    payload["explicit_dataset_license_found"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="license boundary"):
        validate_source_archive_audit(path)


def test_unconfirmed_annotation_freeze_cannot_authorize_inference(
    tmp_path: Path,
) -> None:
    payload = {
        "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
        "status": "BLOCKED_PENDING_HUMAN_POLYGON_CONFIRMATION",
        "formal_inference_authorized": False,
        "polygon_confirmation": False,
        "model_loaded": False,
        "slot_id_order": [f"slot_{index:02d}" for index in range(21)],
    }
    path = tmp_path / "freeze.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    assert validate_annotation_freeze(path)["model_loaded"] is False
    payload["formal_inference_authorized"] = True
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot authorize"):
        validate_annotation_freeze(path)


def test_registry_hash_verification(tmp_path: Path) -> None:
    artifact = tmp_path / "artifact.json"
    artifact.write_text("{}\n", encoding="utf-8")
    record = artifact_record(label="artifact", path=artifact, role="test")
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
                "artifact_count": 1,
                "artifacts": [record],
            }
        ),
        encoding="utf-8",
    )
    result = verify_stage_q_v2_registry(registry)
    assert result["verified"] is True
    assert len(result["registry_sha256"]) == 64


def test_repository_premodel_artifacts_are_bound_and_blocked() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (
        root / "data/stage_q_v2/STAGE_Q_V2_SOURCE_ARCHIVE_AUDIT_20260729.yaml"
    )
    freeze = (
        root / "data/stage_q_v2/STAGE_Q_V2_ANNOTATION_FREEZE_20260729.yaml"
    )
    assert validate_source_archive_audit(source)["model_loaded"] is False
    assert validate_annotation_freeze(freeze)["formal_inference_authorized"] is False


def test_repository_completed_registry_verifies_when_present() -> None:
    root = Path(__file__).resolve().parents[1]
    registry = (
        root
        / "data/stage_q_v2/STAGE_Q_V2_ARTIFACT_REGISTRY_20260729.yaml"
    )
    if not registry.exists():
        pytest.skip("Registry is frozen only after formal outputs complete")
    result = verify_stage_q_v2_registry(registry)
    assert result["verified"] is True
    assert result["artifact_count"] > 750
