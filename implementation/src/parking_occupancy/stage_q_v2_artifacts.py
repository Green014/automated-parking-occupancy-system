from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .stage_n_lmot import sha256_file
from .stage_q_v2_upm import STAGE_Q_V2_PROTOCOL_ID


REQUIRED_LICENSE_BOUNDARY = {
    "official_public_download": True,
    "explicit_dataset_license_found": False,
    "use_scope": "local_noncommercial_course_research",
    "redistribution": "prohibited_by_project_policy",
    "attribution_required": True,
    "legal_interpretation_not_claimed": True,
}


def artifact_record(*, label: str, path: Path, role: str) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "label": label,
        "role": role,
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_artifact_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    errors: list[str] = []
    count = 0
    for count, record in enumerate(records, start=1):
        label = str(record["label"])
        path = Path(str(record["path"]))
        if not path.is_file():
            errors.append(f"missing:{label}")
        elif path.stat().st_size != int(record["bytes"]):
            errors.append(f"bytes:{label}")
        elif sha256_file(path) != str(record["sha256"]):
            errors.append(f"sha256:{label}")
    return {
        "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
        "artifact_count": count,
        "verified": not errors,
        "errors": errors,
    }


def validate_source_archive_audit(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_Q_V2_PROTOCOL_ID:
        raise ValueError("Unexpected Stage Q-v2 protocol ID")
    if payload.get("status") != "ARCHIVE_VALID":
        raise ValueError("Stage Q-v2 source archive is not valid")
    for key, value in REQUIRED_LICENSE_BOUNDARY.items():
        if payload.get(key) != value:
            raise ValueError(f"Stage Q-v2 license boundary differs at {key}")
    archive = payload.get("archive", {})
    if archive.get("archive_bytes") != 250698837:
        raise ValueError("Unexpected UPM test.zip size")
    if (
        archive.get("archive_sha256")
        != "92d61d8f87fe3e7068d8c42ce8dc2c415c08071c92eeddfd4d47260e8922efdc"
    ):
        raise ValueError("Unexpected UPM test.zip SHA-256")
    for key in (
        "zip_readable",
        "crc_verified",
        "path_traversal_safe",
    ):
        if archive.get(key) is not True:
            raise ValueError(f"Archive validation failed at {key}")
    return payload


def validate_annotation_freeze(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_Q_V2_PROTOCOL_ID:
        raise ValueError("Unexpected Stage Q-v2 annotation protocol")
    if payload.get("status") != "BLOCKED_PENDING_HUMAN_POLYGON_CONFIRMATION":
        raise ValueError("Unexpected annotation gate status")
    if payload.get("formal_inference_authorized") is not False:
        raise ValueError("Unconfirmed polygons cannot authorize inference")
    if payload.get("polygon_confirmation") is not False:
        raise ValueError("Polygon confirmation must remain false")
    if payload.get("model_loaded") is not False:
        raise ValueError("Blocked annotation gate cannot load a model")
    if payload.get("slot_id_order") != [
        f"slot_{index:02d}" for index in range(21)
    ]:
        raise ValueError("UPM slot index order changed")
    return payload


def verify_stage_q_v2_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_Q_V2_PROTOCOL_ID:
        raise ValueError("Unexpected Stage Q-v2 registry protocol")
    result = verify_artifact_records(payload["artifacts"])
    if result["artifact_count"] != int(payload["artifact_count"]):
        result["verified"] = False
        result["errors"].append("artifact_count")
    result["registry_path"] = str(path.resolve())
    result["registry_sha256"] = sha256_file(path)
    return result

