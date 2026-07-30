from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .stage_n_lmot import sha256_file
from .stage_q_external import STAGE_Q_PROTOCOL_ID


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
        "protocol_id": STAGE_Q_PROTOCOL_ID,
        "artifact_count": count,
        "verified": not errors,
        "errors": errors,
    }


def verify_stage_q_registry(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_Q_PROTOCOL_ID:
        raise ValueError("Unexpected Stage Q registry protocol")
    if payload.get("status") != "blocked_before_download_no_formal_inference":
        raise ValueError("Unexpected Stage Q registry status")
    if payload.get("formal_inference_executed") is not False:
        raise ValueError("Blocked Stage Q registry cannot record inference")
    result = verify_artifact_records(payload["artifacts"])
    if result["artifact_count"] != int(payload["artifact_count"]):
        result["verified"] = False
        result["errors"].append("artifact_count")
    result["registry_path"] = str(path)
    result["registry_sha256"] = sha256_file(path)
    return result
