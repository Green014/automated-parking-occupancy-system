from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from .stage_n_lmot import sha256_file
from .stage_p_retention import STAGE_P_PROTOCOL_ID


REQUIRED_FORMAL_OUTPUTS = (
    "config_snapshot.yaml",
    "source_image_verification.json",
    "comparison_metrics.json",
    "comparison_summary.json",
    "failure_cases.json",
    "D1_vs_D1_LL_contact_sheet.jpg",
    "D1/detections.jsonl",
    "D1/per_image_statistics.csv",
    "D1/count_predictions.csv",
    "D1/metrics.json",
    "D1/runtime_metadata.json",
    "D1_LL/detections.jsonl",
    "D1_LL/per_image_statistics.csv",
    "D1_LL/count_predictions.csv",
    "D1_LL/metrics.json",
    "D1_LL/runtime_metadata.json",
)


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
        "protocol_id": STAGE_P_PROTOCOL_ID,
        "artifact_count": count,
        "verified": not errors,
        "errors": errors,
    }


def verify_formal_output(output_root: Path) -> dict[str, Any]:
    output_root = output_root.resolve()
    missing = [
        relative
        for relative in REQUIRED_FORMAL_OUTPUTS
        if not (output_root / relative).is_file()
    ]
    return {
        "protocol_id": STAGE_P_PROTOCOL_ID,
        "output_root": str(output_root),
        "verified": not missing,
        "errors": [f"missing:{relative}" for relative in missing],
        "artifact_count": len(REQUIRED_FORMAL_OUTPUTS) - len(missing),
    }


def verify_stage_p_registry(path: Path) -> dict[str, Any]:
    path = path.resolve()
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_P_PROTOCOL_ID:
        raise ValueError("Unexpected Stage P registry protocol")
    result = verify_artifact_records(payload["artifacts"])
    if result["artifact_count"] != int(payload["artifact_count"]):
        result["verified"] = False
        result["errors"].append("artifact_count")
    result["registry_path"] = str(path)
    result["registry_sha256"] = sha256_file(path)
    return result


def authorize_p3_ll_defaults(
    *,
    retention_status: str,
    final_night_gate_status: str,
    real_occupancy_evidence: bool,
    target_path: Path,
) -> bool:
    if target_path.exists():
        raise FileExistsError(f"Refusing to overwrite {target_path}")
    return (
        retention_status == "PASS"
        and final_night_gate_status == "PASS"
        and real_occupancy_evidence is True
    )
