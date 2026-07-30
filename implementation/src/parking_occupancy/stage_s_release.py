from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml

from .integrated_runner import load_integrated_config


STAGE_S_PROTOCOL_ID = "STAGE-S-FINAL-DEFAULT-AND-DEMO-20260729-01"
STAGE_S_CONFIG_NAME = "p3_stage_r_recommended_default_20260729.yaml"
REQUIRED_GITIGNORE_PATTERNS = (
    "implementation/.venv_*/",
    "implementation/data/external/stage_o_training_*/",
    "implementation/data/external/*.partial/",
)
HISTORICAL_REGISTRIES = (
    "data/comparisons/stage_l_integrated_workflow_20260728.yaml",
    "data/stage_m/STAGE_M_ARTIFACT_REGISTRY_20260728.yaml",
    "data/stage_n/STAGE_N_ARTIFACT_REGISTRY_20260728.yaml",
    "data/stage_n_v2/STAGE_N_V2_ARTIFACT_REGISTRY_20260729.yaml",
    "data/stage_n_v3/STAGE_N_V3_ARTIFACT_REGISTRY_20260729.yaml",
    "data/stage_o/STAGE_O_ARTIFACT_REGISTRY_20260729.yaml",
    "data/stage_p/STAGE_P_ARTIFACT_REGISTRY_20260729.yaml",
    "data/stage_q/STAGE_Q_ARTIFACT_REGISTRY_20260729.yaml",
    "data/stage_q_v2/STAGE_Q_V2_ARTIFACT_REGISTRY_20260729.yaml",
    "data/stage_r/STAGE_R_ARTIFACT_REGISTRY_20260729.yaml",
)
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".engine"}
BANNED_DIRECTORY_PARTS = {
    "outputs",
    "runs",
    "datasets",
    "external",
}


class StageSReleaseError(RuntimeError):
    """Raised when the final-default release gate is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_stage_s_config(path: Path) -> dict[str, Any]:
    path = path.resolve()
    config = load_integrated_config(path)
    if config.get("config_id") != "P3-STAGE-R-RECOMMENDED-DEFAULT-20260729-01":
        raise StageSReleaseError("Unexpected Stage S final config ID")
    if config.get("status") != "frozen_for_stage_s_final_default":
        raise StageSReleaseError("Stage S final config is not frozen")
    if config["detector"].get("id") != "D1":
        raise StageSReleaseError("Stage S default detector must be D1")
    if config["mapping"].get("id") != "B1":
        raise StageSReleaseError("Stage S default mapping must be B1")
    if config.get("fusion", {}).get("id") != "F2":
        raise StageSReleaseError("Stage S default fusion must be F2")
    if config["temporal"].get("default_enabled") is not False:
        raise StageSReleaseError("Stage S must disable E4 by default")
    if config["tracking"].get("default_backend") != "none":
        raise StageSReleaseError("Stage S must disable tracking by default")
    if config.get("claims", {}).get("deployment_ready") is not False:
        raise StageSReleaseError("Stage S must not claim deployment readiness")

    provenance = config["parameter_provenance"]
    for field, hash_field in (
        ("source", "source_sha256"),
        ("inherited_runtime", "inherited_runtime_sha256"),
    ):
        bound = (path.parent / str(provenance[field])).resolve()
        if not bound.is_file():
            raise StageSReleaseError(f"Missing Stage S provenance file: {bound}")
        if sha256_file(bound) != str(provenance[hash_field]):
            raise StageSReleaseError(
                f"Stage S provenance hash mismatch: {field}"
            )
    return config


def _git_candidates(repository_root: Path) -> list[Path]:
    result = subprocess.run(
        [
            "git",
            "-c",
            "core.quotepath=false",
            "ls-files",
            "-z",
            "--cached",
            "--others",
            "--exclude-standard",
        ],
        cwd=repository_root,
        check=True,
        capture_output=True,
    )
    relative_paths = [
        Path(value.decode("utf-8"))
        for value in result.stdout.split(b"\0")
        if value
    ]
    return sorted(
        (
            (repository_root / relative).resolve()
            for relative in relative_paths
            if (repository_root / relative).is_file()
        ),
        key=lambda path: str(path).casefold(),
    )


def _candidate_category(relative: Path) -> str:
    suffix = relative.suffix.lower()
    if suffix in {".py", ".toml", ".yaml", ".yml", ".json"}:
        return "code_or_configuration"
    if suffix in {".md", ".txt", ".csv"}:
        return "documentation_or_compact_evidence"
    if suffix in {".png", ".jpg", ".jpeg", ".svg", ".mp4"}:
        return "presentation_artifact"
    return "other"


def submission_candidate_audit(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    gitignore = repository_root / ".gitignore"
    text = gitignore.read_text(encoding="utf-8")
    missing_patterns = [
        pattern for pattern in REQUIRED_GITIGNORE_PATTERNS if pattern not in text
    ]
    if missing_patterns:
        raise StageSReleaseError(
            f"Required .gitignore patterns missing: {missing_patterns}"
        )

    candidates = _git_candidates(repository_root)
    records: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    for path in candidates:
        relative = path.relative_to(repository_root)
        normalized_parts = {part.casefold() for part in relative.parts}
        reason = None
        if relative.suffix.lower() in MODEL_SUFFIXES:
            reason = "model_weight"
        elif any(part.startswith(".venv_") for part in normalized_parts):
            reason = "virtual_environment"
        elif "outputs" in normalized_parts or "runs" in normalized_parts:
            reason = "generated_output_tree"
        elif (
            "implementation" in normalized_parts
            and "data" in normalized_parts
            and "external" in normalized_parts
        ):
            reason = "external_dataset_tree"
        elif "datasets" in normalized_parts:
            reason = "dataset_tree"
        if reason is not None:
            violations.append(
                {"path": relative.as_posix(), "reason": reason}
            )
        records.append(
            {
                "path": relative.as_posix(),
                "bytes": path.stat().st_size,
                "category": _candidate_category(relative),
            }
        )

    total_bytes = sum(record["bytes"] for record in records)
    category_counts: dict[str, dict[str, int]] = {}
    for record in records:
        category = str(record["category"])
        summary = category_counts.setdefault(category, {"files": 0, "bytes": 0})
        summary["files"] += 1
        summary["bytes"] += int(record["bytes"])
    return {
        "schema_version": 1,
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "status": "PASS" if not violations else "FAIL",
        "candidate_files": len(records),
        "candidate_bytes": total_bytes,
        "category_summary": category_counts,
        "required_gitignore_patterns": list(REQUIRED_GITIGNORE_PATTERNS),
        "forbidden_content": {
            "model_weights": False,
            "datasets": False,
            "virtual_environments": False,
            "outputs": False,
        }
        if not violations
        else None,
        "violations": violations,
        "records": records,
    }


def write_submission_audit(
    audit: Mapping[str, Any],
    *,
    json_path: Path,
    csv_path: Path,
) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("path", "bytes", "category"),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(audit["records"])


def _resolve_registry_artifact(
    implementation_root: Path,
    registry_path: Path,
    stored: str,
) -> Path:
    path = Path(stored)
    if path.is_absolute():
        return path
    candidates = (
        implementation_root / path,
        registry_path.parent / path,
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return candidates[0].resolve()


def _binding_records(value: Any) -> list[Mapping[str, Any]]:
    records: list[Mapping[str, Any]] = []
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"}.issubset(value):
            records.append(value)
        for nested in value.values():
            records.extend(_binding_records(nested))
    elif isinstance(value, list):
        for nested in value:
            records.extend(_binding_records(nested))
    return records


def historical_registry_snapshot(
    implementation_root: Path,
) -> dict[str, Any]:
    implementation_root = implementation_root.resolve()
    rows: list[dict[str, Any]] = []
    all_verified = True
    for relative in HISTORICAL_REGISTRIES:
        registry_path = implementation_root / relative
        if not registry_path.is_file():
            raise StageSReleaseError(f"Missing historical registry: {relative}")
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        records = _binding_records(payload)
        errors: list[str] = []
        for record in records:
            artifact = _resolve_registry_artifact(
                implementation_root,
                registry_path,
                str(record["path"]),
            )
            label = str(record.get("label", record["path"]))
            if not artifact.is_file():
                errors.append(f"missing:{label}")
            elif artifact.stat().st_size != int(record["bytes"]):
                errors.append(f"bytes:{label}")
            elif sha256_file(artifact) != str(record["sha256"]):
                errors.append(f"sha256:{label}")
        verified = not errors
        all_verified = all_verified and verified
        rows.append(
            {
                "path": relative,
                "bytes": registry_path.stat().st_size,
                "sha256": sha256_file(registry_path),
                "binding_count": len(records),
                "verified": verified,
                "errors": errors,
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "historical_registry_count": len(rows),
        "all_registry_files_and_artifacts_verified": all_verified,
        "registries": rows,
    }


def compare_registry_snapshots(
    before: Mapping[str, Any],
    after: Mapping[str, Any],
) -> dict[str, Any]:
    before_rows = {row["path"]: row for row in before["registries"]}
    after_rows = {row["path"]: row for row in after["registries"]}
    changes: list[dict[str, Any]] = []
    for path in sorted(set(before_rows) | set(after_rows)):
        old = before_rows.get(path)
        new = after_rows.get(path)
        if old != new:
            changes.append({"path": path, "before": old, "after": new})
    return {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "unchanged": not changes,
        "historical_registry_count": len(after_rows),
        "all_after_verified": bool(
            after["all_registry_files_and_artifacts_verified"]
        ),
        "changes": changes,
    }


def artifact_record(
    path: Path,
    *,
    implementation_root: Path,
    role: str,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": path.relative_to(implementation_root.resolve()).as_posix(),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def verify_stage_s_registry(
    registry_path: Path,
    *,
    implementation_root: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("protocol_id") != STAGE_S_PROTOCOL_ID:
        errors.append("protocol_id")
    records = payload.get("artifacts", [])
    if len(records) != int(payload.get("artifact_count", -1)):
        errors.append("artifact_count")
    for record in records:
        path = implementation_root / str(record["path"])
        if not path.is_file():
            errors.append(f"missing:{record['path']}")
        elif path.stat().st_size != int(record["bytes"]):
            errors.append(f"bytes:{record['path']}")
        elif sha256_file(path) != str(record["sha256"]):
            errors.append(f"sha256:{record['path']}")
    return {
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "artifact_count": len(records),
        "verified": not errors,
        "errors": errors,
        "registry_bytes": registry_path.stat().st_size,
        "registry_sha256": sha256_file(registry_path),
    }


def write_stage_s_registry(
    registry_path: Path,
    *,
    implementation_root: Path,
    artifacts: Iterable[tuple[Path, str]],
) -> dict[str, Any]:
    records = [
        artifact_record(
            path,
            implementation_root=implementation_root,
            role=role,
        )
        for path, role in artifacts
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "registry_id": "STAGE-S-ARTIFACT-REGISTRY-20260729-01",
        "status": "FINAL_DEFAULT_AND_DEMO_COMPLETE_HASH_VERIFIED",
        "created_on": "2026-07-29",
        "artifact_count": len(records),
        "registry_self_hash_included": False,
        "model_inference_run": False,
        "historical_stage_l_to_r_modified": False,
        "artifacts": records,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )
    result = verify_stage_s_registry(
        registry_path,
        implementation_root=implementation_root,
    )
    if not result["verified"]:
        raise StageSReleaseError(f"Stage S registry verification failed: {result}")
    return result


def write_final_evidence_table(path: Path) -> None:
    rows = [
        {
            "decision_area": "default_detector_at_final_R1_pipeline",
            "evidence": "D1 R1 Macro F1=0.706681; D1-LL R1 Macro F1=0.666978",
            "decision": "D1 remains the final default detector",
            "claim_boundary": "post-hoc Stage R attribution of frozen Stage Q-v2",
        },
        {
            "decision_area": "default_pipeline",
            "evidence": "D1 R1 is the strongest frozen default-compatible component result",
            "decision": "D1 -> B1 -> F2 -> Occupancy Output",
            "claim_boundary": "occupied recall remains 0.370927; not deployment-ready",
        },
        {
            "decision_area": "E4",
            "evidence": "D1 R1-to-R2 Macro F1=-0.042363 while occupied recall=+0.075188",
            "decision": "conditional on calibrated continuous video",
            "claim_boundary": "not the default for static or sparse inputs",
        },
        {
            "decision_area": "TrackTrack",
            "evidence": "not run in Stage Q-v2; no frozen occupancy gain",
            "decision": "independent optional MOT module",
            "claim_boundary": "no slot-level occupancy improvement claimed",
        },
        {
            "decision_area": "D1-LL",
            "evidence": "R1 Macro F1 is 0.039703 below D1",
            "decision": "retained negative low-light fine-tuning experiment",
            "claim_boundary": "does not replace D1",
        },
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "decision_area",
                "evidence",
                "decision",
                "claim_boundary",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
