from __future__ import annotations

import hashlib
import csv
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


class ArtifactRegistryError(ValueError):
    """Raised when a release-registry contract is malformed."""


STAGE_U_FROZEN_REGISTRY_SHA256 = (
    "da74ee8c9304724382db385ca6e445689bafd5c41467d534b2a6ab1dae3cdfe9"
)
STAGE_U_SELF_REFERENCE_EXCLUDED = "SELF_REFERENCE_EXCLUDED"
STAGE_S_FROZEN_REGISTRY_SHA256 = (
    "282f30b241c49930888788bcf4f04cde344b7cb03d6f07b23cc955ff181e43aa"
)
STAGE_T_FROZEN_REGISTRY_SHA256 = (
    "2b2989b558619448e5fdc8a54c7efe3812d91c16cb2c7e8806f58824b38d427c"
)
STAGE_V_1_PRE_HARDENING_REGISTRY_SHA256 = (
    "19aec081be8e9707f0025365a136dcd6fc68a005a373ec7a4d6e1e99680bd372"
)
STAGE_W_PRE_HARDENING_REGISTRY_SHA256 = (
    "0a1daf77de33d753f5a79609146f80c5a53a382f78975b03b086b036af410bc9"
)
STAGE_W_1_PRE_W2_REGISTRY_SHA256 = (
    "ac113b84afad1622f75230d3c34c9578b6c980fdad9adf5d079dc3cc60377a27"
)
STAGE_W_2_PRE_W3_REGISTRY_SHA256 = (
    "db496155780b2880149085fe93f16a0c86f5d2e2d3aef67e951f7964bd04a2d4"
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def artifact_record(
    root: Path,
    relative: str,
    *,
    role: str,
    availability: str = "required",
) -> dict[str, Any]:
    path = root.resolve() / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": PurePosixPath(relative).as_posix(),
        "role": role,
        "availability": availability,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _safe_relative(value: str) -> str:
    normalized = value.replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        path.is_absolute()
        or ".." in path.parts
        or normalized.startswith("/")
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise ArtifactRegistryError(
            f"Registry artifact path must be relative and contained: {value}"
        )
    return path.as_posix()


def _managed_files(root: Path, patterns: Sequence[str]) -> set[str]:
    files: set[str] = set()
    for pattern in patterns:
        normalized = _safe_relative(str(pattern))
        for path in root.glob(normalized):
            if path.is_file():
                files.add(path.relative_to(root).as_posix())
    return files


def verify_artifact_registry(
    registry_path: Path,
    *,
    artifact_root: Path,
) -> dict[str, Any]:
    """Verify a Stage V.1/W registry without mutating any artifact.

    Required files must exist and match. ``local_ignored_optional`` files are
    verified when present and reported as unavailable, not failed, when a
    portable checkout intentionally omits ignored runtime outputs. Managed
    glob scopes detect unregistered release artifacts while leaving unrelated
    historical stages outside the new version boundary.
    """

    registry_path = registry_path.resolve()
    artifact_root = artifact_root.resolve()
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ArtifactRegistryError("Registry must contain a mapping")
    records = payload.get("artifacts")
    if not isinstance(records, list):
        raise ArtifactRegistryError("Registry artifacts must be a list")
    errors: list[str] = []
    warnings: list[str] = []
    registered: set[str] = set()
    verified = 0
    optional_unavailable = 0
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("malformed_artifact_record")
            continue
        relative = _safe_relative(str(record.get("path", "")))
        if not relative or relative in registered:
            errors.append(f"duplicate_or_empty:{relative}")
            continue
        registered.add(relative)
        availability = str(record.get("availability", "required"))
        if availability not in {"required", "local_ignored_optional"}:
            errors.append(f"availability:{relative}")
            continue
        path = artifact_root / relative
        if not path.is_file():
            if availability == "local_ignored_optional":
                optional_unavailable += 1
                warnings.append(f"optional_unavailable:{relative}")
                continue
            errors.append(f"missing:{relative}")
            continue
        if path.stat().st_size != int(record.get("bytes", -1)):
            errors.append(f"bytes:{relative}")
            continue
        if sha256_file(path) != str(record.get("sha256", "")):
            errors.append(f"sha256:{relative}")
            continue
        verified += 1

    if int(payload.get("artifact_count", -1)) != len(records):
        errors.append("artifact_count")
    patterns = payload.get("managed_artifact_globs", [])
    if not isinstance(patterns, list):
        raise ArtifactRegistryError("managed_artifact_globs must be a list")
    managed = _managed_files(artifact_root, [str(value) for value in patterns])
    self_path = payload.get("registry_self_path")
    excluded = {_safe_relative(str(self_path))} if self_path else set()
    extras = sorted(managed.difference(registered).difference(excluded))
    for relative in extras:
        errors.append(f"extra:{relative}")

    return {
        "schema_version": 1,
        "registry_id": payload.get("registry_id"),
        "stage": payload.get("stage"),
        "verified": not errors,
        "artifact_count": len(records),
        "verified_artifacts": verified,
        "optional_unavailable": optional_unavailable,
        "managed_artifact_count": len(managed),
        "errors": errors,
        "warnings": warnings,
        "registry_path": str(registry_path),
        "registry_bytes": registry_path.stat().st_size,
        "registry_sha256": sha256_file(registry_path),
    }


def verify_historical_artifact_registry(
    registry_path: Path,
    *,
    artifact_root: Path,
    expected_registry_sha256: str,
    immutable_path_prefixes: Sequence[str],
    classification: str = "pre_hardening_historical_snapshot",
) -> dict[str, Any]:
    """Verify an immutable registry and selected historical evidence.

    A post-stage worktree must not be compared with an earlier release
    manifest as if the old source set were still current. This verifier pins
    the registry file itself, validates its saved schema/count/path safety,
    and checks only explicitly immutable evidence such as frozen media or
    ignored smoke outputs. Current source is owned by the newest registry.
    """

    registry_path = registry_path.resolve()
    artifact_root = artifact_root.resolve()
    errors: list[str] = []
    warnings: list[str] = []
    actual_registry_sha256 = sha256_file(registry_path)
    if actual_registry_sha256 != expected_registry_sha256:
        errors.append("registry_identity")
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ArtifactRegistryError("Registry must contain a mapping")
    records = payload.get("artifacts")
    if not isinstance(records, list):
        raise ArtifactRegistryError("Registry artifacts must be a list")
    if int(payload.get("artifact_count", -1)) != len(records):
        errors.append("artifact_count")

    normalized_prefixes = tuple(
        _safe_relative(str(prefix)).rstrip("/") + "/"
        for prefix in immutable_path_prefixes
    )
    seen: set[str] = set()
    verified_evidence = 0
    optional_unavailable = 0
    for record in records:
        if not isinstance(record, Mapping):
            errors.append("malformed_artifact_record")
            continue
        relative = _safe_relative(str(record.get("path", "")))
        if not relative or relative in seen:
            errors.append(f"duplicate_or_empty:{relative}")
            continue
        seen.add(relative)
        if not relative.startswith(normalized_prefixes):
            continue
        availability = str(record.get("availability", "required"))
        path = artifact_root / relative
        if not path.is_file():
            if availability == "local_ignored_optional":
                optional_unavailable += 1
                warnings.append(f"optional_unavailable:{relative}")
            else:
                errors.append(f"missing_immutable_evidence:{relative}")
            continue
        if path.stat().st_size != int(record.get("bytes", -1)):
            errors.append(f"bytes_immutable_evidence:{relative}")
        elif sha256_file(path) != str(record.get("sha256", "")):
            errors.append(f"sha256_immutable_evidence:{relative}")
        else:
            verified_evidence += 1

    return {
        "schema_version": 1,
        "registry_id": payload.get("registry_id", payload.get("protocol_id")),
        "stage": payload.get("stage"),
        "classification": classification,
        "verified": not errors,
        "errors": errors,
        "warnings": warnings,
        "artifact_count": len(records),
        "verified_immutable_evidence": verified_evidence,
        "optional_unavailable": optional_unavailable,
        "registry_sha256": actual_registry_sha256,
        "live_release_artifacts_compared": False,
    }


def verify_frozen_stage_u_snapshot(repository_root: Path) -> dict[str, Any]:
    """Verify the saved Stage U manifest/audit chain as a historical snapshot.

    This deliberately does not compare the Stage U manifest with the live
    post-Stage-U worktree. Later-stage files are outside that frozen candidate
    set. The immutable registry identity, stored candidate coverage, audit/CSV
    agreement, and saved Stage U evidence files are still verified.
    """

    root = repository_root.resolve()
    stage_u = root / "implementation" / "data" / "stage_u"
    registry_path = (
        stage_u / "STAGE_U_SUBMISSION_ARTIFACT_REGISTRY_20260730.yaml"
    )
    audit_path = stage_u / "STAGE_U_SUBMISSION_AUDIT.json"
    csv_path = stage_u / "STAGE_U_SUBMISSION_CANDIDATES.csv"
    errors: list[str] = []
    if sha256_file(registry_path) != STAGE_U_FROZEN_REGISTRY_SHA256:
        errors.append("registry_identity")
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    records = registry.get("artifacts", [])
    registry_rows = {
        str(row["path"]).replace("\\", "/"): row for row in records
    }
    if len(registry_rows) != int(registry.get("artifact_count", -1)):
        errors.append("registry_artifact_count")
    self_path = str(registry.get("registry_self_path", "")).replace("\\", "/")
    frozen_candidates = set(registry_rows) | {self_path}

    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    audit_rows = {str(row["path"]): row for row in audit.get("records", [])}
    with csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = {str(row["path"]): row for row in csv.DictReader(handle)}
    if set(audit_rows) != frozen_candidates:
        errors.append("audit_candidate_coverage")
    if set(csv_rows) != frozen_candidates:
        errors.append("csv_candidate_coverage")
    if int(audit.get("candidate_files", -1)) != len(frozen_candidates):
        errors.append("audit_candidate_count")

    excluded = {
        "implementation/data/stage_u/STAGE_U_SUBMISSION_AUDIT.json",
        "implementation/data/stage_u/STAGE_U_SUBMISSION_CANDIDATES.csv",
        self_path,
    }
    for relative in frozen_candidates:
        json_hash = str(audit_rows.get(relative, {}).get("sha256", ""))
        csv_hash = str(csv_rows.get(relative, {}).get("sha256", ""))
        if relative in excluded:
            if (
                json_hash != STAGE_U_SELF_REFERENCE_EXCLUDED
                or csv_hash != STAGE_U_SELF_REFERENCE_EXCLUDED
            ):
                errors.append(f"self_reference_policy:{relative}")
            continue
        expected_hash = str(registry_rows[relative]["sha256"])
        if json_hash != expected_hash or csv_hash != expected_hash:
            errors.append(f"saved_hash_chain:{relative}")

    frozen_evidence_prefix = "implementation/data/stage_u/"
    verified_saved_evidence = 0
    for relative, record in registry_rows.items():
        if not relative.startswith(frozen_evidence_prefix):
            continue
        path = root / relative
        if not path.is_file():
            errors.append(f"missing_saved_evidence:{relative}")
        elif path.stat().st_size != int(record["bytes"]):
            errors.append(f"bytes_saved_evidence:{relative}")
        elif sha256_file(path) != str(record["sha256"]):
            errors.append(f"sha256_saved_evidence:{relative}")
        else:
            verified_saved_evidence += 1

    return {
        "schema_version": 1,
        "stage": "U",
        "verified": not errors,
        "errors": errors,
        "frozen_candidate_count": len(frozen_candidates),
        "registry_artifact_count": len(records),
        "verified_saved_evidence": verified_saved_evidence,
        "registry_sha256": sha256_file(registry_path),
        "live_post_stage_u_candidates_compared": False,
    }
