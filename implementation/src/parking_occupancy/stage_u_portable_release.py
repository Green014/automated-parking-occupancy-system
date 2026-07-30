from __future__ import annotations

import csv
import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


STAGE_U_PROTOCOL_ID = "STAGE-U-PORTABLE-FINAL-RELEASE-20260730-01"
PORTABLE_REGISTRY_RELATIVE = (
    "implementation/data/stage_u/"
    "STAGE_U_SUBMISSION_ARTIFACT_REGISTRY_20260730.yaml"
)
SUBMISSION_AUDIT_SELF_PATHS = (
    "implementation/data/stage_u/STAGE_U_SUBMISSION_AUDIT.json",
    "implementation/data/stage_u/STAGE_U_SUBMISSION_CANDIDATES.csv",
)
SELF_REFERENCE_EXCLUDED = "SELF_REFERENCE_EXCLUDED"
SUBMISSION_AUDIT_HASH_EXCLUDED_PATHS = (
    *SUBMISSION_AUDIT_SELF_PATHS,
    PORTABLE_REGISTRY_RELATIVE,
)
LOCAL_ONLY_REGISTRY_PATHS = (
    "implementation/data/stage_m/STAGE_M_ARTIFACT_REGISTRY_20260728.yaml",
    "implementation/data/stage_n/STAGE_N_ARTIFACT_REGISTRY_20260728.yaml",
    "implementation/data/stage_n_v2/STAGE_N_V2_ARTIFACT_REGISTRY_20260729.yaml",
    "implementation/data/stage_n_v3/STAGE_N_V3_ARTIFACT_REGISTRY_20260729.yaml",
    "implementation/data/stage_o/STAGE_O_ARTIFACT_REGISTRY_20260729.yaml",
    "implementation/data/stage_p/STAGE_P_ARTIFACT_REGISTRY_20260729.yaml",
    "implementation/data/stage_q/STAGE_Q_ARTIFACT_REGISTRY_20260729.yaml",
    "implementation/data/stage_q_v2/STAGE_Q_V2_ARTIFACT_REGISTRY_20260729.yaml",
    "implementation/data/stage_t/STAGE_T_ARTIFACT_REGISTRY_20260729.yaml",
)
PORTABLE_HISTORICAL_REGISTRY_PATHS = (
    "implementation/data/stage_r/STAGE_R_ARTIFACT_REGISTRY_20260729.yaml",
    "implementation/data/stage_s/STAGE_S_ARTIFACT_REGISTRY_20260729.yaml",
)
INTENTIONAL_PRESENTATION_PATHS = (
    "implementation/data/stage_s/demo/demo_main.mp4",
    "implementation/data/stage_s/demo/demo_keyframe_default.png",
    "implementation/data/stage_s/demo/demo_keyframe_d1_vs_d1ll.png",
    "implementation/data/stage_s/demo/demo_keyframe_f2_recovery.png",
    "implementation/data/stage_t/demo/demo_tracktrack_optional.mp4",
    "implementation/data/stage_t/demo/demo_tracktrack_optional_keyframe.png",
    (
        "implementation/data/stage_u_1/demo/"
        "demo_tracktrack_identity_diagnostic_presentation.mp4"
    ),
    (
        "implementation/data/stage_u_1/demo/"
        "demo_tracktrack_identity_diagnostic_presentation_keyframe.png"
    ),
)
REQUIRED_RUNTIME_FILES = (
    "README.md",
    "implementation/scripts/run_p3_tt.py",
    "implementation/src/parking_occupancy/p3_tt_runtime.py",
    "implementation/src/parking_occupancy/integrated_runner.py",
    "implementation/src/parking_occupancy/stage_m_tracking.py",
    "implementation/configs/p3_tt_tracktrack_optional_20260729.yaml",
    "implementation/configs/tracktrack_stage_m_frozen_20260728.yaml",
)
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt"}
TEXT_SUFFIXES = {
    ".cfg",
    ".csv",
    ".ini",
    ".json",
    ".md",
    ".py",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
ABSOLUTE_WINDOWS_PATH = re.compile(r"[A-Za-z]:[\\/](?:Users|Program Files)[\\/]")


class StageUPortableError(ValueError):
    """Raised when a portable-release invariant is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_paths(repository_root: Path) -> list[str]:
    result = subprocess.run(
        [
            "git",
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
    return sorted(
        item.decode("utf-8").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    )


def _ignored_even_if_tracked(
    repository_root: Path,
    relative_paths: Sequence[str],
) -> set[str]:
    if not relative_paths:
        return set()
    result = subprocess.run(
        ["git", "check-ignore", "--no-index", "-z", "--stdin"],
        cwd=repository_root,
        input=("\0".join(relative_paths) + "\0").encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if result.returncode not in {0, 1}:
        raise StageUPortableError(
            f"git check-ignore failed: {result.stderr.decode(errors='replace')}"
        )
    return {
        item.decode("utf-8").replace("\\", "/")
        for item in result.stdout.split(b"\0")
        if item
    }


def submission_candidates(repository_root: Path) -> list[str]:
    repository_root = repository_root.resolve()
    local_only = set(LOCAL_ONLY_REGISTRY_PATHS)
    paths = [
        path
        for path in _git_paths(repository_root)
        if path not in local_only
        and (repository_root / Path(path)).is_file()
    ]
    ignored = _ignored_even_if_tracked(repository_root, paths)
    return [path for path in paths if path not in ignored]


def _forbidden_reason(relative: str) -> str | None:
    normalized = "/" + relative.replace("\\", "/").lower().strip("/") + "/"
    suffix = Path(relative).suffix.lower()
    if suffix in MODEL_SUFFIXES:
        return "model_weight"
    if "/implementation/outputs/" in normalized or "/outputs/" in normalized:
        return "runtime_output"
    if "/data/external/" in normalized:
        return "external_data"
    if "/datasets/" in normalized:
        return "dataset"
    if "/.venv" in normalized or "/venv/" in normalized:
        return "virtual_environment"
    return None


def _absolute_path_mentions(path: Path) -> int:
    if path.suffix.lower() not in TEXT_SUFFIXES:
        return 0
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return 0
    return len(ABSOLUTE_WINDOWS_PATH.findall(text))


def audit_submission(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    candidates = submission_candidates(repository_root)
    ignored_returned = sorted(
        _ignored_even_if_tracked(repository_root, candidates)
    )
    records: list[dict[str, Any]] = []
    violations: list[dict[str, str]] = []
    absolute_mentions = 0
    required_runtime_absolute_dependencies: list[str] = []
    presentation = []
    licenses = []
    for relative in candidates:
        path = repository_root / relative
        reason = _forbidden_reason(relative)
        if reason:
            violations.append({"path": relative, "reason": reason})
        mentions = _absolute_path_mentions(path)
        absolute_mentions += mentions
        if relative in REQUIRED_RUNTIME_FILES and mentions:
            required_runtime_absolute_dependencies.append(relative)
        if relative in INTENTIONAL_PRESENTATION_PATHS:
            presentation.append(relative)
        if "license" in path.name.lower() or path.name.lower().startswith("copying"):
            licenses.append(relative)
        records.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": (
                    SELF_REFERENCE_EXCLUDED
                    if relative in SUBMISSION_AUDIT_HASH_EXCLUDED_PATHS
                    else sha256_file(path)
                ),
                "absolute_path_mentions": mentions,
            }
        )
    if ignored_returned:
        violations.extend(
            {"path": path, "reason": "git_ignored_candidate"}
            for path in ignored_returned
        )
    if required_runtime_absolute_dependencies:
        violations.extend(
            {"path": path, "reason": "required_runtime_absolute_path_dependency"}
            for path in required_runtime_absolute_dependencies
        )
    missing_presentations = sorted(
        set(INTENTIONAL_PRESENTATION_PATHS) - set(presentation)
    )
    if missing_presentations:
        violations.extend(
            {"path": path, "reason": "required_presentation_missing"}
            for path in missing_presentations
        )
    sizes = [record["bytes"] for record in records]
    return {
        "schema_version": 1,
        "protocol_id": STAGE_U_PROTOCOL_ID,
        "status": "PASS" if not violations else "FAIL",
        "candidate_files": len(records),
        "candidate_bytes": sum(sizes),
        "largest_file_bytes": max(sizes, default=0),
        "largest_files": sorted(
            records,
            key=lambda record: (-int(record["bytes"]), str(record["path"])),
        )[:20],
        "forbidden_content": {
            "model_weights": any(
                item["reason"] == "model_weight" for item in violations
            ),
            "datasets_or_external_data": any(
                item["reason"] in {"dataset", "external_data"}
                for item in violations
            ),
            "implementation_outputs": any(
                item["reason"] == "runtime_output" for item in violations
            ),
            "virtual_environments": any(
                item["reason"] == "virtual_environment" for item in violations
            ),
            "required_runtime_absolute_path_dependencies": bool(
                required_runtime_absolute_dependencies
            ),
            "git_ignored_files": bool(ignored_returned),
        },
        "absolute_path_mentions_in_historical_text": absolute_mentions,
        "required_runtime_absolute_path_dependencies": (
            required_runtime_absolute_dependencies
        ),
        "intentional_presentation_artifacts": presentation,
        "license_files": licenses,
        "local_only_registries_excluded": list(LOCAL_ONLY_REGISTRY_PATHS),
        "audit_hash_exclusion_policy": {
            "marker": SELF_REFERENCE_EXCLUDED,
            "paths": list(SUBMISSION_AUDIT_HASH_EXCLUDED_PATHS),
            "reason": (
                "Audit JSON/CSV are self-referential and the portable registry "
                "is finalized after the audit. Their current byte sizes remain "
                "recorded, but the audit never claims hashes that would become "
                "stale during final registry generation."
            ),
        },
        "audit_self_paths_excluded_from_self_hashing": list(
            SUBMISSION_AUDIT_HASH_EXCLUDED_PATHS
        ),
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
            fieldnames=[
                "path",
                "bytes",
                "sha256",
                "absolute_path_mentions",
            ],
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(audit["records"])


def verify_saved_submission_evidence(
    *,
    repository_root: Path,
    audit_json_path: Path,
    audit_csv_path: Path,
    registry_path: Path,
) -> dict[str, Any]:
    """Verify the finalized one-way audit -> registry evidence chain."""

    repository_root = repository_root.resolve()
    audit_json_path = audit_json_path.resolve()
    audit_csv_path = audit_csv_path.resolve()
    registry_path = registry_path.resolve()
    audit = json.loads(audit_json_path.read_text(encoding="utf-8"))
    with audit_csv_path.open(encoding="utf-8", newline="") as handle:
        csv_rows = list(csv.DictReader(handle))
    errors: list[str] = []
    candidates = submission_candidates(repository_root)
    candidate_set = set(candidates)
    json_rows = {str(row["path"]): row for row in audit.get("records", [])}
    csv_by_path = {str(row["path"]): row for row in csv_rows}
    if set(json_rows) != candidate_set:
        errors.append("audit_json_candidate_coverage")
    if set(csv_by_path) != candidate_set:
        errors.append("audit_csv_candidate_coverage")
    actual_bytes = sum(
        (repository_root / relative).stat().st_size for relative in candidates
    )
    if int(audit.get("candidate_files", -1)) != len(candidates):
        errors.append("audit_candidate_files")
    if int(audit.get("candidate_bytes", -1)) != actual_bytes:
        errors.append("audit_candidate_bytes")
    for relative in candidates:
        path = repository_root / relative
        json_row = json_rows.get(relative)
        csv_row = csv_by_path.get(relative)
        if json_row is None or csv_row is None:
            continue
        actual_size = path.stat().st_size
        if int(json_row["bytes"]) != actual_size:
            errors.append(f"audit_json_bytes:{relative}")
        if int(csv_row["bytes"]) != actual_size:
            errors.append(f"audit_csv_bytes:{relative}")
        expected_hash = (
            SELF_REFERENCE_EXCLUDED
            if relative in SUBMISSION_AUDIT_HASH_EXCLUDED_PATHS
            else sha256_file(path)
        )
        if str(json_row["sha256"]) != expected_hash:
            errors.append(f"audit_json_sha256:{relative}")
        if str(csv_row["sha256"]) != expected_hash:
            errors.append(f"audit_csv_sha256:{relative}")
        if str(csv_row["absolute_path_mentions"]) != str(
            json_row["absolute_path_mentions"]
        ):
            errors.append(f"audit_csv_absolute_mentions:{relative}")
    registry = verify_portable_registry(
        registry_path,
        package_root=repository_root,
        require_complete_coverage=True,
    )
    if not registry["verified"]:
        errors.extend(f"registry:{error}" for error in registry["errors"])
    return {
        "schema_version": 1,
        "protocol_id": STAGE_U_PROTOCOL_ID,
        "verified": not errors,
        "errors": errors,
        "candidate_file_count": len(candidates),
        "candidate_bytes": actual_bytes,
        "registry_artifact_count": registry["artifact_count"],
        "registry_candidate_file_count": registry["candidate_file_count"],
        "registry_sha256": registry["registry_sha256"],
        "hash_excluded_paths": list(SUBMISSION_AUDIT_HASH_EXCLUDED_PATHS),
    }


def _binding_records(value: Any) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    if isinstance(value, Mapping):
        if {"path", "bytes", "sha256"} <= set(value):
            records.append(dict(value))
        for nested in value.values():
            records.extend(_binding_records(nested))
    elif isinstance(value, list):
        for nested in value:
            records.extend(_binding_records(nested))
    return records


def _resolve_historical_artifact(
    repository_root: Path,
    registry_path: Path,
    value: str,
) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    options = (
        repository_root / path,
        repository_root / "implementation" / path,
        registry_path.parent / path,
    )
    for option in options:
        if option.is_file():
            return option.resolve()
    return options[0].resolve()


def audit_historical_registries(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    candidates = set(submission_candidates(repository_root))
    registry_paths = sorted(
        path
        for path in (repository_root / "implementation" / "data").rglob("*")
        if path.is_file()
        and "artifact_registry" in path.name.lower()
        and path.suffix.lower() in {".yaml", ".yml", ".json"}
        and path.relative_to(repository_root).as_posix()
        != PORTABLE_REGISTRY_RELATIVE
    )
    rows = []
    for registry_path in registry_paths:
        relative = registry_path.relative_to(repository_root).as_posix()
        payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
        bindings = _binding_records(payload)
        reasons: set[str] = set()
        errors: list[str] = []
        for binding in bindings:
            raw_path = str(binding["path"])
            normalized = "/" + raw_path.replace("\\", "/").lower().strip("/") + "/"
            if Path(raw_path).is_absolute():
                reasons.add("machine_absolute_paths")
            if "/outputs/" in normalized:
                reasons.add("ignored_runtime_outputs")
            if "/data/external/" in normalized or "/datasets/" in normalized:
                reasons.add("ignored_dataset_or_external_data")
            if Path(raw_path).suffix.lower() in MODEL_SUFFIXES:
                reasons.add("model_weights")
            artifact = _resolve_historical_artifact(
                repository_root, registry_path, raw_path
            )
            label = str(binding.get("label", raw_path))
            if not artifact.is_file():
                errors.append(f"missing:{label}")
            elif artifact.stat().st_size != int(binding["bytes"]):
                errors.append(f"bytes:{label}")
            elif sha256_file(artifact) != str(binding["sha256"]):
                errors.append(f"sha256:{label}")
            else:
                try:
                    artifact_relative = artifact.relative_to(repository_root).as_posix()
                except ValueError:
                    reasons.add("artifact_outside_repository")
                else:
                    if artifact_relative not in candidates:
                        reasons.add("artifact_not_in_submission_candidates")
        expected_class = (
            "local-only historical registry" if reasons else "portable historical registry"
        )
        rows.append(
            {
                "path": relative,
                "bytes": registry_path.stat().st_size,
                "sha256": sha256_file(registry_path),
                "registry_class": expected_class,
                "classification_reasons": sorted(reasons),
                "binding_count": len(bindings),
                "current_local_verification": {
                    "verified": not errors,
                    "errors": errors,
                },
                "included_in_submission_candidates": relative in candidates,
                "content_modified_by_stage_u": False,
            }
        )
    return {
        "schema_version": 1,
        "protocol_id": STAGE_U_PROTOCOL_ID,
        "status": "AUDITED_WITHOUT_HISTORICAL_REGISTRY_MODIFICATION",
        "registry_count": len(rows),
        "portable_count": sum(
            row["registry_class"] == "portable historical registry" for row in rows
        ),
        "local_only_count": sum(
            row["registry_class"] == "local-only historical registry" for row in rows
        ),
        "registries": rows,
    }


def write_historical_registry_audit(
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
    fields = [
        "path",
        "registry_class",
        "binding_count",
        "included_in_submission_candidates",
        "locally_verified",
        "classification_reasons",
        "sha256",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in audit["registries"]:
            writer.writerow(
                {
                    "path": row["path"],
                    "registry_class": row["registry_class"],
                    "binding_count": row["binding_count"],
                    "included_in_submission_candidates": row[
                        "included_in_submission_candidates"
                    ],
                    "locally_verified": row["current_local_verification"]["verified"],
                    "classification_reasons": "|".join(
                        row["classification_reasons"]
                    ),
                    "sha256": row["sha256"],
                }
            )


def _portable_record(repository_root: Path, relative: str) -> dict[str, Any]:
    path = repository_root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    if _forbidden_reason(relative):
        raise StageUPortableError(f"Forbidden portable artifact: {relative}")
    if Path(relative).is_absolute() or ".." in PurePosixPath(relative).parts:
        raise StageUPortableError(f"Non-portable registry path: {relative}")
    return {
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_portable_registry(
    registry_path: Path,
    *,
    repository_root: Path,
    candidates: Sequence[str],
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    registry_relative = registry_path.resolve().relative_to(
        repository_root
    ).as_posix()
    unique_candidates = sorted(set(candidates))
    if registry_relative not in unique_candidates:
        raise StageUPortableError(
            "Portable registry must itself be a submission candidate"
        )
    records = [
        _portable_record(repository_root, relative)
        for relative in unique_candidates
        if relative != registry_relative
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": STAGE_U_PROTOCOL_ID,
        "registry_id": "STAGE-U-SUBMISSION-ARTIFACT-REGISTRY-20260730-01",
        "registry_type": "portable_submission_registry",
        "created_on": "2026-07-30",
        "submission_root": ".",
        "candidate_file_count": len(unique_candidates),
        "artifact_count": len(records),
        "registry_self_hash_included": False,
        "registry_self_path": registry_relative,
        "coverage": "all submission candidates except this registry file",
        "contains_implementation_outputs": False,
        "contains_datasets_or_external_data": False,
        "contains_model_weights": False,
        "contains_virtual_environments": False,
        "contains_machine_absolute_artifact_paths": False,
        "artifacts": records,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            width=140,
        ),
        encoding="utf-8",
    )
    return verify_portable_registry(
        registry_path,
        package_root=repository_root,
        require_complete_coverage=True,
    )


def _package_files(package_root: Path) -> set[str]:
    files: set[str] = set()
    for path in package_root.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(package_root)
        lowered = {part.lower() for part in relative.parts}
        if ".git" in lowered or ".pytest_cache" in lowered or "__pycache__" in lowered:
            continue
        if path.suffix.lower() in {".pyc", ".pyo"}:
            continue
        files.add(relative.as_posix())
    return files


def verify_portable_registry(
    registry_path: Path,
    *,
    package_root: Path,
    require_complete_coverage: bool = True,
) -> dict[str, Any]:
    package_root = package_root.resolve()
    registry_path = registry_path.resolve()
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("protocol_id") != STAGE_U_PROTOCOL_ID:
        errors.append("protocol_id")
    if payload.get("registry_type") != "portable_submission_registry":
        errors.append("registry_type")
    records = payload.get("artifacts", [])
    if int(payload.get("artifact_count", -1)) != len(records):
        errors.append("artifact_count")
    registered: set[str] = set()
    for record in records:
        relative = str(record["path"]).replace("\\", "/")
        if (
            Path(relative).is_absolute()
            or ".." in PurePosixPath(relative).parts
            or _forbidden_reason(relative)
        ):
            errors.append(f"nonportable:{relative}")
            continue
        registered.add(relative)
        path = package_root / relative
        if not path.is_file():
            errors.append(f"missing:{relative}")
        elif path.stat().st_size != int(record["bytes"]):
            errors.append(f"bytes:{relative}")
        elif sha256_file(path) != str(record["sha256"]):
            errors.append(f"sha256:{relative}")
    registry_relative = str(payload.get("registry_self_path", "")).replace(
        "\\", "/"
    )
    if require_complete_coverage:
        expected = registered | {registry_relative}
        if (package_root / ".git").exists():
            actual = set(submission_candidates(package_root))
        else:
            actual = _package_files(package_root)
        if actual != expected:
            for relative in sorted(expected - actual):
                errors.append(f"coverage_missing:{relative}")
            for relative in sorted(actual - expected):
                errors.append(f"coverage_extra:{relative}")
    return {
        "schema_version": 1,
        "protocol_id": STAGE_U_PROTOCOL_ID,
        "verified": not errors,
        "errors": errors,
        "artifact_count": len(records),
        "candidate_file_count": len(records) + 1,
        "registry_bytes": registry_path.stat().st_size,
        "registry_sha256": sha256_file(registry_path),
        "complete_coverage_checked": require_complete_coverage,
    }


def copy_clean_package(
    *,
    repository_root: Path,
    destination: Path,
    candidates: Sequence[str],
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    destination = destination.resolve()
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    total_bytes = 0
    for relative in sorted(set(candidates)):
        source = repository_root / relative
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        total_bytes += target.stat().st_size
    return {
        "path": str(destination),
        "files": len(set(candidates)),
        "bytes": total_bytes,
    }
