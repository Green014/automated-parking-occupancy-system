from __future__ import annotations

import fnmatch
import hashlib
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import yaml


FORMAL_CONFIG_RELATIVE = (
    "implementation/configs/"
    "p3_stage_r_recommended_default_20260729.yaml"
)
FORMAL_CONFIG_SHA256 = (
    "198f627689cd93f66ca0f087af6686d3afc697ff51e2aa77ee56124187b981b0"
)
SOURCE_MANIFEST_RELATIVE = (
    "implementation/data/STAGE_W_3_PUBLIC_SOURCE_MANIFEST.yaml"
)
PRIVATE_PERMISSION_RELATIVE = (
    "implementation/data/STAGE_W_PERMISSION_AND_PROVENANCE.md"
)
PUBLIC_PERMISSION_RELATIVE = (
    "implementation/data/PUBLIC_PERMISSION_AND_PROVENANCE.md"
)
REQUIRED_PUBLIC_FILES = {
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    PUBLIC_PERMISSION_RELATIVE,
    "implementation/data/MODEL_CARD_D1.md",
    "implementation/data/MODEL_CARD_E1B.md",
    "implementation/data/STAGE_W_3_MODEL_RELEASE_MANIFEST.yaml",
}
MODEL_SUFFIXES = {".pt", ".pth", ".ckpt", ".onnx", ".engine"}
TEXT_SUFFIXES = {
    "",
    ".cfg",
    ".css",
    ".csv",
    ".html",
    ".ini",
    ".json",
    ".jsonl",
    ".md",
    ".py",
    ".svg",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}


class StageW3ReleaseError(ValueError):
    """Raised when the privacy-safe public source contract is violated."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_relative(value: str) -> str:
    normalized = value.replace("\\", "/")
    relative = PurePosixPath(normalized)
    if (
        not normalized
        or relative.is_absolute()
        or ".." in relative.parts
        or re.match(r"^[A-Za-z]:", normalized)
    ):
        raise StageW3ReleaseError(f"Unsafe manifest path: {value}")
    return relative.as_posix()


def load_public_source_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise StageW3ReleaseError("Public source manifest must be a mapping")
    if payload.get("stage") != "W.3":
        raise StageW3ReleaseError("Public source manifest stage must be W.3")
    return dict(payload)


def _available_files(repository_root: Path) -> set[str]:
    if (repository_root / ".git").exists():
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
            capture_output=True,
            check=True,
        )
        return {
            item.decode("utf-8").replace("\\", "/")
            for item in result.stdout.split(b"\0")
            if item and (repository_root / item.decode("utf-8")).is_file()
        }
    return {
        path.relative_to(repository_root).as_posix()
        for path in repository_root.rglob("*")
        if path.is_file() and ".git" not in path.parts
    }


def _matching_files(
    repository_root: Path,
    available_files: set[str],
    group: Mapping[str, Any],
) -> set[str]:
    matches: set[str] = set()
    for raw in group.get("paths", []):
        relative = _safe_relative(str(raw))
        if (repository_root / relative).is_file():
            matches.add(relative)
    for raw in group.get("globs", []):
        pattern = _safe_relative(str(raw))
        variants = {pattern}
        collapsed = pattern
        while "/**/" in collapsed:
            collapsed = collapsed.replace("/**/", "/")
            variants.add(collapsed)
        matches.update(
            relative
            for relative in available_files
            if any(
                fnmatch.fnmatchcase(relative, variant)
                for variant in variants
            )
        )
    return matches


def _privacy_findings(relative: str, content: str) -> list[str]:
    findings: list[str] = []
    windows_home = re.compile(
        r"(?i)[A-Z]:(?:[\\/]|\\\\)+Users(?:[\\/]|\\\\)+"
        r"[^\\/\s\"]+(?:[\\/]|\\\\)+"
    )
    email = re.compile(
        r"(?i)(?<![A-Z0-9._%+-])[A-Z0-9._%+-]+"
        r"@[A-Z0-9.-]+\.[A-Z]{2,}(?![A-Z0-9._%+-])"
    )
    credential_rtsp = re.compile(
        r"(?i)rtsp://[^\s/:@]+:[^\s/@]+@[^\s/]+"
    )
    private_key = re.compile(
        r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"
    )
    token = re.compile(
        r"(?i)(?:gh[pousr]_[A-Za-z0-9]{20,}|"
        r"AKIA[0-9A-Z]{16}|"
        r"(?:api[_-]?key|secret[_-]?key)\s*[:=]\s*['\"]"
        r"[A-Za-z0-9_./+=-]{12,}['\"])"
    )
    for label, pattern in (
        ("machine_user_home", windows_home),
        ("email_address", email),
        ("rtsp_credentials", credential_rtsp),
        ("private_key", private_key),
        ("secret_token", token),
    ):
        if pattern.search(content):
            findings.append(f"{label}:{relative}")
    return findings


def scan_public_candidate(
    repository_root: Path,
    candidates: Sequence[str],
) -> dict[str, Any]:
    findings: list[str] = []
    scanned = 0
    for relative in sorted(set(candidates)):
        path = repository_root / _safe_relative(relative)
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"non_utf8_text:{relative}")
            continue
        scanned += 1
        findings.extend(_privacy_findings(relative, content))
    return {
        "verified": not findings,
        "scanned_text_files": scanned,
        "finding_count": len(findings),
        "findings": findings,
    }


def resolve_public_source_manifest(
    repository_root: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else repository_root / SOURCE_MANIFEST_RELATIVE
    )
    payload = load_public_source_manifest(manifest_path)
    available_files = _available_files(repository_root)
    included_by_category: dict[str, set[str]] = {}
    included: set[str] = set()
    for group in payload.get("include", []):
        category = str(group["category"])
        matches = _matching_files(repository_root, available_files, group)
        included_by_category[category] = matches
        included.update(matches)

    excluded_by_category: dict[str, set[str]] = {}
    excluded: set[str] = set()
    for group in payload.get("exclude", []):
        category = str(group["category"])
        matches = _matching_files(repository_root, available_files, group)
        excluded_by_category[category] = matches
        excluded.update(matches)

    privacy_excluded: set[str] = set()
    privacy_exclusion_reasons: list[str] = []
    for relative in sorted(included.difference(excluded)):
        path = repository_root / relative
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            privacy_excluded.add(relative)
            privacy_exclusion_reasons.append(f"non_utf8_text:{relative}")
            continue
        findings = _privacy_findings(relative, content)
        if findings:
            privacy_excluded.add(relative)
            privacy_exclusion_reasons.extend(findings)
    excluded.update(privacy_excluded)
    excluded_by_category["privacy_sensitive_text"] = privacy_excluded

    candidates = sorted(included.difference(excluded))
    violations: list[str] = []
    candidate_set = set(candidates)
    missing_required = sorted(REQUIRED_PUBLIC_FILES.difference(candidate_set))
    violations.extend(f"missing_required:{item}" for item in missing_required)
    if PRIVATE_PERMISSION_RELATIVE in candidate_set:
        violations.append("private_permission_record_in_candidate")
    for relative in candidates:
        path = repository_root / relative
        if path.suffix.lower() in MODEL_SUFFIXES:
            violations.append(f"model_asset_in_source:{relative}")
        if any(
            part.lower()
            in {
                ".git",
                ".pytest_cache",
                "__pycache__",
                ".venv",
                "datasets",
                "outputs",
                "runs",
                "vendor",
                "weights",
            }
            for part in PurePosixPath(relative).parts
        ):
            violations.append(f"forbidden_path:{relative}")

    formal_path = repository_root / FORMAL_CONFIG_RELATIVE
    formal_sha256 = sha256_file(formal_path)
    if FORMAL_CONFIG_RELATIVE not in candidate_set:
        violations.append("formal_config_not_in_candidate")
    if formal_sha256 != FORMAL_CONFIG_SHA256:
        violations.append("formal_config_sha256")
    if b"\r\n" in formal_path.read_bytes():
        violations.append("formal_config_crlf")

    privacy_scan = scan_public_candidate(repository_root, candidates)
    violations.extend(
        f"privacy_scan:{item}" for item in privacy_scan["findings"]
    )
    candidate_bytes = sum(
        (repository_root / relative).stat().st_size for relative in candidates
    )
    expected = payload.get("resolved_summary", {})
    expected_files = expected.get("candidate_file_count")
    expected_bytes = expected.get("candidate_bytes")
    if expected_files is not None and int(expected_files) != len(candidates):
        violations.append("candidate_file_count")
    if expected_bytes is not None and int(expected_bytes) != candidate_bytes:
        violations.append("candidate_bytes")

    return {
        "schema_version": 1,
        "manifest_id": payload.get("manifest_id"),
        "verified": not violations,
        "source_publication_ready": not violations,
        "violations": violations,
        "candidate_files": candidates,
        "candidate_file_count": len(candidates),
        "candidate_bytes": candidate_bytes,
        "include_category_count": len(included_by_category),
        "exclude_category_count": len(excluded_by_category),
        "included_by_category": {
            key: sorted(value) for key, value in included_by_category.items()
        },
        "excluded_by_category": {
            key: sorted(value) for key, value in excluded_by_category.items()
        },
        "excluded_existing_file_count": len(included.intersection(excluded)),
        "privacy_exclusion_reasons": privacy_exclusion_reasons,
        "privacy_scan": privacy_scan,
        "formal_config_sha256": formal_sha256,
        "public_release_published": False,
        "model_training_run": False,
        "model_inference_run": False,
    }


def copy_public_source_candidate(
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
        safe_relative = _safe_relative(relative)
        source = repository_root / safe_relative
        target = destination / safe_relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        total_bytes += target.stat().st_size
    return {
        "destination": str(destination),
        "file_count": len(set(candidates)),
        "bytes": total_bytes,
    }
