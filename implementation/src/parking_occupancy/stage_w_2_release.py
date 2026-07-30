from __future__ import annotations

import hashlib
import fnmatch
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
    "implementation/data/STAGE_W_2_SOURCE_COMMIT_MANIFEST.yaml"
)


class StageW2ReleaseError(ValueError):
    """Raised when the W.2 portable-source contract is violated."""


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
        raise StageW2ReleaseError(f"Unsafe manifest path: {value}")
    return relative.as_posix()


def load_source_manifest(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise StageW2ReleaseError("Source manifest must contain a mapping")
    if payload.get("stage") != "W.2":
        raise StageW2ReleaseError("Source manifest stage must be W.2")
    return dict(payload)


def _matching_files(
    repository_root: Path,
    available_files: set[str],
    group: Mapping[str, Any],
) -> set[str]:
    matches: set[str] = set()
    for raw in group.get("paths", []):
        relative = _safe_relative(str(raw))
        path = repository_root / relative
        if path.is_file():
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
        if path.is_file()
    }


def resolve_source_manifest(
    repository_root: Path,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    manifest_path = (
        manifest_path.resolve()
        if manifest_path is not None
        else repository_root / SOURCE_MANIFEST_RELATIVE
    )
    payload = load_source_manifest(manifest_path)
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

    content_excluded: dict[str, set[str]] = {}
    for group in payload.get("content_exclude", []):
        category = str(group["category"])
        suffixes = {str(value).lower() for value in group["text_suffixes"]}
        prefixes = tuple(
            _safe_relative(str(value)).rstrip("/") + "/"
            for value in group.get("path_prefixes", [])
        )
        pattern = re.compile(str(group["regex"]))
        matches: set[str] = set()
        for relative in included.difference(excluded):
            path = repository_root / relative
            if prefixes and not relative.startswith(prefixes):
                continue
            if path.suffix.lower() not in suffixes:
                continue
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if pattern.search(text):
                matches.add(relative)
        content_excluded[category] = matches
        excluded.update(matches)

    candidates = sorted(included.difference(excluded))
    integrity = payload.get("integrity", {})
    forbidden_suffixes = {
        str(value).lower() for value in integrity.get("forbidden_suffixes", [])
    }
    violations: list[str] = []
    for relative in candidates:
        path = repository_root / relative
        if path.suffix.lower() in forbidden_suffixes:
            violations.append(f"forbidden_suffix:{relative}")
        if any(
            part.lower() in {
                ".git",
                "__pycache__",
                ".pytest_cache",
                "outputs",
                "runs",
                "weights",
            }
            for part in PurePosixPath(relative).parts
        ):
            violations.append(f"forbidden_path:{relative}")

    formal_relative = _safe_relative(
        str(integrity.get("formal_config", FORMAL_CONFIG_RELATIVE))
    )
    formal_path = repository_root / formal_relative
    if formal_relative not in candidates:
        violations.append("formal_config_not_in_candidate")
    elif sha256_file(formal_path) != str(
        integrity.get("formal_config_sha256", "")
    ):
        violations.append("formal_config_sha256")
    elif b"\r\n" in formal_path.read_bytes():
        violations.append("formal_config_crlf")

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
            **{
                key: sorted(value)
                for key, value in excluded_by_category.items()
            },
            **{
                key: sorted(value) for key, value in content_excluded.items()
            },
        },
        "excluded_existing_file_count": len(included.intersection(excluded)),
        "formal_config_sha256": sha256_file(formal_path),
    }


def copy_source_candidate(
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
