from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any, Sequence

from .stage_u_portable_release import (
    sha256_file,
    submission_candidates,
)


class StageU1ReleaseError(ValueError):
    """Raised when the final submission archive contract is violated."""


def inspect_submission_zip(
    zip_path: Path,
    *,
    expected_candidates: Sequence[str],
) -> dict[str, Any]:
    zip_path = zip_path.resolve()
    expected = sorted(set(expected_candidates))
    with zipfile.ZipFile(zip_path, "r") as archive:
        members = sorted(
            info.filename
            for info in archive.infolist()
            if not info.is_dir()
        )
        unsafe = [
            member
            for member in members
            if Path(member).is_absolute() or ".." in Path(member).parts
        ]
        uncompressed_bytes = sum(
            info.file_size for info in archive.infolist() if not info.is_dir()
        )
        bad_crc = archive.testzip()
    errors: list[str] = []
    if members != expected:
        errors.append("candidate_member_mismatch")
    if unsafe:
        errors.append("unsafe_member_path")
    if bad_crc is not None:
        errors.append(f"crc:{bad_crc}")
    return {
        "verified": not errors,
        "errors": errors,
        "member_count": len(members),
        "uncompressed_bytes": uncompressed_bytes,
        "zip_bytes": zip_path.stat().st_size,
        "zip_sha256": sha256_file(zip_path),
        "unsafe_members": unsafe,
    }


def build_submission_zip(
    *,
    repository_root: Path,
    output_zip: Path,
    candidates: Sequence[str] | None = None,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    output_zip = output_zip.resolve()
    if output_zip.is_relative_to(repository_root):
        raise StageU1ReleaseError(
            "Submission ZIP must be outside the candidate source repository"
        )
    if output_zip.exists():
        raise FileExistsError(f"Refusing to overwrite release ZIP: {output_zip}")
    sha_path = output_zip.parent / "ZIP_SHA256.txt"
    if sha_path.exists():
        raise FileExistsError(f"Refusing to overwrite ZIP digest: {sha_path}")
    selected = sorted(
        set(
            submission_candidates(repository_root)
            if candidates is None
            else candidates
        )
    )
    output_zip.parent.mkdir(parents=True, exist_ok=True)
    candidate_bytes = 0
    with zipfile.ZipFile(
        output_zip,
        "x",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
        allowZip64=True,
    ) as archive:
        for relative in selected:
            source = repository_root / relative
            if not source.is_file():
                raise FileNotFoundError(source)
            archive.write(source, arcname=relative)
            candidate_bytes += source.stat().st_size
    verification = inspect_submission_zip(
        output_zip,
        expected_candidates=selected,
    )
    if not verification["verified"]:
        raise StageU1ReleaseError(f"ZIP verification failed: {verification}")
    sha_path.write_text(
        "\n".join(
            (
                f"file={output_zip.name}",
                f"bytes={output_zip.stat().st_size}",
                f"sha256={verification['zip_sha256']}",
                f"candidate_files={len(selected)}",
                f"candidate_bytes={candidate_bytes}",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    return {
        "schema_version": 1,
        "protocol_id": "STAGE-U.1-FINAL-RELEASE-CORRECTION-20260730-01",
        "status": "SUBMISSION_ZIP_CREATED_AND_VERIFIED",
        "zip_path": str(output_zip),
        "zip_sha256_path": str(sha_path),
        "candidate_files": len(selected),
        "candidate_bytes": candidate_bytes,
        **verification,
    }
