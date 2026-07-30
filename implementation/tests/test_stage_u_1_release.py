from __future__ import annotations

from pathlib import Path

import pytest

from parking_occupancy.stage_u_1_release import (
    StageU1ReleaseError,
    build_submission_zip,
    inspect_submission_zip,
)


def test_release_zip_is_external_complete_and_hashed(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "README.md").write_text("portable\n", encoding="utf-8")
    nested = repository / "implementation" / "src"
    nested.mkdir(parents=True)
    (nested / "module.py").write_text("VALUE = 1\n", encoding="utf-8")
    candidates = ["README.md", "implementation/src/module.py"]
    output = tmp_path / "release" / "submission.zip"
    result = build_submission_zip(
        repository_root=repository,
        output_zip=output,
        candidates=candidates,
    )
    assert result["verified"] is True
    assert result["candidate_files"] == 2
    assert result["member_count"] == 2
    assert (output.parent / "ZIP_SHA256.txt").is_file()
    assert inspect_submission_zip(
        output,
        expected_candidates=candidates,
    )["verified"] is True


def test_release_zip_refuses_output_inside_candidate_root(tmp_path: Path) -> None:
    repository = tmp_path / "repository"
    repository.mkdir()
    (repository / "README.md").write_text("portable\n", encoding="utf-8")
    with pytest.raises(StageU1ReleaseError, match="outside"):
        build_submission_zip(
            repository_root=repository,
            output_zip=repository / "submission.zip",
            candidates=["README.md"],
        )
