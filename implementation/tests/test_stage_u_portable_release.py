from __future__ import annotations

import json
import csv
from pathlib import Path

import yaml

from parking_occupancy.stage_u_portable_release import (
    INTENTIONAL_PRESENTATION_PATHS,
    LOCAL_ONLY_REGISTRY_PATHS,
    PORTABLE_REGISTRY_RELATIVE,
    SELF_REFERENCE_EXCLUDED,
    SUBMISSION_AUDIT_HASH_EXCLUDED_PATHS,
    audit_submission,
)
from parking_occupancy.artifact_registry import verify_frozen_stage_u_snapshot


def _roots() -> tuple[Path, Path]:
    implementation_root = Path(__file__).resolve().parents[1]
    return implementation_root, implementation_root.parent


def test_submission_audit_has_no_forbidden_bulk_artifacts() -> None:
    _, repository_root = _roots()
    audit = audit_submission(repository_root)
    assert audit["status"] == "PASS"
    assert audit["violations"] == []
    assert audit["forbidden_content"] == {
        "model_weights": False,
        "datasets_or_external_data": False,
        "implementation_outputs": False,
        "virtual_environments": False,
        "required_runtime_absolute_path_dependencies": False,
        "git_ignored_files": False,
    }
    candidates = {record["path"] for record in audit["records"]}
    assert not candidates.intersection(LOCAL_ONLY_REGISTRY_PATHS)
    assert set(INTENTIONAL_PRESENTATION_PATHS) <= candidates


def test_historical_registry_classification_is_explicit_when_generated() -> None:
    implementation_root, _ = _roots()
    path = (
        implementation_root
        / "data"
        / "stage_u"
        / "STAGE_U_HISTORICAL_REGISTRY_CLASSIFICATION.json"
    )
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = {row["path"]: row for row in payload["registries"]}
    assert (
        rows[
            "implementation/data/stage_t/"
            "STAGE_T_ARTIFACT_REGISTRY_20260729.yaml"
        ]["registry_class"]
        == "local-only historical registry"
    )
    assert (
        rows[
            "implementation/data/stage_r/"
            "STAGE_R_ARTIFACT_REGISTRY_20260729.yaml"
        ]["registry_class"]
        == "portable historical registry"
    )
    assert (
        rows[
            "implementation/data/stage_s/"
            "STAGE_S_ARTIFACT_REGISTRY_20260729.yaml"
        ]["registry_class"]
        == "portable historical registry"
    )
    assert all(
        row["content_modified_by_stage_u"] is False
        for row in payload["registries"]
    )


def test_historical_registry_audit_excludes_current_stage_u_registry() -> None:
    implementation_root, _ = _roots()
    path = (
        implementation_root
        / "data"
        / "stage_u"
        / "STAGE_U_HISTORICAL_REGISTRY_CLASSIFICATION.json"
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    paths = {row["path"] for row in payload["registries"]}
    assert PORTABLE_REGISTRY_RELATIVE not in paths
    assert payload["portable_count"] == 2
    assert payload["registry_count"] == 11
    assert payload["local_only_count"] == 9


def test_portable_registry_has_relative_non_output_paths_when_generated() -> None:
    _, repository_root = _roots()
    registry = repository_root / PORTABLE_REGISTRY_RELATIVE
    if not registry.exists():
        return
    payload = yaml.safe_load(registry.read_text(encoding="utf-8"))
    assert payload["registry_type"] == "portable_submission_registry"
    for record in payload["artifacts"]:
        relative = str(record["path"]).replace("\\", "/")
        assert not Path(relative).is_absolute()
        assert "implementation/outputs/" not in relative
        assert "data/external/" not in relative
        assert Path(relative).suffix.lower() not in {".pt", ".pth", ".ckpt"}
    result = verify_frozen_stage_u_snapshot(repository_root)
    assert result["verified"] is True
    assert result["live_post_stage_u_candidates_compared"] is False


def test_saved_audit_csv_and_registry_form_stable_one_way_chain() -> None:
    implementation_root, repository_root = _roots()
    stage_u = implementation_root / "data" / "stage_u"
    audit_json = stage_u / "STAGE_U_SUBMISSION_AUDIT.json"
    audit_csv = stage_u / "STAGE_U_SUBMISSION_CANDIDATES.csv"
    registry = repository_root / PORTABLE_REGISTRY_RELATIVE
    if not all(path.exists() for path in (audit_json, audit_csv, registry)):
        return
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    json_rows = {row["path"]: row for row in audit["records"]}
    with audit_csv.open(encoding="utf-8", newline="") as handle:
        csv_rows = {row["path"]: row for row in csv.DictReader(handle)}
    for relative in SUBMISSION_AUDIT_HASH_EXCLUDED_PATHS:
        assert json_rows[relative]["sha256"] == SELF_REFERENCE_EXCLUDED
        assert csv_rows[relative]["sha256"] == SELF_REFERENCE_EXCLUDED
    result = verify_frozen_stage_u_snapshot(repository_root)
    assert result["verified"] is True
    assert result["frozen_candidate_count"] == 599
    assert result["registry_artifact_count"] == 598


def test_frozen_demo_hashes_are_unchanged() -> None:
    implementation_root, _ = _roots()
    stage_s = (
        implementation_root / "data" / "stage_s" / "demo" / "demo_main.mp4"
    )
    stage_t = (
        implementation_root
        / "data"
        / "stage_t"
        / "demo"
        / "demo_tracktrack_optional.mp4"
    )
    from parking_occupancy.stage_u_portable_release import sha256_file

    assert (
        sha256_file(stage_s)
        == "f4e9e59b5bcef1b51f2e94b8443c5f22a69ca850bfc77f5c9b94a1bf947ac608"
    )
    assert (
        sha256_file(stage_t)
        == "b5dfdeb850acdd0a87072a9c48fda44dd5e13725fb7f0e428cfc6164b4d24c1f"
    )


def test_repository_root_readme_is_final_entry() -> None:
    _, repository_root = _roots()
    text = (repository_root / "README.md").read_text(encoding="utf-8")
    assert "D1 detector -> B1 one-to-one polygon mapping" in text
    assert "python implementation/scripts/run_p3_tt.py" in text
    assert "not deployment-ready" in text
    assert "TrackTrack" in text
