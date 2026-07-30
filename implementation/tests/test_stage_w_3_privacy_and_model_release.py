from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

from parking_occupancy.artifact_registry import (
    STAGE_W_2_PRE_W3_REGISTRY_SHA256,
    sha256_file,
    verify_artifact_registry,
)
from parking_occupancy.stage_w_3_release import (
    FORMAL_CONFIG_RELATIVE,
    FORMAL_CONFIG_SHA256,
    PRIVATE_PERMISSION_RELATIVE,
    PUBLIC_PERMISSION_RELATIVE,
    resolve_public_source_manifest,
)


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent
DATA = IMPLEMENTATION_ROOT / "data"
ASSETS = IMPLEMENTATION_ROOT / "outputs" / "stage_w_3_model_release_assets"
W3_REGISTRY = DATA / "STAGE_W_3_ARTIFACT_REGISTRY.yaml"
EXPECTED_MODELS = {
    "D1_NDISPark_best.pt": (
        6255409,
        "0638d50d909d679eb15622632556f6f92052af8eacffb7bf7f398e93efd0ca64",
    ),
    "E1b_CBAM_best.pt": (
        8045704,
        "f6966dabe0801f221cc6e67b9ee117af1b06c93a7e34c96d25771572616ddbe3",
    ),
}


def test_license_is_complete_agpl_text_with_explicit_lf_rule() -> None:
    license_path = REPOSITORY_ROOT / "LICENSE"
    content = license_path.read_bytes()
    assert content.lstrip().startswith(b"GNU AFFERO GENERAL PUBLIC LICENSE")
    assert b"Version 3, 19 November 2007" in content
    assert b"How to Apply These Terms to Your New Programs" in content
    assert b"\r\n" not in content
    attributes = (REPOSITORY_ROOT / ".gitattributes").read_text(
        encoding="utf-8"
    )
    assert "LICENSE text eol=lf" in attributes.splitlines()
    assert (
        hashlib.sha256(content).hexdigest()
        == "0d96a4ff68ad6d4b6f1f30f713b18d5184912ba8dd389f86aa7710db079abcb0"
    )


def test_public_permission_record_is_anonymous_and_bounded() -> None:
    text = (REPOSITORY_ROOT / PUBLIC_PERMISSION_RELATIVE).read_text(
        encoding="utf-8"
    )
    required = {
        "authorization_grantor_role: upstream_code_owner",
        "authorization_status: received_and_privately_retained",
        (
            "authorized_scope: course_integration_and_"
            "public_redistribution_of_adapted_interface"
        ),
        "attestation_recorded_by: project_owner",
        "evidence_storage: retained_privately_outside_repository",
        "evidence_available_to_instructor_if_required: true",
        (
            "adapted_upstream_repository: "
            "https://github.com/prestzy/OpenCV-Car-Parking"
        ),
        "audited_commit: 12271576be39a4ac0eb456526eca122685799e8c",
    }
    assert required.issubset(set(text.splitlines()))
    assert "@" not in text
    assert "C:" + "\\" + "Users" not in text
    assert PRIVATE_PERMISSION_RELATIVE not in resolve_public_source_manifest(
        REPOSITORY_ROOT
    )["candidate_files"]


def test_python_packages_declare_project_license() -> None:
    for path in (
        IMPLEMENTATION_ROOT / "pyproject.toml",
        IMPLEMENTATION_ROOT / "literature_core" / "pyproject.toml",
    ):
        text = path.read_text(encoding="utf-8")
        project_section = text.split("[project]", 1)[1].split("\n[", 1)[0]
        assert 'license = "AGPL-3.0-only"' in project_section.splitlines()


def test_third_party_notices_cover_dependencies_and_datasets() -> None:
    text = (REPOSITORY_ROOT / "THIRD_PARTY_NOTICES.md").read_text(
        encoding="utf-8"
    )
    for value in (
        "Ultralytics",
        "torchvision",
        "OpenCV",
        "Flask",
        "NDISPark",
        "ODC-By-1.0",
        "PKLot",
        "CC-BY-4.0",
        "not a commercial or",
        "deployment-grade product",
    ):
        assert value in text


def test_frozen_model_release_assets_are_exact_and_git_ignored() -> None:
    for filename, (expected_bytes, expected_sha256) in EXPECTED_MODELS.items():
        path = ASSETS / filename
        assert path.stat().st_size == expected_bytes
        assert sha256_file(path) == expected_sha256
        ignored = subprocess.run(
            ["git", "check-ignore", "-q", "--", str(path)],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        assert ignored.returncode == 0


def test_model_release_manifest_has_no_fabricated_url_or_weights_in_source() -> None:
    payload = yaml.safe_load(
        (DATA / "STAGE_W_3_MODEL_RELEASE_MANIFEST.yaml").read_text(
            encoding="utf-8"
        )
    )
    assert payload["release_url"] == "pending"
    assert payload["model_assets_ready_for_github_release"] is True
    assert payload["public_release_published"] is False
    assert payload["model_training_run"] is False
    assert payload["model_inference_run"] is False
    assert payload["source_publication_contains_weights"] is False
    assert {
        item["release_filename"] for item in payload["assets"]
    } == set(EXPECTED_MODELS)
    assert payload["assets"][1]["frozen_experiment_id"].startswith(
        "not_recorded"
    )


def test_public_source_manifest_is_private_data_free_and_model_free() -> None:
    result = resolve_public_source_manifest(REPOSITORY_ROOT)
    manifest_text = (
        DATA / "STAGE_W_3_PUBLIC_SOURCE_MANIFEST.yaml"
    ).read_text(encoding="utf-8")
    assert result["verified"] is True, result["violations"]
    assert result["source_publication_ready"] is True
    assert result["privacy_scan"]["finding_count"] == 0
    assert result["formal_config_sha256"] == FORMAL_CONFIG_SHA256
    assert FORMAL_CONFIG_RELATIVE in result["candidate_files"]
    assert PUBLIC_PERMISSION_RELATIVE in result["candidate_files"]
    assert PRIVATE_PERMISSION_RELATIVE not in result["candidate_files"]
    assert (
        "implementation/tests/"
        "test_stage_w_3_privacy_and_model_release.py"
    ) in result["candidate_files"]
    assert not any(
        Path(relative).suffix.lower()
        in {".pt", ".pth", ".ckpt", ".onnx", ".engine"}
        for relative in result["candidate_files"]
    )
    assert "'**/build/**'" in manifest_text
    assert "'**/dist/**'" in manifest_text


def test_w2_registry_remains_exact_historical_identity() -> None:
    path = DATA / "STAGE_W_2_ARTIFACT_REGISTRY.yaml"
    assert sha256_file(path) == STAGE_W_2_PRE_W3_REGISTRY_SHA256


def test_w3_registry_covers_release_boundary_and_verifies() -> None:
    payload = yaml.safe_load(W3_REGISTRY.read_text(encoding="utf-8"))
    registered = {str(item["path"]) for item in payload["artifacts"]}
    for relative in (
        ".gitattributes",
        "LICENSE",
        "THIRD_PARTY_NOTICES.md",
        PUBLIC_PERMISSION_RELATIVE,
        "implementation/data/MODEL_CARD_D1.md",
        "implementation/data/MODEL_CARD_E1B.md",
        "implementation/data/STAGE_W_3_PUBLIC_SOURCE_MANIFEST.yaml",
        "implementation/tests/test_stage_w_3_privacy_and_model_release.py",
    ):
        assert relative in registered
    result = verify_artifact_registry(
        W3_REGISTRY,
        artifact_root=REPOSITORY_ROOT,
    )
    assert result["verified"] is True, result["errors"]
    assert result["optional_unavailable"] == 0
