from __future__ import annotations

import hashlib
import subprocess
from pathlib import Path

import yaml

from parking_occupancy.artifact_registry import (
    STAGE_W_1_PRE_W2_REGISTRY_SHA256,
    STAGE_W_2_PRE_W3_REGISTRY_SHA256,
    sha256_file,
)


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent
GIT_ATTRIBUTES = REPOSITORY_ROOT / ".gitattributes"
FORMAL_CONFIG_RELATIVE = (
    "implementation/configs/"
    "p3_stage_r_recommended_default_20260729.yaml"
)
FORMAL_CONFIG = REPOSITORY_ROOT / FORMAL_CONFIG_RELATIVE
FORMAL_CONFIG_SHA256 = (
    "198f627689cd93f66ca0f087af6686d3afc697ff51e2aa77ee56124187b981b0"
)
W2_REGISTRY = (
    IMPLEMENTATION_ROOT / "data" / "STAGE_W_2_ARTIFACT_REGISTRY.yaml"
)


def _attribute_lines() -> set[str]:
    return {
        line.strip()
        for line in GIT_ATTRIBUTES.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    }


def test_repository_gitattributes_exists() -> None:
    assert GIT_ATTRIBUTES.is_file()


def test_release_text_extensions_are_explicitly_lf() -> None:
    rules = _attribute_lines()
    for extension in (
        "py",
        "toml",
        "yaml",
        "yml",
        "json",
        "jsonl",
        "csv",
        "md",
        "txt",
        "html",
        "css",
        "ini",
    ):
        assert any(
            line.startswith(f"*.{extension} text eol=lf")
            for line in rules
        )


def test_models_media_images_and_archives_are_binary() -> None:
    rules = _attribute_lines()
    for extension in (
        "pt",
        "pth",
        "ckpt",
        "onnx",
        "engine",
        "mp4",
        "avi",
        "png",
        "jpg",
        "jpeg",
        "pdf",
        "zip",
        "tar",
        "gz",
    ):
        assert f"*.{extension} binary" in rules


def test_frozen_formal_config_is_lf_and_has_exact_identity() -> None:
    content = FORMAL_CONFIG.read_bytes()
    assert b"\r\n" not in content
    assert hashlib.sha256(content).hexdigest() == FORMAL_CONFIG_SHA256


def test_git_attributes_resolve_formal_config_to_text_lf() -> None:
    result = subprocess.run(
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            FORMAL_CONFIG_RELATIVE,
        ],
        cwd=REPOSITORY_ROOT,
        capture_output=True,
        text=True,
        check=True,
    )
    output = result.stdout.replace("\\", "/")
    assert f"{FORMAL_CONFIG_RELATIVE}: text: set" in output
    assert f"{FORMAL_CONFIG_RELATIVE}: eol: lf" in output


def test_w2_registry_contains_cross_platform_boundary_files() -> None:
    payload = yaml.safe_load(W2_REGISTRY.read_text(encoding="utf-8"))
    registered = {str(item["path"]) for item in payload["artifacts"]}
    assert ".gitattributes" in registered
    assert (
        "implementation/tests/"
        "test_stage_w_2_cross_platform_release.py"
    ) in registered


def test_w1_registry_remains_the_exact_pre_w2_snapshot() -> None:
    historical = (
        IMPLEMENTATION_ROOT / "data" / "STAGE_W_1_ARTIFACT_REGISTRY.yaml"
    )
    assert sha256_file(historical) == STAGE_W_1_PRE_W2_REGISTRY_SHA256


def test_w2_manifest_and_registry_remain_exact_pre_w3_snapshots() -> None:
    manifest = yaml.safe_load(
        (
            IMPLEMENTATION_ROOT
            / "data"
            / "STAGE_W_2_SOURCE_COMMIT_MANIFEST.yaml"
        ).read_text(encoding="utf-8")
    )
    assert manifest["stage"] == "W.2"
    assert manifest["resolved_summary"]["candidate_file_count"] == 542
    assert manifest["resolved_summary"]["include_category_count"] == 8
    assert manifest["resolved_summary"]["exclude_category_count"] == 8
    assert sha256_file(W2_REGISTRY) == STAGE_W_2_PRE_W3_REGISTRY_SHA256
