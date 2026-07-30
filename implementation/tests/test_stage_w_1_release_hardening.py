from __future__ import annotations

import tomllib
import subprocess
import sys
from pathlib import Path

from parking_occupancy.artifact_registry import (
    STAGE_V_1_PRE_HARDENING_REGISTRY_SHA256,
    STAGE_W_1_PRE_W2_REGISTRY_SHA256,
    STAGE_W_PRE_HARDENING_REGISTRY_SHA256,
    verify_historical_artifact_registry,
)
from parking_occupancy.integrated_cli import build_parser as final_parser
from parking_occupancy.stage_v_runner import build_parser as compare_parser
from parking_occupancy.stage_w_cli import build_parser as dashboard_parser


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent


def test_dashboard_dependency_and_console_entries_are_declared() -> None:
    pyproject = tomllib.loads(
        (IMPLEMENTATION_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert pyproject["project"]["optional-dependencies"]["dashboard"] == [
        "Flask>=3.1,<4"
    ]
    requirements = [
        line.strip()
        for line in (
            IMPLEMENTATION_ROOT / "stage_w_requirements.txt"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert requirements == ["Flask>=3.1,<4"]
    scripts = pyproject["project"]["scripts"]
    assert scripts["parking-run-final"] == (
        "parking_occupancy.integrated_cli:main"
    )
    assert scripts["parking-compare"] == (
        "parking_occupancy.stage_v_runner:main"
    )
    assert scripts["parking-dashboard"] == (
        "parking_occupancy.stage_w_cli:main"
    )


def test_stage_w_server_import_is_collection_safe_without_flask() -> None:
    script = """
import builtins
real_import = builtins.__import__
def blocked_import(name, *args, **kwargs):
    if name == "flask" or name.startswith("flask."):
        raise ModuleNotFoundError("blocked optional Flask", name="flask")
    return real_import(name, *args, **kwargs)
builtins.__import__ = blocked_import
import parking_occupancy.stage_w_server as server
assert server.Flask is None
try:
    server.create_stage_w_app(object())
except RuntimeError as exc:
    assert "dashboard" in str(exc)
else:
    raise AssertionError("Missing Flask must produce an explicit error")
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=IMPLEMENTATION_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_primary_cli_help_contracts_use_output_dir() -> None:
    for parser in (final_parser(), compare_parser(), dashboard_parser()):
        help_text = parser.format_help()
        assert "--output-dir" in help_text
        assert "--output-root" not in help_text
    assert "--mode" in compare_parser().format_help()
    assert "--host" in dashboard_parser().format_help()
    assert final_parser().prog == "parking-run-final"


def test_readmes_distinguish_runtime_comparison_dashboard_and_tracking() -> None:
    root_readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    implementation_readme = (
        IMPLEMENTATION_ROOT / "README.md"
    ).read_text(encoding="utf-8")
    for text in (root_readme, implementation_readme):
        assert "parking-run-final" in text
        assert "parking-compare" in text
        assert "parking-dashboard" in text
        assert "run_p3_tt.py" in text
        assert ".[integrated,dashboard,dev]" in text
    stage_v_section = implementation_readme.split(
        "## 13. Stage V.1", 1
    )[1].split("## 14.", 1)[0]
    assert "--output-dir" in stage_v_section
    assert "--output-root" not in stage_v_section


def test_w1_unlicensed_gate_is_preserved_as_historical_context() -> None:
    historical = (
        IMPLEMENTATION_ROOT
        / "data"
        / "STAGE_W_1_RELEASE_HARDENING_REPORT.md"
    ).read_text(encoding="utf-8")
    assert "No top-level project licence exists" in historical
    assert "`public_release_ready=false`" in historical
    current_index = (REPOSITORY_ROOT / "FINAL_RELEASE_INDEX.md").read_text(
        encoding="utf-8"
    )
    assert (REPOSITORY_ROOT / "LICENSE").is_file()
    assert "AGPL-3.0-only" in current_index
    assert "`public_release_published=false`" in current_index


def test_pre_hardening_v1_and_w_registries_are_historical_snapshots() -> None:
    for name, expected in (
        (
            "STAGE_V_1_ARTIFACT_REGISTRY.yaml",
            STAGE_V_1_PRE_HARDENING_REGISTRY_SHA256,
        ),
        (
            "STAGE_W_ARTIFACT_REGISTRY.yaml",
            STAGE_W_PRE_HARDENING_REGISTRY_SHA256,
        ),
    ):
        result = verify_historical_artifact_registry(
            IMPLEMENTATION_ROOT / "data" / name,
            artifact_root=IMPLEMENTATION_ROOT,
            expected_registry_sha256=expected,
            immutable_path_prefixes=("outputs",),
        )
        assert result["verified"] is True, result["errors"]
        assert result["classification"] == (
            "pre_hardening_historical_snapshot"
        )
        assert result["live_release_artifacts_compared"] is False


def test_stage_w_1_is_a_pre_w2_historical_source_snapshot() -> None:
    result = verify_historical_artifact_registry(
        IMPLEMENTATION_ROOT
        / "data"
        / "STAGE_W_1_ARTIFACT_REGISTRY.yaml",
        artifact_root=REPOSITORY_ROOT,
        expected_registry_sha256=STAGE_W_1_PRE_W2_REGISTRY_SHA256,
        immutable_path_prefixes=("implementation/outputs",),
        classification="pre_w2_historical_source_snapshot",
    )
    assert result["verified"] is True, result["errors"]
    assert result["classification"] == (
        "pre_w2_historical_source_snapshot"
    )
    assert result["live_release_artifacts_compared"] is False
