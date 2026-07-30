from __future__ import annotations

import json
from pathlib import Path

from parking_occupancy.artifact_registry import (
    STAGE_S_FROZEN_REGISTRY_SHA256,
    verify_historical_artifact_registry,
)
from parking_occupancy.integrated_cli import (
    DEFAULT_FINAL_INTEGRATED_CONFIG,
    build_parser,
)
from parking_occupancy.stage_s_release import (
    REQUIRED_GITIGNORE_PATTERNS,
    compare_registry_snapshots,
    historical_registry_snapshot,
    load_stage_s_config,
    submission_candidate_audit,
)


def _required_cli_args(tmp_path: Path) -> list[str]:
    return [
        "--input",
        str(tmp_path / "input.mp4"),
        "--slots",
        str(tmp_path / "slots.json"),
        "--d1-weights",
        str(tmp_path / "d1.pt"),
        "--e1b-checkpoint",
        str(tmp_path / "e1b.pt"),
        "--output-dir",
        str(tmp_path / "output"),
    ]


def test_final_cli_defaults_to_stage_r_recommendation(tmp_path: Path) -> None:
    args = build_parser().parse_args(_required_cli_args(tmp_path))
    assert args.config.resolve() == DEFAULT_FINAL_INTEGRATED_CONFIG.resolve()
    config = load_stage_s_config(args.config)
    assert config["detector"]["id"] == "D1"
    assert config["mapping"]["id"] == "B1"
    assert config["fusion"]["id"] == "F2"
    assert config["temporal"]["default_enabled"] is False
    assert config["tracking"]["default_backend"] == "none"
    assert args.temporal is None
    assert args.tracker is None


def test_final_cli_allows_explicit_temporal_and_tracker_opt_in(
    tmp_path: Path,
) -> None:
    args = build_parser().parse_args(
        [
            *_required_cli_args(tmp_path),
            "--temporal",
            "--tracker",
            "tracktrack",
        ]
    )
    assert args.temporal is True
    assert args.tracker == "tracktrack"

    disabled = build_parser().parse_args(
        [*_required_cli_args(tmp_path), "--no-temporal", "--tracker", "none"]
    )
    assert disabled.temporal is False
    assert disabled.tracker == "none"


def test_submission_candidate_gate_excludes_local_bulk_artifacts() -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    repository_root = implementation_root.parent
    gitignore = (repository_root / ".gitignore").read_text(encoding="utf-8")
    assert all(pattern in gitignore for pattern in REQUIRED_GITIGNORE_PATTERNS)

    audit = submission_candidate_audit(repository_root)
    assert audit["status"] == "PASS"
    assert audit["violations"] == []
    assert audit["forbidden_content"] == {
        "model_weights": False,
        "datasets": False,
        "virtual_environments": False,
        "outputs": False,
    }


def test_historical_registry_entry_state_is_unchanged() -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    before = json.loads(
        (
            implementation_root
            / "data"
            / "stage_s"
            / "STAGE_S_HISTORICAL_REGISTRY_PRE_SNAPSHOT.json"
        ).read_text(encoding="utf-8")
    )
    current = historical_registry_snapshot(implementation_root)
    gate = compare_registry_snapshots(before, current)
    assert gate["unchanged"] is True


def test_stage_s_registry_when_frozen() -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    registry = (
        implementation_root
        / "data"
        / "stage_s"
        / "STAGE_S_ARTIFACT_REGISTRY_20260729.yaml"
    )
    if not registry.exists():
        return
    result = verify_historical_artifact_registry(
        registry,
        artifact_root=implementation_root.parent,
        expected_registry_sha256=STAGE_S_FROZEN_REGISTRY_SHA256,
        immutable_path_prefixes=("implementation/data/stage_s",),
    )
    assert result["verified"] is True
