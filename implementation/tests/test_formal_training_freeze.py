from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_formal_training_record_matches_frozen_registry() -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load(
        (
            project_root
            / "data"
            / "training"
            / "D1_FORMAL_TRAINING_FROZEN_CHECKSUMS.yaml"
        ).read_text(encoding="utf-8")
    )

    assert registry["registry_id"] == (
        "D1-NDISPARK-FORMAL-FREEZE-20260727-01"
    )
    for artifact in registry["artifacts"]:
        path = project_root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]


def test_formal_record_retains_failure_and_blocks_test() -> None:
    project_root = Path(__file__).resolve().parents[1]
    record = yaml.safe_load(
        (
            project_root
            / "data"
            / "training"
            / "d1_formal_training_20260727.yaml"
        ).read_text(encoding="utf-8")
    )

    assert record["training"]["epochs_completed"] == 47
    assert record["training"]["best_epoch_one_based"] == 37
    assert record["retained_engineering_failure"]["rerun_performed"] is False
    assert record["verification"]["all_size_and_sha256_checks_passed"]
    assert record["gate"]["stage_I_development_detector_comparison_may_start"]
    assert record["gate"]["count_test_allowed_now"] is False
