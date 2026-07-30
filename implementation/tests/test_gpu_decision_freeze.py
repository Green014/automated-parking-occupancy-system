from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_gpu_decision_and_formal_config_match_frozen_registry() -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load(
        (
            project_root
            / "data"
            / "training"
            / "D1_GPU_DECISION_FROZEN_CHECKSUMS.yaml"
        ).read_text(encoding="utf-8")
    )

    assert registry["registry_id"] == (
        "GPU-GATE-NDISPARK-D1-FREEZE-20260727-01"
    )
    for artifact in registry["artifacts"]:
        path = project_root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]


def test_formal_config_is_local_fresh_and_does_not_embed_user_path() -> None:
    project_root = Path(__file__).resolve().parents[1]
    path = (
        project_root
        / "configs"
        / "d1_ndispark_formal_frozen_20260727.yaml"
    )
    raw = path.read_text(encoding="utf-8")
    config = yaml.safe_load(raw)

    assert config["status"] == "frozen_not_executed"
    assert config["model"]["smoke_checkpoint_initialization"] == "prohibited"
    assert config["training"]["imgsz"] == 640
    assert config["training"]["physical_batch"] == 4
    assert config["training"]["post_warmup_accumulation_steps"] == 16
    assert config["resource_gate"]["local_training_selected"]
    assert not config["resource_gate"]["paid_or_remote_gpu_allowed"]
    assert config["execution_gate"]["stage_H_local_execution_allowed"]
    assert "C:\\Users\\" not in raw
