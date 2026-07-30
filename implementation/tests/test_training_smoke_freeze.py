from __future__ import annotations

import hashlib
from pathlib import Path

import yaml


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_d1_smoke_frozen_record_matches_registry() -> None:
    project_root = Path(__file__).resolve().parents[1]
    registry = yaml.safe_load(
        (
            project_root
            / "data"
            / "training"
            / "D1_SMOKE_FROZEN_CHECKSUMS.yaml"
        ).read_text(encoding="utf-8")
    )

    assert registry["registry_id"] == (
        "D1-NDISPARK-SMOKE-FREEZE-20260727-01"
    )
    for artifact in registry["artifacts"]:
        path = project_root / artifact["path"]
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]
