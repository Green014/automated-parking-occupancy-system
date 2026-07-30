from __future__ import annotations

from pathlib import Path
import hashlib

import yaml

from parking_occupancy.stage_k_stratified_analysis import (
    PROTOCOL_ID,
    RECORD_ID,
    load_protocol,
    verify_record,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stage_k_strata_protocol_is_read_only() -> None:
    protocol = load_protocol(
        ROOT
        / "configs"
        / "stage_k_posthoc_stratified_analysis_frozen_20260728.yaml"
    )
    assert protocol["protocol_id"] == PROTOCOL_ID
    assert protocol["scope"]["prediction_allowed"] is False
    assert protocol["scope"]["parameter_selection_allowed"] is False
    assert protocol["metrics"]["strata"] == ["date", "weather"]


def test_stage_k_strata_verifier_checks_registered_hashes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    external = tmp_path / "external"
    source.mkdir()
    external.mkdir()
    artifact = external / "analysis.json"
    artifact.write_text("{}\n", encoding="utf-8")
    record = {
        "record_id": RECORD_ID,
        "artifacts": [
            {
                "role": "analysis",
                "root": "external",
                "path": "analysis.json",
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
            }
        ],
    }
    record_path = tmp_path / "record.yaml"
    record_path.write_text(
        yaml.safe_dump(record),
        encoding="utf-8",
    )

    result = verify_record(
        record_path=record_path,
        source_root=source,
        external_root=external,
    )

    assert result["passed"] is True
    assert result["artifact_count"] == 1
