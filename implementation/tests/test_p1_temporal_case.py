from __future__ import annotations

from pathlib import Path

import pytest

from parking_occupancy.p1_temporal_case import (
    P1_TEMPORAL_RECORD_ID,
    P1_TEMPORAL_PROTOCOL_ID,
    _truth_state,
    load_p1_temporal_protocol,
    verify_p1_temporal_record,
)


ROOT = Path(__file__).resolve().parents[1]
CONFIG = (
    ROOT
    / "configs"
    / "p1_b1_virat_0502_continuous_case_frozen_20260727.yaml"
)


def test_frozen_p1_temporal_protocol_binds_truth_before_prediction() -> None:
    protocol = load_p1_temporal_protocol(CONFIG)
    assert protocol["protocol_id"] == P1_TEMPORAL_PROTOCOL_ID
    assert protocol["scope"]["untouched_claim"] == "prohibited"
    assert protocol["tracking_branch"]["enabled"] is False
    assert protocol["truth"]["transition"]["first_vacant_frame"] == 1660


def test_temporal_truth_uses_half_open_transition_boundary() -> None:
    protocol = load_p1_temporal_protocol(CONFIG)
    assert _truth_state(protocol, 0) == 1
    assert _truth_state(protocol, 1659) == 1
    assert _truth_state(protocol, 1660) == 0
    assert _truth_state(protocol, 1973) == 0
    with pytest.raises(ValueError, match="does not cover"):
        _truth_state(protocol, 1974)


def test_p1_temporal_record_verifier_hashes_artifacts(
    tmp_path: Path,
) -> None:
    import hashlib
    import yaml

    source = tmp_path / "source"
    external = tmp_path / "external"
    source.mkdir()
    external.mkdir()
    artifact = external / "result.txt"
    artifact.write_text("missed\n", encoding="utf-8")
    record_path = tmp_path / "record.yaml"
    record_path.write_text(
        yaml.safe_dump(
            {
                "record_id": P1_TEMPORAL_RECORD_ID,
                "artifacts": [
                    {
                        "role": "result",
                        "root": "external",
                        "path": "result.txt",
                        "bytes": artifact.stat().st_size,
                        "sha256": hashlib.sha256(
                            artifact.read_bytes()
                        ).hexdigest(),
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    result = verify_p1_temporal_record(
        record_path=record_path,
        source_root=source,
        external_root=external,
    )

    assert result["passed"]
    assert result["artifact_count"] == 1
