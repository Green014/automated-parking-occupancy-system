from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import yaml

from parking_occupancy.stage_j_occupancy import (
    PIPELINE_METHOD_IDS,
    STAGE_J_RECORD_ID,
    StageJProtocolError,
    _subset_metrics,
    load_stage_j_protocol,
    verify_stage_j_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGE_J_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_j_p0_p1_p2_pklot_development_frozen_20260727.yaml"
)


def test_frozen_stage_j_protocol_keeps_detector_and_mapping_isolation() -> None:
    protocol = load_stage_j_protocol(STAGE_J_CONFIG)

    assert tuple(protocol["methods"]) == PIPELINE_METHOD_IDS
    assert [
        protocol["methods"][method_id]["detector_id"]
        for method_id in PIPELINE_METHOD_IDS
    ] == ["D0", "D1", "D2"]
    assert [
        protocol["methods"][method_id]["confidence"]
        for method_id in PIPELINE_METHOD_IDS
    ] == [0.10, 0.30, 0.10]
    assert protocol["common_inference"]["agnostic_nms"] is True
    assert protocol["common_inference"]["max_detections"] == 300
    assert protocol["common_mapping"] == {
        "algorithm": "slot_polygon_coverage",
        "mapping_source": "B1",
        "minimum_slot_coverage": 0.40,
        "one_to_one": True,
        "evidence": "detector_confidence_times_slot_coverage",
        "temporal_stabilization": False,
        "test_polygon_edits": "prohibited",
    }
    assert protocol["gates"]["stage_K_untouched_test_available"] is False


def test_stage_j_metrics_report_both_slot_classes_and_macro_f1() -> None:
    rows = [
        {"truth": 1, "prediction": 1, "evidence": 0.9},
        {"truth": 1, "prediction": 0, "evidence": 0.4},
        {"truth": 0, "prediction": 0, "evidence": 0.1},
        {"truth": 0, "prediction": 1, "evidence": 0.8},
    ]

    metrics = _subset_metrics(rows)

    assert metrics["occupied_recall"] == pytest.approx(0.5)
    assert metrics["vacant_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(0.5)
    assert metrics["confusion_matrix"] == {
        "vacant": {"predicted_vacant": 1, "predicted_occupied": 1},
        "occupied": {"predicted_vacant": 1, "predicted_occupied": 1},
    }


def test_stage_j_protocol_rejects_mapping_change(tmp_path: Path) -> None:
    payload = yaml.safe_load(STAGE_J_CONFIG.read_text(encoding="utf-8"))
    payload["common_mapping"]["minimum_slot_coverage"] = 0.41
    payload["data"]["annotations"]["path"] = str(
        (
            PROJECT_ROOT
            / "data"
            / "annotations"
            / "pklot_development_samples.jsonl"
        ).resolve()
    )
    payload["data"]["membership_manifest"]["path"] = str(
        (
            PROJECT_ROOT
            / "data"
            / "manifests"
            / "stage_j_pklot_development_20260727.csv"
        ).resolve()
    )
    changed = tmp_path / "changed.yaml"
    changed.write_text(yaml.safe_dump(payload), encoding="utf-8")

    with pytest.raises(StageJProtocolError, match="mapping"):
        load_stage_j_protocol(changed)


def test_stage_j_artifact_verifier_checks_size_and_hash(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    external = tmp_path / "external"
    source.mkdir()
    external.mkdir()
    artifact = source / "frozen.txt"
    artifact.write_text("frozen\n", encoding="utf-8")
    record = tmp_path / "record.yaml"
    record.write_text(
        yaml.safe_dump(
            {
                "record_id": STAGE_J_RECORD_ID,
                "artifacts": [
                    {
                        "role": "fixture",
                        "root": "source",
                        "path": "frozen.txt",
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

    report = verify_stage_j_record(
        record_path=record,
        source_root=source,
        external_root=external,
    )

    assert report["artifact_count"] == 1
    assert report["passed"] is True
