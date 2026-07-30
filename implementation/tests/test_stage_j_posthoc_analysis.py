from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import pytest
import yaml

from parking_occupancy.stage_j_posthoc_analysis import (
    STAGE_J_POSTHOC_RECORD_ID,
    equal_group_macro,
    grouped_metrics,
    load_occupancy_rows,
    load_stage_j_posthoc_protocol,
    paired_bootstrap_mean_difference,
    verify_stage_j_posthoc_record,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROTOCOL_PATH = (
    PROJECT_ROOT
    / "configs"
    / "stage_j_posthoc_layered_analysis_frozen_20260727.yaml"
)


def _write_occupancy(path: Path, rows: list[dict[str, object]]) -> None:
    fields = [
        "sample_id",
        "slot_id",
        "camera",
        "date",
        "weather",
        "truth",
        "state",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def test_unknown_truth_is_excluded_without_dropping_output_row(
    tmp_path: Path,
) -> None:
    path = tmp_path / "occupancy.csv"
    _write_occupancy(
        path,
        [
            {
                "sample_id": "s1",
                "slot_id": "a",
                "camera": "c1",
                "date": "d1",
                "weather": "sunny",
                "truth": 1,
                "state": 1,
            },
            {
                "sample_id": "s1",
                "slot_id": "b",
                "camera": "c1",
                "date": "d1",
                "weather": "sunny",
                "truth": "",
                "state": 0,
            },
        ],
    )

    all_rows, known_rows, unknown_count = load_occupancy_rows(path)

    assert len(all_rows) == 2
    assert len(known_rows) == 1
    assert unknown_count == 1
    assert all_rows[1]["truth"] is None


def test_camera_metrics_and_camera_macro_weight_cameras_equally() -> None:
    rows = [
        {
            "sample_id": "a",
            "slot_id": "1",
            "camera": "large",
            "truth": 1,
            "prediction": 1,
        },
        {
            "sample_id": "a",
            "slot_id": "2",
            "camera": "large",
            "truth": 0,
            "prediction": 0,
        },
        {
            "sample_id": "a",
            "slot_id": "3",
            "camera": "large",
            "truth": 1,
            "prediction": 1,
        },
        {
            "sample_id": "b",
            "slot_id": "1",
            "camera": "small",
            "truth": 1,
            "prediction": 0,
        },
        {
            "sample_id": "b",
            "slot_id": "2",
            "camera": "small",
            "truth": 0,
            "prediction": 1,
        },
    ]

    by_camera = grouped_metrics(rows, "camera")
    camera_macro = equal_group_macro(by_camera)

    assert by_camera["large"]["macro_f1"] == pytest.approx(1.0)
    assert by_camera["small"]["macro_f1"] == pytest.approx(0.0)
    assert camera_macro["groups"] == 2
    assert camera_macro["macro_f1"] == pytest.approx(0.5)


def test_paired_bootstrap_is_deterministic_and_uses_sample_groups() -> None:
    differences = {
        "image_a": 0.20,
        "image_b": -0.10,
        "image_c": 0.00,
    }

    first = paired_bootstrap_mean_difference(
        differences,
        seed=20260727,
        resamples=2000,
        confidence_level=0.95,
    )
    second = paired_bootstrap_mean_difference(
        differences,
        seed=20260727,
        resamples=2000,
        confidence_level=0.95,
    )

    assert first == second
    assert first["unit"] == "sample_id"
    assert first["sample_count"] == 3
    assert first["estimate"] == pytest.approx(1.0 / 30.0)
    assert first["contains_zero"] is True


def test_committed_posthoc_protocol_freezes_read_only_grouped_analysis() -> None:
    protocol = load_stage_j_posthoc_protocol(PROTOCOL_PATH)

    assert protocol["scope"]["prediction_allowed"] is False
    assert protocol["scope"]["parameter_selection_allowed"] is False
    assert protocol["bootstrap"]["unit"] == "sample_id"
    assert protocol["bootstrap"]["slot_level_resampling"] == "prohibited"
    assert protocol["bootstrap"]["seed"] == 20260727
    assert protocol["bootstrap"]["resamples"] == 20000


def test_posthoc_artifact_verifier_uses_independent_record(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    external = tmp_path / "external"
    source.mkdir()
    external.mkdir()
    artifact = external / "analysis.json"
    artifact.write_bytes(b"{}\n")

    record = tmp_path / "record.yaml"
    record.write_text(
        yaml.safe_dump(
            {
                "record_id": STAGE_J_POSTHOC_RECORD_ID,
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
        ),
        encoding="utf-8",
    )

    report = verify_stage_j_posthoc_record(
        record_path=record,
        source_root=source,
        external_root=external,
    )

    assert report["artifact_count"] == 1
    assert report["passed"] is True
