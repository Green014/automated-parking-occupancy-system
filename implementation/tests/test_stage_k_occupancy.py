from __future__ import annotations

from pathlib import Path

from parking_occupancy.stage_k_occupancy import (
    STAGE_K_RECORD_ID,
    STAGE_K_PROTOCOL_ID,
    _paired_comparison,
    load_stage_k_protocol,
    verify_stage_k_record,
)


ROOT = Path(__file__).resolve().parents[1]


def test_frozen_stage_k_protocol_passes_source_validation() -> None:
    protocol = load_stage_k_protocol(
        ROOT
        / "configs"
        / "stage_k_p0_p1_p2_pklot_test_frozen_20260727.yaml"
    )
    assert protocol["protocol_id"] == STAGE_K_PROTOCOL_ID
    assert protocol["scope"]["data_role"] == "untouched_test"
    assert protocol["data"]["expected"]["images"] == 90
    assert (
        protocol["data"]["expected"][
            "prior_development_image_sha256_overlap"
        ]
        == 0
    )


def test_stage_k_paired_comparison_uses_image_level_scores() -> None:
    metrics = {
        "P0": {
            "overall": {"macro_f1": 0.5},
            "camera_macro": {"macro_f1": 0.5},
            "by_camera": {"cam": {"macro_f1": 0.5}},
        },
        "P1": {
            "overall": {"macro_f1": 0.6},
            "camera_macro": {"macro_f1": 0.6},
            "by_camera": {"cam": {"macro_f1": 0.6}},
        },
    }
    per_sample = {
        "P0": {
            "a": {"macro_f1": 0.4},
            "b": {"macro_f1": 0.6},
            "c": {"macro_f1": 0.8},
        },
        "P1": {
            "a": {"macro_f1": 0.5},
            "b": {"macro_f1": 0.6},
            "c": {"macro_f1": 0.7},
        },
    }
    metadata = {
        sample: {"camera": "cam", "date": "date", "weather": "sunny"}
        for sample in ("a", "b", "c")
    }

    result, rows = _paired_comparison(
        candidate_id="P1",
        metrics_by_method=metrics,
        per_sample_by_method=per_sample,
        sample_metadata=metadata,
        seed=7,
        resamples=100,
    )

    assert result["win_tie_loss"] == {"win": 1, "tie": 1, "loss": 1}
    assert result["paired_bootstrap"]["unit"] == "sample_id"
    assert len(rows) == 3


def test_stage_k_record_verifier_checks_independent_hashes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    external = tmp_path / "external"
    source.mkdir()
    external.mkdir()
    payload = source / "artifact.txt"
    payload.write_text("stage-k\n", encoding="utf-8")
    import hashlib
    import yaml

    record = {
        "record_id": STAGE_K_RECORD_ID,
        "artifacts": [
            {
                "role": "example",
                "root": "source",
                "path": "artifact.txt",
                "bytes": payload.stat().st_size,
                "sha256": hashlib.sha256(payload.read_bytes()).hexdigest(),
            }
        ],
    }
    record_path = tmp_path / "record.yaml"
    record_path.write_text(
        yaml.safe_dump(record),
        encoding="utf-8",
    )

    result = verify_stage_k_record(
        record_path=record_path,
        source_root=source,
        external_root=external,
    )

    assert result["passed"]
    assert result["artifact_count"] == 1
