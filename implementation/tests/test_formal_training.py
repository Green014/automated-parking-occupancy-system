from __future__ import annotations

from pathlib import Path

import pytest

from parking_occupancy.formal_training import (
    FORMAL_EXPERIMENT_ID,
    FormalTrainingResourceProbe,
    FormalTrainingProtocolError,
    best_epoch_indices,
    formal_training_arguments,
    load_formal_settings,
    verify_formal_training_record,
    verify_formal_freeze,
)


def _paths() -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    return (
        root / "configs" / "d1_ndispark_formal_frozen_20260727.yaml",
        root
        / "data"
        / "training"
        / "D1_GPU_DECISION_FROZEN_CHECKSUMS.yaml",
    )


def test_committed_formal_configuration_matches_freeze() -> None:
    config, registry = _paths()
    payload, settings, freeze = load_formal_settings(
        config_path=config,
        freeze_registry_path=registry,
    )

    assert payload["experiment_id"] == FORMAL_EXPERIMENT_ID
    assert settings.max_epochs == 50
    assert settings.patience == 10
    assert settings.imgsz == 640
    assert settings.batch == 4
    assert settings.nbs == 64
    assert settings.expected_accumulation == 16
    assert settings.device == "0"
    assert freeze["registry_id"] == (
        "GPU-GATE-NDISPARK-D1-FREEZE-20260727-01"
    )


def test_formal_arguments_are_exact_and_path_driven(tmp_path: Path) -> None:
    config, registry = _paths()
    _, settings, _ = load_formal_settings(
        config_path=config,
        freeze_registry_path=registry,
    )
    data = tmp_path / "dataset.yaml"
    output = tmp_path / "formal_run"

    arguments = formal_training_arguments(
        settings=settings,
        data_yaml=data,
        output_dir=output,
    )

    assert arguments["data"] == str(data.resolve())
    assert arguments["project"] == str(tmp_path.resolve())
    assert arguments["name"] == "formal_run"
    assert arguments["epochs"] == 50
    assert arguments["patience"] == 10
    assert arguments["batch"] == 4
    assert arguments["nbs"] == 64
    assert arguments["imgsz"] == 640
    assert arguments["save_period"] == 10
    assert arguments["resume"] is False
    assert arguments["mosaic"] == pytest.approx(1.0)


def test_freeze_rejects_tampered_registry(tmp_path: Path) -> None:
    config, registry = _paths()
    changed = tmp_path / "registry.yaml"
    raw = registry.read_text(encoding="utf-8").replace(
        "GPU-GATE-NDISPARK-D1-FREEZE-20260727-01",
        "changed",
    )
    changed.write_text(raw, encoding="utf-8")

    with pytest.raises(
        FormalTrainingProtocolError,
        match="freeze registry",
    ):
        verify_formal_freeze(
            config_path=config,
            freeze_registry_path=changed,
        )


def test_formal_config_forbids_test_and_smoke_initialization() -> None:
    config, registry = _paths()
    payload, _, _ = load_formal_settings(
        config_path=config,
        freeze_registry_path=registry,
    )

    assert payload["model"]["smoke_checkpoint_initialization"] == "prohibited"
    assert payload["data_boundaries"]["forbidden_during_training"] == [
        "NDISPark count-only test",
        "CNR-EXT",
        "PKLot test or historical external result",
        "VIRAT",
    ]
    assert not payload["execution_gate"]["predictions_on_test_allowed"]


def test_best_epoch_is_ultralytics_one_based() -> None:
    zero_based, one_based = best_epoch_indices(37, 47)
    assert zero_based == 36
    assert one_based == 37

    with pytest.raises(RuntimeError, match="outside"):
        best_epoch_indices(0, 47)


def test_resource_probe_skips_final_best_checkpoint_callback() -> None:
    probe = FormalTrainingResourceProbe()
    probe.epochs = [{"epoch": 1}]
    probe._training_epoch_active = False

    probe.on_fit_epoch_end(object())

    assert probe.epochs == [{"epoch": 1}]


def test_formal_artifact_verifier_checks_size_and_hash(
    tmp_path: Path,
) -> None:
    artifact = tmp_path / "outputs" / "best.pt"
    artifact.parent.mkdir()
    artifact.write_bytes(b"formal")
    import hashlib
    import yaml

    record = tmp_path / "record.yaml"
    record.write_text(
        yaml.safe_dump(
            {
                "record_id": "D1-NDISPARK-FORMAL-RECORD-20260727-01",
                "experiment_id": FORMAL_EXPERIMENT_ID,
                "artifacts": [
                    {
                        "role": "best_checkpoint",
                        "path": "outputs/best.pt",
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

    result = verify_formal_training_record(
        record_path=record,
        implementation_root=tmp_path,
    )
    assert result["passed"]
    assert result["artifact_count"] == 1

    artifact.write_bytes(b"changed")
    changed = verify_formal_training_record(
        record_path=record,
        implementation_root=tmp_path,
    )
    assert not changed["passed"]
