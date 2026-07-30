import sys
from pathlib import Path

import pytest

from parking_occupancy.cli import main
from parking_occupancy.method_registry import (
    DEFAULT_REGISTRY_PATH,
    load_method_registry,
    resolve_runnable_method,
)


def test_registry_closes_all_four_baseline_names() -> None:
    registry = load_method_registry()
    assert set(registry["methods"]) == {"B0", "B1", "E0", "T0"}
    assert Path(DEFAULT_REGISTRY_PATH).is_file()


def test_registered_b0_and_b1_differ_only_in_mapping_definition() -> None:
    b0 = resolve_runnable_method("B0")
    b1 = resolve_runnable_method("B1")
    assert (
        b0.weights,
        b0.confidence,
        b0.image_size,
        b0.class_ids,
    ) == (
        b1.weights,
        b1.confidence,
        b1.image_size,
        b1.class_ids,
    )
    assert b0.mapping_type == "bbox_centre_inside_slot_polygon"
    assert b0.minimum_slot_coverage is None
    assert b1.mapping_type == "slot_polygon_coverage"
    assert b1.minimum_slot_coverage == pytest.approx(0.40)


def test_t0_is_raw_temporal_comparator_and_e0_is_historical_only() -> None:
    t0 = resolve_runnable_method("T0")
    assert t0.pipeline_experiment == "t0"
    assert t0.confidence == pytest.approx(0.20)
    assert t0.image_size == 640
    assert "temporal_comparator" in t0.data_role

    with pytest.raises(ValueError, match="historical-only"):
        resolve_runnable_method("E0")


def test_canonical_method_rejects_detector_parameter_override(
    monkeypatch,
    capsys,
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "parking-run",
            "--input",
            "input.mp4",
            "--slots",
            "slots.json",
            "--output-dir",
            "output",
            "--method",
            "B0",
            "--conf",
            "0.50",
        ],
    )
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert "remove overrides: --conf" in capsys.readouterr().err
