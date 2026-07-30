from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parking_occupancy.integrated_runner import (
    DEFAULT_INTEGRATED_CONFIG,
    load_integrated_config,
)
from parking_occupancy.stage_q_v2_evaluation import (
    EXPECTED_SHARED_INFERENCE,
    METHODS,
    REQUIRED_METHOD_OUTPUTS,
    StageQV2EvaluationError,
    evaluate_stage_q_v2_method,
    load_frozen_stage_q_v2_config,
    preflight_stage_q_v2,
    run_stage_q_v2_formal,
    shared_method_signature,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORMAL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_q_v2_external_night_occupancy_frozen_20260729_v2.yaml"
)
BASE_FORMAL_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_q_v2_external_night_occupancy_frozen_20260729.yaml"
)


class FakeDetector:
    def __init__(self) -> None:
        self.sources: list[tuple[str, bool]] = []

    def begin_source(self, source_id: str, *, continuous: bool) -> None:
        self.sources.append((source_id, continuous))

    def detect(self, _frame: np.ndarray):
        return ()

    def metadata(self):
        return {
            "backend": "fake_predict_only",
            "tracker_type": None,
            "generation": len(self.sources),
        }


class FakeClassifier:
    def predict(self, _frame, slots):
        return {slot.slot_id: 0.1 for slot in slots}

    def metadata(self):
        return {"backend": "fake", "device": "cpu"}


class FakeWriter:
    def __init__(self, path: str, *_args) -> None:
        self.path = Path(path)
        self.path.write_bytes(b"reconstructed visualization")

    def isOpened(self) -> bool:
        return True

    def write(self, _frame: np.ndarray) -> None:
        pass

    def release(self) -> None:
        pass


def test_frozen_config_keeps_roles_and_identical_inference() -> None:
    config = load_frozen_stage_q_v2_config(FORMAL_CONFIG)
    assert config["scope"]["D1_remains_project_default"] is True
    assert config["scope"]["D1_LL_role"] == "secondary_frozen_comparison"
    assert config["scope"]["stage_p2_decision_remains"] == "FAIL"
    assert config["shared_inference"]["tracking_backend"] == "none"
    assert {
        key: config["shared_inference"][key]
        for key in EXPECTED_SHARED_INFERENCE
    } == EXPECTED_SHARED_INFERENCE
    assert [row["method_id"] for row in config["formal_runs"]["methods"]] == [
        method_id for method_id, *_ in METHODS
    ]
    signature = shared_method_signature(config)
    assert "D1" not in signature
    assert "D1_LL" not in signature


def test_real_preflight_verifies_confirmation_and_all_manifest_images() -> None:
    result = preflight_stage_q_v2(FORMAL_CONFIG, check_output=False)
    assert result["manifest_verification"]["verified"] is True
    assert result["manifest_verification"]["file_count"] == 376
    assert len(result["truth_rows"]) == 7896
    assert result["output_root"].name == (
        "stage_q_v2_upm_gti_external_20260729_v2"
    )


def test_preflight_refuses_existing_output_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    config = load_frozen_stage_q_v2_config(FORMAL_CONFIG)
    config["formal_runs"]["output_root"] = str(tmp_path)
    monkeypatch.setattr(
        "parking_occupancy.stage_q_v2_evaluation."
        "load_frozen_stage_q_v2_config",
        lambda _path: config,
    )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        preflight_stage_q_v2(FORMAL_CONFIG)


def test_frame_only_metrics_exclude_seconds_and_report_early_delayed_missed(
    tmp_path: Path,
) -> None:
    truth_path = tmp_path / "truth.csv"
    prediction_path = tmp_path / "prediction.csv"
    truth_states = [0, 0, 1, 1, 0, 0, 1, 1]
    prediction_states = [0, 1, 1, 1, 1, 0, 0, 0]
    with truth_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "video_id",
                "frame_index",
                "timestamp_s",
                "slot_id",
                "state",
            ),
        )
        writer.writeheader()
        for frame, state in enumerate(truth_states):
            writer.writerow(
                {
                    "video_id": "sequence",
                    "frame_index": frame * 5,
                    "timestamp_s": "",
                    "slot_id": "slot_00",
                    "state": state,
                }
            )
    fields = (
        "video_id",
        "frame_index",
        "timestamp_s",
        "slot_id",
        "state",
        "evidence",
        "raw_state",
        "filtered_score",
        "detector_occupied",
        "detector_score",
        "classifier_probability",
        "classifier_consulted",
        "gate_branch",
        "track_id",
        "tracker_backend",
        "temporal_enabled",
    )
    with prediction_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for frame, state in enumerate(prediction_states):
            writer.writerow(
                {
                    "video_id": "sequence",
                    "frame_index": frame * 5,
                    "timestamp_s": "",
                    "slot_id": "slot_00",
                    "state": state,
                    "evidence": state,
                    "raw_state": state,
                    "filtered_score": state,
                    "detector_occupied": state,
                    "detector_score": state,
                    "classifier_probability": "",
                    "classifier_consulted": 0,
                    "gate_branch": "test",
                    "track_id": "",
                    "tracker_backend": "none",
                    "temporal_enabled": 1,
                }
            )
    metrics, _failures = evaluate_stage_q_v2_method(
        truth_path=truth_path,
        prediction_path=prediction_path,
        stable_frames=1,
    )
    temporal = metrics["temporal_frame_only"]
    assert temporal["seconds_level_transition_latency_computed"] is False
    assert set(temporal["transition_outcomes"]) == {
        "early",
        "on_time",
        "delayed",
        "missed",
    }
    assert "transition_latency_s" not in temporal
    assert all(
        "signed_error_source_frame_index" in event
        for event in temporal["transition_events"]
    )


def _fake_preflight(tmp_path: Path) -> dict:
    p3 = load_integrated_config(DEFAULT_INTEGRATED_CONFIG)
    source_root = tmp_path / "images"
    source_root.mkdir()
    manifest_rows = []
    truth_rows = []
    for order, sequence_id in enumerate(("source-a", "source-b")):
        image_path = source_root / f"{sequence_id}.jpg"
        image = np.full((600, 800, 3), 25 + order, dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        assert ok
        encoded.tofile(image_path)
        manifest_rows.append(
            {
                "sequence_id": sequence_id,
                "frame_index": str(order * 10),
                "relative_path": image_path.name,
            }
        )
        for slot_index in range(21):
            truth_rows.append(
                {
                    "video_id": sequence_id,
                    "frame_index": order * 10,
                    "timestamp_s": "",
                    "slot_id": f"slot_{slot_index:02d}",
                    "state": 0,
                }
            )
    truth = tmp_path / "truth.csv"
    with truth.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "video_id",
                "frame_index",
                "timestamp_s",
                "slot_id",
                "state",
            ),
        )
        writer.writeheader()
        writer.writerows(truth_rows)
    fake_input = tmp_path / "input.bin"
    fake_input.write_bytes(b"fake")
    polygons = (
        PROJECT_ROOT
        / "data"
        / "stage_q_v2"
        / "STAGE_Q_V2_SLOT_POLYGONS_20260729.json"
    )
    config_path = tmp_path / "formal.yaml"
    config_path.write_text("frozen: true\n", encoding="utf-8")
    output_root = tmp_path / "formal-output"
    config = {
        "shared_inference": {
            **EXPECTED_SHARED_INFERENCE,
            "device": "cpu",
            "B1": {
                "mode": "overlap",
                "minimum_slot_coverage": 0.40,
                "one_to_one": True,
            },
            "E1b_F2": {
                "occupied_threshold": 0.76,
                "detector_negative_slots_only": True,
                "patch_size": [224, 224],
                "perspective_warp": True,
            },
            "E4": {
                "enabled": True,
                "rise_alpha": 0.60,
                "fall_alpha": 0.15,
                "occupied_threshold": 0.58,
                "vacant_threshold": 0.42,
                "raw_threshold": 0.76,
                "stable_frames_for_evaluation": 3,
            },
        },
        "inputs": {
            "manifest": {"sha256": "manifest"},
            "occupancy_truth": {"sha256": "truth"},
            "polygons": {"sha256": "polygons"},
        },
        "models": {"E1b": {"sha256": "e1b"}},
        "temporal_semantics": {"visualization_reconstruction_fps": 2.0},
    }
    return {
        "config": config,
        "config_path": config_path,
        "manifest": fake_input,
        "manifest_rows": manifest_rows,
        "truth": truth,
        "polygons": polygons,
        "p3_runtime": DEFAULT_INTEGRATED_CONFIG,
        "p3": p3,
        "models": {
            "D1": fake_input,
            "D1_LL": fake_input,
            "E1b": fake_input,
        },
        "test_root": source_root,
        "output_root": output_root,
        "confirmation": fake_input,
    }


def test_fake_formal_run_writes_complete_contract_and_resets_sources(
    tmp_path: Path,
    monkeypatch,
) -> None:
    preflight = _fake_preflight(tmp_path)
    monkeypatch.setattr(
        "parking_occupancy.stage_q_v2_evaluation.preflight_stage_q_v2",
        lambda _path: preflight,
    )
    monkeypatch.setattr(
        "parking_occupancy.stage_q_v2_evaluation.cv2.VideoWriter_fourcc",
        lambda *_args: 0,
    )
    monkeypatch.setattr(
        "parking_occupancy.stage_q_v2_evaluation.cv2.VideoWriter",
        FakeWriter,
    )
    detectors = {method_id: FakeDetector() for method_id, *_ in METHODS}
    classifiers = {method_id: FakeClassifier() for method_id, *_ in METHODS}
    result = run_stage_q_v2_formal(
        config_path=preflight["config_path"],
        device="cpu",
        detector_overrides=detectors,
        classifier_overrides=classifiers,
    )
    assert result["status"] == "FORMAL_RUNS_COMPLETE"
    assert result["D1_remains_project_default"] is True
    for method_id, *_ in METHODS:
        method_root = preflight["output_root"] / method_id
        assert REQUIRED_METHOD_OUTPUTS <= {
            path.name for path in method_root.iterdir()
        }
        runtime = json.loads(
            (method_root / "runtime_metadata.json").read_text()
        )
        assert runtime["model_track_called"] is False
        assert len(runtime["source_state_resets"]) == 2
        assert detectors[method_id].sources == [
            ("source-a", True),
            ("source-b", True),
        ]
        metrics = json.loads((method_root / "metrics.json").read_text())
        assert metrics["classification"]["macro_f1"] == pytest.approx(0.5)
        assert (
            metrics["temporal_frame_only"][
                "seconds_level_transition_latency_computed"
            ]
            is False
        )
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        run_stage_q_v2_formal(config_path=preflight["config_path"])


def test_config_rejects_changed_d1_default_role(
    monkeypatch,
) -> None:
    original = BASE_FORMAL_CONFIG.read_text(encoding="utf-8")
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, **_kwargs: original.replace(
            "D1_remains_project_default: true",
            "D1_remains_project_default: false",
        ),
    )
    with pytest.raises(StageQV2EvaluationError, match="D1 default"):
        load_frozen_stage_q_v2_config(BASE_FORMAL_CONFIG)
