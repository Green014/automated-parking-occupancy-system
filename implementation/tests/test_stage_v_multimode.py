from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from parking_occupancy.artifact_registry import (
    ArtifactRegistryError,
    STAGE_V_1_PRE_HARDENING_REGISTRY_SHA256,
    artifact_record,
    verify_artifact_registry,
    verify_historical_artifact_registry,
)
from parking_occupancy.models import Detection, ParkingSlot
from parking_occupancy.stage_v import (
    ClassicPixelBackend,
    ClassicPixelConfig,
    DetectionBackend,
    DetectionCache,
    FrameOccupancyResult,
    FusionBackend,
    OccupancyBackend,
    SlotOccupancyState,
    draw_stage_v_frame,
    validate_frame_result,
    validate_slot_render_coverage,
)
from parking_occupancy.stage_v_runner import (
    EXPECTED_D1,
    EXPECTED_FROZEN_CONFIG,
    FrozenConfigError,
    FrozenArtifactError,
    ModeOutput,
    run_stage_v,
    stage_v_slot_map_from_dict,
    validate_stage_v_config,
    validate_frozen_artifact,
)


SLOTS = (
    ParkingSlot(
        slot_id="slot-a",
        points=((1, 1), (14, 1), (14, 14), (1, 14)),
    ),
    ParkingSlot(
        slot_id="slot-b",
        points=((17, 1), (30, 1), (30, 14), (17, 14)),
    ),
)


class FakeDetector:
    def __init__(self, detections: tuple[Detection, ...] = ()) -> None:
        self.detections = detections
        self.calls = 0
        self.sources: list[tuple[str, bool]] = []

    def begin_source(self, source_id: str, *, continuous: bool) -> None:
        self.sources.append((source_id, continuous))

    def detect(self, _frame: np.ndarray):
        self.calls += 1
        return self.detections

    def metadata(self):
        return {"backend": "fake-detector", "calls": self.calls}


class FakeClassifier:
    def __init__(self, scores: dict[str, float] | None = None) -> None:
        self.scores = scores or {}
        self.calls: list[tuple[str, ...]] = []

    def predict(self, _frame: np.ndarray, slots):
        self.calls.append(tuple(slot.slot_id for slot in slots))
        return {slot.slot_id: self.scores.get(slot.slot_id, 0.1) for slot in slots}

    def metadata(self):
        return {"backend": "fake-classifier"}


def _detection() -> Detection:
    return Detection(
        bbox=(2, 2, 13, 13),
        confidence=0.9,
        class_id=0,
        class_name="vehicle",
        track_id=7,
    )


def _write_inputs(tmp_path: Path) -> tuple[Path, Path]:
    images = tmp_path / "images"
    images.mkdir()
    frame = np.zeros((16, 32, 3), dtype=np.uint8)
    assert cv2.imwrite(str(images / "000.png"), frame)
    assert cv2.imwrite(str(images / "001.png"), frame)
    slots = tmp_path / "slots.json"
    slots.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_width": 32,
                "source_height": 16,
                "coordinate_system": "pixel",
                "slots": [
                    {"id": slot.slot_id, "points": slot.points} for slot in SLOTS
                ],
            }
        ),
        encoding="utf-8",
    )
    return images, slots


def _write_video_inputs(tmp_path: Path, frames: int = 2) -> tuple[Path, Path]:
    video = tmp_path / "input.mp4"
    writer = cv2.VideoWriter(
        str(video),
        cv2.VideoWriter_fourcc(*"mp4v"),
        5.0,
        (32, 16),
    )
    assert writer.isOpened()
    for index in range(frames):
        writer.write(np.full((16, 32, 3), index * 10, dtype=np.uint8))
    writer.release()
    slot_source = tmp_path / "slot-source"
    slot_source.mkdir()
    _, slots = _write_inputs(slot_source)
    return video, slots


def _backends() -> tuple[dict[str, OccupancyBackend], FakeDetector, FakeClassifier]:
    detector = FakeDetector((_detection(),))
    classifier = FakeClassifier({"slot-b": 0.9})
    cache = DetectionCache(detector)
    return (
        {
            "classic": ClassicPixelBackend(
                ClassicPixelConfig(occupied_foreground_ratio=0.30)
            ),
            "detection": DetectionBackend(cache, overlap_threshold=0.40),
            "fusion": FusionBackend(
                cache,
                classifier,
                overlap_threshold=0.40,
                classifier_threshold=0.76,
            ),
        },
        detector,
        classifier,
    )


def test_unified_backend_interface_returns_frame_result() -> None:
    backend: OccupancyBackend = ClassicPixelBackend()
    result = backend.process_frame(
        np.zeros((16, 32, 3), dtype=np.uint8),
        SLOTS,
        3,
        0.3,
    )
    assert isinstance(result, FrameOccupancyResult)
    assert result.frame_index == 3
    assert result.timestamp_s == pytest.approx(0.3)
    assert len(result.slot_states) == len(SLOTS)


def test_rectangle_to_polygon_conversion_supports_list_and_mapping() -> None:
    slot_map = stage_v_slot_map_from_dict(
        {
            "frame_width": 100,
            "frame_height": 50,
            "spaces": [
                {"id": "a", "rect": [10, 5, 20, 10]},
                {
                    "id": "b",
                    "rect": {"x": 40, "y": 10, "width": 30, "height": 20},
                },
            ],
        }
    )
    assert slot_map.slots[0].points == (
        (10.0, 5.0),
        (30.0, 5.0),
        (30.0, 15.0),
        (10.0, 15.0),
    )
    assert slot_map.slots[1].points[-1] == (40.0, 30.0)


def test_classic_occupied_and_vacant_judgement(monkeypatch) -> None:
    backend = ClassicPixelBackend(
        ClassicPixelConfig(occupied_foreground_ratio=0.50)
    )
    binary = np.zeros((16, 32), dtype=np.uint8)
    binary[1:15, 1:15] = 255
    monkeypatch.setattr(backend, "_preprocess", lambda _frame: binary)
    result = backend.process_frame(
        np.zeros((16, 32, 3), dtype=np.uint8),
        SLOTS,
        0,
        0.0,
    )
    states = result.state_by_slot()
    assert states["slot-a"].occupied
    assert not states["slot-b"].occupied
    assert states["slot-a"].details["foreground_pixels"] > 0


def test_every_frame_covers_all_slots_when_detector_is_empty() -> None:
    detector = FakeDetector()
    cache = DetectionCache(detector)
    cache.begin_source("empty", continuous=True)
    backend = DetectionBackend(cache, overlap_threshold=0.40)
    result = backend.process_frame(
        np.zeros((16, 32, 3), dtype=np.uint8),
        SLOTS,
        0,
        0.0,
    )
    validate_frame_result(result, SLOTS)
    assert {state.slot_id for state in result.slot_states} == {
        "slot-a",
        "slot-b",
    }
    assert not any(state.occupied for state in result.slot_states)


def test_detection_adapter_reuses_b1_one_to_one_mapping() -> None:
    detector = FakeDetector((_detection(),))
    cache = DetectionCache(detector)
    cache.begin_source("detection", continuous=True)
    result = DetectionBackend(cache, overlap_threshold=0.40).process_frame(
        np.zeros((16, 32, 3), dtype=np.uint8),
        SLOTS,
        0,
        0.0,
    )
    states = result.state_by_slot()
    assert states["slot-a"].occupied
    assert states["slot-a"].track_id == 7
    assert not states["slot-b"].occupied


def test_fusion_adapter_reviews_only_detector_negative_slots() -> None:
    detector = FakeDetector((_detection(),))
    classifier = FakeClassifier({"slot-b": 0.9})
    cache = DetectionCache(detector)
    cache.begin_source("fusion", continuous=True)
    result = FusionBackend(
        cache,
        classifier,
        overlap_threshold=0.40,
        classifier_threshold=0.76,
    ).process_frame(
        np.zeros((16, 32, 3), dtype=np.uint8),
        SLOTS,
        0,
        0.0,
    )
    assert classifier.calls == [("slot-b",)]
    states = result.state_by_slot()
    assert states["slot-a"].evidence_source.endswith("detector_confirmed")
    assert states["slot-b"].evidence_source.endswith("classifier_recovery")
    assert all(state.occupied for state in states.values())


def test_fake_end_to_end_compare_writes_required_outputs(tmp_path: Path) -> None:
    images, slots = _write_inputs(tmp_path)
    backends, _, _ = _backends()
    output = tmp_path / "stage-v"
    result = run_stage_v(
        input_path=images,
        slots_path=slots,
        output_root=output,
        mode="compare",
        backends=backends,
        config_snapshot={"test": True},
    )
    assert result["frames"] == 2
    for mode in ("classic", "detection", "fusion"):
        mode_dir = output / mode
        assert {
            "annotated_images",
            "configuration_snapshot.yaml",
            "detections.jsonl",
            "events.csv",
            "metrics.json",
            "occupancy.csv",
            "runtime_metadata.json",
            "summary.json",
        } == {path.name for path in mode_dir.iterdir()}
    assert (output / "comparison.json").is_file()
    assert (output / "method_metrics.csv").is_file()
    assert (output / "runtime_comparison.csv").is_file()


def test_output_schema_has_one_row_per_frame_and_slot(tmp_path: Path) -> None:
    images, slots = _write_inputs(tmp_path)
    backends, _, _ = _backends()
    output = tmp_path / "schema"
    run_stage_v(
        input_path=images,
        slots_path=slots,
        output_root=output,
        mode="fusion",
        backends=backends,
        config_snapshot={"test": True},
    )
    with (output / "occupancy.csv").open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        assert set(reader.fieldnames or ()) == set(
            (
                "video_id",
                "frame_index",
                "timestamp_s",
                "slot_id",
                "state",
                "occupied",
                "vacant",
                "evidence",
                "evidence_source",
                "track_id",
                "details_json",
                "warnings",
            )
        )
    assert len(rows) == 4
    assert len({(row["frame_index"], row["slot_id"]) for row in rows}) == 4


def test_visualization_draws_every_configured_slot() -> None:
    states = {
        slot.slot_id: SlotOccupancyState(
            slot_id=slot.slot_id,
            occupied=index == 0,
            evidence_score=1.0 if index == 0 else 0.0,
            evidence_source="test",
            track_id=None,
        )
        for index, slot in enumerate(SLOTS)
    }
    assert validate_slot_render_coverage(SLOTS, states) == 2
    rendered, rendered_count = draw_stage_v_frame(
        frame=np.zeros((80, 160, 3), dtype=np.uint8),
        detections=[],
        slots=SLOTS,
        states=states,
        mode="test",
        processing_fps=10.0,
    )
    assert rendered_count == 2
    assert rendered.any()
    with pytest.raises(ValueError, match="coverage mismatch"):
        validate_slot_render_coverage(SLOTS, {"slot-a": states["slot-a"]})


def test_missing_or_wrong_frozen_weight_has_explicit_error(tmp_path: Path) -> None:
    with pytest.raises(FrozenArtifactError, match="D1 detector is required"):
        validate_frozen_artifact(
            None,
            label="D1 detector",
            expected=EXPECTED_D1,
        )
    wrong = tmp_path / "best.pt"
    wrong.write_bytes(b"not the frozen model")
    with pytest.raises(FrozenArtifactError, match="does not match"):
        validate_frozen_artifact(
            wrong,
            label="D1 detector",
            expected=EXPECTED_D1,
        )


def test_all_modes_use_identical_slot_map(tmp_path: Path) -> None:
    images, slots = _write_inputs(tmp_path)
    backends, _, _ = _backends()
    output = tmp_path / "same-slots"
    run_stage_v(
        input_path=images,
        slots_path=slots,
        output_root=output,
        mode="compare",
        backends=backends,
        config_snapshot={"test": True},
    )
    slot_sets = []
    for mode in ("classic", "detection", "fusion"):
        with (output / mode / "occupancy.csv").open(
            encoding="utf-8", newline=""
        ) as handle:
            slot_sets.append({row["slot_id"] for row in csv.DictReader(handle)})
    assert slot_sets == [{"slot-a", "slot-b"}] * 3


def test_detection_cache_preserves_predictions_and_avoids_duplicate_d1_calls(
    tmp_path: Path,
) -> None:
    images, slots = _write_inputs(tmp_path)
    backends, detector, _ = _backends()
    output = tmp_path / "cache"
    run_stage_v(
        input_path=images,
        slots_path=slots,
        output_root=output,
        mode="compare",
        backends=backends,
        config_snapshot={"test": True},
    )
    assert detector.calls == 2
    with (output / "detection" / "occupancy.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        detection_rows = list(csv.DictReader(handle))
    with (output / "fusion" / "occupancy.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        fusion_rows = list(csv.DictReader(handle))
    detection_slot_a = [
        row["state"] for row in detection_rows if row["slot_id"] == "slot-a"
    ]
    fusion_slot_a = [
        row["state"] for row in fusion_rows if row["slot_id"] == "slot-a"
    ]
    assert detection_slot_a == fusion_slot_a == ["1", "1"]


def test_modified_frozen_config_is_rejected_without_explicit_opt_in(
    tmp_path: Path,
) -> None:
    frozen = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "p3_stage_r_recommended_default_20260729.yaml"
    )
    payload = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    payload["detector"]["confidence"] = 0.31
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    with pytest.raises(FrozenConfigError, match="allow-custom-config"):
        validate_stage_v_config(custom)


def test_formal_and_byte_identical_frozen_config_pass() -> None:
    frozen = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "p3_stage_r_recommended_default_20260729.yaml"
    )
    _, identity = validate_stage_v_config(frozen)
    assert identity["classification"] == "frozen"
    assert identity["method_identity"] == "frozen_C1_C2"
    assert identity["sha256"] == EXPECTED_FROZEN_CONFIG["sha256"]
    assert identity["exact_sha256_match"] is True
    assert identity["hash_mismatch"] is False
    assert identity["critical_parameters_match"] is True
    assert identity["parameter_values_changed"] is False


def test_byte_identical_config_copy_keeps_frozen_identity(
    tmp_path: Path,
) -> None:
    frozen = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "p3_stage_r_recommended_default_20260729.yaml"
    )
    copied = tmp_path / "renamed-frozen-copy.yaml"
    copied.write_bytes(frozen.read_bytes())
    _, identity = validate_stage_v_config(copied)
    assert identity["classification"] == "frozen"
    assert identity["filename"] == copied.name
    assert identity["exact_sha256_match"] is True


def test_hash_only_config_change_is_rejected_then_custom_when_allowed(
    tmp_path: Path,
) -> None:
    frozen = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "p3_stage_r_recommended_default_20260729.yaml"
    )
    commented = tmp_path / "commented.yaml"
    commented.write_bytes(frozen.read_bytes() + b"\n# byte-only change\n")
    with pytest.raises(FrozenConfigError, match="SHA-256 mismatch"):
        validate_stage_v_config(commented)

    config, identity = validate_stage_v_config(
        commented,
        allow_custom_config=True,
    )
    assert config["config_id"] == EXPECTED_FROZEN_CONFIG["config_id"]
    assert identity["classification"] == "custom"
    assert identity["method_identity"] == "custom"
    assert identity["exact_sha256_match"] is False
    assert identity["hash_mismatch"] is True
    assert identity["critical_parameters_match"] is True
    assert identity["parameter_values_changed"] is False
    assert identity["frozen_parameters_changed"] is True
    assert identity["parameter_differences"] == []


def test_custom_config_records_identity_hash_and_parameter_difference(
    tmp_path: Path,
) -> None:
    frozen = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "p3_stage_r_recommended_default_20260729.yaml"
    )
    payload = yaml.safe_load(frozen.read_text(encoding="utf-8"))
    payload["classifier"]["occupied_threshold"] = 0.81
    custom = tmp_path / "custom.yaml"
    custom.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    _, identity = validate_stage_v_config(
        custom,
        allow_custom_config=True,
    )
    assert identity["classification"] == "custom"
    assert identity["method_identity"] == "custom"
    assert identity["exact_sha256_match"] is False
    assert identity["hash_mismatch"] is True
    assert identity["critical_parameters_match"] is False
    assert identity["parameter_values_changed"] is True
    assert identity["frozen_parameters_changed"] is True
    assert len(identity["sha256"]) == 64
    assert identity["parameter_differences"] == [
        {
            "parameter": "classifier.occupied_threshold",
            "expected": 0.76,
            "actual": 0.81,
        }
    ]


class SequenceBackend:
    mode = "fusion"

    def __init__(self, occupied_by_frame: list[tuple[bool, bool]]) -> None:
        self.occupied_by_frame = occupied_by_frame

    def process_frame(self, frame, slots, frame_index, timestamp_s):
        states = tuple(
            SlotOccupancyState(
                slot_id=slot.slot_id,
                occupied=self.occupied_by_frame[frame_index][index],
                evidence_score=0.9,
                evidence_source="sequence-test",
            )
            for index, slot in enumerate(slots)
        )
        return FrameOccupancyResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            slot_states=states,
            vehicle_detections=(),
            timing_ms={
                "backend_total": 10.0,
                "attributed_backend_total": 10.0,
            },
        )

    def metadata(self):
        return {"mode": self.mode, "method_id": "test-sequence"}


def test_first_video_frame_establishes_state_without_arrival(
    tmp_path: Path,
) -> None:
    video, slots = _write_video_inputs(tmp_path)
    output = tmp_path / "first-frame"
    run_stage_v(
        input_path=video,
        slots_path=slots,
        output_root=output,
        mode="fusion",
        backends={"fusion": SequenceBackend([(True, False), (True, False)])},
        config_snapshot={"temporal": {"enabled": False}},
    )
    with (output / "events.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["initial_frame_emits_event"] is False
    assert summary["events_temporally_valid"] is True
    assert summary["event_semantics"] == "raw_frame_level_state_changes"


def test_continuous_video_emits_only_real_post_initial_transition(
    tmp_path: Path,
) -> None:
    video, slots = _write_video_inputs(tmp_path)
    output = tmp_path / "transition"
    run_stage_v(
        input_path=video,
        slots_path=slots,
        output_root=output,
        mode="fusion",
        backends={"fusion": SequenceBackend([(True, False), (False, False)])},
        config_snapshot={"temporal": {"enabled": False}},
    )
    with (output / "events.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    assert rows[0]["frame_index"] == "1"
    assert rows[0]["slot_id"] == "slot-a"
    assert rows[0]["from_state"] == "1"
    assert rows[0]["to_state"] == "0"


def test_image_directory_never_creates_temporal_events(
    tmp_path: Path,
) -> None:
    images, slots = _write_inputs(tmp_path)
    output = tmp_path / "images-no-events"
    run_stage_v(
        input_path=images,
        slots_path=slots,
        output_root=output,
        mode="fusion",
        backends={"fusion": SequenceBackend([(False, False), (True, False)])},
        config_snapshot={"temporal": {"enabled": False}},
    )
    with (output / "events.csv").open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle)) == []
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["events_temporally_valid"] is False
    assert summary["events"] == 0


def test_cache_hit_visual_fps_uses_attributed_backend_total(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured: dict[str, float | str] = {}

    def fake_draw(**kwargs):
        captured["fps"] = kwargs["processing_fps"]
        captured["cache"] = kwargs["cache_status"]
        return kwargs["frame"], len(kwargs["slots"])

    monkeypatch.setattr(
        "parking_occupancy.stage_v_runner.draw_stage_v_frame",
        fake_draw,
    )
    sink = ModeOutput(
        output_dir=tmp_path / "sink",
        mode="fusion",
        source_id="fake",
        input_kind="images",
        fps=1.0,
        frame_size=(32, 16),
        slots=SLOTS,
        backend=SequenceBackend([(False, False)]),
        config_snapshot={"temporal": {"enabled": False}},
    )
    result = SequenceBackend([(False, False)]).process_frame(
        np.zeros((16, 32, 3), dtype=np.uint8),
        SLOTS,
        0,
        0.0,
    )
    result = FrameOccupancyResult(
        frame_index=result.frame_index,
        timestamp_s=result.timestamp_s,
        slot_states=result.slot_states,
        vehicle_detections=result.vehicle_detections,
        timing_ms={
            "backend_total": 5.0,
            "attributed_backend_total": 50.0,
            "cache_hit": 1.0,
        },
    )
    sink.write(np.zeros((16, 32, 3), dtype=np.uint8), result)
    sink._close_handles()
    assert captured == {"fps": pytest.approx(20.0), "cache": "hit"}


def test_registry_verifier_detects_missing_modified_extra_and_optional(
    tmp_path: Path,
) -> None:
    root = tmp_path / "root"
    (root / "src").mkdir(parents=True)
    expected = root / "src" / "stage_v_1.py"
    expected.write_text("frozen\n", encoding="utf-8")
    registry = root / "registry.yaml"
    payload = {
        "schema_version": 1,
        "registry_id": "TEST-V1",
        "stage": "V.1",
        "artifact_count": 2,
        "registry_self_path": "registry.yaml",
        "managed_artifact_globs": ["src/stage_v_*.py"],
        "artifacts": [
            artifact_record(
                root,
                "src/stage_v_1.py",
                role="source",
            ),
            {
                "path": "outputs/local.mp4",
                "role": "smoke",
                "availability": "local_ignored_optional",
                "bytes": 1,
                "sha256": "0" * 64,
            },
        ],
    }
    registry.write_text(yaml.safe_dump(payload), encoding="utf-8")
    verified = verify_artifact_registry(registry, artifact_root=root)
    assert verified["verified"] is True
    assert verified["optional_unavailable"] == 1

    expected.write_text("changed\n", encoding="utf-8")
    modified = verify_artifact_registry(registry, artifact_root=root)
    assert "bytes:src/stage_v_1.py" in modified["errors"]
    expected.write_text("frozen\n", encoding="utf-8")
    (root / "src" / "stage_v_extra.py").write_text("extra\n", encoding="utf-8")
    extra = verify_artifact_registry(registry, artifact_root=root)
    assert "extra:src/stage_v_extra.py" in extra["errors"]
    expected.unlink()
    missing = verify_artifact_registry(registry, artifact_root=root)
    assert "missing:src/stage_v_1.py" in missing["errors"]


def test_registry_rejects_windows_absolute_artifact_path(
    tmp_path: Path,
) -> None:
    registry = tmp_path / "registry.yaml"
    registry.write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "registry_id": "UNSAFE",
                "stage": "V.1",
                "artifact_count": 1,
                "artifacts": [
                    {
                        "path": "C:/outside.txt",
                        "role": "unsafe",
                        "availability": "required",
                        "bytes": 1,
                        "sha256": "0" * 64,
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ArtifactRegistryError):
        verify_artifact_registry(registry, artifact_root=tmp_path)


def test_checked_in_stage_v_1_registry_verifies() -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    result = verify_historical_artifact_registry(
        implementation_root / "data" / "STAGE_V_1_ARTIFACT_REGISTRY.yaml",
        artifact_root=implementation_root,
        expected_registry_sha256=(
            STAGE_V_1_PRE_HARDENING_REGISTRY_SHA256
        ),
        immutable_path_prefixes=("outputs",),
    )
    assert result["verified"] is True, result["errors"]
    assert result["live_release_artifacts_compared"] is False
