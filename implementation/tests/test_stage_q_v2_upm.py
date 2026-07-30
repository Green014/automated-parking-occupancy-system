from __future__ import annotations

import hashlib
import zipfile
from pathlib import Path

import cv2
import numpy as np
import pytest

from parking_occupancy.stage_q_external import (
    FORMAL_METHOD_OUTPUTS,
    ManifestRecord,
    parse_groundtruth_text,
    truth_to_long_form,
    validate_frozen_comparison,
    validate_image_truth_bijection,
    validate_method_output_contract,
    validate_temporal_metric_request,
    verify_manifest_records,
)
from parking_occupancy.stage_q_v2_upm import (
    UPM_SLOT_IDS,
    StageQV2DataError,
    extract_zip_safely,
    inspect_sequence,
    inspect_zip_archive,
    require_human_polygon_confirmation,
    validate_upm_slot_map,
    validate_multi_sequence_polygon_isolation,
)


def _write_image(path: Path, value: int) -> None:
    image = np.full((24, 32, 3), value, dtype=np.uint8)
    ok, encoded = cv2.imencode(".JPG", image)
    assert ok
    encoded.tofile(path)


def _truth_line(name: str, first: int = 1) -> str:
    values = [str(first)] + ["1"] * 20
    return f"{name} {' '.join(values)}"


def _safe_zip(path: Path) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("test/gopro1/groundtruth.txt", _truth_line("a.JPG"))
        archive.writestr("test/gopro1/images/a.JPG", b"image")


def _gate() -> dict:
    return {
        "comparison_roles": {
            "primary": "P3-D1",
            "secondary": "P3-D1-LL",
            "default_change_allowed": False,
        },
        "shared_inference": {
            "imgsz": 640,
            "confidence": 0.30,
            "nms_iou": 0.70,
            "agnostic_nms": True,
            "max_detections": 300,
            "classes": [0],
        },
    }


def _p3_defaults() -> dict:
    return {
        "detector": {
            "id": "D1",
            "image_size": 640,
            "confidence": 0.30,
            "nms_iou": 0.70,
            "agnostic_nms": True,
            "max_detections": 300,
            "class_ids": [0],
        }
    }


def test_zip_structure_is_readable_and_rooted_in_test(tmp_path: Path) -> None:
    archive = tmp_path / "test.zip"
    _safe_zip(archive)
    result = inspect_zip_archive(archive)
    assert result["zip_readable"] is True
    assert result["path_traversal_safe"] is True
    assert result["archive_roots"] == ["test"]
    assert result["file_count"] == 2


@pytest.mark.parametrize(
    "unsafe_name",
    ["../escape.txt", "test/../../escape.txt", "/absolute.txt"],
)
def test_zip_path_traversal_and_unsafe_names_are_rejected(
    tmp_path: Path,
    unsafe_name: str,
) -> None:
    archive = tmp_path / "unsafe.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr(unsafe_name, b"x")
    with pytest.raises(StageQV2DataError, match="Unsafe ZIP member"):
        inspect_zip_archive(archive)


def test_safe_extraction_refuses_overwrite(tmp_path: Path) -> None:
    archive = tmp_path / "test.zip"
    _safe_zip(archive)
    output = tmp_path / "extracted"
    assert extract_zip_safely(archive, output) == output / "test"
    with pytest.raises(FileExistsError, match="overwrite"):
        extract_zip_safely(archive, output)


def test_21_value_vector_and_source_mapping() -> None:
    record = parse_groundtruth_text(_truth_line("a.JPG", first=0))[0]
    assert len(record.source_values) == 21
    assert record.source_values[0] == 0
    assert record.project_states[0] == 1
    assert record.source_values[1] == 1
    assert record.project_states[1] == 0


def test_image_truth_membership_is_one_to_one() -> None:
    records = parse_groundtruth_text(_truth_line("a.JPG"))
    assert validate_image_truth_bijection(["a.JPG"], records)["one_to_one"]
    with pytest.raises(ValueError, match="membership mismatch"):
        validate_image_truth_bijection(["b.JPG"], records)


def test_sequence_inventory_counts_transitions_and_brightness(
    tmp_path: Path,
) -> None:
    sequence = tmp_path / "gopro1"
    images = sequence / "images"
    images.mkdir(parents=True)
    _write_image(images / "GOPR0001.JPG", 25)
    _write_image(images / "GOPR0002.JPG", 35)
    (sequence / "groundtruth.txt").write_text(
        _truth_line("GOPR0002.JPG", first=0)
        + "\n"
        + _truth_line("GOPR0001.JPG", first=1)
        + "\n",
        encoding="utf-8",
    )
    result = inspect_sequence(sequence)
    assert result.image_count == 2
    assert result.image_truth_one_to_one is True
    assert result.binary_vectors_only is True
    assert result.transition_frames == 1
    assert result.slot_state_changes == 1
    assert result.auxiliary_low_light_candidate is True
    assert result.timestamp_available is False
    assert result.reliable_fps_available is False


def test_unknown_truth_is_excluded_from_converted_rows() -> None:
    values = ["?"] + ["0"] * 20
    record = parse_groundtruth_text(
        "a.JPG " + " ".join(values)
    )
    result = truth_to_long_form(record, video_id="gopro1")
    assert result["unknown_excluded"] == 1
    assert len(result["rows"]) == 20


def test_polygon_ids_follow_vector_index_order() -> None:
    slots = []
    for index, slot_id in enumerate(UPM_SLOT_IDS):
        x = (index % 7) * 12
        y = (index // 7) * 15
        slots.append(
            {
                "id": slot_id,
                "points": [[x, y], [x + 8, y], [x + 8, y + 10], [x, y + 10]],
            }
        )
    payload = {
        "schema_version": 1,
        "source_width": 100,
        "source_height": 60,
        "coordinate_system": "pixel",
        "slots": slots,
    }
    assert len(validate_upm_slot_map(payload).slots) == 21
    payload["slots"][0], payload["slots"][1] = (
        payload["slots"][1],
        payload["slots"][0],
    )
    with pytest.raises(ValueError, match="IDs/order"):
        validate_upm_slot_map(payload)


def test_multiple_sequences_require_explicit_polygon_bindings() -> None:
    result = validate_multi_sequence_polygon_isolation(
        ["gopro1", "gopro2"],
        {"gopro1": "scene_a.json", "gopro2": "scene_b.json"},
    )
    assert result["isolated"] is True
    with pytest.raises(StageQV2DataError, match="Every selected"):
        validate_multi_sequence_polygon_isolation(
            ["gopro1", "gopro2"],
            {"gopro1": "scene_a.json"},
        )


def test_manifest_hash_and_size_are_verified(tmp_path: Path) -> None:
    image = tmp_path / "a.JPG"
    image.write_bytes(b"image")
    digest = hashlib.sha256(image.read_bytes()).hexdigest()
    record = ManifestRecord("a.JPG", 5, digest)
    result = verify_manifest_records(tmp_path, [record])
    assert result["verified"] is True
    assert len(result["manifest_sha256"]) == 64


def test_d1_d1_ll_settings_are_identical_and_d1_stays_primary() -> None:
    result = validate_frozen_comparison(_gate(), _p3_defaults())
    assert result["same_inference_settings"] is True
    assert result["primary"] == "P3-D1"
    assert result["secondary"] == "P3-D1-LL"
    assert result["default_detector"] == "D1"


def test_unconfirmed_polygon_blocks_before_model_call() -> None:
    calls = 0

    def run() -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(StageQV2DataError, match="confirmation"):
        require_human_polygon_confirmation(
            gate_status="PASS",
            polygon_confirmation=False,
            run=run,
        )
    assert calls == 0


def test_blocked_night_gate_blocks_before_model_call() -> None:
    calls = 0

    def run() -> None:
        nonlocal calls
        calls += 1

    with pytest.raises(StageQV2DataError, match="does not authorize"):
        require_human_polygon_confirmation(
            gate_status="BLOCKED_NO_QUALIFYING_NIGHT_TEST",
            polygon_confirmation=True,
            run=run,
        )
    assert calls == 0


def test_no_timestamp_prohibits_seconds_latency() -> None:
    with pytest.raises(ValueError, match="seconds-level"):
        validate_temporal_metric_request(
            media_type="low_frame_rate_image_sequence",
            requested_metrics=["transition_latency_seconds"],
        )


def test_formal_output_schema_requires_all_outputs(tmp_path: Path) -> None:
    for relative in FORMAL_METHOD_OUTPUTS:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"x")
    (tmp_path / "annotated_frames").mkdir()
    result = validate_method_output_contract(
        tmp_path,
        orderable_sequence=True,
    )
    assert result["verified"] is False
    assert result["missing"] == ["annotated.mp4"]
