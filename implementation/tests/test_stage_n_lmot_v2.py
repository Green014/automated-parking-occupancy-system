from __future__ import annotations

import io
import json
import tarfile
from pathlib import Path

import pytest

import parking_occupancy.stage_n_lmot_v2 as stage_n_v2
from parking_occupancy.stage_n_lmot import StageNDataGateError
from parking_occupancy.stage_n_lmot_v2 import (
    LmotClassMapV2,
    build_file_manifest,
    discover_split_tar_parts,
    inspect_or_extract_rgb_split_tar,
    load_stage_n_v2_protocol,
    verify_file_manifest,
)


def _write_split_tar(
    directory: Path,
    *,
    archive_stem: str,
    archive_root: str,
    image_directory: str,
    extension: str,
) -> list[Path]:
    payload = io.BytesIO()
    with tarfile.open(fileobj=payload, mode="w") as archive:
        for split, sequence, frame in (
            ("train", "LMOT-02", 1),
            ("val", "LMOT-05", 1),
            ("val", "LMOT-05", 2),
        ):
            data = f"{split}-{frame}".encode("ascii")
            info = tarfile.TarInfo(
                f"{archive_root}/{split}/{sequence}/"
                f"{image_directory}/{frame:06d}{extension}"
            )
            info.size = len(data)
            archive.addfile(info, io.BytesIO(data))
    content = payload.getvalue()
    cut = len(content) // 2
    parts = []
    for code, data in (("aa", content[:cut]), ("ab", content[cut:])):
        path = directory / f"{archive_stem}{code}"
        path.write_bytes(data)
        parts.append(path)
    return parts


def test_split_tar_allows_train_transport_but_extracts_val_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        stage_n_v2, "EXPECTED_VALIDATION_SEQUENCES", ("LMOT-05",)
    )
    monkeypatch.setattr(stage_n_v2, "EXPECTED_FRAMES_PER_SEQUENCE", 2)
    parts = _write_split_tar(
        tmp_path,
        archive_stem="LMOT_light_rgb_trainval.tar",
        archive_root="LMOT_light_rgb_trainval",
        image_directory="img_light_rgb",
        extension=".jpg",
    )
    output = tmp_path / "output"

    report = inspect_or_extract_rgb_split_tar(
        parts=parts,
        archive_root="LMOT_light_rgb_trainval",
        image_directory="img_light_rgb",
        expected_extension=".jpg",
        extract_to=output,
    )

    assert report["transport_train_files_not_extracted"] == 1
    assert report["validation_files"] == 2
    assert report["extracted_files"] == 2
    assert not (output / "LMOT-02").exists()
    assert (
        output / "LMOT-05" / "img_light_rgb" / "000001.jpg"
    ).read_bytes() == b"val-1"


def test_discovery_rejects_missing_split_part(tmp_path: Path) -> None:
    (tmp_path / "sample.taraa").write_bytes(b"a")
    (tmp_path / "sample.tarac").write_bytes(b"c")

    with pytest.raises(StageNDataGateError, match="not contiguous"):
        discover_split_tar_parts(tmp_path, "sample.tar")


def test_stage_n_v2_class_map_requires_empirical_evidence() -> None:
    mapping = LmotClassMapV2(
        id_to_name={
            1: "person",
            2: "bicycle",
            3: "car",
            4: "motorcycle",
            5: "bus",
            6: "truck",
        },
        verification_status="empirical_visual_verified",
        evidence="contact-sheet.yaml",
        evidence_sha256="abc",
        evaluated_mark_values=frozenset({1}),
    )

    assert mapping.is_motor_vehicle(3)
    assert mapping.is_non_motor(1)
    with pytest.raises(StageNDataGateError):
        LmotClassMapV2(
            id_to_name={1: "car"},
            verification_status="official_verified",
            evidence="README",
            evidence_sha256="abc",
            evaluated_mark_values=frozenset({1}),
        )


def test_stage_n_v2_protocol_preserves_original_stage_n() -> None:
    root = Path(__file__).resolve().parents[1]

    payload = load_stage_n_v2_protocol(
        root
        / "configs"
        / "stage_n_v2_lmot_tracking_diagnostic_frozen_20260729.yaml",
        verify_preserved_files=True,
    )

    assert payload["status"] == "acquisition_verified_execution_pending"
    assert list(payload["methods"]) == ["L0", "L1", "L2", "L3"]
    assert payload["lmot"]["extraction"]["train_extraction"] == "prohibited"


def test_extracted_file_manifest_detects_change(tmp_path: Path) -> None:
    data_root = tmp_path / "data"
    data_root.mkdir()
    sample = data_root / "sample.txt"
    sample.write_text("original", encoding="utf-8")
    payload = build_file_manifest(data_root)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    result = verify_file_manifest(
        manifest_path, expected_root=data_root
    )
    assert result["file_count"] == 1
    sample.write_text("modified", encoding="utf-8")
    with pytest.raises(StageNDataGateError, match="hash mismatch"):
        verify_file_manifest(manifest_path, expected_root=data_root)
