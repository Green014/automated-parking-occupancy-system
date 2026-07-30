from __future__ import annotations

import csv
import io
import json
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from parking_occupancy.stage_k_data_gate import (
    ArchivePair,
    CandidateGroup,
    STAGE_K_GATE_V2_RECORD_ID,
    StageKDataGateError,
    parse_pklot_xml,
    render_truth_contact_sheet,
    select_candidate_pairs,
    verify_stage_k_gate_v2_record,
)


def test_evenly_spaced_selection_is_grouped_and_deterministic() -> None:
    pairs = [
        ArchivePair(
            stem=f"2012-10-15_10_{index:02d}_00",
            archive_camera="parking2",
            weather="sunny",
            date="2012-10-15",
            image_member=f"{index}.jpg",
            xml_member=f"{index}.xml",
        )
        for index in range(10)
    ]
    group = CandidateGroup("parking2", "sunny", "2012-10-15", 4)

    selected = select_candidate_pairs(pairs, [group])

    assert [pair.stem for pair in selected] == [
        "2012-10-15_10_00_00",
        "2012-10-15_10_03_00",
        "2012-10-15_10_06_00",
        "2012-10-15_10_09_00",
    ]


def test_selection_refuses_insufficient_complete_pairs() -> None:
    pair = ArchivePair(
        stem="2012-10-15_10_00_00",
        archive_camera="parking2",
        weather="sunny",
        date="2012-10-15",
        image_member="one.jpg",
        xml_member="one.xml",
    )
    with pytest.raises(StageKDataGateError, match="only 1 complete pairs"):
        select_candidate_pairs(
            [pair],
            [CandidateGroup("parking2", "sunny", "2012-10-15", 2)],
        )


def test_pklot_xml_conversion_preserves_truth_and_normalizes_points() -> None:
    root = ET.Element("parking", id="pucpr")
    occupied = ET.SubElement(root, "space", id="1", occupied="1")
    occupied_contour = ET.SubElement(occupied, "contour")
    for x, y in ((0, 0), (1280, 0), (1280, 720), (0, 720)):
        ET.SubElement(occupied_contour, "point", x=str(x), y=str(y))
    unknown = ET.SubElement(root, "space", id="2")
    unknown_contour = ET.SubElement(unknown, "contour")
    for x, y in ((10, 10), (20, 10), (20, 20), (10, 20)):
        ET.SubElement(unknown_contour, "point", x=str(x), y=str(y))

    polylines, counts = parse_pklot_xml(
        ET.tostring(root),
        width=1280,
        height=720,
    )

    assert counts == {
        "known_slots": 1,
        "occupied": 1,
        "vacant": 0,
        "unknown": 1,
    }
    assert polylines[0]["occupancy_status"] == "occupied"
    assert polylines[0]["points"][0][2] == [1.0, 1.0]
    assert polylines[1]["occupancy_status"] == "unknown"


def test_candidate_group_maps_official_camera_directory() -> None:
    assert CandidateGroup(
        "parking1a",
        "cloudy",
        "2013-01-19",
        30,
    ).camera == "ufpr04"
    with pytest.raises(StageKDataGateError, match="Unknown PKLot camera"):
        CandidateGroup("other", "sunny", "2020-01-01", 1).camera


def test_truth_contact_sheet_renders_without_predictions(tmp_path: Path) -> None:
    source_root = tmp_path / "source"
    image_path = source_root / "images" / "pucpr" / "frame.jpg"
    image_path.parent.mkdir(parents=True)
    image = np.zeros((72, 128, 3), dtype=np.uint8)
    encoded, payload = cv2.imencode(".jpg", image)
    assert encoded
    image_path.write_bytes(payload.tobytes())
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(
        json.dumps(
            {
                "sample_id": "sample_001",
                "local_path": "images/pucpr/frame.jpg",
                "sample": {
                    "parking_spaces": {
                        "polylines": [
                            {
                                "occupancy_status": "occupied",
                                "points": [
                                    [
                                        [0.1, 0.1],
                                        [0.4, 0.1],
                                        [0.4, 0.4],
                                        [0.1, 0.4],
                                    ]
                                ],
                            }
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )
    output = tmp_path / "truth.png"

    result = render_truth_contact_sheet(
        annotations_path=annotations,
        source_root=source_root,
        output_path=output,
    )

    assert result["images"] == 1
    assert output.is_file()
    assert result["sha256"]


def test_stage_k_gate_v2_verifier_checks_registered_hashes(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    external = tmp_path / "external"
    source.mkdir()
    external.mkdir()
    artifact = source / "gate.txt"
    artifact.write_text("passed-before-predictions\n", encoding="utf-8")
    import hashlib

    record = {
        "record_id": STAGE_K_GATE_V2_RECORD_ID,
        "artifacts": [
            {
                "role": "fixture",
                "root": "source",
                "path": "gate.txt",
                "bytes": artifact.stat().st_size,
                "sha256": hashlib.sha256(
                    artifact.read_bytes()
                ).hexdigest(),
            }
        ],
    }
    record_path = tmp_path / "record.yaml"
    record_path.write_text(
        yaml.safe_dump(record),
        encoding="utf-8",
    )

    result = verify_stage_k_gate_v2_record(
        record_path=record_path,
        source_root=source,
        external_root=external,
    )

    assert result["passed"] is True
    assert result["artifact_count"] == 1
