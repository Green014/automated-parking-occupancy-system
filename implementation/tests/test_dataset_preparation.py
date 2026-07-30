from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest
import yaml

from parking_occupancy.dataset_preparation import (
    PreparationError,
    convert_annotations,
    prepare_ndispark_dataset,
    validate_split_membership,
)
from parking_occupancy.image_io import write_image
from scripts.freeze_ndispark_protocol import freeze_protocol


def _write_coco_split(
    source_root: Path,
    split: str,
    annotation_name: str,
    file_name: str,
    image_id: int,
) -> None:
    image = np.full(
        (10, 20, 3),
        20 if split == "train" else 40,
        dtype=np.uint8,
    )
    write_image(source_root / split / "imgs" / file_name, image)
    payload = {
        "images": [
            {
                "id": image_id,
                "width": 20,
                "height": 10,
                "file_name": file_name,
            }
        ],
        "annotations": [
            {
                "id": image_id + 100,
                "image_id": image_id,
                "category_id": 3,
                "bbox": [2, 1, 10, 5],
            }
        ],
        "categories": [{"id": 3, "name": "car"}],
    }
    (source_root / split / annotation_name).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    dot_name = annotation_name.replace("coco", "dot").replace(
        ".json",
        ".csv",
    )
    (source_root / split / dot_name).write_text(
        "placeholder\n",
        encoding="utf-8",
    )


def _frozen_source(tmp_path: Path) -> tuple[Path, Path]:
    source_root = tmp_path / "raw"
    _write_coco_split(
        source_root,
        "train",
        "train_coco_annotations.json",
        "60_100.jpg",
        1,
    )
    _write_coco_split(
        source_root,
        "validation",
        "val_coco_annotations.json",
        "60_200.jpg",
        2,
    )
    write_image(
        source_root / "test" / "imgs" / "60_300.jpg",
        np.full((10, 20, 3), 60, dtype=np.uint8),
    )
    (source_root / "test" / "ground_truth_test_counting.csv").write_text(
        "imgName,numVehicles\n60_300,0\n",
        encoding="utf-8",
    )
    (source_root / "README.pdf").write_bytes(b"readme")
    archive = tmp_path / "ndis_park.zip"
    archive.write_bytes(b"archive")

    manifest_root = tmp_path / "data" / "manifests" / "frozen"
    protocol_id = "DPROTO-NDISPARK-ONLY-TEST"
    freeze_protocol(
        source_root=source_root,
        archive_path=archive,
        output_root=manifest_root,
        protocol_id=protocol_id,
        frozen_at="2026-07-27T00:00:00+08:00",
    )
    config_root = tmp_path / "configs"
    config_root.mkdir()
    protocol_path = config_root / "protocol.yaml"
    protocol_path.write_text(
        yaml.safe_dump(
            {
                "protocol_id": protocol_id,
                "status": "frozen",
                "source": {
                    "source_manifest": (
                        "../data/manifests/frozen/"
                        "ndispark_source_manifest_frozen_20260727.yaml"
                    )
                },
                "class_mapping": {
                    "source": {"category_id": 3, "name": "car"},
                    "target": {"category_id": 0, "name": "vehicle"},
                },
                "splits": {
                    split: {
                        "manifest": (
                            "../data/manifests/frozen/"
                            f"ndispark_{split}_frozen_20260727.csv"
                        )
                    }
                    for split in ("train", "validation", "test")
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return source_root, protocol_path


def test_prepare_ndispark_dataset_generates_one_class_data_and_count_test(
    tmp_path: Path,
) -> None:
    source_root, protocol_path = _frozen_source(tmp_path)
    output_root = tmp_path / "processed"

    summary = prepare_ndispark_dataset(
        protocol_path=protocol_path,
        source_root=source_root,
        output_root=output_root,
    )

    assert summary["splits"]["train"]["written_boxes"] == 1
    assert summary["splits"]["validation"]["written_boxes"] == 1
    assert summary["splits"]["test"]["written_boxes"] == 0
    assert (output_root / "images" / "test" / "60_300.jpg").is_file()
    assert not (output_root / "labels" / "test").exists()
    train_label = (
        output_root / "labels" / "train" / "60_100.txt"
    ).read_text(encoding="utf-8")
    assert train_label.startswith("0 ")
    dataset = yaml.safe_load(
        (output_root / "dataset.yaml").read_text(encoding="utf-8")
    )
    assert dataset["names"] == {0: "vehicle"}
    assert dataset["count_test"] == "images/test"
    rows = list(
        csv.DictReader(
            (output_root / "prepared_manifest.csv").open(encoding="utf-8")
        )
    )
    assert len(rows) == 3
    assert next(row for row in rows if row["source_split"] == "test")[
        "vehicle_count"
    ] == "0"


def test_convert_annotations_maps_class_and_logs_repairs_and_duplicates() -> None:
    annotations = [
        {"id": 1, "category_id": 3, "bbox": [-1, 1, 5, 5]},
        {"id": 2, "category_id": 3, "bbox": [-1, 1, 5, 5]},
        {"id": 3, "category_id": 3, "bbox": [30, 1, 2, 2]},
    ]

    lines, actions, counts = convert_annotations(
        annotations,
        protocol_id="TEST",
        split="train",
        file_name="60_100.jpg",
        image_id=1,
        width=20,
        height=10,
        source_category_id=3,
        target_category_id=0,
    )

    assert len(lines) == 1
    assert lines[0].startswith("0 ")
    assert counts["clipped_boxes"] == 1
    assert counts["excluded_duplicate_boxes"] == 1
    assert counts["excluded_invalid_boxes"] == 1
    assert {row["action"] for row in actions} == {
        "clipped_to_image",
        "excluded_duplicate",
        "excluded_invalid",
    }


def test_convert_annotations_rejects_unfrozen_category_mapping() -> None:
    with pytest.raises(PreparationError, match="expected 3"):
        convert_annotations(
            [{"id": 1, "category_id": 7, "bbox": [1, 1, 5, 5]}],
            protocol_id="TEST",
            split="train",
            file_name="60_100.jpg",
            image_id=1,
            width=20,
            height=10,
            source_category_id=3,
            target_category_id=0,
        )


def test_validate_split_membership_rejects_missing_and_unfrozen_images() -> None:
    with pytest.raises(PreparationError, match="membership differs"):
        validate_split_membership(
            expected_file_names=["a.jpg", "b.jpg"],
            actual_file_names=["a.jpg", "c.jpg"],
            split="train",
        )


def test_prepare_ndispark_dataset_refuses_existing_output(tmp_path: Path) -> None:
    source_root, protocol_path = _frozen_source(tmp_path)
    output_root = tmp_path / "processed"
    output_root.mkdir()

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        prepare_ndispark_dataset(
            protocol_path=protocol_path,
            source_root=source_root,
            output_root=output_root,
        )
