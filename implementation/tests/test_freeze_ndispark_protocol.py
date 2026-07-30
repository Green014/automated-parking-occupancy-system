from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
import pytest
import yaml

from scripts.freeze_ndispark_protocol import ProtocolError, freeze_protocol


def _write_image(path: Path, value: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((10, 20, 3), value, dtype=np.uint8)
    encoded, buffer = cv2.imencode(".jpg", image)
    assert encoded
    buffer.tofile(path)


def _write_coco(
    root: Path,
    split: str,
    annotation_name: str,
    file_name: str,
    bbox: list[float],
) -> None:
    _write_image(root / split / "imgs" / file_name, 20 if split == "train" else 40)
    payload = {
        "images": [
            {"id": 1, "width": 20, "height": 10, "file_name": file_name}
        ],
        "annotations": [
            {
                "id": 1,
                "image_id": 1,
                "category_id": 3,
                "bbox": bbox,
            }
        ],
        "categories": [{"id": 3, "name": "car"}],
    }
    (root / split / annotation_name).write_text(
        json.dumps(payload),
        encoding="utf-8",
    )
    (root / split / annotation_name.replace("coco", "dot").replace(".json", ".csv")).write_text(
        "placeholder\n",
        encoding="utf-8",
    )


def _source_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "source"
    _write_coco(
        root,
        "train",
        "train_coco_annotations.json",
        "60_100.jpg",
        [1, 1, 5, 5],
    )
    _write_coco(
        root,
        "validation",
        "val_coco_annotations.json",
        "60_200.jpg",
        [2, 2, 6, 4],
    )
    _write_image(root / "test" / "imgs" / "60_300.jpg", 60)
    with (root / "test" / "ground_truth_test_counting.csv").open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=["imgName", "numVehicles"])
        writer.writeheader()
        writer.writerow({"imgName": "60_300", "numVehicles": 0})
    (root / "README.pdf").write_bytes(b"test readme")
    archive = tmp_path / "ndis_park.zip"
    archive.write_bytes(b"test archive")
    return root, archive


def test_freeze_protocol_writes_relative_manifests_and_keeps_zero_counts(
    tmp_path: Path,
) -> None:
    source_root, archive = _source_tree(tmp_path)
    output_root = tmp_path / "frozen"

    summary = freeze_protocol(
        source_root=source_root,
        archive_path=archive,
        output_root=output_root,
        protocol_id="DPROTO-NDISPARK-ONLY-TEST",
        frozen_at="2026-07-27T00:00:00+08:00",
    )

    assert summary["splits"]["train"]["vehicle_boxes"] == 1
    assert summary["splits"]["test"]["zero_count_images"] == 1
    assert summary["integrity"]["exact_duplicate_sha256_groups_across_splits"] == 0
    test_rows = list(
        csv.DictReader(
            (output_root / "ndispark_test_frozen_20260727.csv").open(
                encoding="utf-8"
            )
        )
    )
    assert test_rows[0]["vehicle_count"] == "0"
    assert test_rows[0]["relative_path"] == "test/imgs/60_300.jpg"
    frozen_yaml = (
        output_root / "ndispark_source_manifest_frozen_20260727.yaml"
    ).read_text(encoding="utf-8")
    assert str(source_root) not in frozen_yaml
    assert yaml.safe_load(frozen_yaml)["status"] == "frozen"


def test_freeze_protocol_rejects_out_of_bounds_bbox(tmp_path: Path) -> None:
    source_root, archive = _source_tree(tmp_path)
    _write_coco(
        source_root,
        "train",
        "train_coco_annotations.json",
        "60_100.jpg",
        [18, 1, 5, 5],
    )

    with pytest.raises(ProtocolError, match="outside"):
        freeze_protocol(
            source_root=source_root,
            archive_path=archive,
            output_root=tmp_path / "frozen",
            protocol_id="DPROTO-NDISPARK-ONLY-TEST",
            frozen_at="2026-07-27T00:00:00+08:00",
        )


def test_freeze_protocol_refuses_to_overwrite_artifacts(tmp_path: Path) -> None:
    source_root, archive = _source_tree(tmp_path)
    output_root = tmp_path / "frozen"
    kwargs = {
        "source_root": source_root,
        "archive_path": archive,
        "output_root": output_root,
        "protocol_id": "DPROTO-NDISPARK-ONLY-TEST",
        "frozen_at": "2026-07-27T00:00:00+08:00",
    }
    freeze_protocol(**kwargs)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        freeze_protocol(**kwargs)


def test_committed_frozen_ndispark_artifacts_match_checksum_registry() -> None:
    repository_root = Path(__file__).resolve().parents[2]
    registry_path = (
        repository_root
        / "implementation"
        / "data"
        / "manifests"
        / "ndispark_only_20260727"
        / "FROZEN_ARTIFACT_CHECKSUMS.yaml"
    )
    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))

    for artifact in registry["artifacts"]:
        path = repository_root / artifact["path"]
        content = path.read_bytes()
        assert len(content) == artifact["bytes"], artifact["path"]
        assert hashlib.sha256(content).hexdigest() == artifact["sha256"], artifact[
            "path"
        ]
