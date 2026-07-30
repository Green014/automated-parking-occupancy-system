from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml


class ProtocolError(ValueError):
    """Raised when source data cannot satisfy the frozen protocol."""


MANIFEST_FIELDS = [
    "protocol_id",
    "split",
    "role",
    "image_id",
    "file_name",
    "relative_path",
    "camera_id",
    "unix_timestamp",
    "width",
    "height",
    "truth_type",
    "vehicle_box_count",
    "vehicle_count",
    "sha256",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_image(path: Path) -> np.ndarray:
    try:
        encoded = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise ProtocolError(f"Could not read image bytes: {path}") from exc
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise ProtocolError(f"OpenCV could not decode image: {path}")
    return image


def _camera_and_timestamp(file_name: str) -> tuple[str, str]:
    parts = Path(file_name).stem.split("_")
    if len(parts) != 2 or not all(part.isdigit() for part in parts):
        raise ProtocolError(
            f"Expected '<camera>_<unix_timestamp>.jpg', got {file_name!r}"
        )
    return parts[0], parts[1]


def _load_coco_split(
    source_root: Path,
    *,
    split: str,
    annotation_name: str,
    role: str,
    protocol_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_root = source_root / split
    annotation_path = split_root / annotation_name
    if not annotation_path.is_file():
        raise ProtocolError(f"Missing COCO annotation: {annotation_path}")

    data = json.loads(annotation_path.read_text(encoding="utf-8"))
    categories = {int(item["id"]): str(item["name"]) for item in data["categories"]}
    annotations_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for annotation in data["annotations"]:
        category_id = int(annotation["category_id"])
        if category_id != 3 or categories.get(category_id) != "car":
            raise ProtocolError(
                f"{split} annotation {annotation.get('id')} uses unsupported "
                f"category {category_id}:{categories.get(category_id)!r}"
            )
        annotations_by_image[int(annotation["image_id"])].append(annotation)

    rows: list[dict[str, Any]] = []
    bbox_count = 0
    for image in sorted(data["images"], key=lambda item: str(item["file_name"])):
        image_id = int(image["id"])
        file_name = str(image["file_name"])
        width = int(image["width"])
        height = int(image["height"])
        path = split_root / "imgs" / file_name
        if not path.is_file():
            raise ProtocolError(f"Missing image declared by COCO: {path}")

        decoded = _read_image(path)
        actual_height, actual_width = decoded.shape[:2]
        if (actual_width, actual_height) != (width, height):
            raise ProtocolError(
                f"COCO/image size mismatch for {file_name}: "
                f"{width}x{height} vs {actual_width}x{actual_height}"
            )

        image_annotations = annotations_by_image.get(image_id, [])
        for annotation in image_annotations:
            bbox = annotation.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                raise ProtocolError(
                    f"Annotation {annotation.get('id')} has invalid bbox structure"
                )
            x, y, box_width, box_height = (float(value) for value in bbox)
            if (
                x < 0
                or y < 0
                or box_width <= 0
                or box_height <= 0
                or x + box_width > width
                or y + box_height > height
            ):
                raise ProtocolError(
                    f"Annotation {annotation.get('id')} bbox is outside "
                    f"{file_name}: {bbox}"
                )
        bbox_count += len(image_annotations)
        camera_id, unix_timestamp = _camera_and_timestamp(file_name)
        rows.append(
            {
                "protocol_id": protocol_id,
                "split": split,
                "role": role,
                "image_id": image_id,
                "file_name": file_name,
                "relative_path": f"{split}/imgs/{file_name}",
                "camera_id": camera_id,
                "unix_timestamp": unix_timestamp,
                "width": width,
                "height": height,
                "truth_type": "vehicle_boxes",
                "vehicle_box_count": len(image_annotations),
                "vehicle_count": "",
                "sha256": sha256_file(path),
            }
        )

    declared_ids = {int(image["id"]) for image in data["images"]}
    orphan_ids = sorted(set(annotations_by_image) - declared_ids)
    if orphan_ids:
        raise ProtocolError(f"{split} contains orphan annotation image IDs: {orphan_ids}")

    return rows, {
        "images": len(rows),
        "vehicle_boxes": bbox_count,
        "cameras": sorted({row["camera_id"] for row in rows}),
        "annotation_file": annotation_name,
        "annotation_bytes": annotation_path.stat().st_size,
        "annotation_sha256": sha256_file(annotation_path),
    }


def _load_count_test(
    source_root: Path,
    *,
    protocol_id: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split = "test"
    split_root = source_root / split
    annotation_name = "ground_truth_test_counting.csv"
    annotation_path = split_root / annotation_name
    if not annotation_path.is_file():
        raise ProtocolError(f"Missing count annotation: {annotation_path}")

    with annotation_path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    if not source_rows or set(source_rows[0]) != {"imgName", "numVehicles"}:
        raise ProtocolError("Unexpected NDISPark count CSV schema")

    counts: dict[str, int] = {}
    for source_row in source_rows:
        stem = str(source_row["imgName"])
        if stem in counts:
            raise ProtocolError(f"Duplicate count row: {stem}")
        try:
            count = int(source_row["numVehicles"])
        except (TypeError, ValueError) as exc:
            raise ProtocolError(f"Invalid vehicle count for {stem}") from exc
        if count < 0:
            raise ProtocolError(f"Negative vehicle count for {stem}")
        counts[stem] = count

    image_paths = sorted((split_root / "imgs").glob("*.jpg"))
    image_stems = {path.stem for path in image_paths}
    if image_stems != set(counts):
        missing_truth = sorted(image_stems - set(counts))
        missing_images = sorted(set(counts) - image_stems)
        raise ProtocolError(
            "Test image/count membership mismatch: "
            f"missing_truth={missing_truth}, missing_images={missing_images}"
        )

    rows: list[dict[str, Any]] = []
    for image_id, path in enumerate(image_paths):
        decoded = _read_image(path)
        height, width = decoded.shape[:2]
        camera_id, unix_timestamp = _camera_and_timestamp(path.name)
        rows.append(
            {
                "protocol_id": protocol_id,
                "split": split,
                "role": "count_only_test",
                "image_id": image_id,
                "file_name": path.name,
                "relative_path": f"test/imgs/{path.name}",
                "camera_id": camera_id,
                "unix_timestamp": unix_timestamp,
                "width": width,
                "height": height,
                "truth_type": "vehicle_count",
                "vehicle_box_count": "",
                "vehicle_count": counts[path.stem],
                "sha256": sha256_file(path),
            }
        )

    return rows, {
        "images": len(rows),
        "vehicle_boxes": None,
        "vehicle_count_sum": sum(counts.values()),
        "zero_count_images": sum(value == 0 for value in counts.values()),
        "cameras": sorted({row["camera_id"] for row in rows}),
        "annotation_file": annotation_name,
        "annotation_bytes": annotation_path.stat().st_size,
        "annotation_sha256": sha256_file(annotation_path),
    }


def _manifest_bytes(rows: list[dict[str, Any]]) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=MANIFEST_FIELDS,
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def _sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def freeze_protocol(
    *,
    source_root: Path,
    archive_path: Path,
    output_root: Path,
    protocol_id: str,
    frozen_at: str,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    archive_path = archive_path.resolve()
    output_root = output_root.resolve()
    if not source_root.is_dir():
        raise ProtocolError(f"Source root does not exist: {source_root}")
    if not archive_path.is_file():
        raise ProtocolError(f"Source archive does not exist: {archive_path}")

    split_specs = [
        ("train", "train_coco_annotations.json", "fine_tuning_train"),
        (
            "validation",
            "val_coco_annotations.json",
            "consumed_development_validation",
        ),
    ]
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    split_summaries: dict[str, Any] = {}
    for split, annotation_name, role in split_specs:
        rows, summary = _load_coco_split(
            source_root,
            split=split,
            annotation_name=annotation_name,
            role=role,
            protocol_id=protocol_id,
        )
        rows_by_split[split] = rows
        split_summaries[split] = summary
    test_rows, test_summary = _load_count_test(
        source_root,
        protocol_id=protocol_id,
    )
    rows_by_split["test"] = test_rows
    split_summaries["test"] = test_summary

    hash_membership: dict[str, list[str]] = defaultdict(list)
    for split, rows in rows_by_split.items():
        for row in rows:
            hash_membership[str(row["sha256"])].append(
                f"{split}/{row['file_name']}"
            )
    duplicates = sorted(
        members
        for members in hash_membership.values()
        if len({member.split("/", 1)[0] for member in members}) > 1
    )
    if duplicates:
        raise ProtocolError(f"Exact duplicate images cross splits: {duplicates}")

    source_files: list[dict[str, Any]] = []
    for relative_path in [
        "README.pdf",
        "train/train_coco_annotations.json",
        "train/train_dot_annotations.csv",
        "validation/val_coco_annotations.json",
        "validation/val_dot_annotations.csv",
        "test/ground_truth_test_counting.csv",
    ]:
        path = source_root / relative_path
        if not path.is_file():
            raise ProtocolError(f"Missing expected source file: {path}")
        source_files.append(
            {
                "relative_path": relative_path,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )

    manifest_payloads = {
        f"ndispark_{split}_frozen_20260727.csv": _manifest_bytes(rows)
        for split, rows in rows_by_split.items()
    }
    summary_name = "ndispark_source_manifest_frozen_20260727.yaml"
    intended_paths = [output_root / name for name in manifest_payloads]
    intended_paths.append(output_root / summary_name)
    existing = [path for path in intended_paths if path.exists()]
    if existing:
        raise FileExistsError(
            "Refusing to overwrite frozen artifact(s): "
            + ", ".join(str(path) for path in existing)
        )

    manifest_summaries = {}
    for name, content in manifest_payloads.items():
        split = name.split("_")[1]
        manifest_summaries[split] = {
            "file": name,
            "rows": len(rows_by_split[split]),
            "bytes": len(content),
            "sha256": _sha256_bytes(content),
        }

    camera_membership: dict[str, list[str]] = defaultdict(list)
    for split, rows in rows_by_split.items():
        for camera_id in {str(row["camera_id"]) for row in rows}:
            camera_membership[camera_id].append(split)

    summary = {
        "schema_version": 1,
        "protocol_id": protocol_id,
        "frozen_at": frozen_at,
        "status": "frozen",
        "source_contract": {
            "dataset": "NDISPark",
            "version": "1.0",
            "doi": "10.5281/zenodo.6560823",
            "source_root": "--source-root or PARKING_DATA_ROOT",
            "committed_absolute_paths": False,
            "archive": {
                "name": archive_path.name,
                "bytes": archive_path.stat().st_size,
                "sha256": sha256_file(archive_path),
            },
            "files": source_files,
        },
        "class_mapping": {
            "source_category_id": 3,
            "source_category_name": "car",
            "target_category_id": 0,
            "target_category_name": "vehicle",
            "mapped_source_categories": ["car"],
        },
        "splits": split_summaries,
        "manifests": manifest_summaries,
        "integrity": {
            "missing_images": 0,
            "decode_failures": 0,
            "bbox_violations": 0,
            "exact_duplicate_sha256_groups_across_splits": 0,
        },
        "leakage_audit": {
            "camera_independent": False,
            "camera_membership": {
                camera: sorted(splits)
                for camera, splits in sorted(camera_membership.items())
            },
            "finding": (
                "The official split is condition-based rather than camera-independent; "
                "six cameras span train, validation, and test."
            ),
            "controls": [
                "Preserve official split membership.",
                "Treat validation as consumed development validation.",
                "Do not select thresholds, epochs, or augmentations from test counts.",
                "Report test count metrics per camera.",
                "Do not report detector mAP, box precision, or box recall on test.",
            ],
        },
    }

    output_root.mkdir(parents=True, exist_ok=True)
    for name, content in manifest_payloads.items():
        (output_root / name).write_bytes(content)
    (output_root / summary_name).write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the official NDISPark-only source protocol."
    )
    parser.add_argument(
        "--source-root",
        help=(
            "Extracted NDISPark root containing train/, validation/, and test/. "
            "Defaults to PARKING_DATA_ROOT."
        ),
    )
    parser.add_argument("--archive-path", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--protocol-id", required=True)
    parser.add_argument("--frozen-at", required=True)
    args = parser.parse_args()

    source_value = args.source_root or os.environ.get("PARKING_DATA_ROOT")
    if not source_value:
        parser.error("--source-root or PARKING_DATA_ROOT is required")
    summary = freeze_protocol(
        source_root=Path(source_value),
        archive_path=Path(args.archive_path),
        output_root=Path(args.output_root),
        protocol_id=args.protocol_id,
        frozen_at=args.frozen_at,
    )
    print(yaml.safe_dump(summary, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
