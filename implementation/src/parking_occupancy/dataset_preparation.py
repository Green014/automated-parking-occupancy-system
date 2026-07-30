from __future__ import annotations

import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from collections.abc import Iterable, Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from .coco_conversion import validate_coco_bbox
from .image_io import read_image


class PreparationError(ValueError):
    """Raised when inputs violate the frozen dataset protocol."""


PREPARED_MANIFEST_FIELDS = [
    "protocol_id",
    "source_split",
    "output_split",
    "role",
    "image_id",
    "file_name",
    "camera_id",
    "unix_timestamp",
    "width",
    "height",
    "truth_type",
    "vehicle_boxes",
    "vehicle_count",
    "source_relative_path",
    "output_relative_path",
    "source_sha256",
    "label_relative_path",
    "label_sha256",
]

ACTION_FIELDS = [
    "protocol_id",
    "split",
    "file_name",
    "image_id",
    "annotation_id",
    "action",
    "reason",
    "source_bbox_xywh",
    "output_bbox_xyxy",
]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _csv_bytes(
    rows: Iterable[dict[str, Any]],
    fieldnames: Sequence[str],
) -> bytes:
    from io import StringIO

    stream = StringIO(newline="")
    writer = csv.DictWriter(
        stream,
        fieldnames=list(fieldnames),
        lineterminator="\n",
    )
    writer.writeheader()
    writer.writerows(rows)
    return stream.getvalue().encode("utf-8")


def validate_split_membership(
    *,
    expected_file_names: Sequence[str],
    actual_file_names: Sequence[str],
    split: str,
) -> None:
    expected_counts = Counter(expected_file_names)
    actual_counts = Counter(actual_file_names)
    duplicate_expected = sorted(
        name for name, count in expected_counts.items() if count > 1
    )
    duplicate_actual = sorted(
        name for name, count in actual_counts.items() if count > 1
    )
    if duplicate_expected or duplicate_actual:
        raise PreparationError(
            f"{split} contains duplicate file names: "
            f"manifest={duplicate_expected}, source={duplicate_actual}"
        )
    if set(expected_counts) != set(actual_counts):
        raise PreparationError(
            f"{split} membership differs from frozen manifest: "
            f"missing_from_source={sorted(set(expected_counts) - set(actual_counts))}, "
            f"unfrozen_source_images={sorted(set(actual_counts) - set(expected_counts))}"
        )


def convert_annotations(
    annotations: Sequence[dict[str, Any]],
    *,
    protocol_id: str,
    split: str,
    file_name: str,
    image_id: int,
    width: int,
    height: int,
    source_category_id: int,
    target_category_id: int,
) -> tuple[list[str], list[dict[str, Any]], dict[str, int]]:
    """Convert one image's annotations and retain every repair/exclusion."""

    lines: list[str] = []
    actions: list[dict[str, Any]] = []
    counts = {
        "source_annotations": 0,
        "written_boxes": 0,
        "clipped_boxes": 0,
        "excluded_invalid_boxes": 0,
        "excluded_duplicate_boxes": 0,
    }
    seen_boxes: dict[tuple[int, str], Any] = {}
    for annotation in sorted(
        annotations,
        key=lambda item: (int(item.get("id", -1)), str(item.get("bbox"))),
    ):
        counts["source_annotations"] += 1
        annotation_id = annotation.get("id")
        category_id = int(annotation.get("category_id", -1))
        if category_id != source_category_id:
            raise PreparationError(
                f"{split}/{file_name} annotation {annotation_id} uses "
                f"category {category_id}, expected {source_category_id}"
            )
        bbox = annotation.get("bbox")
        duplicate_key = (
            category_id,
            json.dumps(bbox, sort_keys=True, separators=(",", ":")),
        )
        if duplicate_key in seen_boxes:
            counts["excluded_duplicate_boxes"] += 1
            actions.append(
                {
                    "protocol_id": protocol_id,
                    "split": split,
                    "file_name": file_name,
                    "image_id": image_id,
                    "annotation_id": annotation_id,
                    "action": "excluded_duplicate",
                    "reason": (
                        "exact duplicate of annotation "
                        f"{seen_boxes[duplicate_key]}"
                    ),
                    "source_bbox_xywh": json.dumps(bbox),
                    "output_bbox_xyxy": "",
                }
            )
            continue
        seen_boxes[duplicate_key] = annotation_id

        try:
            conversion = validate_coco_bbox(bbox, width, height)
        except (TypeError, ValueError) as exc:
            counts["excluded_invalid_boxes"] += 1
            actions.append(
                {
                    "protocol_id": protocol_id,
                    "split": split,
                    "file_name": file_name,
                    "image_id": image_id,
                    "annotation_id": annotation_id,
                    "action": "excluded_invalid",
                    "reason": str(exc),
                    "source_bbox_xywh": json.dumps(bbox),
                    "output_bbox_xyxy": "",
                }
            )
            continue

        if conversion.clipped:
            counts["clipped_boxes"] += 1
            actions.append(
                {
                    "protocol_id": protocol_id,
                    "split": split,
                    "file_name": file_name,
                    "image_id": image_id,
                    "annotation_id": annotation_id,
                    "action": "clipped_to_image",
                    "reason": "source bbox crossed an image boundary",
                    "source_bbox_xywh": json.dumps(
                        conversion.original_xywh
                    ),
                    "output_bbox_xyxy": json.dumps(
                        conversion.clipped_xyxy
                    ),
                }
            )

        center_x, center_y, box_width, box_height = conversion.yolo_xywh
        lines.append(
            f"{target_category_id} {center_x:.8f} {center_y:.8f} "
            f"{box_width:.8f} {box_height:.8f}"
        )
        counts["written_boxes"] += 1
    return lines, actions, counts


def _resolve_protocol_path(protocol_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (protocol_path.parent / path).resolve()


def _load_frozen_inputs(
    protocol_path: Path,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, list[dict[str, str]]]]:
    protocol = yaml.safe_load(protocol_path.read_text(encoding="utf-8"))
    if protocol.get("status") != "frozen":
        raise PreparationError("Dataset protocol is not frozen")
    protocol_id = str(protocol["protocol_id"])

    source_manifest_path = _resolve_protocol_path(
        protocol_path,
        str(protocol["source"]["source_manifest"]),
    )
    source_manifest = yaml.safe_load(
        source_manifest_path.read_text(encoding="utf-8")
    )
    if source_manifest.get("protocol_id") != protocol_id:
        raise PreparationError("Source manifest protocol ID does not match")
    if source_manifest.get("status") != "frozen":
        raise PreparationError("Source manifest is not frozen")

    rows_by_split: dict[str, list[dict[str, str]]] = {}
    for split in ("train", "validation", "test"):
        manifest_path = _resolve_protocol_path(
            protocol_path,
            str(protocol["splits"][split]["manifest"]),
        )
        expected = source_manifest["manifests"][split]
        if manifest_path.stat().st_size != int(expected["bytes"]):
            raise PreparationError(f"{split} frozen manifest size mismatch")
        if sha256_file(manifest_path) != str(expected["sha256"]):
            raise PreparationError(f"{split} frozen manifest SHA-256 mismatch")
        rows = _read_csv(manifest_path)
        if len(rows) != int(expected["rows"]):
            raise PreparationError(f"{split} frozen manifest row mismatch")
        if any(row.get("protocol_id") != protocol_id for row in rows):
            raise PreparationError(f"{split} contains a foreign protocol ID")
        rows_by_split[split] = rows
    return protocol, source_manifest, rows_by_split


def prepare_ndispark_dataset(
    *,
    protocol_path: Path,
    source_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    protocol_path = protocol_path.resolve()
    source_root = source_root.resolve()
    output_root = output_root.resolve()
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite generated dataset: {output_root}"
        )
    if not source_root.is_dir():
        raise PreparationError(f"Source root does not exist: {source_root}")

    protocol, source_manifest, rows_by_split = _load_frozen_inputs(protocol_path)
    protocol_id = str(protocol["protocol_id"])
    mapping = protocol["class_mapping"]
    source_category_id = int(mapping["source"]["category_id"])
    target_category_id = int(mapping["target"]["category_id"])
    if target_category_id != 0 or mapping["target"]["name"] != "vehicle":
        raise PreparationError("Frozen target mapping must be class 0 vehicle")

    hash_membership: dict[str, list[str]] = defaultdict(list)
    for split, rows in rows_by_split.items():
        for row in rows:
            hash_membership[row["sha256"]].append(
                f"{split}/{row['file_name']}"
            )
    duplicate_image_groups = [
        members for members in hash_membership.values() if len(members) > 1
    ]
    if duplicate_image_groups:
        raise PreparationError(
            f"Frozen manifests contain duplicate images: {duplicate_image_groups}"
        )

    planned_images: list[dict[str, Any]] = []
    all_actions: list[dict[str, Any]] = []
    aggregate_box_counts = Counter()
    split_summaries: dict[str, dict[str, Any]] = {}

    for source_split, output_split in (
        ("train", "train"),
        ("validation", "val"),
        ("test", "test"),
    ):
        frozen_rows = rows_by_split[source_split]
        frozen_by_name = {row["file_name"]: row for row in frozen_rows}
        annotation_by_image: dict[int, list[dict[str, Any]]] = defaultdict(list)
        source_images: list[dict[str, Any]] = []
        annotation_path: Path | None = None
        if source_split != "test":
            annotation_name = str(
                source_manifest["splits"][source_split]["annotation_file"]
            )
            annotation_path = source_root / source_split / annotation_name
            if sha256_file(annotation_path) != str(
                source_manifest["splits"][source_split]["annotation_sha256"]
            ):
                raise PreparationError(
                    f"{source_split} annotation SHA-256 mismatch"
                )
            coco = json.loads(annotation_path.read_text(encoding="utf-8"))
            source_images = list(coco["images"])
            for annotation in coco["annotations"]:
                annotation_by_image[int(annotation["image_id"])].append(annotation)
            validate_split_membership(
                expected_file_names=list(frozen_by_name),
                actual_file_names=[
                    str(image["file_name"]) for image in source_images
                ],
                split=source_split,
            )
        else:
            source_images = [
                {
                    "id": int(row["image_id"]),
                    "file_name": row["file_name"],
                    "width": int(row["width"]),
                    "height": int(row["height"]),
                }
                for row in frozen_rows
            ]

        split_box_counts = Counter()
        for source_image in sorted(
            source_images,
            key=lambda item: str(item["file_name"]),
        ):
            file_name = str(source_image["file_name"])
            frozen = frozen_by_name[file_name]
            source_path = source_root / frozen["relative_path"]
            if not source_path.is_file():
                raise PreparationError(f"Missing source image: {source_path}")
            source_sha256 = sha256_file(source_path)
            if source_sha256 != frozen["sha256"]:
                raise PreparationError(
                    f"Source image SHA-256 mismatch: {source_split}/{file_name}"
                )
            image = read_image(source_path)
            actual_height, actual_width = image.shape[:2]
            width = int(source_image["width"])
            height = int(source_image["height"])
            if (actual_width, actual_height) != (width, height):
                raise PreparationError(
                    f"Image dimensions differ for {source_split}/{file_name}"
                )
            if (width, height) != (
                int(frozen["width"]),
                int(frozen["height"]),
            ):
                raise PreparationError(
                    f"Frozen dimensions differ for {source_split}/{file_name}"
                )

            label_lines: list[str] = []
            action_rows: list[dict[str, Any]] = []
            box_counts = Counter()
            if source_split != "test":
                label_lines, action_rows, raw_counts = convert_annotations(
                    annotation_by_image[int(source_image["id"])],
                    protocol_id=protocol_id,
                    split=source_split,
                    file_name=file_name,
                    image_id=int(source_image["id"]),
                    width=width,
                    height=height,
                    source_category_id=source_category_id,
                    target_category_id=target_category_id,
                )
                box_counts.update(raw_counts)
                if int(frozen["vehicle_box_count"]) != raw_counts[
                    "source_annotations"
                ]:
                    raise PreparationError(
                        f"Frozen box count differs for {source_split}/{file_name}"
                    )
            all_actions.extend(action_rows)
            aggregate_box_counts.update(box_counts)
            split_box_counts.update(box_counts)
            planned_images.append(
                {
                    "source_split": source_split,
                    "output_split": output_split,
                    "source_image": source_path,
                    "source_sha256": source_sha256,
                    "image_id": int(source_image["id"]),
                    "file_name": file_name,
                    "camera_id": frozen["camera_id"],
                    "unix_timestamp": frozen["unix_timestamp"],
                    "width": width,
                    "height": height,
                    "role": frozen["role"],
                    "truth_type": frozen["truth_type"],
                    "vehicle_count": frozen["vehicle_count"],
                    "label_lines": label_lines,
                }
            )

        split_summaries[source_split] = {
            "images": len(source_images),
            "source_annotations": split_box_counts["source_annotations"],
            "written_boxes": split_box_counts["written_boxes"],
            "clipped_boxes": split_box_counts["clipped_boxes"],
            "excluded_invalid_boxes": split_box_counts[
                "excluded_invalid_boxes"
            ],
            "excluded_duplicate_boxes": split_box_counts[
                "excluded_duplicate_boxes"
            ],
        }

    output_root.mkdir(parents=True)
    prepared_rows: list[dict[str, Any]] = []
    for planned in planned_images:
        output_split = planned["output_split"]
        destination = output_root / "images" / output_split / planned["file_name"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(planned["source_image"], destination)
        if sha256_file(destination) != planned["source_sha256"]:
            raise RuntimeError(f"Copied image hash mismatch: {destination}")

        label_relative_path = ""
        label_sha256 = ""
        if planned["source_split"] != "test":
            label_path = (
                output_root
                / "labels"
                / output_split
                / f"{Path(planned['file_name']).stem}.txt"
            )
            label_path.parent.mkdir(parents=True, exist_ok=True)
            label_content = (
                "\n".join(planned["label_lines"])
                + ("\n" if planned["label_lines"] else "")
            ).encode("utf-8")
            label_path.write_bytes(label_content)
            label_relative_path = label_path.relative_to(output_root).as_posix()
            label_sha256 = sha256_bytes(label_content)

        prepared_rows.append(
            {
                "protocol_id": protocol_id,
                "source_split": planned["source_split"],
                "output_split": output_split,
                "role": planned["role"],
                "image_id": planned["image_id"],
                "file_name": planned["file_name"],
                "camera_id": planned["camera_id"],
                "unix_timestamp": planned["unix_timestamp"],
                "width": planned["width"],
                "height": planned["height"],
                "truth_type": planned["truth_type"],
                "vehicle_boxes": len(planned["label_lines"])
                if planned["source_split"] != "test"
                else "",
                "vehicle_count": planned["vehicle_count"],
                "source_relative_path": (
                    f"{planned['source_split']}/imgs/{planned['file_name']}"
                ),
                "output_relative_path": destination.relative_to(
                    output_root
                ).as_posix(),
                "source_sha256": planned["source_sha256"],
                "label_relative_path": label_relative_path,
                "label_sha256": label_sha256,
            }
        )

    dataset_yaml = {
        "path": str(output_root),
        "train": "images/train",
        "val": "images/val",
        "count_test": "images/test",
        "names": {0: "vehicle"},
    }
    dataset_path = output_root / "dataset.yaml"
    dataset_path.write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    class_mapping_path = output_root / "class_mapping.yaml"
    class_mapping_path.write_text(
        yaml.safe_dump(
            {
                "protocol_id": protocol_id,
                "source": {"3": "car"},
                "target": {"0": "vehicle"},
                "mapping": {"3": 0},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
        newline="\n",
    )
    manifest_path = output_root / "prepared_manifest.csv"
    manifest_path.write_bytes(
        _csv_bytes(prepared_rows, PREPARED_MANIFEST_FIELDS)
    )
    actions_path = output_root / "annotation_actions.csv"
    actions_path.write_bytes(_csv_bytes(all_actions, ACTION_FIELDS))

    summary = {
        "schema_version": 1,
        "preparation_id": "DPREP-NDISPARK-ONLY-20260727-01",
        "protocol_id": protocol_id,
        "generated_at": datetime.now().astimezone().isoformat(),
        "status": "complete",
        "path_contract": {
            "source": "--source-root or PARKING_DATA_ROOT",
            "output": "--output-root",
            "committed_absolute_paths": False,
            "note": (
                "The ignored generated dataset.yaml contains the runtime "
                "output path required by Ultralytics."
            ),
        },
        "class_mapping": {
            "source_category_id": source_category_id,
            "source_name": mapping["source"]["name"],
            "target_category_id": target_category_id,
            "target_name": mapping["target"]["name"],
        },
        "splits": split_summaries,
        "integrity": {
            "frozen_source_images_verified": len(planned_images),
            "duplicate_image_groups": 0,
            **dict(aggregate_box_counts),
        },
        "artifacts": {
            "dataset_yaml": {
                "path": "dataset.yaml",
                "bytes": dataset_path.stat().st_size,
                "sha256": sha256_file(dataset_path),
            },
            "class_mapping": {
                "path": "class_mapping.yaml",
                "bytes": class_mapping_path.stat().st_size,
                "sha256": sha256_file(class_mapping_path),
            },
            "prepared_manifest": {
                "path": "prepared_manifest.csv",
                "bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
            },
            "annotation_actions": {
                "path": "annotation_actions.csv",
                "bytes": actions_path.stat().st_size,
                "sha256": sha256_file(actions_path),
            },
        },
    }
    summary_path = output_root / "preparation_summary.yaml"
    summary_path.write_text(
        yaml.safe_dump(summary, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
        newline="\n",
    )
    return summary
