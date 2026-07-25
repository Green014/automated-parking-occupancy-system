from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
from collections import defaultdict
from pathlib import Path

import yaml

from parking_occupancy.coco_conversion import coco_bbox_to_yolo

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a COCO detection split to Ultralytics YOLO format"
    )
    parser.add_argument("--coco", required=True)
    parser.add_argument("--images", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--category-names", default="car")
    parser.add_argument(
        "--model-weights",
        help=(
            "Optional Ultralytics weights whose class indices/names must be "
            "preserved, e.g. yolov8n.pt"
        ),
    )
    args = parser.parse_args()

    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    allowed = {
        name.strip()
        for name in args.category_names.split(",")
        if name.strip()
    }
    categories = [
        category
        for category in coco["categories"]
        if category["name"] in allowed
    ]
    if not categories:
        raise ValueError(f"No COCO categories match {sorted(allowed)}")
    categories.sort(key=lambda category: int(category["id"]))
    if args.model_weights:
        from ultralytics import YOLO

        model_names = {
            int(index): str(name)
            for index, name in YOLO(args.model_weights).names.items()
        }
        model_index_by_name = {
            name: index for index, name in model_names.items()
        }
        missing_names = [
            category["name"]
            for category in categories
            if category["name"] not in model_index_by_name
        ]
        if missing_names:
            raise ValueError(
                f"Categories are absent from model names: {missing_names}"
            )
        class_by_category = {
            int(category["id"]): model_index_by_name[category["name"]]
            for category in categories
        }
        output_names = model_names
    else:
        class_by_category = {
            int(category["id"]): class_index
            for class_index, category in enumerate(categories)
        }
        output_names = {
            index: category["name"]
            for index, category in enumerate(categories)
        }

    annotations_by_image: dict[int, list[dict]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if int(annotation["category_id"]) in class_by_category:
            annotations_by_image[int(annotation["image_id"])].append(annotation)

    output_root = Path(args.output_root).resolve()
    image_output = output_root / "images" / args.split
    label_output = output_root / "labels" / args.split
    image_output.mkdir(parents=True, exist_ok=True)
    label_output.mkdir(parents=True, exist_ok=True)
    image_source = Path(args.images)
    manifest_rows = []
    label_count = 0
    for image in coco["images"]:
        source = image_source / image["file_name"]
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = image_output / Path(image["file_name"]).name
        if not destination.is_file():
            shutil.copy2(source, destination)
        lines = []
        for annotation in annotations_by_image[int(image["id"])]:
            try:
                center_x, center_y, width, height = coco_bbox_to_yolo(
                    annotation["bbox"],
                    int(image["width"]),
                    int(image["height"]),
                )
            except ValueError:
                continue
            lines.append(
                f"{class_by_category[int(annotation['category_id'])]} "
                f"{center_x:.8f} {center_y:.8f} {width:.8f} {height:.8f}"
            )
        (label_output / f"{destination.stem}.txt").write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        label_count += len(lines)
        manifest_rows.append(
            {
                "image_id": image["id"],
                "file_name": destination.name,
                "width": image["width"],
                "height": image["height"],
                "labels": len(lines),
                "sha256": _sha256(destination),
            }
        )

    dataset_yaml = {
        "path": str(output_root),
        "train": f"images/{args.split}",
        "val": f"images/{args.split}",
        "names": output_names,
    }
    (output_root / "dataset.yaml").write_text(
        yaml.safe_dump(dataset_yaml, sort_keys=False),
        encoding="utf-8",
    )
    with (output_root / "manifest.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)
    print(
        f"Converted {len(manifest_rows)} images and {label_count} boxes "
        f"for classes {[category['name'] for category in categories]}"
    )


if __name__ == "__main__":
    main()
