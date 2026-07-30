from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export machine preannotations for a selected image manifest"
    )
    parser.add_argument("--coco", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--supercategory",
        help="Optionally retain only categories in this supercategory",
    )
    parser.add_argument(
        "--category-names",
        help="Optional comma-separated allow-list of category names",
    )
    args = parser.parse_args()

    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    image_ids = {int(row["image_id"]) for row in rows}
    categories = coco["categories"]
    if args.supercategory:
        categories = [
            category
            for category in categories
            if category.get("supercategory") == args.supercategory
        ]
    if args.category_names:
        allowed_names = {
            name.strip()
            for name in args.category_names.split(",")
            if name.strip()
        }
        categories = [
            category
            for category in categories
            if category["name"] in allowed_names
        ]
    category_ids = {int(category["id"]) for category in categories}
    images = [
        image for image in coco["images"] if int(image["id"]) in image_ids
    ]
    annotations = [
        annotation
        for annotation in coco["annotations"]
        if int(annotation["image_id"]) in image_ids
        and int(annotation["category_id"]) in category_ids
    ]
    subset = {
        "info": {
            **coco.get("info", {}),
            "annotation_status": "machine_preannotations_require_manual_correction",
            "leakage_control": (
                "train/validation/holdout are disjoint video_source groups"
            ),
        },
        "licenses": coco.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": categories,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(subset, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"Exported {len(images)} images and {len(annotations)} "
        "machine preannotations; manual review is mandatory"
    )


if __name__ == "__main__":
    main()
