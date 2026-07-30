from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from urllib.parse import quote

BASE_URL = (
    "https://huggingface.co/datasets/shivam11/"
    "grand-bassin-traffic/resolve/main/images"
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract one Grand Bassin segment and its COCO annotations"
    )
    parser.add_argument("--coco", required=True)
    parser.add_argument("--video-source", required=True)
    parser.add_argument("--manifest-output", required=True)
    parser.add_argument("--coco-output", required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    images = sorted(
        (
            image
            for image in data["images"]
            if image.get("video_source") == args.video_source
        ),
        key=lambda image: int(image["frame_index"]),
    )
    if not images:
        raise ValueError(f"No frames found for {args.video_source}")
    image_ids = {int(image["id"]) for image in images}
    annotations = [
        annotation
        for annotation in data["annotations"]
        if int(annotation["image_id"]) in image_ids
    ]
    annotations_per_image: dict[int, int] = {image_id: 0 for image_id in image_ids}
    for annotation in annotations:
        annotations_per_image[int(annotation["image_id"])] += 1

    manifest_path = Path(args.manifest_output)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "sequence_id",
        "split",
        "image_id",
        "frame_index",
        "timestamp_s",
        "camera_view",
        "annotations",
        "remote_path",
        "download_url",
        "local_path",
    ]
    sequence_id = Path(args.video_source).stem
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for output_index, image in enumerate(images):
            remote_path = str(image["file_name"])
            local_path = f"data/raw/grand_bassin/images/{remote_path}"
            writer.writerow(
                {
                    "sequence_id": sequence_id,
                    "split": "development",
                    "image_id": image["id"],
                    "frame_index": image["frame_index"],
                    "timestamp_s": f"{output_index / 2.0:.3f}",
                    "camera_view": image.get("camera_view", ""),
                    "annotations": annotations_per_image[int(image["id"])],
                    "remote_path": remote_path,
                    "download_url": (
                        f"{BASE_URL}/{quote(remote_path, safe='/')}?download=true"
                    ),
                    "local_path": local_path,
                }
            )

    subset = {
        "info": {
            **data.get("info", {}),
            "subset_note": (
                "Development sequence extracted without changing the original "
                "machine-generated annotations."
            ),
            "video_source": args.video_source,
            "sampled_fps": 2.0,
        },
        "licenses": data.get("licenses", []),
        "images": images,
        "annotations": annotations,
        "categories": data["categories"],
    }
    coco_output = Path(args.coco_output)
    coco_output.parent.mkdir(parents=True, exist_ok=True)
    coco_output.write_text(json.dumps(subset) + "\n", encoding="utf-8")
    print(
        f"Selected {len(images)} frames and {len(annotations)} annotations "
        f"from {args.video_source}"
    )


if __name__ == "__main__":
    main()
