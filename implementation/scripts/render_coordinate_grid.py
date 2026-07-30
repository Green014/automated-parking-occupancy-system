from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from parking_occupancy.image_io import read_image, write_image


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Overlay an image-coordinate grid for polygon annotation"
    )
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--spacing", type=int, default=50)
    parser.add_argument(
        "--crop",
        nargs=4,
        type=int,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
    )
    args = parser.parse_args()

    image = read_image(Path(args.input))
    source_height, source_width = image.shape[:2]
    crop_x, crop_y = 0, 0
    if args.crop:
        crop_x, crop_y, crop_width, crop_height = args.crop
        image = image[
            crop_y : min(crop_y + crop_height, source_height),
            crop_x : min(crop_x + crop_width, source_width),
        ].copy()
    height, width = image.shape[:2]
    first_x = ((crop_x + args.spacing - 1) // args.spacing) * args.spacing
    for global_x in range(first_x, crop_x + width, args.spacing):
        x = global_x - crop_x
        cv2.line(image, (x, 0), (x, height - 1), (255, 255, 0), 1)
        cv2.putText(
            image,
            str(global_x),
            (x + 2, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(x),
            (x + 2, 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    first_y = ((crop_y + args.spacing - 1) // args.spacing) * args.spacing
    for global_y in range(first_y, crop_y + height, args.spacing):
        y = global_y - crop_y
        cv2.line(image, (0, y), (width - 1, y), (255, 255, 0), 1)
        cv2.putText(
            image,
            str(global_y),
            (2, min(y + 17, height - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(global_y),
            (2, min(y + 17, height - 2)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 0),
            1,
            cv2.LINE_AA,
        )
    write_image(Path(args.output), image)


if __name__ == "__main__":
    main()
