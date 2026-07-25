from __future__ import annotations

import argparse
from pathlib import Path

import cv2


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat an image into a smoke video")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--frames", type=int, default=30)
    parser.add_argument("--fps", type=float, default=10.0)
    args = parser.parse_args()

    image = cv2.imread(args.input)
    if image is None:
        raise RuntimeError(f"Could not read image: {args.input}")
    height, width = image.shape[:2]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create video: {output}")
    try:
        for _ in range(args.frames):
            writer.write(image)
    finally:
        writer.release()
    print(f"Wrote {args.frames} frames to {output}")


if __name__ == "__main__":
    main()
