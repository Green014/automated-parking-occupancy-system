"""Render a labeled parking-slot polygon map over one source image."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument(
        "--frame-index",
        type=int,
        default=0,
        help="Zero-based source frame used with --video.",
    )
    parser.add_argument("--slots", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-size", type=int, default=0)
    args = parser.parse_args()

    if args.image is not None:
        image = cv2.imread(str(args.image))
        if image is None:
            raise RuntimeError(f"could not read {args.image}")
    else:
        if args.frame_index < 0:
            parser.error("--frame-index must be non-negative")
        capture = cv2.VideoCapture(str(args.video))
        try:
            if not capture.isOpened():
                raise RuntimeError(f"could not open {args.video}")
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
            if args.frame_index >= frame_count:
                raise ValueError(
                    f"frame index {args.frame_index} exceeds the last source "
                    f"frame {frame_count - 1}"
                )
            capture.set(cv2.CAP_PROP_POS_FRAMES, args.frame_index)
            ok, image = capture.read()
            if not ok:
                raise RuntimeError(
                    f"could not read frame {args.frame_index} from {args.video}"
                )
        finally:
            capture.release()
    payload = json.loads(args.slots.read_text(encoding="utf-8"))

    overlay = image.copy()
    for slot in payload["slots"]:
        contour = np.asarray(slot["points"], dtype=np.int32)
        cv2.fillPoly(overlay, [contour], (0, 255, 255))
    canvas = cv2.addWeighted(overlay, 0.25, image, 0.75, 0)
    if args.grid_size > 0:
        height, width = canvas.shape[:2]
        for x in range(0, width, args.grid_size):
            cv2.line(canvas, (x, 0), (x, height), (255, 0, 255), 1)
            cv2.putText(
                canvas,
                str(x),
                (x + 2, 16),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
        for y in range(0, height, args.grid_size):
            cv2.line(canvas, (0, y), (width, y), (255, 0, 255), 1)
            cv2.putText(
                canvas,
                str(y),
                (2, y + 14),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.38,
                (255, 0, 255),
                1,
                cv2.LINE_AA,
            )
    for slot in payload["slots"]:
        contour = np.asarray(slot["points"], dtype=np.int32)
        cv2.polylines(canvas, [contour], True, (0, 255, 255), 2)
        center = contour.mean(axis=0).astype(int)
        cv2.putText(
            canvas,
            str(slot["id"]),
            (int(center[0]) - 28, int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            str(slot["id"]),
            (int(center[0]) - 28, int(center[1])),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if not cv2.imwrite(str(args.output), canvas):
        raise RuntimeError(f"could not write {args.output}")
    print(f"Wrote {len(payload['slots'])} slots to {args.output}")


if __name__ == "__main__":
    main()
