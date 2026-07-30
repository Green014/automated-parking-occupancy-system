from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import cv2
import numpy as np

from parking_occupancy.image_io import read_image, write_image
from parking_occupancy.sequence_io import evenly_spaced_indices


def load_frame_paths(manifest_path: Path, project_root: Path) -> list[Path]:
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"Manifest has no rows: {manifest_path}")
    return [project_root / row["local_path"] for row in rows]


def make_contact_sheet(
    frame_paths: list[Path],
    output_path: Path,
    fps: float,
    count: int,
    columns: int,
) -> None:
    indices = evenly_spaced_indices(len(frame_paths), count)
    thumbnails: list[np.ndarray] = []
    for frame_index in indices:
        frame = read_image(frame_paths[frame_index])
        thumbnail = cv2.resize(frame, (640, 360), interpolation=cv2.INTER_AREA)
        label = f"frame {frame_index:04d} | {frame_index / fps:6.1f} s"
        cv2.rectangle(thumbnail, (0, 0), (260, 32), (0, 0, 0), -1)
        cv2.putText(
            thumbnail,
            label,
            (10, 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.62,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        thumbnails.append(thumbnail)

    rows = math.ceil(len(thumbnails) / columns)
    sheet = np.zeros((rows * 360, columns * 640, 3), dtype=np.uint8)
    for index, thumbnail in enumerate(thumbnails):
        row, column = divmod(index, columns)
        sheet[row * 360 : (row + 1) * 360, column * 640 : (column + 1) * 640] = (
            thumbnail
        )
    write_image(output_path, sheet)


def write_video(frame_paths: list[Path], output_path: Path, fps: float) -> None:
    first = read_image(frame_paths[0])
    height, width = first.shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"OpenCV could not create video: {output_path}")
    try:
        for frame_path in frame_paths:
            frame = read_image(frame_path)
            if frame.shape[:2] != (height, width):
                raise ValueError(
                    f"Unexpected dimensions {frame.shape[:2]} in {frame_path}"
                )
            writer.write(frame)
    finally:
        writer.release()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a continuous MP4 and contact sheet from an image manifest"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--video-output", required=True)
    parser.add_argument("--contact-sheet-output")
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--contact-count", type=int, default=16)
    parser.add_argument("--contact-columns", type=int, default=4)
    args = parser.parse_args()

    if args.fps <= 0:
        raise ValueError("--fps must be positive")
    frame_paths = load_frame_paths(
        Path(args.manifest),
        Path(args.project_root).resolve(),
    )
    missing = [str(path) for path in frame_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"{len(missing)} manifest frames are missing; first: {missing[0]}"
        )

    write_video(frame_paths, Path(args.video_output), args.fps)
    if args.contact_sheet_output:
        make_contact_sheet(
            frame_paths=frame_paths,
            output_path=Path(args.contact_sheet_output),
            fps=args.fps,
            count=args.contact_count,
            columns=args.contact_columns,
        )
    print(
        f"Wrote {len(frame_paths)} frames at {args.fps:g} FPS "
        f"({len(frame_paths) / args.fps:.1f} seconds)"
    )


if __name__ == "__main__":
    main()
