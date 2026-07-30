"""Build a visual-screening contact sheet and machine-readable video inventory."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sample_video(path: Path, samples: int) -> tuple[dict[str, object], list[np.ndarray]]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"cannot open video: {path}")
    try:
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        if frame_count < samples or fps <= 0 or width <= 0 or height <= 0:
            raise ValueError(f"invalid video properties: {path}")
        indices = np.linspace(0, frame_count - 1, samples, dtype=int).tolist()
        frames: list[np.ndarray] = []
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"cannot decode frame {index}: {path}")
            frames.append(frame)
    finally:
        capture.release()

    return (
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": _sha256(path),
            "width": width,
            "height": height,
            "fps": fps,
            "frame_count": frame_count,
            "duration_seconds": frame_count / fps,
            "sampled_frames": indices,
        },
        frames,
    )


def _tile(frame: np.ndarray, label: str, width: int = 480) -> np.ndarray:
    scale = width / frame.shape[1]
    resized = cv2.resize(
        frame,
        (width, round(frame.shape[0] * scale)),
        interpolation=cv2.INTER_AREA,
    )
    canvas = cv2.copyMakeBorder(
        resized,
        42,
        0,
        0,
        0,
        cv2.BORDER_CONSTANT,
        value=(245, 245, 245),
    )
    cv2.putText(
        canvas,
        label,
        (10, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.58,
        (20, 20, 20),
        1,
        cv2.LINE_AA,
    )
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--video-dir",
        type=Path,
        default=Path("datasets/virat/screening/videos"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/virat_screening"),
    )
    parser.add_argument("--samples", type=int, default=3)
    args = parser.parse_args()
    if args.samples < 2:
        parser.error("--samples must be at least 2")

    videos = sorted(args.video_dir.glob("*.mp4"))
    if not videos:
        parser.error(f"no MP4 files found under {args.video_dir}")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    inventory_path = args.output_dir / "screening_inventory.json"
    sheet_path = args.output_dir / "contact_sheet.jpg"
    for output in (inventory_path, sheet_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite existing output: {output}")

    records: list[dict[str, object]] = []
    rows: list[np.ndarray] = []
    for video in videos:
        record, frames = _sample_video(video, args.samples)
        records.append(record)
        row = np.hstack(
            [
                _tile(frame, f"{video.stem} | frame {frame_index}")
                for frame, frame_index in zip(
                    frames,
                    record["sampled_frames"],
                    strict=True,
                )
            ]
        )
        rows.append(row)
    sheet = np.vstack(rows)

    if not cv2.imwrite(str(sheet_path), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92]):
        raise OSError(f"failed to write {sheet_path}")
    inventory_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "purpose": "candidate visual screening only; not occupancy truth",
                "videos": records,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    print(inventory_path)
    print(sheet_path)


if __name__ == "__main__":
    main()
