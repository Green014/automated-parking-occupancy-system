from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from parking_occupancy.image_io import read_image, write_image


def _crop_with_padding(
    image: np.ndarray,
    center_x: int,
    center_y: int,
    width: int,
    height: int,
) -> np.ndarray:
    padded = cv2.copyMakeBorder(
        image,
        height,
        height,
        width,
        width,
        cv2.BORDER_CONSTANT,
    )
    x = center_x + width
    y = center_y + height
    return padded[
        y - height // 2 : y + (height + 1) // 2,
        x - width // 2 : x + (width + 1) // 2,
    ]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render crops around candidate state transitions"
    )
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--candidate-ids", nargs="*")
    parser.add_argument("--offsets", default="-30,-10,-1,1,10,30")
    parser.add_argument("--crop-width", type=int, default=240)
    parser.add_argument("--crop-height", type=int, default=160)
    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()
    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    with Path(args.candidates).open(newline="", encoding="utf-8") as handle:
        candidates = list(csv.DictReader(handle))
    if args.candidate_ids:
        selected = set(args.candidate_ids)
        candidates = [
            candidate
            for candidate in candidates
            if candidate["candidate_id"] in selected
        ]
    offsets = [int(value) for value in args.offsets.split(",")]

    rows: list[np.ndarray] = []
    for candidate in candidates:
        transitions = [
            int(value)
            for value in candidate["transition_frames"].split(";")
            if value
        ]
        for transition in transitions:
            indices = [
                min(max(transition + offset, 0), len(manifest) - 1)
                for offset in offsets
            ]
            tiles: list[np.ndarray] = []
            for frame_index in indices:
                image_path = project_root / manifest[frame_index]["local_path"]
                image = read_image(image_path)
                crop = _crop_with_padding(
                    image,
                    int(float(candidate["x"])),
                    int(float(candidate["y"])),
                    args.crop_width,
                    args.crop_height,
                )
                crop = cv2.resize(crop, (360, 240), interpolation=cv2.INTER_CUBIC)
                center = (180, 120)
                cv2.circle(crop, center, 20, (0, 255, 255), 3)
                label = (
                    f"{candidate['candidate_id']} | f{frame_index:04d} | "
                    f"{frame_index / 2.0:.1f}s"
                )
                cv2.rectangle(crop, (0, 0), (330, 31), (0, 0, 0), -1)
                cv2.putText(
                    crop,
                    label,
                    (8, 22),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.56,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )
                tiles.append(crop)
            rows.append(np.hstack(tiles))
    if not rows:
        raise ValueError("No candidate transitions were selected")
    write_image(Path(args.output), np.vstack(rows))
    print(f"Rendered {len(rows)} transition rows")


if __name__ == "__main__":
    main()
