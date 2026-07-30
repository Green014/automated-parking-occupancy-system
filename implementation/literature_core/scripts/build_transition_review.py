"""Build visual review sheets for machine-proposed temporal state transitions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from literature_core.annotation_review import (  # noqa: E402
    nearest_available_indices,
    parse_transition_frames,
    temporal_review_indices,
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    x: float
    y: float
    width: int
    height: int
    transitions: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--video", type=Path)
    source.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Base directory for local_path values in an image manifest.",
    )
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=10)
    parser.add_argument("--uniform-samples", type=int, default=12)
    parser.add_argument("--padding", type=int, default=35)
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Source-video frames per candidate-timeline sample.",
    )
    parser.add_argument(
        "--coordinate-mode",
        choices=("center", "top-left"),
        default="center",
        help="Meaning of x/y in the candidate CSV.",
    )
    parser.add_argument(
        "--draw-roi-on-crop",
        action="store_true",
        help="Draw a thin ROI outline on crop sheets for boundary adjudication.",
    )
    parser.add_argument("--crop-cell-width", type=int, default=280)
    parser.add_argument("--crop-cell-height", type=int, default=230)
    parser.add_argument(
        "--crop-grid-size",
        type=int,
        default=0,
        help="Draw a globally labeled coordinate grid on crop images.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_candidates(path: Path) -> list[Candidate]:
    candidates: list[Candidate] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            candidates.append(
                Candidate(
                    candidate_id=row["candidate_id"],
                    x=float(row["x"]),
                    y=float(row["y"]),
                    width=round(float(row["width"])),
                    height=round(float(row["height"])),
                    transitions=tuple(
                        parse_transition_frames(row.get("transition_frames"))
                    ),
                )
            )
    if not candidates:
        raise ValueError(f"no candidates found in {path}")
    return candidates


def read_frames(
    capture: cv2.VideoCapture,
    sample_indices: list[int],
    frame_stride: int,
    total_source_frames: int,
    fps: float,
) -> dict[int, tuple[int, float, np.ndarray]]:
    frames: dict[int, tuple[int, float, np.ndarray]] = {}
    for sample_index in sample_indices:
        source_index = min(sample_index * frame_stride, total_source_frames - 1)
        capture.set(cv2.CAP_PROP_POS_FRAMES, source_index)
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"could not read source frame {source_index}")
        timestamp = source_index / fps if fps > 0 else 0.0
        frames[sample_index] = (source_index, timestamp, frame)
    return frames


def load_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty frame manifest: {path}")
    return rows


def read_manifest_frames(
    rows: list[dict[str, str]],
    project_root: Path,
    sample_indices: list[int],
) -> dict[int, tuple[int, float, np.ndarray]]:
    frames: dict[int, tuple[int, float, np.ndarray]] = {}
    for sample_index in sample_indices:
        row = rows[sample_index]
        source_index = int(row.get("frame_index", sample_index))
        timestamp = float(row.get("timestamp_s", 0.0))
        image_path = project_root / row["local_path"]
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise RuntimeError(f"could not read image {image_path}")
        frames[sample_index] = (source_index, timestamp, frame)
    return frames


def fit_image(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized = cv2.resize(
        image,
        (
            max(1, round(image.shape[1] * scale)),
            max(1, round(image.shape[0] * scale)),
        ),
        interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC,
    )
    canvas = np.full((height, width, 3), 245, dtype=np.uint8)
    top = (height - resized.shape[0]) // 2
    left = (width - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized
    return canvas


def tile(
    images: list[tuple[int, int, float, np.ndarray]],
    *,
    columns: int,
    cell_width: int,
    cell_height: int,
) -> np.ndarray:
    title_height = 30
    rows = (len(images) + columns - 1) // columns
    sheet = np.full(
        (rows * (cell_height + title_height), columns * cell_width, 3),
        255,
        dtype=np.uint8,
    )
    for ordinal, (sample_index, source_index, timestamp, image) in enumerate(images):
        row, column = divmod(ordinal, columns)
        y = row * (cell_height + title_height)
        x = column * cell_width
        fitted = fit_image(image, cell_width, cell_height)
        sheet[y + title_height : y + title_height + cell_height, x : x + cell_width] = (
            fitted
        )
        cv2.putText(
            sheet,
            f"sample {sample_index:03d} / f{source_index:04d} / {timestamp:5.1f}s",
            (x + 7, y + 21),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    return sheet


def main() -> None:
    args = parse_args()
    if args.frame_stride <= 0:
        raise ValueError("--frame-stride must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    candidates = load_candidates(args.candidates)

    capture: cv2.VideoCapture | None = None
    manifest_rows: list[dict[str, str]] | None = None
    available_manifest_indices: list[int] | None = None
    if args.video is not None:
        capture = cv2.VideoCapture(str(args.video))
        if not capture.isOpened():
            raise RuntimeError(f"could not open {args.video}")
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        total_samples = (total_frames + args.frame_stride - 1) // args.frame_stride
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        source_file = args.video
        source_kind = "video"
    else:
        manifest_rows = load_manifest(args.manifest)
        total_samples = len(manifest_rows)
        total_frames = total_samples
        available_manifest_indices = [
            index
            for index, row in enumerate(manifest_rows)
            if (args.project_root / row["local_path"]).is_file()
        ]
        if not available_manifest_indices:
            raise RuntimeError("none of the manifest images are available locally")
        timestamps = [float(row.get("timestamp_s", 0.0)) for row in manifest_rows]
        positive_steps = [
            right - left
            for left, right in zip(timestamps, timestamps[1:])
            if right > left
        ]
        fps = 1.0 / float(np.median(positive_steps)) if positive_steps else 0.0
        first_path = (
            args.project_root
            / manifest_rows[available_manifest_indices[0]]["local_path"]
        )
        first_frame = cv2.imread(str(first_path))
        if first_frame is None:
            raise RuntimeError(f"could not read image {first_path}")
        frame_height, frame_width = first_frame.shape[:2]
        source_file = args.manifest
        source_kind = "image_manifest"

    review_rows: list[dict[str, object]] = []
    candidate_summaries: list[dict[str, object]] = []
    for candidate in candidates:
        indices = temporal_review_indices(
            total_samples,
            candidate.transitions,
            radius=args.radius,
            uniform_samples=args.uniform_samples,
        )
        if available_manifest_indices is not None:
            indices = nearest_available_indices(indices, available_manifest_indices)
        if capture is not None:
            frames = read_frames(
                capture,
                indices,
                args.frame_stride,
                total_frames,
                fps,
            )
        else:
            assert manifest_rows is not None
            frames = read_manifest_frames(
                manifest_rows,
                args.project_root,
                indices,
            )
        full_views: list[tuple[int, int, float, np.ndarray]] = []
        crop_views: list[tuple[int, int, float, np.ndarray]] = []
        if args.coordinate_mode == "center":
            bbox_x = round(candidate.x - candidate.width / 2)
            bbox_y = round(candidate.y - candidate.height / 2)
        else:
            bbox_x = round(candidate.x)
            bbox_y = round(candidate.y)
        bbox_x = max(0, min(bbox_x, frame_width - 1))
        bbox_y = max(0, min(bbox_y, frame_height - 1))
        bbox_width = min(candidate.width, frame_width - bbox_x)
        bbox_height = min(candidate.height, frame_height - bbox_y)
        x0 = max(0, bbox_x - args.padding)
        y0 = max(0, bbox_y - args.padding)
        x1 = min(frame_width, bbox_x + bbox_width + args.padding)
        y1 = min(frame_height, bbox_y + bbox_height + args.padding)

        for sample_index in indices:
            source_index, timestamp, frame = frames[sample_index]
            overlay = frame.copy()
            cv2.rectangle(
                overlay,
                (bbox_x, bbox_y),
                (bbox_x + bbox_width, bbox_y + bbox_height),
                (0, 0, 255),
                4,
            )
            full_views.append((sample_index, source_index, timestamp, overlay))

            crop = frame[y0:y1, x0:x1].copy()
            if args.draw_roi_on_crop:
                cv2.rectangle(
                    crop,
                    (bbox_x - x0, bbox_y - y0),
                    (
                        bbox_x + bbox_width - x0,
                        bbox_y + bbox_height - y0,
                    ),
                    (0, 255, 255),
                    1,
                )
            if args.crop_grid_size > 0:
                for grid_x in range(
                    ((x0 + args.crop_grid_size - 1) // args.crop_grid_size)
                    * args.crop_grid_size,
                    x1,
                    args.crop_grid_size,
                ):
                    local_x = grid_x - x0
                    cv2.line(
                        crop,
                        (local_x, 0),
                        (local_x, crop.shape[0] - 1),
                        (255, 0, 255),
                        1,
                    )
                    cv2.putText(
                        crop,
                        str(grid_x),
                        (local_x + 2, 13),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
                for grid_y in range(
                    ((y0 + args.crop_grid_size - 1) // args.crop_grid_size)
                    * args.crop_grid_size,
                    y1,
                    args.crop_grid_size,
                ):
                    local_y = grid_y - y0
                    cv2.line(
                        crop,
                        (0, local_y),
                        (crop.shape[1] - 1, local_y),
                        (255, 0, 255),
                        1,
                    )
                    cv2.putText(
                        crop,
                        str(grid_y),
                        (2, max(12, local_y - 2)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.35,
                        (255, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
            crop_views.append((sample_index, source_index, timestamp, crop))
            review_rows.append(
                {
                    "candidate_id": candidate.candidate_id,
                    "sample_index": sample_index,
                    "source_frame_index": source_index,
                    "timestamp_s": f"{timestamp:.3f}",
                    "near_proposed_transition": int(
                        any(
                            abs(sample_index - transition) <= args.radius
                            for transition in candidate.transitions
                        )
                    ),
                    "manual_state": "",
                    "manual_note": "",
                }
            )

        safe_id = "".join(
            character if character.isalnum() or character in "-_" else "_"
            for character in candidate.candidate_id
        )
        full_path = args.output_dir / f"{safe_id}_full_context.jpg"
        crop_path = args.output_dir / f"{safe_id}_roi_context.jpg"
        cv2.imwrite(
            str(full_path),
            tile(
                full_views,
                columns=3,
                cell_width=480,
                cell_height=270,
            ),
        )
        cv2.imwrite(
            str(crop_path),
            tile(
                crop_views,
                columns=5,
                cell_width=args.crop_cell_width,
                cell_height=args.crop_cell_height,
            ),
        )
        candidate_summaries.append(
            {
                "candidate_id": candidate.candidate_id,
                "bbox_xywh": [
                    bbox_x,
                    bbox_y,
                    bbox_width,
                    bbox_height,
                ],
                "proposed_transition_frames": list(candidate.transitions),
                "review_frame_count": len(indices),
                "full_context_sheet": full_path.name,
                "roi_context_sheet": crop_path.name,
            }
        )

    if capture is not None:
        capture.release()
    with (args.output_dir / "review_manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(review_rows[0]))
        writer.writeheader()
        writer.writerows(review_rows)

    summary = {
        "source_kind": source_kind,
        "source_file": str(source_file.resolve()),
        "source_file_sha256": sha256(source_file),
        "candidate_source": str(args.candidates.resolve()),
        "video": {
            "total_frames": total_frames,
            "candidate_timeline_samples": total_samples,
            "locally_available_samples": (
                len(available_manifest_indices)
                if available_manifest_indices is not None
                else total_samples
            ),
            "fps": fps,
            "width": frame_width,
            "height": frame_height,
        },
        "selection": {
            "transition_radius": args.radius,
            "uniform_samples": args.uniform_samples,
            "padding": args.padding,
            "frame_stride": args.frame_stride,
            "coordinate_mode": args.coordinate_mode,
            "draw_roi_on_crop": args.draw_roi_on_crop,
        },
        "candidates": candidate_summaries,
    }
    (args.output_dir / "review_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
