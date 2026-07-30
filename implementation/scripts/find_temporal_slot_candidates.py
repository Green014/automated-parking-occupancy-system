from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from parking_occupancy.image_io import read_image, write_image


@dataclass(slots=True)
class Candidate:
    candidate_id: str
    x: float
    y: float
    width: float
    height: float


def _vehicle_boxes_by_image(
    coco: dict,
    include_buses: bool = False,
) -> dict[int, list[list[float]]]:
    vehicle_ids = {
        int(category["id"])
        for category in coco["categories"]
        if category.get("supercategory") == "vehicle"
        and (include_buses or category["name"] != "bus")
    }
    boxes: dict[int, list[list[float]]] = defaultdict(list)
    for annotation in coco["annotations"]:
        if int(annotation["category_id"]) in vehicle_ids:
            boxes[int(annotation["image_id"])].append(annotation["bbox"])
    return boxes


def _seed_candidates(
    ordered_images: list[dict],
    boxes_by_image: dict[int, list[list[float]]],
    merge_radius: float,
) -> list[Candidate]:
    seed_indices = sorted(
        {0, len(ordered_images) // 4, len(ordered_images) // 2,
         3 * len(ordered_images) // 4, len(ordered_images) - 1}
    )
    candidates: list[Candidate] = []
    for frame_index in seed_indices:
        image_id = int(ordered_images[frame_index]["id"])
        for x, y, width, height in boxes_by_image[image_id]:
            center_x = x + width / 2.0
            center_y = y + height / 2.0
            match = next(
                (
                    candidate
                    for candidate in candidates
                    if (candidate.x - center_x) ** 2
                    + (candidate.y - center_y) ** 2
                    <= merge_radius**2
                ),
                None,
            )
            if match is None:
                candidates.append(
                    Candidate(
                        candidate_id="",
                        x=center_x,
                        y=center_y,
                        width=width,
                        height=height,
                    )
                )
            else:
                match.x = (match.x + center_x) / 2.0
                match.y = (match.y + center_y) / 2.0
                match.width = max(match.width, width)
                match.height = max(match.height, height)
    for index, candidate in enumerate(candidates, start=1):
        candidate.candidate_id = f"C{index:04d}"
    return candidates


def _median_boolean(states: list[bool], window: int) -> list[bool]:
    radius = window // 2
    filtered: list[bool] = []
    for index in range(len(states)):
        start = max(0, index - radius)
        end = min(len(states), index + radius + 1)
        values = states[start:end]
        filtered.append(sum(values) * 2 >= len(values))
    return filtered


def _run_statistics(states: list[bool]) -> tuple[int, int, int, list[int]]:
    longest_occupied = 0
    longest_vacant = 0
    transitions: list[int] = []
    run_start = 0
    for index in range(1, len(states) + 1):
        if index == len(states) or states[index] != states[run_start]:
            run_length = index - run_start
            if states[run_start]:
                longest_occupied = max(longest_occupied, run_length)
            else:
                longest_vacant = max(longest_vacant, run_length)
            if index < len(states):
                transitions.append(index)
            run_start = index
    return (
        longest_occupied,
        longest_vacant,
        len(transitions),
        transitions,
    )


def analyse_candidates(
    ordered_images: list[dict],
    boxes_by_image: dict[int, list[list[float]]],
    candidates: list[Candidate],
    match_radius: float,
    median_window: int,
) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    radius_squared = match_radius**2
    for candidate in candidates:
        states: list[bool] = []
        for image in ordered_images:
            boxes = boxes_by_image[int(image["id"])]
            occupied = any(
                (candidate.x - (x + width / 2.0)) ** 2
                + (candidate.y - (y + height / 2.0)) ** 2
                <= radius_squared
                for x, y, width, height in boxes
            )
            states.append(occupied)
        states = _median_boolean(states, median_window)
        longest_occupied, longest_vacant, transitions, transition_frames = (
            _run_statistics(states)
        )
        occupied_fraction = sum(states) / len(states)
        results.append(
            {
                "candidate_id": candidate.candidate_id,
                "x": round(candidate.x, 2),
                "y": round(candidate.y, 2),
                "width": round(candidate.width, 2),
                "height": round(candidate.height, 2),
                "occupied_fraction": round(occupied_fraction, 6),
                "longest_occupied": longest_occupied,
                "longest_vacant": longest_vacant,
                "transitions": transitions,
                "transition_frames": ";".join(map(str, transition_frames)),
                "states": states,
            }
        )
    return results


def draw_candidates(
    image_path: Path,
    candidates: list[dict[str, object]],
    output_path: Path,
) -> None:
    image = read_image(image_path)
    for candidate in candidates:
        x = int(float(candidate["x"]))
        y = int(float(candidate["y"]))
        cv2.circle(image, (x, y), 12, (0, 255, 255), 2)
        cv2.putText(
            image,
            str(candidate["candidate_id"]),
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(candidate["candidate_id"]),
            (x + 5, y - 8),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    write_image(output_path, image)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Find fixed vehicle-centre locations with long state changes"
    )
    parser.add_argument("--coco", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--overlay-output", required=True)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--merge-radius", type=float, default=12.0)
    parser.add_argument("--match-radius", type=float, default=13.0)
    parser.add_argument("--median-window", type=int, default=9)
    parser.add_argument("--minimum-run", type=int, default=30)
    parser.add_argument("--maximum-transitions", type=int, default=8)
    parser.add_argument(
        "--include-buses",
        action="store_true",
        help="Include buses when looking for fixed dwell locations",
    )
    args = parser.parse_args()

    coco = json.loads(Path(args.coco).read_text(encoding="utf-8"))
    ordered_images = sorted(
        coco["images"],
        key=lambda image: int(image["frame_index"]),
    )
    boxes_by_image = _vehicle_boxes_by_image(
        coco,
        include_buses=args.include_buses,
    )
    candidates = _seed_candidates(
        ordered_images,
        boxes_by_image,
        args.merge_radius,
    )
    results = analyse_candidates(
        ordered_images,
        boxes_by_image,
        candidates,
        args.match_radius,
        args.median_window,
    )
    shortlisted = [
        result
        for result in results
        if int(result["longest_occupied"]) >= args.minimum_run
        and int(result["longest_vacant"]) >= args.minimum_run
        and 0 < int(result["transitions"]) <= args.maximum_transitions
    ]
    shortlisted.sort(
        key=lambda result: (
            int(result["transitions"]),
            -min(
                int(result["longest_occupied"]),
                int(result["longest_vacant"]),
            ),
        )
    )

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "candidate_id",
        "x",
        "y",
        "width",
        "height",
        "occupied_fraction",
        "longest_occupied",
        "longest_vacant",
        "transitions",
        "transition_frames",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for result in shortlisted:
            writer.writerow({name: result[name] for name in fieldnames})

    with Path(args.manifest).open(newline="", encoding="utf-8") as handle:
        first_row = next(csv.DictReader(handle))
    first_image = Path(args.project_root).resolve() / first_row["local_path"]
    draw_candidates(first_image, shortlisted, Path(args.overlay_output))
    print(
        f"Found {len(shortlisted)} temporal candidates "
        f"from {len(candidates)} spatial seeds"
    )


if __name__ == "__main__":
    main()
