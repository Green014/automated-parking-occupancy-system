from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .models import ParkingSlot

Split = Literal["train", "development", "test"]


@dataclass(frozen=True, slots=True)
class SlotSample:
    sample_id: str
    source: str
    group_id: str
    split: Split
    image_path: Path
    slot_id: str
    points: tuple[tuple[float, float], ...]
    label: int


def read_image(path: str | Path) -> np.ndarray:
    """Read an image from a Unicode-safe Windows path."""

    image_path = Path(path)
    encoded = np.fromfile(image_path, dtype=np.uint8)
    image = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"OpenCV could not read image: {image_path}")
    return image


def write_image(path: str | Path, image: np.ndarray) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    extension = output_path.suffix or ".png"
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise RuntimeError(f"OpenCV could not encode image: {output_path}")
    encoded.tofile(output_path)


def load_split_config(path: str | Path) -> dict[str, Split]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported split schema")
    source_to_split: dict[str, Split] = {}
    for split in ("train", "development", "test"):
        sources = payload.get(split, [])
        if not sources:
            raise ValueError(f"Split {split} must contain at least one source")
        for source in sources:
            normalized = str(source).lower()
            if normalized in source_to_split:
                raise ValueError(f"Source appears in multiple splits: {normalized}")
            source_to_split[normalized] = split  # type: ignore[assignment]
    return source_to_split


def load_pklot_slot_samples(
    annotations_path: str | Path,
    project_root: str | Path,
    split_config_path: str | Path,
) -> list[SlotSample]:
    """Load FiftyOne-exported PKLot polygons without treating unknown as truth."""

    source_to_split = load_split_config(split_config_path)
    project_root = Path(project_root).resolve()
    samples: list[SlotSample] = []
    with Path(annotations_path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row["source"]).lower()
            if source not in source_to_split:
                raise ValueError(f"Source is absent from split config: {source}")
            split = source_to_split[source]
            date = str(row["date"])
            image_path = (project_root / row["local_path"]).resolve()
            if not image_path.is_file():
                raise FileNotFoundError(image_path)
            metadata = row["sample"].get("metadata", {})
            width = int(metadata.get("width", 0))
            height = int(metadata.get("height", 0))
            if width <= 0 or height <= 0:
                image = read_image(image_path)
                height, width = image.shape[:2]
            polygons = row["sample"]["parking_spaces"]["polylines"]
            for item in polygons:
                status = str(item.get("occupancy_status", "")).lower()
                if status == "unknown":
                    continue
                if status not in {"occupied", "not occupied"}:
                    raise ValueError(f"Unexpected PKLot status: {status}")
                normalized_points = item["points"][0]
                points = tuple(
                    (float(point[0]) * width, float(point[1]) * height)
                    for point in normalized_points
                )
                samples.append(
                    SlotSample(
                        sample_id=str(row["sample_id"]),
                        source=source,
                        group_id=f"{source}/{date}",
                        split=split,
                        image_path=image_path,
                        slot_id=str(item["space_id"]),
                        points=points,
                        label=int(status == "occupied"),
                    )
                )
    if not samples:
        raise ValueError("No known PKLot slot samples were loaded")
    return samples


def load_slot_map(
    path: str | Path,
    frame_size: tuple[int, int],
) -> tuple[ParkingSlot, ...]:
    """Load the baseline slot-map JSON schema without importing baseline code."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload: dict[str, Any] = json.load(handle)
    if int(payload.get("schema_version", 1)) != 1:
        raise ValueError("Unsupported slot-map schema")
    source_width = int(payload["source_width"])
    source_height = int(payload["source_height"])
    target_width, target_height = frame_size
    if min(source_width, source_height, target_width, target_height) <= 0:
        raise ValueError("Slot-map and frame dimensions must be positive")
    coordinate_system = payload.get("coordinate_system", "pixel")
    if coordinate_system not in {"pixel", "normalized"}:
        raise ValueError("Unsupported coordinate system")

    slots: list[ParkingSlot] = []
    for item in payload.get("slots", []):
        raw_points = item["points"]
        if coordinate_system == "normalized":
            source_points = [
                (float(point[0]) * source_width, float(point[1]) * source_height)
                for point in raw_points
            ]
        else:
            source_points = [
                (float(point[0]), float(point[1])) for point in raw_points
            ]
        points = tuple(
            (
                x * target_width / source_width,
                y * target_height / source_height,
            )
            for x, y in source_points
        )
        slots.append(ParkingSlot(str(item["id"]), points))
    if not slots:
        raise ValueError("Slot map contains no slots")
    if len({slot.slot_id for slot in slots}) != len(slots):
        raise ValueError("Slot IDs must be unique")
    return tuple(slots)

