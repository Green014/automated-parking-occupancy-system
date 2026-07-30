from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .models import ParkingSlot


@dataclass(frozen=True, slots=True)
class SlotMap:
    schema_version: int
    source_width: int
    source_height: int
    source: str
    slots: tuple[ParkingSlot, ...]

    def scaled_to(self, width: int, height: int) -> "SlotMap":
        if width <= 0 or height <= 0:
            raise ValueError("Target frame size must be positive")
        sx = width / self.source_width
        sy = height / self.source_height
        scaled = tuple(
            ParkingSlot(
                slot_id=slot.slot_id,
                points=tuple((x * sx, y * sy) for x, y in slot.points),
            )
            for slot in self.slots
        )
        return SlotMap(
            schema_version=self.schema_version,
            source_width=width,
            source_height=height,
            source=self.source,
            slots=scaled,
        )


def _validate_convex(slot: ParkingSlot) -> None:
    contour = np.asarray(slot.points, dtype=np.float32).reshape((-1, 1, 2))
    if not cv2.isContourConvex(contour):
        raise ValueError(
            f"Slot {slot.slot_id} is not convex; overlap mapping requires convex polygons"
        )
    if abs(float(cv2.contourArea(contour))) <= 1e-6:
        raise ValueError(f"Slot {slot.slot_id} has zero area")


def slot_map_from_dict(payload: dict[str, Any]) -> SlotMap:
    version = int(payload.get("schema_version", 1))
    if version != 1:
        raise ValueError(f"Unsupported slot-map schema_version: {version}")

    width = int(payload["source_width"])
    height = int(payload["source_height"])
    if width <= 0 or height <= 0:
        raise ValueError("source_width and source_height must be positive")

    coordinate_system = payload.get("coordinate_system", "pixel")
    if coordinate_system not in {"pixel", "normalized"}:
        raise ValueError("coordinate_system must be 'pixel' or 'normalized'")

    slots: list[ParkingSlot] = []
    seen_ids: set[str] = set()
    for item in payload.get("slots", []):
        slot_id = str(item["id"])
        if slot_id in seen_ids:
            raise ValueError(f"Duplicate slot id: {slot_id}")
        seen_ids.add(slot_id)
        raw_points = item["points"]
        if coordinate_system == "normalized":
            points = tuple(
                (float(point[0]) * width, float(point[1]) * height)
                for point in raw_points
            )
        else:
            points = tuple((float(point[0]), float(point[1])) for point in raw_points)
        slot = ParkingSlot(slot_id=slot_id, points=points)
        _validate_convex(slot)
        slots.append(slot)

    if not slots:
        raise ValueError("Slot map does not contain any slots")

    return SlotMap(
        schema_version=version,
        source_width=width,
        source_height=height,
        source=str(payload.get("source", "")),
        slots=tuple(slots),
    )


def load_slot_map(
    path: str | Path,
    frame_size: tuple[int, int] | None = None,
) -> SlotMap:
    slot_path = Path(path)
    with slot_path.open("r", encoding="utf-8") as handle:
        slot_map = slot_map_from_dict(json.load(handle))
    if frame_size is not None:
        slot_map = slot_map.scaled_to(*frame_size)
    return slot_map


def slot_map_to_dict(slot_map: SlotMap) -> dict[str, Any]:
    return {
        "schema_version": slot_map.schema_version,
        "source": slot_map.source,
        "source_width": slot_map.source_width,
        "source_height": slot_map.source_height,
        "coordinate_system": "pixel",
        "slots": [
            {
                "id": slot.slot_id,
                "points": [[round(x, 3), round(y, 3)] for x, y in slot.points],
            }
            for slot in slot_map.slots
        ],
    }


def save_slot_map(slot_map: SlotMap, path: str | Path) -> None:
    slot_path = Path(path)
    slot_path.parent.mkdir(parents=True, exist_ok=True)
    with slot_path.open("w", encoding="utf-8") as handle:
        json.dump(slot_map_to_dict(slot_map), handle, indent=2, ensure_ascii=False)
        handle.write("\n")
