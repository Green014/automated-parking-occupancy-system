"""CNRPark+EXT external-holdout metadata and geometry helpers."""

from __future__ import annotations

import csv
from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .models import ParkingSlot


WEATHER_FOLDERS = {"S": "SUNNY", "O": "OVERCAST", "R": "RAINY"}


@dataclass(frozen=True, slots=True)
class CNRSlotLabel:
    camera: str
    datetime: str
    slot_id: str
    occupancy: int
    weather: str

    @property
    def group_id(self) -> str:
        return f"{self.camera}/{self.datetime}"

    @property
    def relative_frame_path(self) -> Path:
        date, capture_time = self.datetime.split("_", 1)
        compact_time = capture_time.replace(".", "")
        camera_number = int(self.camera)
        return (
            Path("FULL_IMAGE_1000x750")
            / WEATHER_FOLDERS[self.weather]
            / date
            / f"camera{camera_number}"
            / f"{date}_{compact_time}.jpg"
        )


def load_cnr_ext_metadata(
    path: str | Path,
    *,
    cameras: Iterable[str] = tuple(f"{index:02d}" for index in range(1, 10)),
) -> dict[str, tuple[CNRSlotLabel, ...]]:
    """Load CNR-EXT only and group complete slot labels by source image."""

    allowed_cameras = {str(camera).zfill(2) for camera in cameras}
    groups: dict[str, list[CNRSlotLabel]] = defaultdict(list)
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"camera", "datetime", "occupancy", "slot_id", "weather"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError("CNR metadata CSV is missing required columns")
        for row in reader:
            camera = str(row["camera"]).zfill(2)
            if camera not in allowed_cameras:
                continue
            occupancy = int(row["occupancy"])
            weather = str(row["weather"])
            if occupancy not in {0, 1}:
                raise ValueError("CNR occupancy must be 0 or 1")
            if weather not in WEATHER_FOLDERS:
                raise ValueError(f"Unsupported CNR weather code: {weather}")
            label = CNRSlotLabel(
                camera=camera,
                datetime=str(row["datetime"]),
                slot_id=str(row["slot_id"]),
                occupancy=occupancy,
                weather=weather,
            )
            groups[label.group_id].append(label)
    if not groups:
        raise ValueError("No CNR-EXT rows matched the requested cameras")
    return {
        group_id: tuple(sorted(labels, key=lambda label: int(label.slot_id)))
        for group_id, labels in sorted(groups.items())
    }


def scaled_axis_aligned_slot(
    slot_id: str,
    box: tuple[float, float, float, float],
    *,
    source_size: tuple[int, int] = (2592, 1944),
    target_size: tuple[int, int] = (1000, 750),
) -> ParkingSlot:
    """Scale an official source-resolution x/y/w/h box into released frames."""

    x, y, width, height = box
    if width <= 0 or height <= 0:
        raise ValueError("CNR slot box must have positive width and height")
    scale_x = target_size[0] / source_size[0]
    scale_y = target_size[1] / source_size[1]
    x1, y1 = x * scale_x, y * scale_y
    x2, y2 = (x + width) * scale_x, (y + height) * scale_y
    return ParkingSlot(
        str(slot_id),
        ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
    )


def load_cnr_ext_boxes(
    dataset_root: str | Path,
) -> dict[str, dict[str, ParkingSlot]]:
    """Load the nine official ``cameraN.csv`` x/y/w/h geometry files."""

    root = Path(dataset_root)
    result: dict[str, dict[str, ParkingSlot]] = {}
    for camera_number in range(1, 10):
        path = root / f"camera{camera_number}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing official CNR geometry: {path}")
        slots: dict[str, ParkingSlot] = {}
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != ["SlotId", "X", "Y", "W", "H"]:
                raise ValueError(
                    f"Unexpected CNR geometry schema in {path}: "
                    f"{reader.fieldnames}"
                )
            for row in reader:
                slot_id = str(row["SlotId"])
                if slot_id in slots:
                    raise ValueError(
                        f"Duplicate slot {slot_id} in official geometry {path}"
                    )
                slots[slot_id] = scaled_axis_aligned_slot(
                    slot_id,
                    (
                        float(row["X"]),
                        float(row["Y"]),
                        float(row["W"]),
                        float(row["H"]),
                    ),
                )
        if not slots:
            raise ValueError(f"Official CNR geometry is empty: {path}")
        result[f"{camera_number:02d}"] = slots
    return result
