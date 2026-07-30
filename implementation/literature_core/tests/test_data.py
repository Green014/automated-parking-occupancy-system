import json

import numpy as np

from literature_core.data import (
    load_pklot_slot_samples,
    load_slot_map,
    write_image,
)


def test_pklot_loader_splits_by_camera_and_excludes_unknown(tmp_path) -> None:
    image_path = tmp_path / "数据" / "frame.jpg"
    write_image(image_path, np.zeros((10, 20, 3), dtype=np.uint8))
    payload = {
        "sample_id": "sample_1",
        "source": "pucpr",
        "date": "2026-01-01",
        "local_path": str(image_path.relative_to(tmp_path)),
        "sample": {
            "metadata": {"width": 20, "height": 10},
            "parking_spaces": {
                "polylines": [
                    {
                        "space_id": 1,
                        "occupancy_status": "occupied",
                        "points": [[[0.1, 0.2], [0.5, 0.2], [0.5, 0.8], [0.1, 0.8]]],
                    },
                    {
                        "space_id": 2,
                        "occupancy_status": "unknown",
                        "points": [[[0.5, 0.2], [0.9, 0.2], [0.9, 0.8], [0.5, 0.8]]],
                    },
                ]
            },
        },
    }
    annotations = tmp_path / "annotations.jsonl"
    annotations.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    split = tmp_path / "split.json"
    split.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "train": ["pucpr"],
                "development": ["ufpr04"],
                "test": ["ufpr05"],
            }
        ),
        encoding="utf-8",
    )
    samples = load_pklot_slot_samples(annotations, tmp_path, split)
    assert len(samples) == 1
    assert samples[0].split == "train"
    assert samples[0].label == 1
    assert samples[0].points[0] == (2.0, 2.0)


def test_slot_map_scales_pixel_coordinates(tmp_path) -> None:
    path = tmp_path / "slots.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_width": 100,
                "source_height": 50,
                "coordinate_system": "pixel",
                "slots": [
                    {
                        "id": "A",
                        "points": [[10, 10], [20, 10], [20, 20], [10, 20]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    slots = load_slot_map(path, (200, 100))
    assert slots[0].points == (
        (20.0, 20.0),
        (40.0, 20.0),
        (40.0, 40.0),
        (20.0, 40.0),
    )

