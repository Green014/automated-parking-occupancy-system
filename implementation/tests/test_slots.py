import json

import pytest

from parking_occupancy.slots import load_slot_map, slot_map_from_dict


def test_normalized_slot_map_is_converted_to_pixels() -> None:
    slot_map = slot_map_from_dict(
        {
            "schema_version": 1,
            "source_width": 100,
            "source_height": 50,
            "coordinate_system": "normalized",
            "slots": [
                {
                    "id": "s1",
                    "points": [[0.1, 0.2], [0.3, 0.2], [0.3, 0.4], [0.1, 0.4]],
                }
            ],
        }
    )
    assert slot_map.slots[0].points[0] == (10.0, 10.0)
    assert slot_map.slots[0].points[2] == (30.0, 20.0)


def test_slot_map_scales_to_video_resolution(tmp_path) -> None:
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
                        "id": "s1",
                        "points": [[10, 10], [30, 10], [30, 20], [10, 20]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    slot_map = load_slot_map(path, frame_size=(200, 100))
    assert slot_map.slots[0].points[0] == (20.0, 20.0)
    assert slot_map.slots[0].points[2] == (60.0, 40.0)


def test_non_convex_slot_is_rejected() -> None:
    with pytest.raises(ValueError, match="not convex"):
        slot_map_from_dict(
            {
                "schema_version": 1,
                "source_width": 100,
                "source_height": 100,
                "slots": [
                    {
                        "id": "bad",
                        "points": [[0, 0], [10, 10], [0, 10], [10, 0]],
                    }
                ],
            }
        )

