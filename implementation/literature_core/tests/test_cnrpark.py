from pathlib import Path

import pytest

from literature_core.cnrpark import (
    CNRSlotLabel,
    load_cnr_ext_boxes,
    load_cnr_ext_metadata,
    scaled_axis_aligned_slot,
)


def test_cnr_label_builds_official_full_frame_path() -> None:
    label = CNRSlotLabel("06", "2015-11-22_09.47", "205", 1, "S")
    assert label.group_id == "06/2015-11-22_09.47"
    assert label.relative_frame_path == Path(
        "FULL_IMAGE_1000x750/SUNNY/2015-11-22/"
        "camera6/2015-11-22_0947.jpg"
    )


def test_load_cnr_ext_metadata_filters_preliminary_subset(tmp_path: Path) -> None:
    metadata = tmp_path / "metadata.csv"
    metadata.write_text(
        "camera,datetime,occupancy,slot_id,weather\n"
        "A,20150703_0805,0,1,S\n"
        "01,2015-11-12_07.09,1,185,S\n"
        "01,2015-11-12_07.09,0,184,S\n",
        encoding="utf-8",
    )
    groups = load_cnr_ext_metadata(metadata, cameras=("01",))
    assert list(groups) == ["01/2015-11-12_07.09"]
    assert [label.slot_id for label in next(iter(groups.values()))] == ["184", "185"]


def test_scaled_axis_aligned_slot_uses_published_resolution_ratio() -> None:
    slot = scaled_axis_aligned_slot("184", (259.2, 194.4, 259.2, 194.4))
    expected = (
        (100.0, 75.0),
        (200.0, 75.0),
        (200.0, 150.0),
        (100.0, 150.0),
    )
    for point, expected_point in zip(slot.points, expected, strict=True):
        assert point == pytest.approx(expected_point)


def test_load_cnr_ext_boxes_requires_and_scales_all_nine_files(
    tmp_path: Path,
) -> None:
    for camera in range(1, 10):
        (tmp_path / f"camera{camera}.csv").write_text(
            "SlotId,X,Y,W,H\n"
            f"{camera},259.2,194.4,259.2,194.4\n",
            encoding="utf-8",
        )
    boxes = load_cnr_ext_boxes(tmp_path)
    assert sorted(boxes) == [f"{camera:02d}" for camera in range(1, 10)]
    assert boxes["01"]["1"].points[0] == pytest.approx((100.0, 75.0))
