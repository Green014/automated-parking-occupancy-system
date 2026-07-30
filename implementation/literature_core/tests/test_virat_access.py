from io import BytesIO
import hashlib
from pathlib import Path
from urllib.error import HTTPError

import pytest
import yaml

from literature_core.virat_access import (
    ViratItem,
    download_item,
    fetch_folder_items,
    parse_annotation_item,
    parse_video_item,
    select_cross_scene_candidates,
    select_named_items,
    verify_screening_manifest,
)


def _item(scene: str, size: int, suffix: str = "00") -> ViratItem:
    return ViratItem(
        item_id=f"id-{scene}-{suffix}",
        name=f"VIRAT_S_{scene}01_{suffix}_000000_000010.mp4",
        size=size,
        scene_id=scene,
        sequence_id="01",
    )


def test_parse_video_item_extracts_scene() -> None:
    item = parse_video_item(
        {
            "_id": "official-id",
            "name": "VIRAT_S_010003_07_000608_000636.mp4",
            "size": 1_618_350,
        }
    )
    assert item is not None
    assert item.scene_id == "0100"
    assert item.sequence_id == "03"


def test_parse_annotation_item_extracts_scene() -> None:
    item = parse_annotation_item(
        {
            "_id": "annotation-id",
            "name": (
                "VIRAT_S_050300_04_001057_001122."
                "viratdata.events.txt"
            ),
            "size": 124_606,
        }
    )
    assert item is not None
    assert item.scene_id == "0503"
    assert item.sequence_id == "00"


def test_cross_scene_selection_respects_count_and_byte_budget() -> None:
    selected = select_cross_scene_candidates(
        [
            _item("0001", 8, "00"),
            _item("0001", 3, "01"),
            _item("0002", 4),
            _item("0003", 10),
        ],
        max_candidates=3,
        max_total_bytes=12,
        max_item_bytes=9,
    )
    assert [(item.scene_id, item.size) for item in selected] == [
        ("0001", 3),
        ("0002", 4),
    ]


def test_download_is_atomic_and_reuses_only_size_matching_file(
    tmp_path: Path,
) -> None:
    item = ViratItem("abc", "sample.mp4", 4, "scene", "sequence")

    class Response(BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            self.close()

    result = download_item(item, tmp_path, opener=lambda *_args, **_kwargs: Response(b"data"))
    assert result["status"] == "downloaded"
    assert (tmp_path / "sample.mp4").read_bytes() == b"data"
    assert not (tmp_path / "sample.mp4.part").exists()

    existing = download_item(
        item,
        tmp_path,
        opener=lambda *_args, **_kwargs: pytest.fail("network should not be used"),
    )
    assert existing["status"] == "existing"


def test_download_refuses_existing_size_mismatch(tmp_path: Path) -> None:
    item = ViratItem("abc", "sample.mp4", 4, "scene", "sequence")
    (tmp_path / "sample.mp4").write_bytes(b"wrong")
    with pytest.raises(FileExistsError, match="official size"):
        download_item(item, tmp_path)


def test_download_retries_transient_http_error_and_stays_atomic(
    tmp_path: Path,
) -> None:
    item = ViratItem("abc", "sample.mp4", 4, "scene", "sequence")
    responses: list[object] = [
        HTTPError("https://example.invalid", 502, "temporary", {}, None),
        BytesIO(b"data"),
    ]
    delays: list[float] = []

    def opener(*_args, **_kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    result = download_item(
        item,
        tmp_path,
        opener=opener,
        max_attempts=3,
        retry_delay_s=0.5,
        sleeper=delays.append,
    )

    assert result["status"] == "downloaded"
    assert (tmp_path / "sample.mp4").read_bytes() == b"data"
    assert not (tmp_path / "sample.mp4.part").exists()
    assert delays == [0.5]


def test_named_selection_requires_official_names_and_budget() -> None:
    items = [_item("0001", 8), _item("0002", 4)]
    with pytest.raises(ValueError, match="not present"):
        select_named_items(
            items,
            ["unknown.mp4"],
            max_total_bytes=20,
            max_item_bytes=10,
        )
    with pytest.raises(ValueError, match="exceeds budget"):
        select_named_items(
            items,
            [item.name for item in items],
            max_total_bytes=10,
            max_item_bytes=10,
        )


def test_catalog_fetch_retries_transient_http_error_without_duplicate_records() -> None:
    responses: list[object] = [
        HTTPError("https://example.invalid", 502, "temporary", {}, None),
        BytesIO(b'[{"_id": "one"}]'),
    ]
    delays: list[float] = []

    def opener(*_args, **_kwargs):
        response = responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    records = fetch_folder_items(
        "folder",
        page_size=2,
        max_attempts=3,
        retry_delay_s=0.25,
        opener=opener,
        sleeper=delays.append,
    )

    assert records == [{"_id": "one"}]
    assert delays == [0.25]
    assert responses == []


def test_catalog_fetch_does_not_retry_non_transient_http_error() -> None:
    calls = 0

    def opener(*_args, **_kwargs):
        nonlocal calls
        calls += 1
        raise HTTPError("https://example.invalid", 404, "missing", {}, None)

    with pytest.raises(HTTPError) as error:
        fetch_folder_items(
            "folder",
            max_attempts=3,
            opener=opener,
            sleeper=lambda _delay: pytest.fail("must not sleep"),
        )
    assert error.value.code == 404
    assert calls == 1


def test_screening_manifest_reconciles_bytes_hashes_and_official_grouping() -> None:
    path = (
        Path(__file__).parents[1]
        / "data"
        / "manifests"
        / "virat_screening_20260726.yaml"
    )
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    candidates = payload["candidates"]
    assert len(candidates) == payload["screening"]["downloaded_video_count"]
    assert sum(item["bytes"] for item in candidates) == payload["screening"][
        "downloaded_bytes"
    ]
    assert len({item["name"] for item in candidates}) == len(candidates)
    assert sum(
        item["decision"] == "eligible_frame_truth_verified_candidate"
        for item in candidates
    ) == 1
    for record in candidates:
        parsed = parse_video_item(
            {
                "_id": record["item_id"],
                "name": record["name"],
                "size": record["bytes"],
            }
        )
        assert parsed is not None
        assert parsed.scene_id == record["physical_scene_id"]
        assert parsed.sequence_id == record["sequence_id"]
        assert len(record["sha256"]) == 64


def test_local_screening_verifier_checks_size_hash_and_grouping(
    tmp_path: Path,
) -> None:
    video = tmp_path / "round" / "VIRAT_S_050202_10_002159_002233.mp4"
    video.parent.mkdir()
    video.write_bytes(b"video")
    payload = {
        "screening": {
            "downloaded_video_count": 1,
            "downloaded_bytes": 5,
            "budget_bytes": 10,
        },
        "candidates": [
            {
                "name": video.name,
                "item_id": "official",
                "physical_scene_id": "0502",
                "sequence_id": "02",
                "bytes": 5,
                "sha256": hashlib.sha256(b"video").hexdigest(),
            }
        ],
    }
    report = verify_screening_manifest(payload, tmp_path)
    assert report["valid"] is True
    assert report["checked_files"] == 1

    payload["candidates"][0]["physical_scene_id"] = "050202"
    report = verify_screening_manifest(payload, tmp_path)
    assert report["valid"] is False
    assert any("physical scene" in error for error in report["errors"])
