from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from parking_occupancy.artifact_registry import (
    STAGE_S_FROZEN_REGISTRY_SHA256,
    verify_historical_artifact_registry,
)
from parking_occupancy.integrated_cli import (
    DEFAULT_FINAL_INTEGRATED_CONFIG,
    build_parser as build_stage_s_parser,
)
from parking_occupancy.p3_tt_runtime import (
    DEFAULT_P3_TT_CONFIG,
    run_generic_p3_tt,
)
from parking_occupancy.stage_s_release import (
    load_stage_s_config,
)
from parking_occupancy.stage_t_tracktrack import (
    StageTError,
    load_p3_tt_config,
    run_stage_t_variant,
    validate_tracks_schema,
)


def _inputs(tmp_path: Path) -> dict[str, Path]:
    paths = {
        "video": tmp_path / "arbitrary-camera-video.mp4",
        "slots": tmp_path / "camera-polygons.json",
        "d1": tmp_path / "custom-d1.pt",
        "e1b": tmp_path / "custom-e1b.pt",
    }
    paths["video"].write_bytes(b"not VIRAT and intentionally arbitrary")
    paths["slots"].write_text(
        '{"schema_version":1,"source_width":32,"source_height":24,'
        '"coordinate_system":"pixel","slots":[{"id":"s1",'
        '"points":[[2,2],[22,2],[22,22],[2,22]]}]}',
        encoding="utf-8",
    )
    paths["d1"].write_bytes(b"custom D1")
    paths["e1b"].write_bytes(b"custom E1b")
    return paths


def _fake_integrated(calls: list[dict[str, object]]):
    def run(**kwargs):
        calls.append(kwargs)
        output = Path(kwargs["output_root"]).resolve()
        output.mkdir(parents=True)
        source_id = str(kwargs["source_id"])
        with (output / "occupancy.csv").open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=[
                    "video_id",
                    "frame_index",
                    "timestamp_s",
                    "slot_id",
                    "state",
                    "track_id",
                ],
                lineterminator="\n",
            )
            writer.writeheader()
            writer.writerow(
                {
                    "video_id": source_id,
                    "frame_index": 0,
                    "timestamp_s": 0,
                    "slot_id": "s1",
                    "state": 1,
                    "track_id": 7,
                }
            )
        (output / "events.csv").write_text(
            "video_id,frame_index,timestamp_s,slot_id,from_state,to_state\n",
            encoding="utf-8",
        )
        (output / "detections.jsonl").write_text(
            json.dumps(
                {
                    "video_id": source_id,
                    "frame_index": 0,
                    "timestamp_s": 0.0,
                    "detections": [
                        {
                            "bbox": [1, 1, 9, 9],
                            "confidence": 0.9,
                            "class_id": 0,
                            "class_name": "vehicle",
                            "track_id": 7,
                        }
                    ],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        (output / "annotated.mp4").write_bytes(b"mock annotated")
        (output / "metrics.json").write_text(
            '{"status":"not_computed_no_truth"}\n',
            encoding="utf-8",
        )
        (output / "summary.json").write_text(
            json.dumps(
                {
                    "method_id": "P3",
                    "source_id": source_id,
                    "temporal_enabled": kwargs["temporal_enabled"],
                    "tracker_backend": kwargs["tracker_backend"],
                }
            ),
            encoding="utf-8",
        )
        (output / "runtime_metadata.json").write_text(
            json.dumps(
                {
                    "source_state_reset": {
                        "source_id": source_id,
                        "detector_begin_source_called": True,
                        "event_state_reinitialized": True,
                    },
                    "timing": {},
                }
            ),
            encoding="utf-8",
        )
        return {"source_id": source_id}

    return run


def test_generic_runner_accepts_non_virat_and_forces_tracktrack(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    calls: list[dict[str, object]] = []
    output = tmp_path / "generic-output"
    summary = run_generic_p3_tt(
        input_path=paths["video"],
        slots_path=paths["slots"],
        detector_weights=paths["d1"],
        classifier_checkpoint=paths["e1b"],
        source_id="camera-arbitrary",
        output_root=output,
        integrated_runner_fn=_fake_integrated(calls),
    )
    assert calls[0]["temporal_enabled"] is False
    assert calls[0]["tracker_backend"] == "tracktrack"
    assert summary["custom_weights"] is True
    assert summary["stage_t_result_comparison_applicable"] is False
    assert summary["tracker_state_reused"] is False
    assert not (output / "metrics.json").exists()


def test_generic_runner_resets_by_constructing_each_source_run_fresh(
    tmp_path: Path,
) -> None:
    paths = _inputs(tmp_path)
    calls: list[dict[str, object]] = []
    fake = _fake_integrated(calls)
    for index, source_id in enumerate(("camera-a", "camera-b")):
        run_generic_p3_tt(
            input_path=paths["video"],
            slots_path=paths["slots"],
            detector_weights=paths["d1"],
            classifier_checkpoint=paths["e1b"],
            source_id=source_id,
            output_root=tmp_path / f"output-{index}",
            integrated_runner_fn=fake,
        )
    assert [call["source_id"] for call in calls] == ["camera-a", "camera-b"]
    assert all(call["temporal_enabled"] is False for call in calls)
    for index, source_id in enumerate(("camera-a", "camera-b")):
        runtime = json.loads(
            (tmp_path / f"output-{index}" / "runtime_metadata.json").read_text(
                encoding="utf-8"
            )
        )
        assert runtime["source_state_reset"]["source_id"] == source_id
        assert runtime["tracker_state_reused"] is False
        assert runtime["event_state_reused"] is False
        assert runtime["temporal_state_reused"] is False


def test_generic_tracks_jsonl_schema_uses_existing_writer(tmp_path: Path) -> None:
    paths = _inputs(tmp_path)
    output = tmp_path / "generic-output"
    run_generic_p3_tt(
        input_path=paths["video"],
        slots_path=paths["slots"],
        detector_weights=paths["d1"],
        classifier_checkpoint=paths["e1b"],
        source_id="camera-a",
        output_root=output,
        integrated_runner_fn=_fake_integrated([]),
    )
    records = [
        json.loads(line)
        for line in (output / "tracks.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    validate_tracks_schema(records)
    assert records[0]["tracks"][0]["track_id"] == 7
    assert records[0]["tracks"][0]["assigned_slot_ids"] == ["s1"]


def test_frozen_stage_t_runner_still_rejects_wrong_virat_hash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = _inputs(tmp_path)
    frozen_config = load_p3_tt_config(DEFAULT_P3_TT_CONFIG)
    monkeypatch.setattr(
        "parking_occupancy.stage_t_tracktrack.load_p3_tt_config",
        lambda _: frozen_config,
    )
    monkeypatch.setattr(
        "parking_occupancy.stage_t_tracktrack.sha256_file",
        lambda _: "wrong-video-hash",
    )
    with pytest.raises(StageTError, match="VIRAT.*SHA-256 changed"):
        run_stage_t_variant(
            variant_id="TT0",
            tracker_backend="none",
            input_path=paths["video"],
            slots_path=paths["slots"],
            detector_weights=paths["d1"],
            classifier_checkpoint=paths["e1b"],
            output_root=tmp_path / "frozen-output",
            config_path=DEFAULT_P3_TT_CONFIG,
            truth_path=None,
            source_id="wrong-virat",
        )


def test_stage_s_default_and_registry_remain_unchanged(tmp_path: Path) -> None:
    args = build_stage_s_parser().parse_args(
        [
            "--input",
            str(tmp_path / "input.mp4"),
            "--slots",
            str(tmp_path / "slots.json"),
            "--d1-weights",
            str(tmp_path / "d1.pt"),
            "--e1b-checkpoint",
            str(tmp_path / "e1b.pt"),
            "--output-dir",
            str(tmp_path / "output"),
        ]
    )
    assert args.config.resolve() == DEFAULT_FINAL_INTEGRATED_CONFIG.resolve()
    config = load_stage_s_config(args.config)
    assert config["temporal"]["default_enabled"] is False
    assert config["tracking"]["default_backend"] == "none"
    implementation_root = Path(__file__).resolve().parents[1]
    registry = verify_historical_artifact_registry(
        implementation_root
        / "data"
        / "stage_s"
        / "STAGE_S_ARTIFACT_REGISTRY_20260729.yaml",
        artifact_root=implementation_root.parent,
        expected_registry_sha256=STAGE_S_FROZEN_REGISTRY_SHA256,
        immutable_path_prefixes=("implementation/data/stage_s",),
    )
    assert registry["verified"] is True
