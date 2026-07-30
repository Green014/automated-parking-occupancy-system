import hashlib
import json
from pathlib import Path

from literature_core.temporal_protocol import validate_temporal_protocol


def _protocol(status: str = "pending_access") -> dict:
    return {
        "schema_version": 1,
        "status": status,
        "dataset": {
            "id": "example",
            "access": {
                "requires_individual_acceptance": True,
                "user_acceptance_recorded": False,
            },
        },
        "split_policy": {
            "group_unit": "scene",
            "allow_same_scene_fallback": False,
            "minimum_guard_frames": 250,
        },
        "holdout_locked_before_development": False,
        "truth_requirements": {
            "slot_polygons_verified": False,
            "mixed_classes_verified": False,
            "transition_verified": False,
            "annotation_consistency_checked": False,
        },
        "validation_requirements": {
            "minimum_slots": 1,
            "minimum_occupied_slot_frames": 1,
            "minimum_vacant_slot_frames": 1,
            "minimum_transitions_per_partition": 1,
        },
        "selections": {"development": None, "holdout": None},
    }


def _selection(
    partition_id: str,
    scene_id: str,
    start_frame: int,
    end_frame: int,
    *,
    source_video_id: str = "video",
    source_path: str = "datasets/video.bin",
    truth_path: str = "data/truth.json",
    source_sha256: str = "a" * 64,
) -> dict:
    return {
        "dataset_id": "example",
        "scene_id": scene_id,
        "source_video_id": source_video_id,
        "partition_id": partition_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "source_path": source_path,
        "truth_path": truth_path,
        "source_sha256": source_sha256,
    }


def _write_artifacts(
    root: Path,
    *,
    scene_id: str,
    source_video_id: str,
    source_name: str,
    truth_name: str,
) -> tuple[str, str, str]:
    source = root / "datasets" / source_name
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_bytes(f"{scene_id}-{source_video_id}".encode())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    truth = {
        "schema_version": 1,
        "dataset_id": "example",
        "scene_id": scene_id,
        "source_video_id": source_video_id,
        "source_sha256": digest,
        "video": {"frame_count": 1_000, "width": 100, "height": 80},
        "review": {"status": "verified"},
        "slots": [
            {
                "slot_id": "slot-1",
                "polygon": [[10, 10], [30, 10], [30, 40], [10, 40]],
                "intervals": [
                    {
                        "start_frame": 0,
                        "end_frame": 500,
                        "state": "occupied",
                    },
                    {
                        "start_frame": 500,
                        "end_frame": 1_000,
                        "state": "vacant",
                    },
                ],
            }
        ],
    }
    truth_path = root / "data" / truth_name
    truth_path.parent.mkdir(parents=True, exist_ok=True)
    truth_path.write_text(json.dumps(truth), encoding="utf-8")
    return (
        str(source.relative_to(root)),
        str(truth_path.relative_to(root)),
        digest,
    )


def _probe(_: Path) -> dict:
    return {"frame_count": 1_000, "width": 100, "height": 80, "fps": 25.0}


def test_pending_protocol_is_valid_but_not_experiment_ready() -> None:
    report = validate_temporal_protocol(_protocol())
    assert report["schema_valid"] is True
    assert report["ready_for_experiment"] is False


def test_pending_candidate_artifacts_are_verified_without_opening_gate(
    tmp_path: Path,
) -> None:
    payload = _protocol("candidate_screening")
    source, truth, digest = _write_artifacts(
        tmp_path,
        scene_id="scene-1",
        source_video_id="video-1",
        source_name="candidate.bin",
        truth_name="candidate.json",
    )
    payload["selections"]["candidate"] = _selection(
        "unassigned",
        "scene-1",
        0,
        1_000,
        source_video_id="video-1",
        source_path=source,
        truth_path=truth,
        source_sha256=digest,
    )
    report = validate_temporal_protocol(
        payload,
        project_root=tmp_path,
        video_probe=_probe,
    )
    assert report["schema_valid"] is True
    assert report["ready_for_experiment"] is False
    assert report["artifact_validation"]["candidate"]["verified"] is True


def test_slot_level_grouping_is_rejected() -> None:
    payload = _protocol()
    payload["split_policy"]["group_unit"] = "slot"
    report = validate_temporal_protocol(payload)
    assert report["schema_valid"] is False
    assert any("slot/frame-level" in error for error in report["errors"])


def test_same_video_adjacent_partitions_violate_guard() -> None:
    payload = _protocol("frozen")
    payload["dataset"]["access"]["user_acceptance_recorded"] = True
    payload["holdout_locked_before_development"] = True
    payload["truth_requirements"] = {
        key: True for key in payload["truth_requirements"]
    }
    payload["split_policy"]["allow_same_scene_fallback"] = True
    payload["selections"] = {
        "development": _selection("dev", "scene-1", 0, 1_000),
        "holdout": _selection("holdout", "scene-1", 1_100, 2_000),
    }
    report = validate_temporal_protocol(payload)
    assert report["schema_valid"] is False
    assert any("temporal guard" in error for error in report["errors"])


def test_frozen_protocol_without_artifact_root_is_not_ready() -> None:
    payload = _protocol("frozen")
    payload["dataset"]["access"]["user_acceptance_recorded"] = True
    payload["holdout_locked_before_development"] = True
    payload["truth_requirements"] = {
        key: True for key in payload["truth_requirements"]
    }
    payload["selections"] = {
        "development": _selection("dev", "scene-1", 0, 1_000),
        "holdout": _selection(
            "holdout",
            "scene-2",
            0,
            1_000,
            source_video_id="video-2",
        ),
    }
    report = validate_temporal_protocol(payload)
    assert report["schema_valid"] is False
    assert report["ready_for_experiment"] is False
    assert any("requires project_root" in error for error in report["errors"])


def test_distinct_scene_frozen_protocol_requires_verified_artifacts(
    tmp_path: Path,
) -> None:
    payload = _protocol("frozen")
    payload["dataset"]["access"]["user_acceptance_recorded"] = True
    payload["holdout_locked_before_development"] = True
    payload["truth_requirements"] = {
        key: True for key in payload["truth_requirements"]
    }
    dev_source, dev_truth, dev_digest = _write_artifacts(
        tmp_path,
        scene_id="scene-1",
        source_video_id="video-1",
        source_name="dev.bin",
        truth_name="dev.json",
    )
    hold_source, hold_truth, hold_digest = _write_artifacts(
        tmp_path,
        scene_id="scene-2",
        source_video_id="video-2",
        source_name="holdout.bin",
        truth_name="holdout.json",
    )
    payload["selections"] = {
        "development": _selection(
            "dev",
            "scene-1",
            0,
            1_000,
            source_video_id="video-1",
            source_path=dev_source,
            truth_path=dev_truth,
            source_sha256=dev_digest,
        ),
        "holdout": _selection(
            "holdout",
            "scene-2",
            0,
            1_000,
            source_video_id="video-2",
            source_path=hold_source,
            truth_path=hold_truth,
            source_sha256=hold_digest,
        ),
    }
    report = validate_temporal_protocol(
        payload,
        project_root=tmp_path,
        video_probe=_probe,
    )
    assert report["schema_valid"] is True
    assert report["ready_for_experiment"] is True
    assert report["artifact_validation"]["development"]["verified"] is True
    assert report["artifact_validation"]["holdout"]["truth"][
        "transition_count"
    ] == 1


def test_frozen_protocol_rejects_hash_bounds_and_missing_transition(
    tmp_path: Path,
) -> None:
    payload = _protocol("frozen")
    payload["dataset"]["access"]["user_acceptance_recorded"] = True
    payload["holdout_locked_before_development"] = True
    payload["truth_requirements"] = {
        key: True for key in payload["truth_requirements"]
    }
    dev_source, dev_truth, _ = _write_artifacts(
        tmp_path,
        scene_id="scene-1",
        source_video_id="video-1",
        source_name="dev.bin",
        truth_name="dev.json",
    )
    hold_source, hold_truth, hold_digest = _write_artifacts(
        tmp_path,
        scene_id="scene-2",
        source_video_id="video-2",
        source_name="holdout.bin",
        truth_name="holdout.json",
    )
    hold_truth_path = tmp_path / hold_truth
    hold_payload = json.loads(hold_truth_path.read_text(encoding="utf-8"))
    hold_payload["slots"][0]["intervals"] = [
        {"start_frame": 0, "end_frame": 1_000, "state": "occupied"}
    ]
    hold_truth_path.write_text(json.dumps(hold_payload), encoding="utf-8")
    payload["selections"] = {
        "development": _selection(
            "dev",
            "scene-1",
            0,
            1_001,
            source_video_id="video-1",
            source_path=dev_source,
            truth_path=dev_truth,
            source_sha256="0" * 64,
        ),
        "holdout": _selection(
            "holdout",
            "scene-2",
            0,
            1_000,
            source_video_id="video-2",
            source_path=hold_source,
            truth_path=hold_truth,
            source_sha256=hold_digest,
        ),
    }
    report = validate_temporal_protocol(
        payload,
        project_root=tmp_path,
        video_probe=_probe,
    )
    assert report["schema_valid"] is False
    assert report["ready_for_experiment"] is False
    assert any("does not match the file" in error for error in report["errors"])
    assert any("exceeds source frame_count" in error for error in report["errors"])
    assert any("transition_count" in error for error in report["errors"])
