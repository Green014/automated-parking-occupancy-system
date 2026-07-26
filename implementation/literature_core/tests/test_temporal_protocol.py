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
        "selections": {"development": None, "holdout": None},
    }


def _selection(
    partition_id: str,
    scene_id: str,
    start_frame: int,
    end_frame: int,
    *,
    source_video_id: str = "video",
) -> dict:
    return {
        "dataset_id": "example",
        "scene_id": scene_id,
        "source_video_id": source_video_id,
        "partition_id": partition_id,
        "start_frame": start_frame,
        "end_frame": end_frame,
        "source_sha256": "a" * 64,
    }


def test_pending_protocol_is_valid_but_not_experiment_ready() -> None:
    report = validate_temporal_protocol(_protocol())
    assert report["schema_valid"] is True
    assert report["ready_for_experiment"] is False


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


def test_distinct_scene_frozen_protocol_is_ready() -> None:
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
    assert report["schema_valid"] is True
    assert report["ready_for_experiment"] is True
