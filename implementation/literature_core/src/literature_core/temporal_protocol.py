"""Validation rules for leakage-safe temporal dataset protocols."""

from __future__ import annotations

from collections.abc import Callable
import hashlib
import json
import math
from pathlib import Path
import re
from typing import Any

import yaml

_ALLOWED_STATUSES = {"pending_access", "candidate_screening", "frozen"}
_ALLOWED_GROUP_UNITS = {"camera", "video", "scene"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_STATES = {"occupied", "vacant"}
VideoProbe = Callable[[Path], dict[str, Any]]


def _selection_errors(
    role: str,
    selection: Any,
    *,
    require_artifacts: bool,
) -> list[str]:
    if not isinstance(selection, dict):
        return [f"selections.{role} must be a mapping"]

    errors: list[str] = []
    for key in (
        "dataset_id",
        "scene_id",
        "source_video_id",
        "partition_id",
        "start_frame",
        "end_frame",
    ):
        if key not in selection:
            errors.append(f"selections.{role}.{key} is required")

    if errors:
        return errors

    start = selection["start_frame"]
    end = selection["end_frame"]
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        errors.append(f"selections.{role}.start_frame must be a non-negative integer")
    if not isinstance(end, int) or isinstance(end, bool):
        errors.append(f"selections.{role}.end_frame must be an integer")
    elif isinstance(start, int) and not isinstance(start, bool) and end <= start:
        errors.append(
            f"selections.{role}.end_frame must be greater than start_frame"
        )

    if require_artifacts:
        for key in ("source_path", "truth_path"):
            value = selection.get(key)
            if not isinstance(value, str) or not value.strip():
                errors.append(
                    f"selections.{role}.{key} is required for artifact verification"
                )
        digest = str(selection.get("source_sha256", "")).lower()
        if not _SHA256.fullmatch(digest):
            errors.append(
                f"selections.{role}.source_sha256 must be a 64-character "
                "SHA-256 digest for artifact verification"
            )
    return errors


def _sha256_file(path: Path, *, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_project_path(
    project_root: Path,
    raw_path: Any,
    *,
    field: str,
) -> tuple[Path | None, list[str]]:
    if not isinstance(raw_path, str) or not raw_path.strip():
        return None, [f"{field} must be a non-empty relative path"]
    candidate = Path(raw_path)
    if candidate.is_absolute():
        return None, [f"{field} must be relative to the project root"]
    root = project_root.resolve()
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        return None, [f"{field} must remain inside the project root"]
    return resolved, []


def _default_video_probe(path: Path) -> dict[str, Any]:
    import cv2

    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError("OpenCV could not open the source video")
    try:
        return {
            "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
            "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        }
    finally:
        capture.release()


def _read_truth(path: Path) -> dict[str, Any]:
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("truth root must be a mapping")
    return payload


def _positive_requirement(
    requirements: dict[str, Any],
    key: str,
    *,
    default: int,
    errors: list[str],
) -> int:
    value = requirements.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 1:
        errors.append(f"validation_requirements.{key} must be a positive integer")
        return default
    return value


def _truth_errors(
    role: str,
    truth: dict[str, Any],
    selection: dict[str, Any],
    video: dict[str, Any],
    requirements: dict[str, int],
) -> tuple[list[str], dict[str, Any]]:
    prefix = f"selections.{role}.truth"
    errors: list[str] = []
    summary: dict[str, Any] = {
        "slot_count": 0,
        "occupied_slot_frames": 0,
        "vacant_slot_frames": 0,
        "transition_count": 0,
    }

    if truth.get("schema_version") != 1:
        errors.append(f"{prefix}.schema_version must equal 1")
    for key in ("dataset_id", "scene_id", "source_video_id", "source_sha256"):
        expected = (
            str(selection.get(key, "")).lower()
            if key == "source_sha256"
            else selection.get(key)
        )
        actual = (
            str(truth.get(key, "")).lower()
            if key == "source_sha256"
            else truth.get(key)
        )
        if actual != expected:
            errors.append(f"{prefix}.{key} must match selections.{role}.{key}")

    review = truth.get("review")
    if not isinstance(review, dict) or review.get("status") != "verified":
        errors.append(f"{prefix}.review.status must equal verified")

    video_truth = truth.get("video")
    if not isinstance(video_truth, dict):
        errors.append(f"{prefix}.video must be a mapping")
        video_truth = {}
    for key in ("frame_count", "width", "height"):
        actual = video.get(key)
        if (
            not isinstance(actual, int)
            or isinstance(actual, bool)
            or actual < 1
        ):
            errors.append(f"{prefix}: probed video {key} must be positive")
        if video_truth.get(key) != actual:
            errors.append(f"{prefix}.video.{key} does not match the source video")

    frame_count = video.get("frame_count")
    width = video.get("width")
    height = video.get("height")
    start_frame = selection.get("start_frame")
    end_frame = selection.get("end_frame")
    valid_selection_bounds = (
        isinstance(start_frame, int)
        and not isinstance(start_frame, bool)
        and isinstance(end_frame, int)
        and not isinstance(end_frame, bool)
        and 0 <= start_frame < end_frame
    )
    if not valid_selection_bounds:
        errors.append(
            f"selections.{role} needs valid frame bounds before truth "
            "coverage can be verified"
        )
        return errors, summary
    if (
        isinstance(frame_count, int)
        and end_frame > frame_count
    ):
        errors.append(
            f"selections.{role}.end_frame exceeds source frame_count "
            f"({end_frame} > {frame_count})"
        )

    slots = truth.get("slots")
    if not isinstance(slots, list):
        errors.append(f"{prefix}.slots must be a list")
        return errors, summary
    slot_ids: set[str] = set()
    for index, slot in enumerate(slots):
        slot_prefix = f"{prefix}.slots[{index}]"
        if not isinstance(slot, dict):
            errors.append(f"{slot_prefix} must be a mapping")
            continue
        slot_id = slot.get("slot_id")
        if not isinstance(slot_id, str) or not slot_id.strip():
            errors.append(f"{slot_prefix}.slot_id is required")
        elif slot_id in slot_ids:
            errors.append(f"{slot_prefix}.slot_id must be unique")
        else:
            slot_ids.add(slot_id)

        polygon = slot.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            errors.append(f"{slot_prefix}.polygon must contain at least 3 points")
        else:
            valid_polygon_points: list[tuple[float, float]] = []
            for point_index, point in enumerate(polygon):
                valid_point = (
                    isinstance(point, list)
                    and len(point) == 2
                    and all(
                        isinstance(value, (int, float))
                        and not isinstance(value, bool)
                        and math.isfinite(float(value))
                        for value in point
                    )
                )
                if not valid_point:
                    errors.append(
                        f"{slot_prefix}.polygon[{point_index}] must be [x, y]"
                    )
                    continue
                x, y = point
                valid_polygon_points.append((float(x), float(y)))
                if (
                    isinstance(width, int)
                    and isinstance(height, int)
                    and not (0 <= x < width and 0 <= y < height)
                ):
                    errors.append(
                        f"{slot_prefix}.polygon[{point_index}] is outside "
                        "the source image"
                    )
            if len(valid_polygon_points) == len(polygon):
                twice_area = abs(
                    sum(
                        x1 * y2 - x2 * y1
                        for (x1, y1), (x2, y2) in zip(
                            valid_polygon_points,
                            valid_polygon_points[1:] + valid_polygon_points[:1],
                        )
                    )
                )
                if twice_area == 0:
                    errors.append(f"{slot_prefix}.polygon must have non-zero area")

        intervals = slot.get("intervals")
        if not isinstance(intervals, list) or not intervals:
            errors.append(f"{slot_prefix}.intervals must be a non-empty list")
            continue
        previous_end: int | None = None
        previous_state: str | None = None
        covered_start: int | None = None
        covered_end: int | None = None
        for interval_index, interval in enumerate(intervals):
            interval_prefix = f"{slot_prefix}.intervals[{interval_index}]"
            if not isinstance(interval, dict):
                errors.append(f"{interval_prefix} must be a mapping")
                continue
            interval_start = interval.get("start_frame")
            interval_end = interval.get("end_frame")
            state = interval.get("state")
            valid_bounds = (
                isinstance(interval_start, int)
                and not isinstance(interval_start, bool)
                and isinstance(interval_end, int)
                and not isinstance(interval_end, bool)
                and interval_start >= 0
                and interval_end > interval_start
            )
            if not valid_bounds:
                errors.append(
                    f"{interval_prefix} must have an increasing non-negative "
                    "half-open frame interval"
                )
                continue
            if state not in _STATES:
                errors.append(
                    f"{interval_prefix}.state must be occupied or vacant"
                )
                continue
            if isinstance(frame_count, int) and interval_end > frame_count:
                errors.append(
                    f"{interval_prefix}.end_frame exceeds source frame_count"
                )
            if previous_end is not None and interval_start != previous_end:
                errors.append(
                    f"{interval_prefix} must be contiguous with the prior interval"
                )
            overlap_start = max(interval_start, start_frame)
            overlap_end = min(interval_end, end_frame)
            if overlap_end > overlap_start:
                frames = overlap_end - overlap_start
                summary[f"{state}_slot_frames"] += frames
                covered_start = (
                    overlap_start if covered_start is None else min(covered_start, overlap_start)
                )
                covered_end = (
                    overlap_end if covered_end is None else max(covered_end, overlap_end)
                )
                if (
                    previous_state is not None
                    and previous_state != state
                    and start_frame < interval_start < end_frame
                ):
                    summary["transition_count"] += 1
            previous_end = interval_end
            previous_state = state
        if covered_start != start_frame or covered_end != end_frame:
            errors.append(
                f"{slot_prefix}.intervals must cover the complete selected frame range"
            )

    summary["slot_count"] = len(slot_ids)
    checks = (
        ("slot_count", "minimum_slots"),
        ("occupied_slot_frames", "minimum_occupied_slot_frames"),
        ("vacant_slot_frames", "minimum_vacant_slot_frames"),
        ("transition_count", "minimum_transitions_per_partition"),
    )
    for value_key, requirement_key in checks:
        if summary[value_key] < requirements[requirement_key]:
            errors.append(
                f"selections.{role} has {summary[value_key]} {value_key}, below "
                f"validation_requirements.{requirement_key}="
                f"{requirements[requirement_key]}"
            )
    return errors, summary


def _artifact_errors(
    role: str,
    selection: dict[str, Any],
    *,
    project_root: Path,
    video_probe: VideoProbe,
    requirements: dict[str, int],
) -> tuple[list[str], dict[str, Any]]:
    errors: list[str] = []
    summary: dict[str, Any] = {"verified": False}
    source, source_errors = _resolve_project_path(
        project_root,
        selection.get("source_path"),
        field=f"selections.{role}.source_path",
    )
    truth_path, truth_path_errors = _resolve_project_path(
        project_root,
        selection.get("truth_path"),
        field=f"selections.{role}.truth_path",
    )
    errors.extend(source_errors)
    errors.extend(truth_path_errors)
    if errors or source is None or truth_path is None:
        return errors, summary
    if not source.is_file():
        errors.append(f"selections.{role}.source_path does not exist: {source}")
    if not truth_path.is_file():
        errors.append(f"selections.{role}.truth_path does not exist: {truth_path}")
    if errors:
        return errors, summary

    expected_digest = str(selection.get("source_sha256", "")).lower()
    actual_digest = _sha256_file(source)
    summary["source_sha256"] = actual_digest
    if actual_digest != expected_digest:
        errors.append(f"selections.{role}.source_sha256 does not match the file")
    try:
        video = video_probe(source)
    except Exception as error:
        errors.append(f"selections.{role}.source_path probe failed: {error}")
        return errors, summary
    if not isinstance(video, dict):
        errors.append(
            f"selections.{role}.source_path probe must return a metadata mapping"
        )
        return errors, summary
    summary["video"] = video
    try:
        truth = _read_truth(truth_path)
    except Exception as error:
        errors.append(f"selections.{role}.truth_path could not be read: {error}")
        return errors, summary
    truth_validation_errors, truth_summary = _truth_errors(
        role,
        truth,
        selection,
        video,
        requirements,
    )
    errors.extend(truth_validation_errors)
    summary["truth"] = truth_summary
    summary["verified"] = not errors
    return errors, summary


def validate_temporal_protocol(
    payload: dict[str, Any],
    *,
    project_root: Path | None = None,
    video_probe: VideoProbe | None = None,
) -> dict[str, Any]:
    """Validate access, grouping, and development/holdout separation.

    Frame intervals use half-open ``[start_frame, end_frame)`` semantics.
    A valid pending protocol is intentionally not ready for experiments.
    """

    errors: list[str] = []
    warnings: list[str] = []

    if payload.get("schema_version") != 1:
        errors.append("schema_version must equal 1")

    status = payload.get("status")
    if status not in _ALLOWED_STATUSES:
        errors.append(
            "status must be one of pending_access, candidate_screening, or frozen"
        )

    dataset = payload.get("dataset")
    if not isinstance(dataset, dict):
        errors.append("dataset must be a mapping")
        dataset = {}
    dataset_id = dataset.get("id")
    if not isinstance(dataset_id, str) or not dataset_id.strip():
        errors.append("dataset.id is required")

    access = dataset.get("access")
    if not isinstance(access, dict):
        errors.append("dataset.access must be a mapping")
        access = {}
    requires_acceptance = access.get("requires_individual_acceptance", False)
    acceptance_recorded = access.get("user_acceptance_recorded", False)
    if not isinstance(requires_acceptance, bool):
        errors.append(
            "dataset.access.requires_individual_acceptance must be boolean"
        )
    if not isinstance(acceptance_recorded, bool):
        errors.append("dataset.access.user_acceptance_recorded must be boolean")

    split_policy = payload.get("split_policy")
    if not isinstance(split_policy, dict):
        errors.append("split_policy must be a mapping")
        split_policy = {}
    group_unit = split_policy.get("group_unit")
    if group_unit not in _ALLOWED_GROUP_UNITS:
        errors.append(
            "split_policy.group_unit must be camera, video, or scene; "
            "slot/frame-level grouping is prohibited"
        )
    allow_same_scene = split_policy.get("allow_same_scene_fallback", False)
    if not isinstance(allow_same_scene, bool):
        errors.append("split_policy.allow_same_scene_fallback must be boolean")
        allow_same_scene = False
    guard_frames = split_policy.get("minimum_guard_frames", 0)
    if (
        not isinstance(guard_frames, int)
        or isinstance(guard_frames, bool)
        or guard_frames < 0
    ):
        errors.append("split_policy.minimum_guard_frames must be non-negative")
        guard_frames = 0

    selections = payload.get("selections")
    if not isinstance(selections, dict):
        errors.append("selections must be a mapping")
        selections = {}

    truth_requirements = payload.get("truth_requirements")
    if not isinstance(truth_requirements, dict):
        errors.append("truth_requirements must be a mapping")
        truth_requirements = {}
    truth_keys = (
        "slot_polygons_verified",
        "mixed_classes_verified",
        "transition_verified",
        "annotation_consistency_checked",
    )
    for key in truth_keys:
        if not isinstance(truth_requirements.get(key), bool):
            errors.append(f"truth_requirements.{key} must be boolean")

    raw_validation_requirements = payload.get("validation_requirements", {})
    if not isinstance(raw_validation_requirements, dict):
        errors.append("validation_requirements must be a mapping")
        raw_validation_requirements = {}
    requirement_errors: list[str] = []
    validation_requirements = {
        key: _positive_requirement(
            raw_validation_requirements,
            key,
            default=1,
            errors=requirement_errors,
        )
        for key in (
            "minimum_slots",
            "minimum_occupied_slot_frames",
            "minimum_vacant_slot_frames",
            "minimum_transitions_per_partition",
        )
    }
    if status == "frozen" or selections.get("candidate") is not None:
        errors.extend(requirement_errors)

    require_selection = status == "frozen"
    parsed: dict[str, dict[str, Any]] = {}
    for role in ("development", "holdout"):
        selection = selections.get(role)
        if selection is None and not require_selection:
            continue
        errors.extend(
            _selection_errors(
                role,
                selection,
                require_artifacts=require_selection,
            )
        )
        if isinstance(selection, dict):
            parsed[role] = selection

    candidate_selection = selections.get("candidate")
    if candidate_selection is not None:
        errors.extend(
            _selection_errors(
                "candidate",
                candidate_selection,
                require_artifacts=True,
            )
        )
        if isinstance(candidate_selection, dict):
            if candidate_selection.get("dataset_id") != dataset_id:
                errors.append("selections.candidate.dataset_id must equal dataset.id")
            parsed["candidate"] = candidate_selection

    if status == "frozen":
        if requires_acceptance and not acceptance_recorded:
            errors.append(
                "the dataset requires individual acceptance, but no user "
                "acceptance is recorded"
            )
        if payload.get("holdout_locked_before_development") is not True:
            errors.append(
                "holdout_locked_before_development must be true for a frozen "
                "protocol"
            )
        missing_truth = [
            key for key in truth_keys if truth_requirements.get(key) is not True
        ]
        if missing_truth:
            errors.append(
                "frozen protocol requires verified temporal truth: "
                + ", ".join(missing_truth)
            )

    if {"development", "holdout"}.issubset(parsed):
        development = parsed["development"]
        holdout = parsed["holdout"]
        for role, selection in parsed.items():
            if selection.get("dataset_id") != dataset_id:
                errors.append(
                    f"selections.{role}.dataset_id must equal dataset.id"
                )
        if development.get("partition_id") == holdout.get("partition_id"):
            errors.append("development and holdout partition_id values must differ")

        same_dataset = development.get("dataset_id") == holdout.get("dataset_id")
        same_scene = (
            same_dataset
            and development.get("scene_id") == holdout.get("scene_id")
        )
        same_video = (
            same_scene
            and development.get("source_video_id")
            == holdout.get("source_video_id")
        )
        if same_scene and not allow_same_scene:
            errors.append(
                "development and holdout use the same scene while "
                "allow_same_scene_fallback is false"
            )
        elif same_scene and not same_video:
            errors.append(
                "same-scene fallback requires one source_video_id so temporal "
                "separation can be verified"
            )
        elif same_video:
            dev_start = development.get("start_frame")
            dev_end = development.get("end_frame")
            hold_start = holdout.get("start_frame")
            hold_end = holdout.get("end_frame")
            if all(
                isinstance(value, int) and not isinstance(value, bool)
                for value in (dev_start, dev_end, hold_start, hold_end)
            ):
                separated = (
                    dev_end + guard_frames <= hold_start
                    or hold_end + guard_frames <= dev_start
                )
                if not separated:
                    errors.append(
                        "development/holdout intervals overlap or violate the "
                        f"{guard_frames}-frame temporal guard"
                    )
                else:
                    warnings.append(
                        "same-scene temporal fallback is separated by the "
                        "declared frame guard; report this weaker boundary"
                    )

    artifacts: dict[str, Any] = {}
    if "candidate" in parsed:
        if project_root is None:
            errors.append(
                "declared candidate verification requires project_root so the "
                "source video, hash, frame bounds, and truth file can be verified"
            )
        else:
            role_errors, role_summary = _artifact_errors(
                "candidate",
                parsed["candidate"],
                project_root=project_root,
                video_probe=video_probe or _default_video_probe,
                requirements=validation_requirements,
            )
            errors.extend(role_errors)
            artifacts["candidate"] = role_summary
    if status == "frozen":
        if project_root is None:
            errors.append(
                "frozen protocol requires project_root so source videos, hashes, "
                "frame bounds, and truth files can be verified"
            )
        else:
            probe = video_probe or _default_video_probe
            for role in ("development", "holdout"):
                selection = parsed.get(role)
                if not isinstance(selection, dict):
                    continue
                role_errors, role_summary = _artifact_errors(
                    role,
                    selection,
                    project_root=project_root,
                    video_probe=probe,
                    requirements=validation_requirements,
                )
                errors.extend(role_errors)
                artifacts[role] = role_summary

    schema_valid = not errors
    ready_for_experiment = schema_valid and status == "frozen"
    if schema_valid and not ready_for_experiment:
        warnings.append(
            "protocol is structurally valid but not frozen; E4/E5/Fusion V2 "
            "experiments must not start"
        )

    return {
        "schema_version": 1,
        "protocol_status": status,
        "schema_valid": schema_valid,
        "ready_for_experiment": ready_for_experiment,
        "validation_requirements": validation_requirements,
        "artifact_validation": artifacts,
        "errors": errors,
        "warnings": warnings,
    }
