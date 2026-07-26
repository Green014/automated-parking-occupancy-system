"""Validation rules for leakage-safe temporal dataset protocols."""

from __future__ import annotations

import re
from typing import Any

_ALLOWED_STATUSES = {"pending_access", "candidate_screening", "frozen"}
_ALLOWED_GROUP_UNITS = {"camera", "video", "scene"}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def _selection_errors(
    role: str,
    selection: Any,
    *,
    require_hash: bool,
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

    if require_hash:
        digest = str(selection.get("source_sha256", "")).lower()
        if not _SHA256.fullmatch(digest):
            errors.append(
                f"selections.{role}.source_sha256 must be a 64-character "
                "SHA-256 digest when the protocol is frozen"
            )
    return errors


def validate_temporal_protocol(payload: dict[str, Any]) -> dict[str, Any]:
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
                require_hash=require_selection,
            )
        )
        if isinstance(selection, dict):
            parsed[role] = selection

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
        "errors": errors,
        "warnings": warnings,
    }
