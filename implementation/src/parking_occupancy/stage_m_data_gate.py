from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import yaml

from .stage_m_tracking import StageMProtocolError, sha256_file


@dataclass(frozen=True, slots=True)
class GateDecision:
    gate_id: str
    status: str
    allowed_claim: str
    reasons: tuple[str, ...]


def _timestamp(value: Any, label: str) -> datetime:
    if not value:
        raise StageMProtocolError(f"Missing {label}")
    try:
        return datetime.fromisoformat(str(value))
    except ValueError as exc:
        raise StageMProtocolError(f"Invalid {label}: {value}") from exc


def _verify_artifact(
    item: Mapping[str, Any],
    *,
    base_dir: Path,
    label: str,
) -> None:
    for key in ("path", "bytes", "sha256"):
        if key not in item:
            raise StageMProtocolError(f"{label} misses {key}")
    path = Path(str(item["path"]))
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    if not path.is_file():
        raise StageMProtocolError(f"Missing {label}: {path}")
    if path.stat().st_size != int(item["bytes"]):
        raise StageMProtocolError(f"{label} byte count mismatch")
    if sha256_file(path) != str(item["sha256"]):
        raise StageMProtocolError(f"{label} SHA-256 mismatch")


def validate_formal_parking_gate(
    payload: Mapping[str, Any],
    *,
    base_dir: Path,
    verify_files: bool = True,
) -> GateDecision:
    """Validate the pre-run gate for a formal continuous parking test."""

    gate_id = str(payload.get("gate_id", ""))
    if not gate_id:
        raise StageMProtocolError("Formal data gate has no gate_id")
    reasons: list[str] = []
    if payload.get("license_status") != "verified":
        reasons.append("dataset_license_not_verified")
    if payload.get("camera_type") != "fixed":
        reasons.append("camera_is_not_fixed")
    if payload.get("annotation_review") != "human_reviewed":
        reasons.append("truth_not_human_reviewed")
    if payload.get("threshold_selection_after_freeze") is not False:
        reasons.append("post_freeze_threshold_selection_not_prohibited")

    scenes = payload.get("scenes", {})
    development = scenes.get("development")
    test = scenes.get("test")
    if not isinstance(development, Mapping) or not isinstance(test, Mapping):
        reasons.append("development_and_test_scenes_required")
    else:
        if not development.get("physical_scene_id") or not test.get(
            "physical_scene_id"
        ):
            reasons.append("physical_scene_ids_required")
        elif development["physical_scene_id"] == test["physical_scene_id"]:
            reasons.append("test_scene_must_be_physically_distinct")
        for role, scene in (("development", development), ("test", test)):
            for key in ("video", "polygons", "truth"):
                if key not in scene:
                    reasons.append(f"{role}_{key}_artifact_required")
                elif verify_files:
                    _verify_artifact(
                        scene[key],
                        base_dir=base_dir,
                        label=f"{role} {key}",
                    )
            event_types = set(scene.get("event_types", []))
            if not event_types.intersection(
                {"entry", "departure", "pass_by", "occlusion"}
            ):
                reasons.append(f"{role}_event_required")
        for key in ("video", "polygons", "truth"):
            development_item = development.get(key)
            test_item = test.get(key)
            if (
                isinstance(development_item, Mapping)
                and isinstance(test_item, Mapping)
                and development_item.get("sha256")
                == test_item.get("sha256")
            ):
                reasons.append(f"development_and_test_{key}_must_be_distinct")

    freeze = payload.get("freeze", {})
    if not isinstance(freeze, Mapping):
        reasons.append("freeze_record_required")
    else:
        frozen_at = _timestamp(freeze.get("frozen_at"), "freeze.frozen_at")
        reviewed_at = _timestamp(
            freeze.get("truth_reviewed_at"), "freeze.truth_reviewed_at"
        )
        if reviewed_at > frozen_at:
            reasons.append("truth_review_must_precede_configuration_freeze")
        if freeze.get("test_runs_after_freeze") != 0:
            reasons.append(
                "formal_test_requires_zero_prior_test_runs_after_freeze"
            )

    status = "eligible" if not reasons else "blocked"
    return GateDecision(
        gate_id=gate_id,
        status=status,
        allowed_claim=(
            "formal_continuous_slot_occupancy_test"
            if status == "eligible"
            else "gate_audit_only"
        ),
        reasons=tuple(reasons),
    )


def load_stage_m_gate_audit(path: Path) -> tuple[GateDecision, ...]:
    """Load the Stage M audit and enforce task-specific claim boundaries."""

    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise StageMProtocolError("Unsupported Stage M data-gate schema")
    decisions: list[GateDecision] = []
    for item in payload.get("gates", []):
        gate_id = str(item.get("gate_id", ""))
        status = str(item.get("status", ""))
        allowed_claim = str(item.get("allowed_claim", ""))
        reasons = tuple(str(value) for value in item.get("reasons", []))
        if not gate_id or status not in {"eligible", "blocked", "deferred"}:
            raise StageMProtocolError(f"Invalid gate record: {item}")
        if status != "eligible" and not reasons:
            raise StageMProtocolError(f"Blocked gate {gate_id} needs reasons")
        if gate_id == "aodraw_detector_diagnostic":
            if allowed_claim not in {
                "detector_only_supporting_experiment",
                "audit_only",
            }:
                raise StageMProtocolError(
                    "AODRaw cannot support occupancy or tracking claims"
                )
        if gate_id == "lmot_tracking_diagnostic":
            if allowed_claim not in {
                "validation_tracking_diagnostic_only",
                "audit_only",
            }:
                raise StageMProtocolError(
                    "LMOT cannot support parking-slot occupancy claims"
                )
        decisions.append(
            GateDecision(
                gate_id=gate_id,
                status=status,
                allowed_claim=allowed_claim,
                reasons=reasons,
            )
        )
    if not decisions:
        raise StageMProtocolError("Stage M data-gate audit is empty")
    return tuple(decisions)
