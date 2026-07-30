from __future__ import annotations

import csv
import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping, Sequence, TypeVar

import yaml

from .slots import SlotMap, slot_map_from_dict
from .stage_n_lmot import sha256_file


STAGE_Q_PROTOCOL_ID = (
    "STAGE-Q-INDEPENDENT-EXTERNAL-NIGHT-OCCUPANCY-20260729-01"
)
STAGE_Q_GATE_ID = "STAGE-Q-CANDIDATE-GATE-20260729-01"
SOURCE_SLOT_COUNT = 21
SOURCE_SLOT_IDS = tuple(f"slot_{index:02d}" for index in range(1, 22))
SOURCE_VALUE_MEANING = {0: "occupied", 1: "vacant"}
PROJECT_STATE = {"occupied": 1, "vacant": 0}
BLOCKING_GATE_STATUSES = {
    "BLOCKED",
    "BLOCKED_BEFORE_DOWNLOAD",
    "INELIGIBLE",
}
LOW_RATE_MEDIA_TYPES = {
    "low_frame_rate_image_sequence",
    "ordered_low_frame_rate_image_sequence",
}
SECOND_BASED_TEMPORAL_METRICS = {
    "transition_latency_seconds",
    "signed_transition_error_seconds",
    "early_seconds",
    "delayed_seconds",
}
FRAME_INDEX_TEMPORAL_METRICS = {
    "state_change_agreement",
    "frame_index_transition_difference",
}
FORMAL_METHOD_OUTPUTS = (
    "occupancy.csv",
    "events.csv",
    "detections.jsonl",
    "summary.json",
    "metrics.json",
    "runtime_metadata.json",
    "qualitative_contact_sheet.jpg",
)
CONDITIONAL_SEQUENCE_OUTPUTS = (
    "annotated.mp4",
    "annotated_frames",
)


class StageQDataGateError(ValueError):
    """Raised before Stage Q crosses a frozen data or evaluation boundary."""


@dataclass(frozen=True, slots=True)
class OccupancyTruthRecord:
    image_name: str
    source_values: tuple[int | None, ...]
    project_states: tuple[int | None, ...]


@dataclass(frozen=True, slots=True)
class ManifestRecord:
    relative_path: str
    bytes: int
    sha256: str


def load_candidate_gate(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise StageQDataGateError("Stage Q candidate gate must be a mapping")
    if payload.get("gate_id") != STAGE_Q_GATE_ID:
        raise StageQDataGateError("Unexpected Stage Q candidate gate ID")
    if payload.get("frozen_prior_conclusions", {}).get("stage_p2_status") != "FAIL":
        raise StageQDataGateError("Stage P2 FAIL must remain frozen")
    if payload.get("frozen_prior_conclusions", {}).get("default_detector") != "D1":
        raise StageQDataGateError("D1 must remain the default detector")
    if payload.get("comparison_roles") != {
        "primary": "P3-D1",
        "secondary": "P3-D1-LL",
        "default_change_allowed": False,
        "purpose": "independent_external_occupancy_evidence_not_model_selection",
    }:
        raise StageQDataGateError("Unexpected Stage Q comparison roles")
    return payload


def _decode_source_value(token: str) -> int | None:
    normalized = token.strip().lower()
    if normalized in {"?", "unknown", "na", "n/a", "-1"}:
        return None
    if normalized not in {"0", "1"}:
        raise StageQDataGateError(
            f"Unsupported occupancy value {token!r}; expected 0, 1, or unknown"
        )
    return int(normalized)


def parse_groundtruth_line(line: str, *, line_number: int = 1) -> OccupancyTruthRecord:
    stripped = line.strip()
    if not stripped:
        raise StageQDataGateError(f"Empty truth line at {line_number}")
    match = re.match(r"^([^,\s]+)[,\s]+(.+)$", stripped)
    if match is None:
        raise StageQDataGateError(
            f"Missing image/vector separator at truth line {line_number}"
        )
    image_name, raw_vector = match.groups()
    raw_vector = raw_vector.strip().strip("[]()")
    split_values = [
        token
        for token in re.split(r"[\s,;]+", raw_vector)
        if token
    ]
    if len(split_values) == 1:
        contiguous = split_values[0]
        if len(contiguous) == SOURCE_SLOT_COUNT and set(contiguous) <= {
            "0",
            "1",
            "?",
        }:
            split_values = list(contiguous)
    if len(split_values) != SOURCE_SLOT_COUNT:
        raise StageQDataGateError(
            f"Truth line {line_number} has {len(split_values)} values; "
            f"expected {SOURCE_SLOT_COUNT}"
        )
    source_values = tuple(_decode_source_value(value) for value in split_values)
    project_states = tuple(
        None
        if value is None
        else PROJECT_STATE[SOURCE_VALUE_MEANING[value]]
        for value in source_values
    )
    return OccupancyTruthRecord(
        image_name=image_name,
        source_values=source_values,
        project_states=project_states,
    )


def parse_groundtruth_text(text: str) -> list[OccupancyTruthRecord]:
    records: list[OccupancyTruthRecord] = []
    seen: set[str] = set()
    for line_number, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        record = parse_groundtruth_line(line, line_number=line_number)
        if record.image_name in seen:
            raise StageQDataGateError(
                f"Duplicate truth image name: {record.image_name}"
            )
        seen.add(record.image_name)
        records.append(record)
    if not records:
        raise StageQDataGateError("Ground-truth file contains no records")
    return records


def parse_groundtruth_file(path: Path) -> list[OccupancyTruthRecord]:
    return parse_groundtruth_text(path.read_text(encoding="utf-8-sig"))


def validate_image_truth_bijection(
    image_names: Iterable[str],
    records: Sequence[OccupancyTruthRecord],
) -> dict[str, Any]:
    names = [str(name) for name in image_names]
    if len(names) != len(set(names)):
        raise StageQDataGateError("Duplicate image name in image manifest")
    truth_names = [record.image_name for record in records]
    image_set = set(names)
    truth_set = set(truth_names)
    missing_truth = sorted(image_set - truth_set)
    missing_images = sorted(truth_set - image_set)
    if missing_truth or missing_images:
        raise StageQDataGateError(
            "Image/truth membership mismatch: "
            f"missing_truth={missing_truth[:5]}, "
            f"missing_images={missing_images[:5]}"
        )
    return {
        "image_count": len(names),
        "truth_record_count": len(records),
        "one_to_one": True,
    }


def truth_to_long_form(
    records: Sequence[OccupancyTruthRecord],
    *,
    video_id: str,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    unknown_excluded = 0
    for frame_index, record in enumerate(records):
        for slot_id, state in zip(SOURCE_SLOT_IDS, record.project_states):
            if state is None:
                unknown_excluded += 1
                continue
            rows.append(
                {
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "timestamp_s": "",
                    "slot_id": slot_id,
                    "state": state,
                }
            )
    return {
        "rows": rows,
        "unknown_excluded": unknown_excluded,
        "timestamp_semantics": "unavailable_low_frame_rate_sequence_index_only",
    }


def write_long_form_truth(
    path: Path,
    records: Sequence[OccupancyTruthRecord],
    *,
    video_id: str,
) -> dict[str, Any]:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    result = truth_to_long_form(records, video_id=video_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "video_id",
                "frame_index",
                "timestamp_s",
                "slot_id",
                "state",
            ),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(result["rows"])
    return result


def validate_slot_map_contract(
    payload: Mapping[str, Any],
    *,
    truth_slot_ids: Sequence[str] = SOURCE_SLOT_IDS,
) -> SlotMap:
    slot_map = slot_map_from_dict(dict(payload))
    actual_ids = tuple(slot.slot_id for slot in slot_map.slots)
    expected_ids = tuple(truth_slot_ids)
    if len(actual_ids) != SOURCE_SLOT_COUNT:
        raise StageQDataGateError(
            f"Stage Q requires exactly {SOURCE_SLOT_COUNT} slot polygons"
        )
    if actual_ids != expected_ids:
        raise StageQDataGateError(
            "Polygon IDs/order do not match occupancy-truth slot IDs"
        )
    width = slot_map.source_width
    height = slot_map.source_height
    for slot in slot_map.slots:
        for x, y in slot.points:
            if not (0.0 <= x < width and 0.0 <= y < height):
                raise StageQDataGateError(
                    f"Polygon coordinate out of range for {slot.slot_id}: "
                    f"({x}, {y}) outside {width}x{height}"
                )
    return slot_map


def manifest_fingerprint(records: Sequence[ManifestRecord]) -> str:
    if not records:
        raise StageQDataGateError("Cannot fingerprint an empty manifest")
    canonical = [
        {
            "relative_path": record.relative_path.replace("\\", "/"),
            "bytes": int(record.bytes),
            "sha256": record.sha256.lower(),
        }
        for record in records
    ]
    encoded = json.dumps(
        canonical,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def verify_manifest_records(
    root: Path,
    records: Sequence[ManifestRecord],
    *,
    expected_manifest_sha256: str | None = None,
) -> dict[str, Any]:
    root = root.resolve()
    if not records:
        raise StageQDataGateError("Empty Stage Q image manifest")
    seen: set[str] = set()
    for record in records:
        normalized = record.relative_path.replace("\\", "/")
        if normalized in seen:
            raise StageQDataGateError(
                f"Duplicate manifest path: {record.relative_path}"
            )
        seen.add(normalized)
        path = (root / record.relative_path).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise StageQDataGateError(
                f"Manifest path escapes dataset root: {record.relative_path}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(path)
        if path.stat().st_size != int(record.bytes):
            raise StageQDataGateError(f"File-size mismatch: {path}")
        if sha256_file(path) != record.sha256.lower():
            raise StageQDataGateError(f"SHA-256 mismatch: {path}")
    fingerprint = manifest_fingerprint(records)
    if (
        expected_manifest_sha256 is not None
        and fingerprint != expected_manifest_sha256.lower()
    ):
        raise StageQDataGateError("Scene/manifest fingerprint mismatch")
    return {
        "verified": True,
        "file_count": len(records),
        "manifest_sha256": fingerprint,
    }


def validate_frozen_comparison(
    gate: Mapping[str, Any],
    p3_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    roles = gate.get("comparison_roles", {})
    if roles.get("primary") != "P3-D1":
        raise StageQDataGateError("P3-D1 must be the primary method")
    if roles.get("secondary") != "P3-D1-LL":
        raise StageQDataGateError("P3-D1-LL must remain secondary")
    if roles.get("default_change_allowed") is not False:
        raise StageQDataGateError("Stage Q cannot change the default detector")
    detector = p3_defaults.get("detector", {})
    if detector.get("id") != "D1":
        raise StageQDataGateError("P3 default detector must remain D1")
    shared = gate.get("shared_inference", {})
    expected = {
        "imgsz": detector.get("image_size"),
        "confidence": detector.get("confidence"),
        "nms_iou": detector.get("nms_iou"),
        "agnostic_nms": detector.get("agnostic_nms"),
        "max_detections": detector.get("max_detections"),
        "classes": detector.get("class_ids"),
    }
    mismatches = {
        key: {"gate": shared.get(key), "p3_defaults": value}
        for key, value in expected.items()
        if shared.get(key) != value
    }
    if mismatches:
        raise StageQDataGateError(
            f"Stage Q/P3 inference setting mismatch: {mismatches}"
        )
    return {
        "primary": "P3-D1",
        "secondary": "P3-D1-LL",
        "same_inference_settings": True,
        "default_detector": "D1",
        "default_change_allowed": False,
    }


def validate_temporal_metric_request(
    *,
    media_type: str,
    requested_metrics: Iterable[str],
) -> dict[str, Any]:
    requested = set(requested_metrics)
    if media_type in LOW_RATE_MEDIA_TYPES:
        prohibited = sorted(requested & SECOND_BASED_TEMPORAL_METRICS)
        if prohibited:
            raise StageQDataGateError(
                "Low-frame-rate image sequences cannot support seconds-level "
                f"transition latency: {prohibited}"
            )
        unsupported = sorted(requested - FRAME_INDEX_TEMPORAL_METRICS)
        if unsupported:
            raise StageQDataGateError(
                f"Unsupported low-frame-rate temporal metrics: {unsupported}"
            )
        return {
            "media_type": media_type,
            "seconds_level_latency_supported": False,
            "metrics": sorted(requested),
            "interpretation": "sequence_index_only_not_realtime_latency",
        }
    return {
        "media_type": media_type,
        "seconds_level_latency_supported": True,
        "metrics": sorted(requested),
    }


def validate_method_output_contract(
    output_root: Path,
    *,
    orderable_sequence: bool,
) -> dict[str, Any]:
    required = list(FORMAL_METHOD_OUTPUTS)
    required.append("annotated_frames")
    if orderable_sequence:
        required.append("annotated.mp4")
    missing = [
        relative
        for relative in required
        if not (output_root / relative).exists()
    ]
    return {
        "protocol_id": STAGE_Q_PROTOCOL_ID,
        "verified": not missing,
        "required": required,
        "missing": missing,
        "annotated_video_required": orderable_sequence,
    }


_T = TypeVar("_T")


def execute_formal_run_if_authorized(
    gate: Mapping[str, Any],
    run: Callable[[], _T],
) -> _T:
    status = str(gate.get("status", "")).upper()
    authorized = gate.get("formal_inference_authorized") is True
    if status in BLOCKING_GATE_STATUSES or not authorized:
        raise StageQDataGateError(
            f"Formal Stage Q inference is not authorized (status={status})"
        )
    return run()
