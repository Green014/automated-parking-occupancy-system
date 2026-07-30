from __future__ import annotations

import re
from collections import deque
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import urlsplit

from .models import ParkingSlot
from .stage_v import FrameOccupancyResult, validate_frame_result


WINDOWS_ABSOLUTE_PATH = re.compile(
    r"(?<![A-Za-z0-9+.-])[A-Za-z]:[\\/](?![\\/])[^\s\"']+"
)
POSIX_USER_PATH = re.compile(r"/(?:home|Users)/[^\s\"']+")


def redact_source(value: str | int) -> str:
    if isinstance(value, int) or str(value).isdigit():
        return f"camera:{int(value)}"
    text = str(value)
    parsed = urlsplit(text)
    if parsed.scheme and parsed.netloc:
        host = parsed.hostname or "source"
        port = f":{parsed.port}" if parsed.port else ""
        return f"{parsed.scheme}://{host}{port}/<redacted>"
    normalized = text.replace("\\", "/")
    return normalized.rsplit("/", 1)[-1]


def _safe_text(value: Any) -> str:
    text = str(value)
    text = WINDOWS_ABSOLUTE_PATH.sub("<local-path-redacted>", text)
    text = POSIX_USER_PATH.sub("<local-path-redacted>", text)
    parsed = urlsplit(text)
    if parsed.scheme.lower() in {"rtsp", "rtsps"} and parsed.netloc:
        return redact_source(text)
    return text


def sanitize_for_api(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): sanitize_for_api(item)
            for key, item in value.items()
            if not any(
                token in str(key).casefold()
                for token in ("checkpoint_path", "weights_path", "model_path")
            )
        }
    if isinstance(value, (list, tuple)):
        return [sanitize_for_api(item) for item in value]
    if isinstance(value, str):
        return _safe_text(value)
    return value


class StateEventTracker:
    """In-memory UI events/sessions with explicit temporal validity."""

    def __init__(
        self,
        *,
        continuous: bool,
        temporal_enabled: bool,
        max_events: int = 100,
    ) -> None:
        self.continuous = bool(continuous)
        self.temporal_enabled = bool(temporal_enabled)
        self.previous: dict[str, bool] | None = None
        self._events: deque[dict[str, Any]] = deque(maxlen=max_events)
        self._active_sessions: dict[str, dict[str, Any]] = {}

    @property
    def semantics(self) -> str:
        if not self.continuous:
            return "not_temporally_valid_noncontinuous_input"
        return (
            "e4_stabilized_frame_state_changes"
            if self.temporal_enabled
            else "raw_frame_level_state_changes"
        )

    def update(self, result: FrameOccupancyResult) -> list[dict[str, Any]]:
        current = {
            state.slot_id: state.occupied for state in result.slot_states
        }
        if self.previous is None:
            self.previous = current
            return []
        if not self.continuous:
            self.previous = current
            return []
        emitted: list[dict[str, Any]] = []
        by_slot = result.state_by_slot()
        for slot_id in sorted(current):
            before = self.previous[slot_id]
            after = current[slot_id]
            if before == after:
                continue
            state = by_slot[slot_id]
            event = {
                "event_type": "arrival" if after else "departure",
                "slot_id": slot_id,
                "track_id": state.track_id,
                "frame_index": result.frame_index,
                "timestamp_s": result.timestamp_s,
                "from_state": "occupied" if before else "vacant",
                "to_state": "occupied" if after else "vacant",
                "evidence": state.evidence_score,
                "evidence_source": state.evidence_source,
                "temporal_semantics": self.semantics,
            }
            emitted.append(event)
            self._events.append(event)
            if after:
                self._active_sessions[slot_id] = {
                    "slot_id": slot_id,
                    "track_id": state.track_id,
                    "start_frame": result.frame_index,
                    "start_timestamp_s": result.timestamp_s,
                    "status": "active",
                    "temporal_semantics": self.semantics,
                }
            else:
                self._active_sessions.pop(slot_id, None)
        self.previous = current
        return emitted

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        return list(self._events)[-limit:][::-1]

    def sessions(self) -> list[dict[str, Any]]:
        return [
            dict(value)
            for _, value in sorted(self._active_sessions.items())
        ]


def frame_result_to_ui_payload(
    result: FrameOccupancyResult,
    configured_slots: Sequence[ParkingSlot],
    *,
    mode: str,
    temporal_enabled: bool,
    tracker_enabled: bool,
    cache_status: str | None = None,
    health: str = "running",
    message: str = "Processing",
    source: str | None = None,
) -> dict[str, Any]:
    validate_frame_result(result, configured_slots)
    by_slot = result.state_by_slot()
    occupied = sum(state.occupied for state in result.slot_states)
    total = len(configured_slots)
    vacant = total - occupied
    attributed_ms = float(
        result.timing_ms.get(
            "attributed_backend_total",
            result.timing_ms.get("backend_total", 0.0),
        )
    )
    if cache_status is None:
        cache_value = result.timing_ms.get("cache_hit")
        cache_status = (
            "not-used"
            if cache_value is None
            else ("hit" if bool(cache_value) else "miss")
        )
    payload = {
        "schema_version": 1,
        "stage": "W",
        "health": health,
        "message": _safe_text(message),
        "frame_index": result.frame_index,
        "timestamp_s": result.timestamp_s,
        "occupied": occupied,
        "vacant": vacant,
        "total": total,
        "rendered_slots": total,
        "mode": mode,
        "temporal_enabled": bool(temporal_enabled),
        "tracker_enabled": bool(tracker_enabled),
        "slots": [
            {
                "slot_id": slot.slot_id,
                "state": (
                    "occupied"
                    if by_slot[slot.slot_id].occupied
                    else "vacant"
                ),
                "evidence": by_slot[slot.slot_id].evidence_score,
                "evidence_source": by_slot[slot.slot_id].evidence_source,
                "track_id": (
                    by_slot[slot.slot_id].track_id
                    if tracker_enabled
                    else None
                ),
            }
            for slot in configured_slots
        ],
        "detections": [
            {
                "bbox": [float(value) for value in detection.bbox],
                "confidence": detection.confidence,
                "class_id": detection.class_id,
                "class_name": detection.class_name,
                "track_id": detection.track_id if tracker_enabled else None,
            }
            for detection in result.vehicle_detections
        ],
        "runtime": {
            "attributed_processing_ms": attributed_ms,
            "attributed_fps": (
                1000.0 / attributed_ms if attributed_ms > 0 else None
            ),
            "cache": cache_status,
            "timing_ms": dict(result.timing_ms),
        },
        "warnings": [_safe_text(value) for value in result.warnings],
    }
    if source is not None:
        payload["source"] = redact_source(source)
    if occupied + vacant != total:
        raise ValueError("UI count invariant failed")
    return sanitize_for_api(payload)


def error_payload(
    message: str,
    *,
    mode: str,
    total: int = 0,
    source: str | int | None = None,
) -> dict[str, Any]:
    payload = {
        "schema_version": 1,
        "stage": "W",
        "health": "error",
        "running": False,
        "message": _safe_text(message),
        "mode": mode,
        "frame_index": None,
        "timestamp_s": None,
        "occupied": 0,
        "vacant": total,
        "total": total,
        "rendered_slots": 0,
        "temporal_enabled": False,
        "tracker_enabled": False,
        "slots": [],
        "detections": [],
        "runtime": {
            "attributed_processing_ms": None,
            "attributed_fps": None,
            "cache": "not-used",
            "timing_ms": {},
        },
        "warnings": [_safe_text(message)],
    }
    if source is not None:
        payload["source"] = redact_source(source)
    return sanitize_for_api(payload)
