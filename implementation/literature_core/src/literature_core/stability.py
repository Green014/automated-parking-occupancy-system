from __future__ import annotations

import statistics
from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import Any


def probability_summary(values: Iterable[float]) -> dict[str, float | int]:
    data = sorted(float(value) for value in values)
    if not data:
        raise ValueError("At least one probability is required")
    if any(not 0.0 <= value <= 1.0 for value in data):
        raise ValueError("Probabilities must be in [0, 1]")
    return {
        "count": len(data),
        "mean": statistics.fmean(data),
        "median": statistics.median(data),
        "p10": data[round((len(data) - 1) * 0.1)],
        "p90": data[round((len(data) - 1) * 0.9)],
        "minimum": data[0],
        "maximum": data[-1],
        "zero_fraction": sum(value == 0.0 for value in data) / len(data),
    }


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "median": statistics.median(ordered),
        "p90": ordered[round((len(ordered) - 1) * 0.9)],
        "maximum": ordered[-1],
    }


def positive_only_stability_metrics(
    rows: Iterable[Mapping[str, Any]],
    *,
    state_key: str,
    fps: float,
    warmup_frames: int = 6,
) -> dict[str, Any]:
    """Evaluate constant-positive sequences without implying negative truth."""

    if fps <= 0:
        raise ValueError("fps must be positive")
    if warmup_frames < 0:
        raise ValueError("warmup_frames must be non-negative")
    data = list(rows)
    if not data:
        raise ValueError("At least one row is required")
    if any(int(row["truth"]) != 1 for row in data):
        raise ValueError("This evaluator is restricted to positive-only truth")

    grouped: dict[str, list[tuple[int, int]]] = defaultdict(list)
    for row in data:
        state = int(row[state_key])
        if state not in {0, 1}:
            raise ValueError("Predicted states must be binary")
        grouped[str(row["slot_id"])].append((int(row["frame_index"]), state))

    occupied = sum(int(row[state_key]) for row in data)
    recall = occupied / len(data)
    changes = 0
    slot_minutes = 0.0
    acquisition_s: list[float] = []
    per_slot: dict[str, Any] = {}
    for slot_id, values in sorted(grouped.items()):
        values.sort()
        frames = [frame for frame, _ in values]
        states = [state for _, state in values]
        first_occupied = next(
            (frame for frame, state in values if state == 1),
            None,
        )
        if first_occupied is not None:
            acquisition_s.append((first_occupied - frames[0]) / fps)
        stable_states = [
            state
            for frame, state in values
            if frame >= frames[0] + warmup_frames
        ]
        slot_changes = sum(
            current != previous
            for previous, current in zip(
                stable_states,
                stable_states[1:],
                strict=False,
            )
        )
        changes += slot_changes
        slot_minutes += len(stable_states) / fps / 60.0
        per_slot[slot_id] = {
            "frames": len(states),
            "occupied_recall": sum(states) / len(states),
            "false_free_rate": 1.0 - sum(states) / len(states),
            "post_warmup_state_changes": slot_changes,
            "initial_acquisition_s": (
                None
                if first_occupied is None
                else (first_occupied - frames[0]) / fps
            ),
        }

    return {
        "truth_scope": "positive_only_continuously_occupied",
        "samples": len(data),
        "slots": len(grouped),
        "frames_per_slot": sorted(
            {len(values) for values in grouped.values()}
        ),
        "occupied_recall": recall,
        "positive_only_f1": (
            2.0 * recall / (1.0 + recall) if recall else 0.0
        ),
        "false_free_rate": 1.0 - recall,
        "post_warmup_unsupported_changes": changes,
        "post_warmup_flicker_per_slot_minute": (
            changes / slot_minutes if slot_minutes else 0.0
        ),
        "initial_acquisition_s": _summary(acquisition_s),
        "warmup_frames": warmup_frames,
        "fps": fps,
        "per_slot": per_slot,
        "excluded_claims": [
            "vacant recall",
            "false-occupied rate",
            "transition latency",
            "IDF1/HOTA",
        ],
    }
