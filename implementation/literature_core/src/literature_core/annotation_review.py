"""Utilities for selecting and reviewing temporal annotation candidates."""

from __future__ import annotations

import re
from bisect import bisect_left
from collections.abc import Iterable, Sequence


def parse_transition_frames(value: str | None) -> list[int]:
    """Parse a delimited list of non-negative transition frame indices."""

    if value is None or not value.strip():
        return []
    frames: list[int] = []
    for token in re.split(r"[\s,;|]+", value.strip()):
        if not token:
            continue
        frame = int(token)
        if frame < 0:
            raise ValueError("transition frame indices must be non-negative")
        frames.append(frame)
    return sorted(set(frames))


def uniform_frame_indices(total_frames: int, sample_count: int) -> list[int]:
    """Return evenly distributed integer indices including both endpoints."""

    if total_frames <= 0 or sample_count <= 0:
        return []
    if total_frames == 1 or sample_count == 1:
        return [0]
    count = min(total_frames, sample_count)
    last = total_frames - 1
    return sorted({round(index * last / (count - 1)) for index in range(count)})


def temporal_review_indices(
    total_frames: int,
    transition_frames: Sequence[int] | Iterable[int],
    *,
    radius: int = 8,
    uniform_samples: int = 12,
) -> list[int]:
    """Combine uniform coverage with dense windows around candidate transitions."""

    if radius < 0:
        raise ValueError("radius must be non-negative")
    indices = set(uniform_frame_indices(total_frames, uniform_samples))
    for transition in transition_frames:
        if transition < 0:
            raise ValueError("transition frame indices must be non-negative")
        start = max(0, transition - radius)
        stop = min(total_frames, transition + radius + 1)
        indices.update(range(start, stop))
    return sorted(indices)


def nearest_available_indices(
    desired_indices: Sequence[int] | Iterable[int],
    available_indices: Sequence[int],
) -> list[int]:
    """Map requested indices to their nearest locally available frame indices."""

    available = sorted(set(available_indices))
    if not available:
        return []
    selected: set[int] = set()
    for desired in desired_indices:
        position = bisect_left(available, desired)
        choices = available[max(0, position - 1) : min(len(available), position + 1)]
        selected.add(min(choices, key=lambda value: (abs(value - desired), value)))
    return sorted(selected)


def state_intervals(states: Sequence[int]) -> list[tuple[int, int, int]]:
    """Compress binary frame states into inclusive ``(start, end, state)`` runs."""

    if not states:
        return []
    if any(state not in (0, 1) for state in states):
        raise ValueError("states must contain only 0 and 1")

    intervals: list[tuple[int, int, int]] = []
    start = 0
    current = states[0]
    for index, state in enumerate(states[1:], start=1):
        if state != current:
            intervals.append((start, index - 1, current))
            start = index
            current = state
    intervals.append((start, len(states) - 1, current))
    return intervals
