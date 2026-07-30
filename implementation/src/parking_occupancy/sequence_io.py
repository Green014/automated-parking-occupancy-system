from __future__ import annotations


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """Return unique, evenly spaced indices including both endpoints."""

    if total <= 0:
        return []
    count = min(max(count, 1), total)
    return sorted(
        {
            int(round(index * (total - 1) / max(count - 1, 1)))
            for index in range(count)
        }
    )
