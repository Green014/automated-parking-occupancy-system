from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from .models import SlotEvidence


@dataclass(frozen=True, slots=True)
class HysteresisConfig:
    rise_alpha: float = 0.60
    fall_alpha: float = 0.15
    occupied_threshold: float = 0.18
    vacant_threshold: float = 0.06

    def __post_init__(self) -> None:
        for name, value in (
            ("rise_alpha", self.rise_alpha),
            ("fall_alpha", self.fall_alpha),
            ("occupied_threshold", self.occupied_threshold),
            ("vacant_threshold", self.vacant_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.vacant_threshold >= self.occupied_threshold:
            raise ValueError(
                "vacant_threshold must be smaller than occupied_threshold"
            )


@dataclass(frozen=True, slots=True)
class FilteredSlotState:
    slot_id: str
    occupied: bool
    filtered_score: float
    raw_occupied: bool
    raw_evidence_score: float
    changed: bool
    track_id: int | None


class TemporalOccupancyFilter:
    """Confidence-aware EMA with separate ON/OFF thresholds."""

    def __init__(
        self,
        slot_ids: Iterable[str],
        config: HysteresisConfig | None = None,
    ) -> None:
        self.config = config or HysteresisConfig()
        frozen_slot_ids = tuple(slot_ids)
        if not frozen_slot_ids:
            raise ValueError("At least one slot ID is required")
        if len(set(frozen_slot_ids)) != len(frozen_slot_ids):
            raise ValueError("Slot IDs must be unique")
        self._score = {slot_id: 0.0 for slot_id in frozen_slot_ids}
        self._occupied = {slot_id: False for slot_id in frozen_slot_ids}

    def update(
        self,
        evidence_by_slot: dict[str, SlotEvidence],
    ) -> dict[str, FilteredSlotState]:
        states: dict[str, FilteredSlotState] = {}
        for slot_id in self._score:
            evidence = evidence_by_slot[slot_id]
            previous_score = self._score[slot_id]
            target = evidence.evidence_score if evidence.occupied else 0.0
            alpha = (
                self.config.rise_alpha
                if target >= previous_score
                else self.config.fall_alpha
            )
            score = alpha * target + (1.0 - alpha) * previous_score
            previous_state = self._occupied[slot_id]
            occupied = previous_state
            if not previous_state and score >= self.config.occupied_threshold:
                occupied = True
            elif previous_state and score <= self.config.vacant_threshold:
                occupied = False

            self._score[slot_id] = score
            self._occupied[slot_id] = occupied
            states[slot_id] = FilteredSlotState(
                slot_id=slot_id,
                occupied=occupied,
                filtered_score=score,
                raw_occupied=evidence.occupied,
                raw_evidence_score=evidence.evidence_score,
                changed=occupied != previous_state,
                track_id=evidence.track_id,
            )
        return states
