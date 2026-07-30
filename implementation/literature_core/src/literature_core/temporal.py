from __future__ import annotations

from dataclasses import dataclass

from .models import FusedEvidence


@dataclass(frozen=True, slots=True)
class TemporalConfig:
    rise_alpha: float = 0.60
    fall_alpha: float = 0.15
    occupied_threshold: float = 0.58
    vacant_threshold: float = 0.42
    raw_threshold: float = 0.50

    def __post_init__(self) -> None:
        for name, value in (
            ("rise_alpha", self.rise_alpha),
            ("fall_alpha", self.fall_alpha),
            ("occupied_threshold", self.occupied_threshold),
            ("vacant_threshold", self.vacant_threshold),
            ("raw_threshold", self.raw_threshold),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")
        if self.vacant_threshold >= self.occupied_threshold:
            raise ValueError("vacant_threshold must be below occupied_threshold")


@dataclass(frozen=True, slots=True)
class TemporalState:
    slot_id: str
    occupied: bool
    filtered_probability: float
    raw_occupied: bool
    changed: bool


class TemporalFusionFilter:
    """Asymmetric probability EMA followed by ON/OFF hysteresis."""

    def __init__(
        self,
        slot_ids: tuple[str, ...],
        config: TemporalConfig | None = None,
    ) -> None:
        if not slot_ids or len(set(slot_ids)) != len(slot_ids):
            raise ValueError("slot_ids must be a non-empty unique tuple")
        self.config = config or TemporalConfig()
        self._probability = {slot_id: 0.0 for slot_id in slot_ids}
        self._occupied = {slot_id: False for slot_id in slot_ids}

    def update(self, evidence: FusedEvidence) -> TemporalState:
        if evidence.slot_id not in self._probability:
            raise KeyError(evidence.slot_id)
        previous_probability = self._probability[evidence.slot_id]
        alpha = (
            self.config.rise_alpha
            if evidence.probability >= previous_probability
            else self.config.fall_alpha
        )
        probability = (
            alpha * evidence.probability
            + (1.0 - alpha) * previous_probability
        )
        previous_state = self._occupied[evidence.slot_id]
        occupied = previous_state
        if not occupied and probability >= self.config.occupied_threshold:
            occupied = True
        elif occupied and probability <= self.config.vacant_threshold:
            occupied = False
        self._probability[evidence.slot_id] = probability
        self._occupied[evidence.slot_id] = occupied
        return TemporalState(
            slot_id=evidence.slot_id,
            occupied=occupied,
            filtered_probability=probability,
            raw_occupied=evidence.probability >= self.config.raw_threshold,
            changed=occupied != previous_state,
        )

