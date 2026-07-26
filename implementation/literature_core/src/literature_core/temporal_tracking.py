"""Interpretable track-to-slot gating primitives for synthetic development.

This module does not run a tracker and does not constitute an E5 result. It
accepts already computed track/slot coverage evidence so association, dwell,
and moving-vehicle suppression can be tested without using temporal truth.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
import math


@dataclass(frozen=True, slots=True)
class TrackSlotEvidence:
    """One track's geometric and motion evidence for one parking slot."""

    slot_id: str
    track_id: str
    coverage: float
    center_displacement_px: float

    def __post_init__(self) -> None:
        if not self.slot_id or not self.track_id:
            raise ValueError("slot_id and track_id must not be empty")
        if not math.isfinite(self.coverage) or not 0.0 <= self.coverage <= 1.0:
            raise ValueError("coverage must be finite and in [0, 1]")
        if (
            not math.isfinite(self.center_displacement_px)
            or self.center_displacement_px < 0.0
        ):
            raise ValueError(
                "center_displacement_px must be finite and non-negative"
            )


@dataclass(frozen=True, slots=True)
class TrackSlotGateConfig:
    """Predeclared rule thresholds; real values require development selection."""

    minimum_coverage: float
    maximum_stationary_displacement_px: float
    occupied_dwell_frames: int
    vacant_dwell_frames: int

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.minimum_coverage)
            or not 0.0 <= self.minimum_coverage <= 1.0
        ):
            raise ValueError("minimum_coverage must be finite and in [0, 1]")
        if (
            not math.isfinite(self.maximum_stationary_displacement_px)
            or self.maximum_stationary_displacement_px < 0.0
        ):
            raise ValueError(
                "maximum_stationary_displacement_px must be finite and "
                "non-negative"
            )
        for name, value in (
            ("occupied_dwell_frames", self.occupied_dwell_frames),
            ("vacant_dwell_frames", self.vacant_dwell_frames),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class TrackSlotState:
    """Auditable slot state after one synthetic or real frame update."""

    slot_id: str
    occupied: bool
    changed: bool
    assigned_track_id: str | None
    confirmed_track_id: str | None
    candidate_dwell_frames: int
    clear_dwell_frames: int
    suppressed_moving_track_ids: tuple[str, ...]


def associate_tracks_to_slots(
    evidence: Iterable[TrackSlotEvidence],
    *,
    slot_ids: tuple[str, ...],
    minimum_coverage: float,
    maximum_stationary_displacement_px: float,
) -> dict[str, TrackSlotEvidence]:
    """Greedily select a deterministic one-track/one-slot assignment."""

    known_slots = set(slot_ids)
    if not slot_ids or len(known_slots) != len(slot_ids):
        raise ValueError("slot_ids must be a non-empty unique tuple")
    candidates = []
    for item in evidence:
        if item.slot_id not in known_slots:
            raise KeyError(item.slot_id)
        if (
            item.coverage >= minimum_coverage
            and item.center_displacement_px
            <= maximum_stationary_displacement_px
        ):
            candidates.append(item)

    assigned_slots: set[str] = set()
    assigned_tracks: set[str] = set()
    assignments: dict[str, TrackSlotEvidence] = {}
    for item in sorted(
        candidates,
        key=lambda value: (-value.coverage, value.track_id, value.slot_id),
    ):
        if item.slot_id in assigned_slots or item.track_id in assigned_tracks:
            continue
        assignments[item.slot_id] = item
        assigned_slots.add(item.slot_id)
        assigned_tracks.add(item.track_id)
    return assignments


class TrackSlotGate:
    """State machine for stationary-track dwell and delayed vacancy."""

    def __init__(
        self,
        slot_ids: tuple[str, ...],
        config: TrackSlotGateConfig,
    ) -> None:
        if not slot_ids or len(set(slot_ids)) != len(slot_ids):
            raise ValueError("slot_ids must be a non-empty unique tuple")
        self.slot_ids = slot_ids
        self.config = config
        self._occupied = {slot_id: False for slot_id in slot_ids}
        self._confirmed_track: dict[str, str | None] = {
            slot_id: None for slot_id in slot_ids
        }
        self._candidate_track: dict[str, str | None] = {
            slot_id: None for slot_id in slot_ids
        }
        self._candidate_dwell = {slot_id: 0 for slot_id in slot_ids}
        self._clear_dwell = {slot_id: 0 for slot_id in slot_ids}

    def update(
        self,
        evidence: Iterable[TrackSlotEvidence],
    ) -> tuple[TrackSlotState, ...]:
        """Advance every slot by one frame and return states in slot order."""

        frame_evidence = tuple(evidence)
        assignments = associate_tracks_to_slots(
            frame_evidence,
            slot_ids=self.slot_ids,
            minimum_coverage=self.config.minimum_coverage,
            maximum_stationary_displacement_px=(
                self.config.maximum_stationary_displacement_px
            ),
        )
        moving_by_slot = {
            slot_id: tuple(
                sorted(
                    {
                        item.track_id
                        for item in frame_evidence
                        if item.slot_id == slot_id
                        and item.coverage >= self.config.minimum_coverage
                        and item.center_displacement_px
                        > self.config.maximum_stationary_displacement_px
                    }
                )
            )
            for slot_id in self.slot_ids
        }

        states: list[TrackSlotState] = []
        for slot_id in self.slot_ids:
            previous = self._occupied[slot_id]
            assigned = assignments.get(slot_id)
            if assigned is None:
                self._candidate_track[slot_id] = None
                self._candidate_dwell[slot_id] = 0
                if previous:
                    self._clear_dwell[slot_id] += 1
                    if (
                        self._clear_dwell[slot_id]
                        >= self.config.vacant_dwell_frames
                    ):
                        self._occupied[slot_id] = False
                        self._confirmed_track[slot_id] = None
                else:
                    self._clear_dwell[slot_id] = 0
            else:
                self._clear_dwell[slot_id] = 0
                if self._candidate_track[slot_id] == assigned.track_id:
                    self._candidate_dwell[slot_id] += 1
                else:
                    self._candidate_track[slot_id] = assigned.track_id
                    self._candidate_dwell[slot_id] = 1

                if previous:
                    if (
                        self._confirmed_track[slot_id] != assigned.track_id
                        and self._candidate_dwell[slot_id]
                        >= self.config.occupied_dwell_frames
                    ):
                        self._confirmed_track[slot_id] = assigned.track_id
                elif (
                    self._candidate_dwell[slot_id]
                    >= self.config.occupied_dwell_frames
                ):
                    self._occupied[slot_id] = True
                    self._confirmed_track[slot_id] = assigned.track_id

            states.append(
                TrackSlotState(
                    slot_id=slot_id,
                    occupied=self._occupied[slot_id],
                    changed=self._occupied[slot_id] != previous,
                    assigned_track_id=(
                        assigned.track_id if assigned is not None else None
                    ),
                    confirmed_track_id=self._confirmed_track[slot_id],
                    candidate_dwell_frames=self._candidate_dwell[slot_id],
                    clear_dwell_frames=self._clear_dwell[slot_id],
                    suppressed_moving_track_ids=moving_by_slot[slot_id],
                )
            )
        return tuple(states)
