"""Literature-aligned parking occupancy components."""

from .fusion import FusionConfig, fuse_evidence
from .mapping import map_detections_to_slots
from .models import Detection, ParkingSlot, SlotDecision
from .patches import extract_slot_patch
from .temporal import TemporalConfig, TemporalFusionFilter
from .temporal_tracking import (
    TrackSlotEvidence,
    TrackSlotGate,
    TrackSlotGateConfig,
    TrackSlotState,
    associate_tracks_to_slots,
)

__all__ = [
    "Detection",
    "FusionConfig",
    "ParkingSlot",
    "SlotDecision",
    "TemporalConfig",
    "TemporalFusionFilter",
    "TrackSlotEvidence",
    "TrackSlotGate",
    "TrackSlotGateConfig",
    "TrackSlotState",
    "associate_tracks_to_slots",
    "extract_slot_patch",
    "fuse_evidence",
    "map_detections_to_slots",
]
