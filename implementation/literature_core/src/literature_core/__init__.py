"""Literature-aligned parking occupancy components."""

from .fusion import FusionConfig, fuse_evidence
from .mapping import map_detections_to_slots
from .models import Detection, ParkingSlot, SlotDecision
from .patches import extract_slot_patch
from .temporal import TemporalConfig, TemporalFusionFilter

__all__ = [
    "Detection",
    "FusionConfig",
    "ParkingSlot",
    "SlotDecision",
    "TemporalConfig",
    "TemporalFusionFilter",
    "extract_slot_patch",
    "fuse_evidence",
    "map_detections_to_slots",
]

