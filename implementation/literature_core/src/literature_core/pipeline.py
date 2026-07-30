from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from .detector import ObjectDetector
from .fusion import FusionConfig, fuse_evidence
from .mapping import map_detections_to_slots
from .models import FrameResult, ParkingSlot, SlotDecision
from .patches import extract_slot_patch
from .temporal import TemporalConfig, TemporalFusionFilter


class PatchClassifier(Protocol):
    patch_size: tuple[int, int]

    def predict_patches(self, patches: list[np.ndarray]) -> list[float]:
        """Return occupied probability for every patch."""


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    patch_size: tuple[int, int] = (224, 224)
    perspective_warp: bool = True
    minimum_slot_coverage: float = 0.10
    one_to_one: bool = True
    decision_threshold: float = 0.50
    fusion: FusionConfig = FusionConfig()
    temporal: TemporalConfig = TemporalConfig()
    use_temporal: bool = True

    def __post_init__(self) -> None:
        if not 0.0 <= self.decision_threshold <= 1.0:
            raise ValueError("decision_threshold must be in [0, 1]")


class LiteratureCorePipeline:
    """Frame-level minimum closure with every branch output retained."""

    def __init__(
        self,
        slots: tuple[ParkingSlot, ...],
        classifier: PatchClassifier | None,
        detector: ObjectDetector | None,
        config: PipelineConfig | None = None,
    ) -> None:
        if not slots or len({slot.slot_id for slot in slots}) != len(slots):
            raise ValueError("slots must be non-empty with unique IDs")
        if classifier is None and detector is None:
            raise ValueError("At least one evidence branch is required")
        self.slots = slots
        self.classifier = classifier
        self.detector = detector
        self.config = config or PipelineConfig()
        self._previous = {slot.slot_id: False for slot in slots}
        self._temporal = (
            TemporalFusionFilter(
                tuple(slot.slot_id for slot in slots),
                self.config.temporal,
            )
            if self.config.use_temporal
            else None
        )

    def process_frame(
        self,
        frame: np.ndarray,
        frame_index: int,
        timestamp_s: float,
    ) -> FrameResult:
        patches = (
            [
                extract_slot_patch(
                    frame,
                    slot.points,
                    output_size=self.config.patch_size,
                    perspective_warp=self.config.perspective_warp,
                )
                for slot in self.slots
            ]
            if self.classifier is not None
            else []
        )
        p_cls = (
            self.classifier.predict_patches(patches)
            if self.classifier is not None
            else [None] * len(self.slots)
        )
        detections = self.detector.detect(frame) if self.detector is not None else []
        detector_evidence = map_detections_to_slots(
            detections,
            self.slots,
            minimum_slot_coverage=self.config.minimum_slot_coverage,
            one_to_one=self.config.one_to_one,
        )
        p_det: list[float | None] = (
            [item.probability for item in detector_evidence]
            if self.detector is not None
            else [None] * len(self.slots)
        )

        decisions: list[SlotDecision] = []
        for slot, classifier_probability, detector_probability in zip(
            self.slots,
            p_cls,
            p_det,
            strict=True,
        ):
            fused = fuse_evidence(
                slot.slot_id,
                classifier_probability,
                detector_probability,
                config=self.config.fusion,
            )
            if self._temporal is not None:
                temporal = self._temporal.update(fused)
                occupied = temporal.occupied
                filtered_probability = temporal.filtered_probability
                raw_occupied = temporal.raw_occupied
                changed = temporal.changed
            else:
                occupied = fused.probability >= self.config.decision_threshold
                filtered_probability = fused.probability
                raw_occupied = occupied
                changed = occupied != self._previous[slot.slot_id]
            self._previous[slot.slot_id] = occupied
            decisions.append(
                SlotDecision(
                    slot_id=slot.slot_id,
                    occupied=occupied,
                    probability=fused.probability,
                    filtered_probability=filtered_probability,
                    p_cls=classifier_probability,
                    p_det=detector_probability,
                    p_track=None,
                    raw_occupied=raw_occupied,
                    changed=changed,
                )
            )
        return FrameResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            detections=tuple(detections),
            detector_evidence=detector_evidence,
            decisions=tuple(decisions),
        )
