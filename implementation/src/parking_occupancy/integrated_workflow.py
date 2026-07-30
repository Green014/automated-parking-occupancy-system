from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace

from .models import SlotEvidence


@dataclass(frozen=True, slots=True)
class UncertaintyGateConfig:
    """Asymmetric detector-first fusion for slot occupancy.

    A mapped vehicle is treated as positive evidence that the classifier
    cannot overturn. A slot without a mapped vehicle is uncertain because it
    may be genuinely vacant or contain a missed vehicle, so the classifier is
    consulted only for that branch.
    """

    classifier_occupied_threshold: float = 0.76

    def __post_init__(self) -> None:
        if not 0.0 <= self.classifier_occupied_threshold <= 1.0:
            raise ValueError(
                "classifier_occupied_threshold must be in [0, 1]"
            )


@dataclass(frozen=True, slots=True)
class IntegratedSlotDecision:
    slot_id: str
    occupied: bool
    score: float
    branch: str
    detector_occupied: bool
    detector_score: float
    classifier_probability: float | None
    classifier_consulted: bool
    track_id: int | None


def uncertainty_gated_fusion(
    detector_evidence: Mapping[str, SlotEvidence],
    classifier_probabilities: Mapping[str, float],
    config: UncertaintyGateConfig | None = None,
) -> dict[str, IntegratedSlotDecision]:
    """Fuse detector and classifier evidence without symmetric score averaging."""

    config = config or UncertaintyGateConfig()
    unknown_classifier_slots = set(classifier_probabilities).difference(
        detector_evidence
    )
    if unknown_classifier_slots:
        raise KeyError(
            "Classifier scores contain unknown slots: "
            + ", ".join(sorted(unknown_classifier_slots))
        )

    decisions: dict[str, IntegratedSlotDecision] = {}
    for slot_id, detector in detector_evidence.items():
        if detector.occupied:
            decisions[slot_id] = IntegratedSlotDecision(
                slot_id=slot_id,
                occupied=True,
                score=1.0,
                branch="detector_confirmed",
                detector_occupied=True,
                detector_score=detector.evidence_score,
                classifier_probability=None,
                classifier_consulted=False,
                track_id=detector.track_id,
            )
            continue

        if slot_id not in classifier_probabilities:
            raise KeyError(
                f"Detector-negative slot {slot_id} needs a classifier score"
            )
        classifier_probability = float(classifier_probabilities[slot_id])
        if not 0.0 <= classifier_probability <= 1.0:
            raise ValueError(
                f"Classifier probability for {slot_id} must be in [0, 1]"
            )
        occupied = (
            classifier_probability >= config.classifier_occupied_threshold
        )
        decisions[slot_id] = IntegratedSlotDecision(
            slot_id=slot_id,
            occupied=occupied,
            score=classifier_probability,
            branch=(
                "classifier_recovery"
                if occupied
                else "classifier_rejected"
            ),
            detector_occupied=False,
            detector_score=detector.evidence_score,
            classifier_probability=classifier_probability,
            classifier_consulted=True,
            track_id=None,
        )
    return decisions


def apply_track_override(
    decision: IntegratedSlotDecision,
    *,
    stationary_track_confirmed: bool,
    moving_track_overlaps: bool,
) -> IntegratedSlotDecision:
    """Apply the optional tracking branch before temporal stabilization."""

    if stationary_track_confirmed:
        return replace(
            decision,
            occupied=True,
            score=1.0,
            branch="stationary_track_confirmed",
        )
    if moving_track_overlaps:
        return replace(
            decision,
            occupied=False,
            score=0.0,
            branch="moving_track_suppressed",
        )
    return decision


def decisions_as_slot_evidence(
    decisions: Mapping[str, IntegratedSlotDecision],
) -> dict[str, SlotEvidence]:
    """Adapt fused decisions to the existing temporal-filter interface."""

    return {
        slot_id: SlotEvidence(
            slot_id=slot_id,
            occupied=decision.occupied,
            geometric_score=(
                1.0 if decision.occupied else 0.0
            ),
            evidence_score=decision.score,
            track_id=decision.track_id,
        )
        for slot_id, decision in decisions.items()
    }
