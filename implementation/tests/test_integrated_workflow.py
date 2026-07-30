from __future__ import annotations

import pytest

from parking_occupancy.integrated_workflow import (
    IntegratedSlotDecision,
    UncertaintyGateConfig,
    apply_track_override,
    decisions_as_slot_evidence,
    uncertainty_gated_fusion,
)
from parking_occupancy.models import SlotEvidence


def detector(slot_id: str, occupied: bool, score: float = 0.0) -> SlotEvidence:
    return SlotEvidence(
        slot_id=slot_id,
        occupied=occupied,
        geometric_score=1.0 if occupied else 0.0,
        evidence_score=score,
        track_id=7 if occupied else None,
    )


def test_detector_positive_cannot_be_overturned_by_classifier() -> None:
    decisions = uncertainty_gated_fusion(
        {"s1": detector("s1", True, 0.21)},
        {},
    )

    assert decisions["s1"].occupied
    assert decisions["s1"].branch == "detector_confirmed"
    assert decisions["s1"].classifier_consulted is False
    assert decisions["s1"].score == 1.0


def test_classifier_only_recovers_detector_negative_above_threshold() -> None:
    evidence = {
        "recovered": detector("recovered", False),
        "vacant": detector("vacant", False),
    }
    decisions = uncertainty_gated_fusion(
        evidence,
        {"recovered": 0.76, "vacant": 0.75},
        UncertaintyGateConfig(classifier_occupied_threshold=0.76),
    )

    assert decisions["recovered"].occupied
    assert decisions["recovered"].branch == "classifier_recovery"
    assert not decisions["vacant"].occupied
    assert decisions["vacant"].branch == "classifier_rejected"


def test_detector_negative_requires_classifier_score() -> None:
    with pytest.raises(KeyError, match="needs a classifier score"):
        uncertainty_gated_fusion(
            {"s1": detector("s1", False)},
            {},
        )


def test_track_override_suppresses_moving_track_and_confirms_stationary() -> None:
    decision = IntegratedSlotDecision(
        slot_id="s1",
        occupied=True,
        score=1.0,
        branch="detector_confirmed",
        detector_occupied=True,
        detector_score=0.4,
        classifier_probability=None,
        classifier_consulted=False,
        track_id=7,
    )

    moving = apply_track_override(
        decision,
        stationary_track_confirmed=False,
        moving_track_overlaps=True,
    )
    stationary = apply_track_override(
        moving,
        stationary_track_confirmed=True,
        moving_track_overlaps=True,
    )

    assert not moving.occupied
    assert moving.branch == "moving_track_suppressed"
    assert stationary.occupied
    assert stationary.branch == "stationary_track_confirmed"


def test_integrated_decisions_feed_existing_temporal_filter() -> None:
    decision = IntegratedSlotDecision(
        slot_id="s1",
        occupied=True,
        score=0.91,
        branch="classifier_recovery",
        detector_occupied=False,
        detector_score=0.0,
        classifier_probability=0.91,
        classifier_consulted=True,
        track_id=None,
    )

    evidence = decisions_as_slot_evidence({"s1": decision})

    assert evidence["s1"].occupied
    assert evidence["s1"].evidence_score == pytest.approx(0.91)
