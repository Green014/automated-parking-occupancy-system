from __future__ import annotations

from dataclasses import dataclass

from .models import FusedEvidence


@dataclass(frozen=True, slots=True)
class FusionConfig:
    classifier_weight: float = 0.65
    detector_weight: float = 0.35
    track_weight: float = 0.0

    def __post_init__(self) -> None:
        weights = (
            self.classifier_weight,
            self.detector_weight,
            self.track_weight,
        )
        if any(weight < 0.0 for weight in weights):
            raise ValueError("Fusion weights must be non-negative")
        if sum(weights) <= 0.0:
            raise ValueError("At least one fusion weight must be positive")


def _probability(value: float | None, name: str) -> float | None:
    if value is not None and not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be in [0, 1]")
    return value


def fuse_evidence(
    slot_id: str,
    p_cls: float | None,
    p_det: float | None,
    p_track: float | None = None,
    config: FusionConfig | None = None,
) -> FusedEvidence:
    """Compute an interpretable normalized weighted sum of available branches."""

    config = config or FusionConfig()
    values = (
        _probability(p_cls, "p_cls"),
        _probability(p_det, "p_det"),
        _probability(p_track, "p_track"),
    )
    configured = (
        config.classifier_weight,
        config.detector_weight,
        config.track_weight,
    )
    active = tuple(
        weight if value is not None else 0.0
        for value, weight in zip(values, configured, strict=True)
    )
    denominator = sum(active)
    if denominator <= 0.0:
        raise ValueError("No available evidence branch has a positive weight")
    effective = tuple(weight / denominator for weight in active)
    probability = sum(
        (0.0 if value is None else value) * weight
        for value, weight in zip(values, effective, strict=True)
    )
    return FusedEvidence(
        slot_id=slot_id,
        p_cls=p_cls,
        p_det=p_det,
        p_track=p_track,
        probability=probability,
        effective_weights=effective,
    )

