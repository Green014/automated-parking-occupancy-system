from literature_core.fusion import fuse_evidence
from literature_core.temporal import TemporalConfig, TemporalFusionFilter


def test_hysteresis_changes_only_after_thresholds() -> None:
    filter_ = TemporalFusionFilter(
        ("A",),
        TemporalConfig(
            rise_alpha=1.0,
            fall_alpha=1.0,
            occupied_threshold=0.6,
            vacant_threshold=0.4,
        ),
    )
    assert not filter_.update(fuse_evidence("A", 0.5, None)).occupied
    occupied = filter_.update(fuse_evidence("A", 0.8, None))
    assert occupied.occupied
    assert occupied.changed
    assert filter_.update(fuse_evidence("A", 0.5, None)).occupied
    vacant = filter_.update(fuse_evidence("A", 0.2, None))
    assert not vacant.occupied
    assert vacant.changed


def test_asymmetric_ema_suppresses_single_low_frame() -> None:
    filter_ = TemporalFusionFilter(
        ("A",),
        TemporalConfig(
            rise_alpha=1.0,
            fall_alpha=0.1,
            occupied_threshold=0.6,
            vacant_threshold=0.3,
        ),
    )
    assert filter_.update(fuse_evidence("A", 0.9, None)).occupied
    state = filter_.update(fuse_evidence("A", 0.0, None))
    assert state.occupied
    assert state.filtered_probability > 0.3

