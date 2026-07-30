import pytest

from literature_core.fusion import FusionConfig, fuse_evidence


def test_weighted_fusion_retains_components() -> None:
    result = fuse_evidence(
        "A",
        p_cls=0.8,
        p_det=0.2,
        config=FusionConfig(0.75, 0.25, 0.0),
    )
    assert result.probability == pytest.approx(0.65)
    assert result.p_cls == 0.8
    assert result.p_det == 0.2
    assert result.effective_weights == (0.75, 0.25, 0.0)


def test_missing_branch_renormalizes_available_evidence() -> None:
    result = fuse_evidence(
        "A",
        p_cls=0.7,
        p_det=None,
        config=FusionConfig(0.6, 0.4, 0.0),
    )
    assert result.probability == pytest.approx(0.7)
    assert result.effective_weights == (1.0, 0.0, 0.0)


def test_invalid_probability_is_rejected() -> None:
    with pytest.raises(ValueError, match="p_cls"):
        fuse_evidence("A", 1.2, 0.4)

