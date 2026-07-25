import pytest

from literature_core.stability import (
    positive_only_stability_metrics,
    probability_summary,
)


def test_positive_only_stability_separates_startup_and_flicker() -> None:
    rows = [
        {
            "slot_id": "A",
            "frame_index": frame,
            "truth": 1,
            "raw_state": raw,
            "state": final,
        }
        for frame, (raw, final) in enumerate(
            [(1, 0), (0, 1), (1, 1), (1, 1)]
        )
    ]
    raw = positive_only_stability_metrics(
        rows,
        state_key="raw_state",
        fps=2.0,
        warmup_frames=1,
    )
    final = positive_only_stability_metrics(
        rows,
        state_key="state",
        fps=2.0,
        warmup_frames=1,
    )
    assert raw["occupied_recall"] == pytest.approx(0.75)
    assert raw["post_warmup_unsupported_changes"] == 1
    assert raw["initial_acquisition_s"]["median"] == 0.0
    assert final["post_warmup_unsupported_changes"] == 0
    assert final["initial_acquisition_s"]["median"] == 0.5


def test_positive_only_stability_rejects_negative_truth() -> None:
    with pytest.raises(ValueError):
        positive_only_stability_metrics(
            [{"slot_id": "A", "frame_index": 0, "truth": 0, "state": 0}],
            state_key="state",
            fps=2.0,
        )


def test_probability_summary_preserves_zero_evidence_rate() -> None:
    summary = probability_summary([0.0, 0.0, 0.5, 1.0])
    assert summary["mean"] == pytest.approx(0.375)
    assert summary["zero_fraction"] == pytest.approx(0.5)
    assert summary["maximum"] == 1.0
