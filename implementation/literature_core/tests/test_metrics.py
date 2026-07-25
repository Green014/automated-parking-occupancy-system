import pytest

from literature_core.metrics import (
    binary_metrics,
    select_threshold,
    sequence_temporal_metrics,
)


def test_binary_metrics_include_both_class_recalls_and_macro_f1() -> None:
    metrics = binary_metrics([0, 0, 1, 1], [0, 1, 1, 1])
    assert metrics["vacant_recall"] == pytest.approx(0.5)
    assert metrics["occupied_recall"] == pytest.approx(1.0)
    assert metrics["balanced_accuracy"] == pytest.approx(0.75)
    assert metrics["confusion_matrix"] == [[1, 1], [0, 2]]


def test_threshold_selection_uses_development_values() -> None:
    threshold, rows = select_threshold(
        [0, 0, 1, 1],
        [0.1, 0.2, 0.8, 0.9],
        thresholds=[0.2, 0.5, 0.8],
    )
    assert threshold == 0.5
    assert len(rows) == 3


def test_delayed_correct_transition_is_not_regular_flicker() -> None:
    metrics = sequence_temporal_metrics(
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        fps=2.0,
        stable_frames=2,
    )
    assert metrics["matched_transitions"] == 1
    assert metrics["unsupported_flicker_count"] == 0
    assert metrics["transition_latency_s"]["entry"]["median"] == 1.0


def test_unrelated_change_remains_flicker() -> None:
    metrics = sequence_temporal_metrics(
        [0, 0, 0, 1, 1, 1],
        [0, 1, 0, 0, 1, 1],
        fps=2.0,
        stable_frames=2,
    )
    assert metrics["matched_transitions"] == 1
    assert metrics["unsupported_flicker_count"] == 2

