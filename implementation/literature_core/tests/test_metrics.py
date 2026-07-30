import pytest

from literature_core.metrics import (
    binary_metrics,
    grouped_bootstrap_binary_metrics,
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


def test_on_time_transition_has_zero_signed_error() -> None:
    metrics = sequence_temporal_metrics(
        [0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1],
        fps=2.0,
        stable_frames=2,
    )
    assert metrics["transition_outcomes"] == {
        "early": 0,
        "on_time": 1,
        "delayed": 0,
        "missed": 0,
    }
    assert metrics["signed_transition_error_values_s"]["entry"] == [0.0]


def test_early_transition_retains_negative_signed_error() -> None:
    metrics = sequence_temporal_metrics(
        [0, 0, 0, 0, 1, 1, 1],
        [0, 0, 1, 1, 1, 1, 1],
        fps=2.0,
        stable_frames=2,
    )
    assert metrics["early_transitions"] == 1
    assert metrics["transition_events"][0]["predicted_transition_frame"] == 2
    assert metrics["transition_events"][0]["signed_error_frames"] == -2
    assert metrics["signed_transition_error_s"]["entry"]["median"] == -1.0
    assert metrics["transition_latency_s"]["entry"]["count"] == 0


def test_delayed_correct_transition_is_not_regular_flicker() -> None:
    metrics = sequence_temporal_metrics(
        [0, 0, 1, 1, 1, 1],
        [0, 0, 0, 0, 1, 1],
        fps=2.0,
        stable_frames=2,
    )
    assert metrics["matched_transitions"] == 1
    assert metrics["delayed_transitions"] == 1
    assert metrics["unsupported_flicker_count"] == 0
    assert metrics["transition_latency_s"]["entry"]["median"] == 1.0


def test_completely_missed_transition_is_reported() -> None:
    metrics = sequence_temporal_metrics(
        [0, 0, 1, 1, 1],
        [0, 0, 0, 0, 0],
        fps=2.0,
        stable_frames=2,
    )
    assert metrics["matched_transitions"] == 0
    assert metrics["missed_transitions"] == 1
    assert metrics["transition_outcomes"]["missed"] == 1
    assert metrics["transition_events"][0]["predicted_transition_frame"] is None


def test_brief_correct_state_after_transition_is_not_stable_match() -> None:
    metrics = sequence_temporal_metrics(
        [0, 0, 1, 1, 1, 1],
        [0, 0, 1, 0, 0, 0],
        fps=2.0,
        stable_frames=2,
    )
    assert metrics["matched_transitions"] == 0
    assert metrics["missed_transitions"] == 1
    assert metrics["unsupported_flicker_count"] == 2


def test_old_target_state_after_next_truth_transition_is_not_matched() -> None:
    metrics = sequence_temporal_metrics(
        [0, 0, 1, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1, 1],
        fps=2.0,
        stable_frames=2,
    )

    first_event = metrics["transition_events"][0]
    assert first_event["truth_transition_frame"] == 2
    assert first_event["event_window_end_frame_exclusive"] == 4
    assert first_event["predicted_transition_frame"] is None
    assert first_event["outcome"] == "missed"


def test_multiple_prediction_jumps_are_separated_from_matched_transition() -> None:
    metrics = sequence_temporal_metrics(
        [0, 0, 0, 1, 1, 1, 1],
        [0, 1, 0, 1, 0, 1, 1],
        fps=2.0,
        stable_frames=2,
        tolerance_frames=2,
    )
    assert metrics["matched_transitions"] == 1
    assert metrics["delayed_transitions"] == 0
    assert metrics["on_time_transitions"] == 1
    assert metrics["transition_events"][0]["predicted_transition_frame"] == 5
    assert metrics["transition_instability_changes"] == 4
    assert metrics["unsupported_flicker_count"] == 0


def test_grouped_bootstrap_resamples_complete_groups_deterministically() -> None:
    arguments = (
        [0, 1, 0, 1, 0, 1],
        [0, 1, 1, 1, 0, 0],
        ["image_a", "image_a", "image_b", "image_b", "image_c", "image_c"],
    )
    result = grouped_bootstrap_binary_metrics(
        *arguments,
        iterations=50,
        seed=7,
    )
    repeated = grouped_bootstrap_binary_metrics(
        *arguments,
        iterations=50,
        seed=7,
    )
    assert result == repeated
    assert result["macro_f1"]["lower"] <= result["macro_f1"]["estimate"]
    assert result["macro_f1"]["estimate"] <= result["macro_f1"]["upper"]
