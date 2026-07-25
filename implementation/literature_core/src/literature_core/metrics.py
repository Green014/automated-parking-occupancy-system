from __future__ import annotations

import statistics
from collections.abc import Sequence
from typing import Any


def binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> dict[str, Any]:
    """Return class-aware slot metrics without external metric dependencies."""

    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Expected equally sized, non-empty label sequences")
    if any(value not in {0, 1} for value in (*y_true, *y_pred)):
        raise ValueError("Labels and predictions must be binary")
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred, strict=True))

    occupied_precision = tp / (tp + fp) if tp + fp else 0.0
    occupied_recall = tp / (tp + fn) if tp + fn else 0.0
    occupied_f1 = (
        2 * occupied_precision * occupied_recall
        / (occupied_precision + occupied_recall)
        if occupied_precision + occupied_recall
        else 0.0
    )
    vacant_precision = tn / (tn + fn) if tn + fn else 0.0
    vacant_recall = tn / (tn + fp) if tn + fp else 0.0
    vacant_f1 = (
        2 * vacant_precision * vacant_recall
        / (vacant_precision + vacant_recall)
        if vacant_precision + vacant_recall
        else 0.0
    )
    return {
        "samples": len(y_true),
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / len(y_true),
        "precision": occupied_precision,
        "recall": occupied_recall,
        "f1": occupied_f1,
        "occupied_recall": occupied_recall,
        "vacant_recall": vacant_recall,
        "balanced_accuracy": (occupied_recall + vacant_recall) / 2.0,
        "macro_f1": (occupied_f1 + vacant_f1) / 2.0,
        "false_free_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_occupied_rate": fp / (tn + fp) if tn + fp else 0.0,
    }


def evaluate_probabilities(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    threshold: float,
) -> dict[str, Any]:
    if len(y_true) != len(probabilities):
        raise ValueError("Label/probability lengths differ")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be in [0, 1]")
    if any(not 0.0 <= value <= 1.0 for value in probabilities):
        raise ValueError("probabilities must be in [0, 1]")
    result = binary_metrics(
        y_true,
        [int(value >= threshold) for value in probabilities],
    )
    result["threshold"] = threshold
    return result


def select_threshold(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    thresholds: Sequence[float] | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Select macro-F1 threshold with a false-free and 0.5 tie-break."""

    thresholds = thresholds or tuple(index / 100 for index in range(1, 100))
    rows = [
        evaluate_probabilities(y_true, probabilities, threshold)
        for threshold in thresholds
    ]
    selected = max(
        rows,
        key=lambda row: (
            row["macro_f1"],
            -row["false_free_rate"],
            -abs(row["threshold"] - 0.5),
        ),
    )
    return float(selected["threshold"]), rows


def _summary(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    ordered = sorted(values)
    p90_index = round((len(ordered) - 1) * 0.9)
    return {
        "count": len(values),
        "median": statistics.median(ordered),
        "p90": ordered[p90_index],
        "maximum": ordered[-1],
    }


def sequence_temporal_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    fps: float,
    stable_frames: int = 3,
    tolerance_frames: int = 0,
) -> dict[str, Any]:
    """Measure flicker and transition latency for one ordered slot sequence.

    A delayed transition that reaches a stable correct state is assigned to
    the ground-truth transition window and is not counted as ordinary flicker.
    Extra changes inside that window are reported separately as transition
    instability.
    """

    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Expected equally sized, non-empty sequences")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if stable_frames <= 0 or tolerance_frames < 0:
        raise ValueError("Invalid temporal window")
    if any(value not in {0, 1} for value in (*y_true, *y_pred)):
        raise ValueError("Temporal states must be binary")

    truth_changes = [
        index
        for index in range(1, len(y_true))
        if y_true[index] != y_true[index - 1]
    ]
    prediction_changes = [
        index
        for index in range(1, len(y_pred))
        if y_pred[index] != y_pred[index - 1]
    ]
    supported_windows: list[tuple[int, int]] = []
    entry_latency: list[float] = []
    exit_latency: list[float] = []
    missed = 0

    for transition_index, start in enumerate(truth_changes):
        target = y_true[start]
        stop = (
            truth_changes[transition_index + 1]
            if transition_index + 1 < len(truth_changes)
            else len(y_true)
        )
        stable_start = next(
            (
                candidate
                for candidate in range(start, stop)
                if candidate + stable_frames <= stop
                and all(
                    y_pred[offset] == target
                    for offset in range(candidate, candidate + stable_frames)
                )
            ),
            None,
        )
        if stable_start is None:
            missed += 1
            continue
        latency = (stable_start - start) / fps
        (entry_latency if target == 1 else exit_latency).append(latency)
        supported_windows.append(
            (max(1, start - tolerance_frames), stable_start)
        )

    window_change_counts = [
        sum(begin <= change <= end for change in prediction_changes)
        for begin, end in supported_windows
    ]
    transition_instability = sum(
        max(0, count - 1) for count in window_change_counts
    )
    unsupported = [
        change
        for change in prediction_changes
        if not any(begin <= change <= end for begin, end in supported_windows)
    ]
    slot_minutes = len(y_true) / fps / 60.0
    all_latency = entry_latency + exit_latency
    return {
        "frames": len(y_true),
        "ground_truth_transitions": len(truth_changes),
        "matched_transitions": len(all_latency),
        "missed_transitions": missed,
        "unsupported_flicker_count": len(unsupported),
        "flicker_rate_per_slot_minute": (
            len(unsupported) / slot_minutes if slot_minutes else 0.0
        ),
        "transition_instability_changes": transition_instability,
        "transition_latency_s": {
            "all": _summary(all_latency),
            "entry": _summary(entry_latency),
            "exit": _summary(exit_latency),
        },
        "transition_latency_values_s": {
            "entry": entry_latency,
            "exit": exit_latency,
        },
        "state_stabilization_time_s": _summary(all_latency),
        "stable_frames": stable_frames,
        "tolerance_frames": tolerance_frames,
    }
