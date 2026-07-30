from __future__ import annotations

import random
import statistics
from collections import defaultdict
from collections.abc import Sequence
from typing import Any


def _metrics_from_counts(
    tn: int,
    fp: int,
    fn: int,
    tp: int,
) -> dict[str, Any]:
    samples = tn + fp + fn + tp
    if samples <= 0:
        raise ValueError("Confusion counts must contain at least one sample")
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
        "samples": samples,
        "confusion_matrix": [[tn, fp], [fn, tp]],
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "accuracy": (tp + tn) / samples,
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

    return _metrics_from_counts(tn, fp, fn, tp)


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


def grouped_bootstrap_binary_metrics(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    group_ids: Sequence[str],
    *,
    iterations: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260725,
) -> dict[str, dict[str, float]]:
    """Bootstrap complete image/video groups rather than individual slots."""

    if len(y_true) != len(y_pred) or len(y_true) != len(group_ids) or not y_true:
        raise ValueError("truth, predictions, and groups must be equally sized")
    if iterations <= 0 or not 0.0 < confidence < 1.0:
        raise ValueError("invalid bootstrap settings")
    group_counts: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0, 0])
    for truth, prediction, group_id in zip(
        y_true,
        y_pred,
        group_ids,
        strict=True,
    ):
        if truth not in {0, 1} or prediction not in {0, 1}:
            raise ValueError("Labels and predictions must be binary")
        index = (
            0
            if truth == 0 and prediction == 0
            else 1
            if truth == 0
            else 2
            if prediction == 0
            else 3
        )
        group_counts[str(group_id)][index] += 1
    groups = sorted(group_counts)
    if len(groups) < 2:
        raise ValueError("grouped bootstrap needs at least two groups")

    metric_names = (
        "macro_f1",
        "occupied_recall",
        "vacant_recall",
        "false_free_rate",
        "false_occupied_rate",
    )
    observed = binary_metrics(y_true, y_pred)
    samples = {name: [] for name in metric_names}
    generator = random.Random(seed)
    for _ in range(iterations):
        sampled_groups = generator.choices(groups, k=len(groups))
        counts = [
            sum(group_counts[group][index] for group in sampled_groups)
            for index in range(4)
        ]
        metrics = _metrics_from_counts(
            counts[0],
            counts[1],
            counts[2],
            counts[3],
        )
        for name in metric_names:
            samples[name].append(float(metrics[name]))

    alpha = (1.0 - confidence) / 2.0

    def percentile(values: list[float], quantile: float) -> float:
        ordered = sorted(values)
        position = quantile * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        weight = position - lower
        return ordered[lower] * (1.0 - weight) + ordered[upper] * weight

    return {
        name: {
            "estimate": float(observed[name]),
            "lower": percentile(values, alpha),
            "upper": percentile(values, 1.0 - alpha),
        }
        for name, values in samples.items()
    }


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
    warmup_frames: int = 0,
) -> dict[str, Any]:
    """Measure flicker and signed transition timing for one slot sequence.

    A ground-truth transition is matched to the nearest *observed prediction
    change* into the target state that remains correct for ``stable_frames``.
    Each search is bounded by the adjacent ground-truth transitions. Searching
    on both sides of the current transition prevents an early change that is
    already active at the truth frame from being reported as zero latency,
    while the next transition prevents a late prediction of the old target
    state from being credited to the wrong event. Extra changes inside a
    matched event window are reported as transition instability rather than
    ordinary flicker. ``warmup_frames`` filters flicker only; it does not
    remove or shift truth events.

    ``transition_latency_s`` remains a non-negative, post-truth latency
    summary for compatibility.  Early events are excluded from that summary
    and are represented by ``signed_transition_error_s`` and
    ``transition_events`` instead of a misleading zero.
    """

    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Expected equally sized, non-empty sequences")
    if fps <= 0:
        raise ValueError("fps must be positive")
    if (
        stable_frames <= 0
        or tolerance_frames < 0
        or warmup_frames < 0
    ):
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
    entry_signed_error: list[float] = []
    exit_signed_error: list[float] = []
    transition_events: list[dict[str, Any]] = []
    outcome_counts = {
        "early": 0,
        "on_time": 0,
        "delayed": 0,
        "missed": 0,
    }
    missed = 0

    for transition_index, start in enumerate(truth_changes):
        target = y_true[start]
        direction = "entry" if target == 1 else "exit"
        search_start = (
            truth_changes[transition_index - 1]
            if transition_index > 0
            else 1
        )
        stop = (
            truth_changes[transition_index + 1]
            if transition_index + 1 < len(truth_changes)
            else len(y_true)
        )
        stable_candidates = [
            candidate
            for candidate in prediction_changes
            if search_start <= candidate < stop
            and y_pred[candidate] == target
            and candidate + stable_frames <= stop
            and all(
                y_pred[offset] == target
                for offset in range(candidate, candidate + stable_frames)
            )
            and (
                candidate >= start
                or all(
                    y_pred[offset] == target
                    for offset in range(candidate, start + 1)
                )
            )
        ]
        stable_start = (
            min(
                stable_candidates,
                key=lambda candidate: (abs(candidate - start), candidate),
            )
            if stable_candidates
            else None
        )
        if stable_start is None:
            missed += 1
            outcome_counts["missed"] += 1
            transition_events.append(
                {
                    "truth_transition_frame": start,
                    "predicted_transition_frame": None,
                    "event_window_start_frame": search_start,
                    "event_window_end_frame_exclusive": stop,
                    "direction": direction,
                    "from_state": y_true[start - 1],
                    "to_state": target,
                    "outcome": "missed",
                    "signed_error_frames": None,
                    "signed_error_s": None,
                }
            )
            continue

        signed_error_frames = stable_start - start
        signed_error_s = signed_error_frames / fps
        if signed_error_frames < -tolerance_frames:
            outcome = "early"
        elif signed_error_frames > tolerance_frames:
            outcome = "delayed"
        else:
            outcome = "on_time"
        outcome_counts[outcome] += 1
        (
            entry_signed_error if target == 1 else exit_signed_error
        ).append(signed_error_s)
        if signed_error_frames >= 0:
            (entry_latency if target == 1 else exit_latency).append(
                signed_error_s
            )
        transition_events.append(
            {
                "truth_transition_frame": start,
                "predicted_transition_frame": stable_start,
                "event_window_start_frame": search_start,
                "event_window_end_frame_exclusive": stop,
                "direction": direction,
                "from_state": y_true[start - 1],
                "to_state": target,
                "outcome": outcome,
                "signed_error_frames": signed_error_frames,
                "signed_error_s": signed_error_s,
            }
        )
        supported_windows.append(
            (
                min(stable_start, max(1, start - tolerance_frames)),
                max(stable_start, start),
            )
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
        and change >= warmup_frames
    ]
    flicker_frames = max(0, len(y_true) - warmup_frames)
    slot_minutes = flicker_frames / fps / 60.0
    all_latency = entry_latency + exit_latency
    all_signed_error = entry_signed_error + exit_signed_error
    return {
        "frames": len(y_true),
        "ground_truth_transitions": len(truth_changes),
        "matched_transitions": len(all_signed_error),
        "missed_transitions": missed,
        "early_transitions": outcome_counts["early"],
        "on_time_transitions": outcome_counts["on_time"],
        "delayed_transitions": outcome_counts["delayed"],
        "transition_outcomes": outcome_counts,
        "transition_events": transition_events,
        "unsupported_flicker_count": len(unsupported),
        "unsupported_flicker_frames": unsupported,
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
        "signed_transition_error_s": {
            "all": _summary(all_signed_error),
            "entry": _summary(entry_signed_error),
            "exit": _summary(exit_signed_error),
        },
        "signed_transition_error_values_s": {
            "entry": entry_signed_error,
            "exit": exit_signed_error,
        },
        "state_stabilization_time_s": _summary(all_latency),
        "stable_frames": stable_frames,
        "tolerance_frames": tolerance_frames,
        "warmup_frames_excluded_from_flicker_only": warmup_frames,
    }
