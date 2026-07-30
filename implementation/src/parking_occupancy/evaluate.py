from __future__ import annotations

import argparse
import csv
import json
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

Key = tuple[str, int, str]


def _read_states(path: Path) -> dict[Key, dict[str, Any]]:
    rows: dict[Key, dict[str, Any]] = {}
    with path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (row["video_id"], int(row["frame_index"]), row["slot_id"])
            rows[key] = {
                "state": int(row["state"]),
                "evidence": float(row.get("evidence", row["state"])),
                "timestamp_s": float(row.get("timestamp_s", 0.0)),
            }
    return rows


def binary_metrics(y_true: list[int], y_pred: list[int]) -> dict[str, float | int]:
    if len(y_true) != len(y_pred) or not y_true:
        raise ValueError("Expected equally sized, non-empty label lists")
    tp = sum(t == 1 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    tn = sum(t == 0 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    fp = sum(t == 0 and p == 1 for t, p in zip(y_true, y_pred, strict=True))
    fn = sum(t == 1 and p == 0 for t, p in zip(y_true, y_pred, strict=True))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    occupied_f1 = (
        2 * precision * recall / (precision + recall)
        if precision + recall
        else 0.0
    )
    vacant_precision = tn / (tn + fn) if tn + fn else 0.0
    vacant_recall = tn / (tn + fp) if tn + fp else 0.0
    vacant_f1 = (
        2
        * vacant_precision
        * vacant_recall
        / (vacant_precision + vacant_recall)
        if vacant_precision + vacant_recall
        else 0.0
    )
    return {
        "samples": len(y_true),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "accuracy": (tp + tn) / len(y_true),
        "precision": precision,
        "recall": recall,
        "f1": occupied_f1,
        "occupied_precision": precision,
        "occupied_recall": recall,
        "occupied_f1": occupied_f1,
        "vacant_precision": vacant_precision,
        "vacant_recall": vacant_recall,
        "vacant_f1": vacant_f1,
        "macro_f1": (occupied_f1 + vacant_f1) / 2.0,
        "false_free_rate": fn / (tp + fn) if tp + fn else 0.0,
        "false_occupied_rate": fp / (tn + fp) if tn + fp else 0.0,
    }


def precision_recall_curve(
    y_true: list[int],
    evidence: list[float],
) -> tuple[list[float], list[float], list[float], float]:
    thresholds = sorted(set(evidence), reverse=True)
    if not thresholds:
        return [], [], [], 0.0
    thresholds = [float("inf"), *thresholds]
    precision: list[float] = [1.0]
    recall: list[float] = [0.0]
    for threshold in thresholds:
        if threshold == float("inf"):
            continue
        predicted = [int(score >= threshold) for score in evidence]
        metrics = binary_metrics(y_true, predicted)
        precision.append(float(metrics["precision"]))
        recall.append(float(metrics["recall"]))

    precision_envelope = precision.copy()
    for index in range(len(precision_envelope) - 2, -1, -1):
        precision_envelope[index] = max(
            precision_envelope[index],
            precision_envelope[index + 1],
        )
    ap = sum(
        max(0.0, recall[index] - recall[index - 1])
        * precision_envelope[index]
        for index in range(1, len(recall))
    )
    return precision, recall, thresholds, ap


def temporal_metrics(
    truth: dict[Key, dict[str, Any]],
    prediction: dict[Key, dict[str, Any]],
    fps: float,
    warmup_frames: int = 0,
    stable_frames: int = 3,
    tolerance_frames: int = 0,
) -> dict[str, Any]:
    """Aggregate the canonical signed event-window metric across slots.

    The compatibility ``transition_latency_s`` summary contains only
    non-negative post-truth matches. Early matches remain negative in
    ``signed_transition_error_s`` and are never inserted as zero latency.
    """

    try:
        from literature_core.metrics import sequence_temporal_metrics
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "literature_core is required for canonical temporal evaluation"
        ) from error
    if fps <= 0:
        raise ValueError("fps must be positive")

    grouped: dict[tuple[str, str], list[int]] = defaultdict(list)
    for video_id, frame_index, slot_id in truth:
        grouped[(video_id, slot_id)].append(frame_index)
    if not grouped:
        raise ValueError("Expected at least one slot sequence")

    sequence_rows: dict[str, dict[str, Any]] = {}
    total_frames = 0
    flicker_frames = 0
    count_keys = (
        "ground_truth_transitions",
        "matched_transitions",
        "missed_transitions",
        "early_transitions",
        "on_time_transitions",
        "delayed_transitions",
        "unsupported_flicker_count",
        "transition_instability_changes",
    )
    aggregate_counts = {key: 0 for key in count_keys}
    latency_values = {"entry": [], "exit": []}
    signed_values = {"entry": [], "exit": []}
    events: list[dict[str, Any]] = []
    for video_slot, frames in grouped.items():
        frames.sort()
        video_id, slot_id = video_slot
        y_true = [
            int(truth[(video_id, frame, slot_id)]["state"])
            for frame in frames
        ]
        y_pred = [
            int(prediction[(video_id, frame, slot_id)]["state"])
            for frame in frames
        ]
        row = sequence_temporal_metrics(
            y_true,
            y_pred,
            fps,
            stable_frames=stable_frames,
            tolerance_frames=tolerance_frames,
            warmup_frames=warmup_frames,
        )
        row_id = f"{video_id}::{slot_id}"
        sequence_rows[row_id] = row
        total_frames += len(frames)
        flicker_frames += max(0, len(frames) - warmup_frames)
        for key in count_keys:
            aggregate_counts[key] += int(row[key])
        for direction in ("entry", "exit"):
            latency_values[direction].extend(
                row["transition_latency_values_s"][direction]
            )
            signed_values[direction].extend(
                row["signed_transition_error_values_s"][direction]
            )
        for event in row["transition_events"]:
            translated = dict(event)
            translated["video_id"] = video_id
            translated["slot_id"] = slot_id
            for key in (
                "truth_transition_frame",
                "predicted_transition_frame",
                "event_window_start_frame",
                "event_window_end_frame_exclusive",
            ):
                position = translated.get(key)
                if position is None:
                    continue
                if key == "event_window_end_frame_exclusive" and position >= len(
                    frames
                ):
                    translated[f"{key}_source"] = frames[-1] + 1
                else:
                    translated[f"{key}_source"] = frames[int(position)]
            events.append(translated)

    def summary(values: list[float]) -> dict[str, float | int | None]:
        if not values:
            return {
                "count": 0,
                "median": None,
                "p90": None,
                "maximum": None,
            }
        ordered = sorted(values)
        return {
            "count": len(ordered),
            "median": statistics.median(ordered),
            "p90": ordered[round((len(ordered) - 1) * 0.9)],
            "maximum": ordered[-1],
        }

    all_latency = latency_values["entry"] + latency_values["exit"]
    all_signed = signed_values["entry"] + signed_values["exit"]
    latency_by_direction = {
        "all": summary(all_latency),
        "entry": summary(latency_values["entry"]),
        "exit": summary(latency_values["exit"]),
    }
    signed_by_direction = {
        "all": summary(all_signed),
        "entry": summary(signed_values["entry"]),
        "exit": summary(signed_values["exit"]),
    }
    compatibility_latency = {
        **latency_by_direction["all"],
        "definition": (
            "non-negative post-truth latency only; early matches excluded"
        ),
    }
    slot_minutes = flicker_frames / fps / 60.0

    return {
        "definition": "literature_core.metrics.sequence_temporal_metrics",
        "sequence_count": len(sequence_rows),
        "frames": total_frames,
        "warmup_frames_excluded": warmup_frames,
        **aggregate_counts,
        "flicker_rate_per_slot_minute": (
            aggregate_counts["unsupported_flicker_count"] / slot_minutes
            if slot_minutes
            else 0.0
        ),
        "transition_outcomes": {
            "early": aggregate_counts["early_transitions"],
            "on_time": aggregate_counts["on_time_transitions"],
            "delayed": aggregate_counts["delayed_transitions"],
            "missed": aggregate_counts["missed_transitions"],
        },
        "transition_events": events,
        "transition_latency_s": compatibility_latency,
        "transition_latency_by_direction_s": latency_by_direction,
        "transition_latency_values_s": latency_values,
        "signed_transition_error_s": signed_by_direction,
        "signed_transition_error_values_s": signed_values,
        "state_stabilization_time_s": summary(all_latency),
        "stable_frames": stable_frames,
        "tolerance_frames": tolerance_frames,
        "per_sequence": sequence_rows,
    }


def _plot_confusion(metrics: dict[str, Any], output_path: Path) -> None:
    matrix = [[metrics["tn"], metrics["fp"]], [metrics["fn"], metrics["tp"]]]
    figure, axis = plt.subplots(figsize=(4.5, 4))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1], labels=["Vacant", "Occupied"])
    axis.set_yticks([0, 1], labels=["Vacant", "Occupied"])
    axis.set_xlabel("Predicted")
    axis.set_ylabel("Ground truth")
    axis.set_title("Slot-level confusion matrix")
    for row in range(2):
        for column in range(2):
            axis.text(column, row, str(matrix[row][column]), ha="center", va="center")
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def _plot_pr(
    precision: list[float],
    recall: list[float],
    ap: float,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(5, 4))
    axis.plot(recall, precision, marker=".", linewidth=1)
    axis.set_xlim(0, 1)
    axis.set_ylim(0, 1.02)
    axis.set_xlabel("Recall")
    axis.set_ylabel("Precision")
    axis.set_title(f"Slot-level PR curve (AP={ap:.3f})")
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def evaluate(
    ground_truth_path: Path,
    predictions_path: Path,
    output_dir: Path,
    fps: float,
    warmup_frames: int = 0,
    stable_frames: int = 3,
    tolerance_frames: int = 0,
) -> dict[str, Any]:
    truth = _read_states(ground_truth_path)
    prediction = _read_states(predictions_path)
    missing = sorted(set(truth) - set(prediction))
    extra = sorted(set(prediction) - set(truth))
    if missing or extra:
        raise ValueError(
            f"Prediction/ground-truth keys differ: missing={len(missing)}, "
            f"extra={len(extra)}"
        )

    keys = sorted(truth)
    y_true = [truth[key]["state"] for key in keys]
    y_pred = [prediction[key]["state"] for key in keys]
    evidence = [prediction[key]["evidence"] for key in keys]
    classification = binary_metrics(y_true, y_pred)
    precision, recall, _thresholds, ap = precision_recall_curve(y_true, evidence)
    temporal = temporal_metrics(
        truth,
        prediction,
        fps,
        warmup_frames=warmup_frames,
        stable_frames=stable_frames,
        tolerance_frames=tolerance_frames,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    _plot_confusion(classification, output_dir / "confusion_matrix.png")
    _plot_pr(precision, recall, ap, output_dir / "pr_curve.png")
    errors_path = output_dir / "errors.csv"
    with errors_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "video_id",
                "frame_index",
                "slot_id",
                "ground_truth",
                "prediction",
                "evidence",
                "error_type",
            ],
        )
        writer.writeheader()
        for key, expected, predicted, score in zip(
            keys,
            y_true,
            y_pred,
            evidence,
            strict=True,
        ):
            if expected == predicted:
                continue
            video_id, frame_index, slot_id = key
            writer.writerow(
                {
                    "video_id": video_id,
                    "frame_index": frame_index,
                    "slot_id": slot_id,
                    "ground_truth": expected,
                    "prediction": predicted,
                    "evidence": f"{score:.6f}",
                    "error_type": (
                        "false_occupied" if predicted else "false_free"
                    ),
                }
            )
    report = {
        "classification": classification,
        "slot_average_precision": ap,
        "temporal": temporal,
        "ground_truth": str(ground_truth_path.resolve()),
        "predictions": str(predictions_path.resolve()),
        "fps": fps,
        "failure_cases_csv": str(errors_path.resolve()),
    }
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate slot-state predictions")
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument(
        "--warmup-frames",
        type=int,
        default=0,
        help="Ignore early state-establishment changes in temporal metrics only",
    )
    parser.add_argument(
        "--stable-frames",
        type=int,
        default=3,
        help="Frames a predicted target state must persist to match an event",
    )
    parser.add_argument(
        "--transition-tolerance-frames",
        type=int,
        default=0,
        help="Absolute signed-error tolerance classified as on-time",
    )
    args = parser.parse_args()
    report = evaluate(
        Path(args.ground_truth),
        Path(args.predictions),
        Path(args.output_dir),
        args.fps,
        warmup_frames=args.warmup_frames,
        stable_frames=args.stable_frames,
        tolerance_frames=args.transition_tolerance_frames,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
