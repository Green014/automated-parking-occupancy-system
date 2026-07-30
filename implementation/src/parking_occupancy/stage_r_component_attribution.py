from __future__ import annotations

import csv
import hashlib
import json
import math
import statistics
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import yaml


STAGE_R_PROTOCOL_ID = "STAGE-R-QV2-POSTHOC-COMPONENT-ATTRIBUTION-20260729-01"
STAGE_Q_V2_PROTOCOL_ID = (
    "STAGE-Q-V2-UPM-GTI-EXTERNAL-NIGHT-OCCUPANCY-20260729-01"
)
METHODS = {
    "D1": "QV2-0",
    "D1-LL": "QV2-1",
}
COMPONENTS = {
    "R0": {
        "prediction_field": "detector_occupied",
        "pipeline": "B1",
    },
    "R1": {
        "prediction_field": "raw_state",
        "pipeline": "B1 + F2",
    },
    "R2": {
        "prediction_field": "state",
        "pipeline": "B1 + F2 + E4",
    },
}
DELTA_PAIRS = (("R0", "R1"), ("R1", "R2"))
REQUIRED_OCCUPANCY_FIELDS = {
    "video_id",
    "frame_index",
    "slot_id",
    "detector_occupied",
    "raw_state",
    "state",
}
METRIC_FIELDS = (
    "tp",
    "tn",
    "fp",
    "fn",
    "macro_f1",
    "occupied_precision",
    "occupied_recall",
    "occupied_f1",
    "vacant_precision",
    "vacant_recall",
    "vacant_f1",
    "accuracy",
    "balanced_accuracy",
    "false_free_rate",
    "false_occupied_rate",
    "occupied_count_mae",
    "occupied_count_rmse",
    "occupied_count_mean_signed_error",
)
COMPARISON_FIELDS = (
    "detector",
    "method_id",
    "component",
    "pipeline",
    "prediction_field",
    "scope_type",
    "scope_id",
    "samples",
    "frames",
    "truth_occupied",
    "truth_vacant",
    *METRIC_FIELDS,
)


class StageRAnalysisError(RuntimeError):
    """Raised when a frozen Stage Q-v2 input fails Stage R validation."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_divide(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def binary_metrics(
    truth: Sequence[int],
    prediction: Sequence[int],
) -> dict[str, int | float]:
    if len(truth) != len(prediction) or not truth:
        raise ValueError("truth and prediction must be equally sized and non-empty")
    if any(value not in {0, 1} for value in (*truth, *prediction)):
        raise ValueError("Stage R classification values must be binary")

    tp = sum(
        expected == 1 and actual == 1
        for expected, actual in zip(truth, prediction, strict=True)
    )
    tn = sum(
        expected == 0 and actual == 0
        for expected, actual in zip(truth, prediction, strict=True)
    )
    fp = sum(
        expected == 0 and actual == 1
        for expected, actual in zip(truth, prediction, strict=True)
    )
    fn = sum(
        expected == 1 and actual == 0
        for expected, actual in zip(truth, prediction, strict=True)
    )
    occupied_precision = _safe_divide(tp, tp + fp)
    occupied_recall = _safe_divide(tp, tp + fn)
    occupied_f1 = _safe_divide(
        2 * occupied_precision * occupied_recall,
        occupied_precision + occupied_recall,
    )
    vacant_precision = _safe_divide(tn, tn + fn)
    vacant_recall = _safe_divide(tn, tn + fp)
    vacant_f1 = _safe_divide(
        2 * vacant_precision * vacant_recall,
        vacant_precision + vacant_recall,
    )
    samples = len(truth)
    return {
        "samples": samples,
        "truth_occupied": tp + fn,
        "truth_vacant": tn + fp,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "macro_f1": (occupied_f1 + vacant_f1) / 2,
        "occupied_precision": occupied_precision,
        "occupied_recall": occupied_recall,
        "occupied_f1": occupied_f1,
        "vacant_precision": vacant_precision,
        "vacant_recall": vacant_recall,
        "vacant_f1": vacant_f1,
        "accuracy": (tp + tn) / samples,
        "balanced_accuracy": (occupied_recall + vacant_recall) / 2,
        "false_free_rate": _safe_divide(fn, tp + fn),
        "false_occupied_rate": _safe_divide(fp, tn + fp),
    }


def occupied_count_metrics(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    if not records:
        raise ValueError("occupied-count metrics need at least one record")
    frame_counts: dict[tuple[str, int], list[int]] = defaultdict(
        lambda: [0, 0]
    )
    for record in records:
        key = (str(record["video_id"]), int(record["frame_index"]))
        frame_counts[key][0] += int(record["truth"])
        frame_counts[key][1] += int(record["prediction"])
    errors = [
        predicted - expected
        for expected, predicted in frame_counts.values()
    ]
    return {
        "frames": len(frame_counts),
        "occupied_count_mae": statistics.fmean(abs(error) for error in errors),
        "occupied_count_rmse": math.sqrt(
            statistics.fmean(error * error for error in errors)
        ),
        "occupied_count_mean_signed_error": statistics.fmean(errors),
    }


def evaluate_records(
    records: Sequence[Mapping[str, Any]],
) -> dict[str, int | float]:
    result = binary_metrics(
        [int(record["truth"]) for record in records],
        [int(record["prediction"]) for record in records],
    )
    result.update(occupied_count_metrics(records))
    return result


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise StageRAnalysisError(f"CSV has no header: {path}")
        return list(reader.fieldnames), list(reader)


def _record_for_path(
    registry: Mapping[str, Any],
    expected: Path,
) -> Mapping[str, Any]:
    expected_resolved = str(expected.resolve()).casefold()
    expected_suffix = str(expected).replace("\\", "/").casefold()
    for record in registry["artifacts"]:
        stored = str(record["path"])
        if str(Path(stored).resolve()).casefold() == expected_resolved:
            return record
        normalized = stored.replace("\\", "/").casefold()
        if normalized.endswith(expected_suffix):
            return record
    raise StageRAnalysisError(f"Frozen registry has no binding for {expected}")


def _verify_record(path: Path, record: Mapping[str, Any]) -> dict[str, Any]:
    if not path.is_file():
        raise StageRAnalysisError(f"Frozen input is missing: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != int(record["bytes"]):
        raise StageRAnalysisError(f"Frozen byte count changed: {path}")
    if actual_sha256 != str(record["sha256"]):
        raise StageRAnalysisError(f"Frozen SHA-256 changed: {path}")
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
    }


def _resolve_config_binding(config_path: Path, record: Mapping[str, Any]) -> Path:
    return (config_path.parent / str(record["path"])).resolve()


def _key(row: Mapping[str, str]) -> tuple[str, int, str]:
    return (
        row["video_id"],
        int(row["frame_index"]),
        row["slot_id"],
    )


def _relative(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def validate_frozen_inputs(project_root: Path) -> dict[str, Any]:
    project_root = project_root.resolve()
    output_root = (
        project_root
        / "outputs"
        / "stage_q_v2_upm_gti_external_20260729_v2"
    )
    report_path = (
        project_root / "data" / "STAGE_Q_V2_UPM_GTI_EXTERNAL_EVALUATION_REPORT.md"
    )
    config_path = (
        project_root
        / "configs"
        / "stage_q_v2_external_night_occupancy_frozen_20260729_v2.yaml"
    )
    registry_path = (
        project_root
        / "data"
        / "stage_q_v2"
        / "STAGE_Q_V2_ARTIFACT_REGISTRY_20260729.yaml"
    )
    for path in (output_root, report_path, config_path, registry_path):
        if not path.exists():
            raise StageRAnalysisError(f"Required frozen input is missing: {path}")

    registry = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if registry.get("protocol_id") != STAGE_Q_V2_PROTOCOL_ID:
        raise StageRAnalysisError("Unexpected Stage Q-v2 registry protocol")
    if registry.get("status") != "FORMAL_RUNS_COMPLETE_AND_HASH_VERIFIED":
        raise StageRAnalysisError("Stage Q-v2 registry is not formally complete")
    if registry.get("model_track_called") is not False:
        raise StageRAnalysisError("Frozen Stage Q-v2 unexpectedly called tracking")

    additive_config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if additive_config.get("protocol_id") != STAGE_Q_V2_PROTOCOL_ID:
        raise StageRAnalysisError("Unexpected Stage Q-v2 additive config protocol")
    if additive_config.get("status") != "frozen_before_formal_model_inference":
        raise StageRAnalysisError("Stage Q-v2 additive config is not frozen")
    invariants = additive_config.get("invariants", {})
    required_invariants = {
        "detector_weights_changed": False,
        "E1b_checkpoint_changed": False,
        "manifest_changed": False,
        "truth_changed": False,
        "polygons_changed": False,
        "P3_parameters_changed": False,
        "method_order_changed": False,
        "D1_remains_project_default": True,
    }
    for name, expected in required_invariants.items():
        if invariants.get(name) is not expected:
            raise StageRAnalysisError(f"Stage Q-v2 invariant changed: {name}")

    input_audit: dict[str, Any] = {
        "stage_q_v2_registry": {
            "path": _relative(registry_path, project_root),
            "bytes": registry_path.stat().st_size,
            "sha256": sha256_file(registry_path),
            "artifact_count": int(registry["artifact_count"]),
            "status": registry["status"],
        }
    }
    for label, path in (
        ("stage_q_v2_report", report_path),
        ("stage_q_v2_additive_config", config_path),
    ):
        verified = _verify_record(path, _record_for_path(registry, path))
        verified["path"] = _relative(path, project_root)
        input_audit[label] = verified
    if (
        input_audit["stage_q_v2_additive_config"]["sha256"]
        != registry.get("formal_config_sha256")
    ):
        raise StageRAnalysisError("Registry formal-config SHA-256 is inconsistent")

    base_binding = additive_config["additive_retry_of"]
    base_config_path = _resolve_config_binding(config_path, base_binding)
    if base_config_path.stat().st_size != int(base_binding["bytes"]):
        raise StageRAnalysisError("Base frozen config byte count changed")
    if sha256_file(base_config_path) != str(base_binding["sha256"]):
        raise StageRAnalysisError("Base frozen config SHA-256 changed")
    base_config = yaml.safe_load(base_config_path.read_text(encoding="utf-8"))
    if base_config.get("protocol_id") != STAGE_Q_V2_PROTOCOL_ID:
        raise StageRAnalysisError("Unexpected Stage Q-v2 base config protocol")

    bound_inputs: dict[str, Path] = {}
    for label in ("manifest", "occupancy_truth", "polygons"):
        binding = base_config["inputs"][label]
        path = _resolve_config_binding(base_config_path, binding)
        if path.stat().st_size != int(binding["bytes"]):
            raise StageRAnalysisError(f"Frozen {label} byte count changed")
        if sha256_file(path) != str(binding["sha256"]):
            raise StageRAnalysisError(f"Frozen {label} SHA-256 changed")
        bound_inputs[label] = path
        input_audit[label] = {
            "path": _relative(path, project_root),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }

    truth_fields, truth_rows = _read_csv(bound_inputs["occupancy_truth"])
    manifest_fields, manifest_rows = _read_csv(bound_inputs["manifest"])
    if truth_fields != ["video_id", "frame_index", "timestamp_s", "slot_id", "state"]:
        raise StageRAnalysisError("Frozen truth CSV fields changed")
    required_manifest_fields = {
        "sequence_id",
        "frame_index",
        "source_occupancy_vector",
    }
    if not required_manifest_fields.issubset(manifest_fields):
        raise StageRAnalysisError("Frozen manifest fields are incomplete")
    if len(truth_rows) != int(base_config["inputs"]["occupancy_truth"]["rows"]):
        raise StageRAnalysisError("Frozen truth row count changed")
    if len(manifest_rows) != int(base_config["inputs"]["manifest"]["rows"]):
        raise StageRAnalysisError("Frozen manifest row count changed")

    truth: dict[tuple[str, int, str], dict[str, str]] = {}
    for row in truth_rows:
        key = _key(row)
        if key in truth:
            raise StageRAnalysisError(f"Duplicate frozen truth key: {key}")
        if row["state"] not in {"0", "1"}:
            raise StageRAnalysisError(f"Non-binary frozen truth at {key}")
        truth[key] = row

    manifest: dict[tuple[str, int], dict[str, str]] = {}
    for row in manifest_rows:
        frame_key = (row["sequence_id"], int(row["frame_index"]))
        if frame_key in manifest:
            raise StageRAnalysisError(f"Duplicate manifest frame: {frame_key}")
        vector = row["source_occupancy_vector"]
        if len(vector) != 21 or set(vector) - {"0", "1"}:
            raise StageRAnalysisError(f"Invalid source occupancy vector: {frame_key}")
        manifest[frame_key] = row

    expected_slots = [f"slot_{index:02d}" for index in range(21)]
    if sorted({key[2] for key in truth}) != expected_slots:
        raise StageRAnalysisError("Frozen truth slot IDs changed")
    if {(key[0], key[1]) for key in truth} != set(manifest):
        raise StageRAnalysisError("Frozen truth and manifest frame keys differ")
    for (video_id, frame_index, slot_id), row in truth.items():
        source_value = manifest[(video_id, frame_index)][
            "source_occupancy_vector"
        ][int(slot_id.split("_")[1])]
        expected = 1 if source_value == "0" else 0
        if int(row["state"]) != expected:
            raise StageRAnalysisError(
                f"Truth/source encoding mismatch at {(video_id, frame_index, slot_id)}"
            )

    predictions: dict[str, dict[tuple[str, int, str], dict[str, str]]] = {}
    occupancy_fields: dict[str, list[str]] = {}
    for detector, method_id in METHODS.items():
        occupancy_path = output_root / method_id / "occupancy.csv"
        verified = _verify_record(
            occupancy_path,
            _record_for_path(registry, occupancy_path),
        )
        verified["path"] = _relative(occupancy_path, project_root)
        input_audit[f"{detector}_occupancy"] = verified
        fields, rows = _read_csv(occupancy_path)
        if not REQUIRED_OCCUPANCY_FIELDS.issubset(fields):
            missing = sorted(REQUIRED_OCCUPANCY_FIELDS - set(fields))
            raise StageRAnalysisError(
                f"{detector} occupancy fields are missing: {missing}"
            )
        prediction: dict[tuple[str, int, str], dict[str, str]] = {}
        for row in rows:
            key = _key(row)
            if key in prediction:
                raise StageRAnalysisError(
                    f"Duplicate {detector} occupancy key: {key}"
                )
            for component in COMPONENTS.values():
                field = str(component["prediction_field"])
                if row[field] not in {"0", "1"}:
                    raise StageRAnalysisError(
                        f"Non-binary {detector} {field} at {key}"
                    )
            if row.get("tracker_backend") != "none" or row.get("track_id") != "":
                raise StageRAnalysisError(
                    f"Unexpected tracker evidence in {detector} at {key}"
                )
            if row.get("temporal_enabled") != "1":
                raise StageRAnalysisError(
                    f"Frozen E4 flag changed in {detector} at {key}"
                )
            prediction[key] = row
        if set(prediction) != set(truth):
            raise StageRAnalysisError(
                f"{detector} occupancy keys differ from frozen truth"
            )
        predictions[detector] = prediction
        occupancy_fields[detector] = fields

    sequences = sorted({key[0] for key in truth})
    input_audit["schema_and_key_validation"] = {
        "truth_fields": truth_fields,
        "manifest_fields": manifest_fields,
        "occupancy_fields": occupancy_fields,
        "truth_encoding": {
            "canonical_state_1": "occupied",
            "canonical_state_0": "vacant",
            "source_value_0": "occupied",
            "source_value_1": "vacant",
            "all_rows_cross_checked": True,
        },
        "truth_rows": len(truth),
        "selected_frames": len(manifest),
        "sequences": len(sequences),
        "sequence_ids": sequences,
        "slots": len(expected_slots),
        "slot_ids": expected_slots,
        "key_sets_identical": True,
        "source_timestamps_available": False,
        "source_fps_available": False,
        "tracker_backend": "none",
    }
    return {
        "project_root": project_root,
        "output_root": output_root,
        "truth": truth,
        "manifest": manifest,
        "predictions": predictions,
        "input_audit": input_audit,
    }


def _scopes(
    truth: Mapping[tuple[str, int, str], Mapping[str, str]],
) -> list[tuple[str, str, list[tuple[str, int, str]]]]:
    keys = sorted(truth)
    rows: list[tuple[str, str, list[tuple[str, int, str]]]] = [
        ("overall", "all", keys)
    ]
    for sequence_id in sorted({key[0] for key in keys}):
        rows.append(
            (
                "sequence",
                sequence_id,
                [key for key in keys if key[0] == sequence_id],
            )
        )
    for slot_id in sorted({key[2] for key in keys}):
        rows.append(
            (
                "slot",
                slot_id,
                [key for key in keys if key[2] == slot_id],
            )
        )
    return rows


def component_comparison(
    truth: Mapping[tuple[str, int, str], Mapping[str, str]],
    predictions: Mapping[
        str,
        Mapping[tuple[str, int, str], Mapping[str, str]],
    ],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    scopes = _scopes(truth)
    for detector, method_id in METHODS.items():
        for component, definition in COMPONENTS.items():
            prediction_field = str(definition["prediction_field"])
            for scope_type, scope_id, keys in scopes:
                records = [
                    {
                        "video_id": key[0],
                        "frame_index": key[1],
                        "slot_id": key[2],
                        "truth": int(truth[key]["state"]),
                        "prediction": int(
                            predictions[detector][key][prediction_field]
                        ),
                    }
                    for key in keys
                ]
                metrics = evaluate_records(records)
                results.append(
                    {
                        "detector": detector,
                        "method_id": method_id,
                        "component": component,
                        "pipeline": definition["pipeline"],
                        "prediction_field": prediction_field,
                        "scope_type": scope_type,
                        "scope_id": scope_id,
                        **metrics,
                    }
                )
    return results


def component_deltas(
    comparison: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    indexed = {
        (
            str(row["detector"]),
            str(row["component"]),
            str(row["scope_type"]),
            str(row["scope_id"]),
        ): row
        for row in comparison
    }
    results: list[dict[str, Any]] = []
    scope_keys = sorted(
        {
            (
                str(row["detector"]),
                str(row["scope_type"]),
                str(row["scope_id"]),
            )
            for row in comparison
        }
    )
    for detector, scope_type, scope_id in scope_keys:
        for source, target in DELTA_PAIRS:
            before = indexed[(detector, source, scope_type, scope_id)]
            after = indexed[(detector, target, scope_type, scope_id)]
            row: dict[str, Any] = {
                "detector": detector,
                "scope_type": scope_type,
                "scope_id": scope_id,
                "transition": f"{source}_to_{target}",
                "definition": "signed_absolute_change_target_minus_source",
            }
            for metric in METRIC_FIELDS:
                row[f"delta_{metric}"] = (
                    float(after[metric]) - float(before[metric])
                )
            results.append(row)
    return results


def _outcome(truth: int, prediction: int) -> str:
    if truth == 1:
        return "TP" if prediction == 1 else "FN"
    return "FP" if prediction == 1 else "TN"


def error_transitions(
    truth: Mapping[tuple[str, int, str], Mapping[str, str]],
    predictions: Mapping[
        str,
        Mapping[tuple[str, int, str], Mapping[str, str]],
    ],
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    keys = sorted(truth)
    for detector in METHODS:
        for source, target in DELTA_PAIRS:
            source_field = str(COMPONENTS[source]["prediction_field"])
            target_field = str(COMPONENTS[target]["prediction_field"])
            counts: Counter[str] = Counter()
            for key in keys:
                expected = int(truth[key]["state"])
                before = int(predictions[detector][key][source_field])
                after = int(predictions[detector][key][target_field])
                counts[f"{_outcome(expected, before)}_to_{_outcome(expected, after)}"] += 1
            for transition, count in sorted(counts.items()):
                results.append(
                    {
                        "detector": detector,
                        "component_transition": f"{source}_to_{target}",
                        "error_transition": transition,
                        "count": count,
                    }
                )
    return results


def _continuous_segments(
    frame_indices: Sequence[int],
) -> list[list[int]]:
    if not frame_indices:
        return []
    segments: list[list[int]] = []
    current = [int(frame_indices[0])]
    for frame_index in frame_indices[1:]:
        frame_index = int(frame_index)
        if frame_index - current[-1] == 1:
            current.append(frame_index)
        else:
            if len(current) >= 2:
                segments.append(current)
            current = [frame_index]
    if len(current) >= 2:
        segments.append(current)
    return segments


def temporal_validity_audit(
    manifest: Mapping[tuple[str, int], Mapping[str, str]],
    predictions: Mapping[
        str,
        Mapping[tuple[str, int, str], Mapping[str, str]],
    ],
) -> dict[str, Any]:
    frames_by_sequence: dict[str, list[int]] = defaultdict(list)
    for sequence_id, frame_index in manifest:
        frames_by_sequence[sequence_id].append(frame_index)
    for frame_indices in frames_by_sequence.values():
        frame_indices.sort()

    all_gaps: list[int] = []
    sequence_rows: list[dict[str, Any]] = []
    segments: list[dict[str, Any]] = []
    for sequence_id in sorted(frames_by_sequence):
        frames = frames_by_sequence[sequence_id]
        gaps = [
            current - previous
            for previous, current in zip(frames, frames[1:])
        ]
        all_gaps.extend(gaps)
        continuous = _continuous_segments(frames)
        for index, segment in enumerate(continuous, start=1):
            segments.append(
                {
                    "sequence_id": sequence_id,
                    "segment_id": f"{sequence_id}_continuous_{index:02d}",
                    "start_source_frame_index": segment[0],
                    "end_source_frame_index": segment[-1],
                    "length_frames": len(segment),
                    "within_segment_gap_1_boundaries": len(segment) - 1,
                }
            )
        sequence_rows.append(
            {
                "sequence_id": sequence_id,
                "selected_frames": len(frames),
                "adjacent_selected_boundaries": len(gaps),
                "gap_1_boundaries": sum(gap == 1 for gap in gaps),
                "gap_gt_1_boundaries": sum(gap > 1 for gap in gaps),
                "maximum_gap_frames": max(gaps) if gaps else None,
                "continuous_segment_count": len(continuous),
                "continuous_segment_lengths_frames": [
                    len(segment) for segment in continuous
                ],
            }
        )

    state_audit: dict[str, dict[str, Any]] = {}
    slots = sorted({key[2] for rows in predictions.values() for key in rows})
    for detector, rows in predictions.items():
        raw_changes = 0
        state_changes = 0
        state_changes_at_gap_gt_1 = 0
        state_changes_at_gap_1 = 0
        e4_overrides = 0
        e4_overrides_at_first_frame_after_gap_gt_1 = 0
        for sequence_id, frames in sorted(frames_by_sequence.items()):
            for slot_id in slots:
                for previous, current in zip(frames, frames[1:]):
                    previous_row = rows[(sequence_id, previous, slot_id)]
                    current_row = rows[(sequence_id, current, slot_id)]
                    gap = current - previous
                    if int(current_row["raw_state"]) != int(
                        previous_row["raw_state"]
                    ):
                        raw_changes += 1
                    if int(current_row["state"]) != int(previous_row["state"]):
                        state_changes += 1
                        if gap == 1:
                            state_changes_at_gap_1 += 1
                        elif gap > 1:
                            state_changes_at_gap_gt_1 += 1
                for index, frame_index in enumerate(frames):
                    row = rows[(sequence_id, frame_index, slot_id)]
                    if int(row["state"]) != int(row["raw_state"]):
                        e4_overrides += 1
                        if (
                            index > 0
                            and frame_index - frames[index - 1] > 1
                        ):
                            e4_overrides_at_first_frame_after_gap_gt_1 += 1
        state_audit[detector] = {
            "observed_raw_state_changes": raw_changes,
            "observed_E4_state_changes": state_changes,
            "E4_state_changes_on_gap_1_boundaries": state_changes_at_gap_1,
            "E4_state_changes_on_gap_gt_1_boundaries": (
                state_changes_at_gap_gt_1
            ),
            "fraction_E4_state_changes_on_gap_gt_1_boundaries": _safe_divide(
                state_changes_at_gap_gt_1,
                state_changes,
            ),
            "rows_where_state_differs_from_raw_state": e4_overrides,
            "such_rows_at_first_frame_after_gap_gt_1": (
                e4_overrides_at_first_frame_after_gap_gt_1
            ),
        }

    distribution = {
        str(gap): count for gap, count in sorted(Counter(all_gaps).items())
    }
    return {
        "units": "source_frames",
        "seconds_conversion_performed": False,
        "source_timestamps_available": False,
        "reliable_source_fps_available": False,
        "selected_frames": sum(len(value) for value in frames_by_sequence.values()),
        "sequences": len(frames_by_sequence),
        "adjacent_selected_boundaries": len(all_gaps),
        "gap_distribution": distribution,
        "gap_1_boundaries": sum(gap == 1 for gap in all_gaps),
        "gap_gt_1_boundaries": sum(gap > 1 for gap in all_gaps),
        "maximum_gap_frames": max(all_gaps) if all_gaps else None,
        "continuous_segment_definition": (
            "maximal within-sequence selected-frame runs with every source-frame "
            "gap equal to 1 and at least 2 frames"
        ),
        "continuous_segment_count": len(segments),
        "continuous_segment_lengths_frames": [
            row["length_frames"] for row in segments
        ],
        "continuous_segment_selected_frames": sum(
            int(row["length_frames"]) for row in segments
        ),
        "continuous_segments": segments,
        "per_sequence": sequence_rows,
        "detector_state_audit": state_audit,
        "interpretation_boundary": (
            "Frozen E4 output was not regenerated. Sparse selected frames can "
            "carry prior E4 state across unobserved source-frame intervals; "
            "this audit is post-hoc and cannot establish continuous-video or "
            "TrackTrack performance."
        ),
    }


def _overall_rows(
    comparison: Sequence[Mapping[str, Any]],
) -> dict[tuple[str, str], Mapping[str, Any]]:
    return {
        (str(row["detector"]), str(row["component"])): row
        for row in comparison
        if row["scope_type"] == "overall"
    }


def final_system_evidence(
    comparison: Sequence[Mapping[str, Any]],
    temporal_audit: Mapping[str, Any],
) -> list[dict[str, str]]:
    overall = _overall_rows(comparison)
    d1_r1 = overall[("D1", "R1")]
    d1_r2 = overall[("D1", "R2")]
    d1_ll_r2 = overall[("D1-LL", "R2")]
    return [
        {
            "decision_area": "default_detector",
            "evidence": (
                f"D1 R2 Macro F1={d1_r2['macro_f1']:.6f}; D1-LL "
                f"R2 Macro F1={d1_ll_r2['macro_f1']:.6f}."
            ),
            "decision": "Keep D1 as the default detector.",
            "claim_boundary": (
                "D1-LL remains a frozen negative low-light fine-tuning experiment."
            ),
        },
        {
            "decision_area": "fusion_F2",
            "evidence": (
                f"D1 B1-to-F2 Macro F1 change="
                f"{float(d1_r1['macro_f1']) - float(overall[('D1', 'R0')]['macro_f1']):+.6f}."
            ),
            "decision": "Use B1 + F2 in the default occupancy path.",
            "claim_boundary": "Post-hoc attribution on frozen Stage Q-v2 outputs.",
        },
        {
            "decision_area": "temporal_E4",
            "evidence": (
                f"D1 F2-to-E4 occupied-recall change="
                f"{float(d1_r2['occupied_recall']) - float(d1_r1['occupied_recall']):+.6f}; "
                f"Macro-F1 change="
                f"{float(d1_r2['macro_f1']) - float(d1_r1['macro_f1']):+.6f}."
            ),
            "decision": "Make E4 conditional on genuinely continuous video.",
            "claim_boundary": "E4 needs separate continuous-video calibration.",
        },
        {
            "decision_area": "temporal_validity",
            "evidence": (
                f"{temporal_audit['gap_gt_1_boundaries']} selected-frame "
                f"boundaries have source-frame gap > 1; maximum gap="
                f"{temporal_audit['maximum_gap_frames']} frames."
            ),
            "decision": "Do not use Stage Q-v2 as continuous-video proof.",
            "claim_boundary": "No timestamps/FPS or seconds-level latency claimed.",
        },
        {
            "decision_area": "TrackTrack",
            "evidence": "Stage Q-v2 tracker backend is none; TrackTrack was not run.",
            "decision": "Retain TrackTrack as an independent optional MOT module.",
            "claim_boundary": (
                "No claim that TrackTrack improves slot-level occupancy."
            ),
        },
    ]


def build_stage_r_analysis(project_root: Path) -> dict[str, Any]:
    frozen = validate_frozen_inputs(project_root)
    comparison = component_comparison(
        frozen["truth"],
        frozen["predictions"],
    )
    deltas = component_deltas(comparison)
    transitions = error_transitions(
        frozen["truth"],
        frozen["predictions"],
    )
    temporal = temporal_validity_audit(
        frozen["manifest"],
        frozen["predictions"],
    )
    truth_values = [int(row["state"]) for row in frozen["truth"].values()]
    occupied = sum(truth_values)
    vacant = len(truth_values) - occupied
    evidence = final_system_evidence(comparison, temporal)
    return {
        "schema_version": 1,
        "protocol_id": STAGE_R_PROTOCOL_ID,
        "analysis_date": "2026-07-29",
        "status": "POSTHOC_ANALYSIS_OF_FROZEN_STAGE_Q_V2_OUTPUTS",
        "untouched_test_claim": False,
        "model_inference_run": False,
        "training_or_tuning_run": False,
        "parameters_changed": False,
        "frozen_outputs_modified": False,
        "metric_scope": "slot_level_occupancy",
        "truth_class_distribution": {
            "samples": len(truth_values),
            "occupied": occupied,
            "vacant": vacant,
            "occupied_prevalence": occupied / len(truth_values),
            "vacant_prevalence": vacant / len(truth_values),
            "class_imbalance_requires_class_aware_metrics": True,
        },
        "component_definitions": COMPONENTS,
        "input_audit": frozen["input_audit"],
        "comparison": comparison,
        "deltas": deltas,
        "error_transitions": transitions,
        "temporal_validity_audit": temporal,
        "final_system_evidence": evidence,
        "conclusion": {
            "default_pipeline": "D1 -> B1 -> F2 -> Occupancy Output",
            "conditional_module": (
                "F2 -> E4 only for genuinely continuous video after calibration"
            ),
            "negative_experiment": "D1-LL replaces D1",
            "independent_experimental_module": (
                "TrackTrack after detection for MOT research; not default occupancy"
            ),
            "TrackTrack_slot_occupancy_improvement_claimed": False,
        },
    }


def _write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fieldnames: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _temporal_csv_rows(temporal: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for gap, count in temporal["gap_distribution"].items():
        rows.append(
            {
                "record_type": "gap_distribution",
                "sequence_id": "",
                "detector": "",
                "metric": f"source_frame_gap_{gap}",
                "value": count,
                "details": "within-sequence adjacent selected frames",
            }
        )
    for sequence in temporal["per_sequence"]:
        for metric in (
            "selected_frames",
            "adjacent_selected_boundaries",
            "gap_1_boundaries",
            "gap_gt_1_boundaries",
            "maximum_gap_frames",
            "continuous_segment_count",
        ):
            rows.append(
                {
                    "record_type": "per_sequence",
                    "sequence_id": sequence["sequence_id"],
                    "detector": "",
                    "metric": metric,
                    "value": sequence[metric],
                    "details": "",
                }
            )
    for segment in temporal["continuous_segments"]:
        rows.append(
            {
                "record_type": "continuous_segment",
                "sequence_id": segment["sequence_id"],
                "detector": "",
                "metric": segment["segment_id"],
                "value": segment["length_frames"],
                "details": (
                    f"source_frame_index {segment['start_source_frame_index']}"
                    f"..{segment['end_source_frame_index']}"
                ),
            }
        )
    for detector, audit in temporal["detector_state_audit"].items():
        for metric, value in audit.items():
            rows.append(
                {
                    "record_type": "detector_state_audit",
                    "sequence_id": "",
                    "detector": detector,
                    "metric": metric,
                    "value": value,
                    "details": "frozen E4 output; no regeneration",
                }
            )
    return rows


def _system_flow_svg() -> str:
    return """<svg xmlns="http://www.w3.org/2000/svg" width="1400" height="760" viewBox="0 0 1400 760">
<defs>
  <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="8" markerHeight="8" orient="auto-start-reverse">
    <path d="M 0 0 L 10 5 L 0 10 z" fill="#334155"/>
  </marker>
  <style>
    text { font-family: "Segoe UI", Arial, sans-serif; fill: #172033; }
    .title { font-size: 30px; font-weight: 700; }
    .section { font-size: 20px; font-weight: 700; }
    .node-title { font-size: 22px; font-weight: 700; }
    .node-sub { font-size: 16px; }
    .arrow { stroke: #334155; stroke-width: 3; fill: none; marker-end: url(#arrow); }
    .dash { stroke-dasharray: 10 8; }
  </style>
</defs>
<rect width="1400" height="760" fill="#f8fafc"/>
<text x="60" y="58" class="title">Automated Parking Lot Occupancy and Tracking System — Final Evidence Architecture</text>

<rect x="45" y="90" width="1310" height="250" rx="18" fill="#ecfdf5" stroke="#16a34a" stroke-width="3"/>
<text x="70" y="125" class="section" fill="#166534">DEFAULT OCCUPANCY PATH</text>
<g>
  <rect x="90" y="165" width="220" height="110" rx="14" fill="#ffffff" stroke="#16a34a" stroke-width="3"/>
  <text x="200" y="208" text-anchor="middle" class="node-title">D1</text>
  <text x="200" y="238" text-anchor="middle" class="node-sub">YOLO detector</text>
  <rect x="400" y="165" width="220" height="110" rx="14" fill="#ffffff" stroke="#16a34a" stroke-width="3"/>
  <text x="510" y="208" text-anchor="middle" class="node-title">B1</text>
  <text x="510" y="238" text-anchor="middle" class="node-sub">polygon + 1:1 mapping</text>
  <rect x="710" y="165" width="220" height="110" rx="14" fill="#ffffff" stroke="#16a34a" stroke-width="3"/>
  <text x="820" y="208" text-anchor="middle" class="node-title">F2</text>
  <text x="820" y="238" text-anchor="middle" class="node-sub">E1b gated fusion</text>
  <rect x="1020" y="165" width="270" height="110" rx="14" fill="#dcfce7" stroke="#16a34a" stroke-width="3"/>
  <text x="1155" y="208" text-anchor="middle" class="node-title">Occupancy Output</text>
  <text x="1155" y="238" text-anchor="middle" class="node-sub">slot-level occupied / vacant</text>
  <path d="M310 220 H395" class="arrow"/>
  <path d="M620 220 H705" class="arrow"/>
  <path d="M930 220 H1015" class="arrow"/>
</g>

<rect x="45" y="370" width="820" height="310" rx="18" fill="#eff6ff" stroke="#2563eb" stroke-width="3"/>
<text x="70" y="405" class="section" fill="#1d4ed8">CONDITIONAL — GENUINELY CONTINUOUS VIDEO ONLY</text>
<rect x="105" y="455" width="220" height="110" rx="14" fill="#ffffff" stroke="#2563eb" stroke-width="3"/>
<text x="215" y="498" text-anchor="middle" class="node-title">F2</text>
<text x="215" y="528" text-anchor="middle" class="node-sub">raw occupancy state</text>
<rect x="420" y="455" width="250" height="110" rx="14" fill="#dbeafe" stroke="#2563eb" stroke-width="3"/>
<text x="545" y="498" text-anchor="middle" class="node-title">E4 (optional)</text>
<text x="545" y="528" text-anchor="middle" class="node-sub">temporal stabilization</text>
<path d="M325 510 H415" class="arrow dash"/>
<text x="105" y="615" class="node-sub">Requires separate calibration on continuous video.</text>
<text x="105" y="642" class="node-sub">E4 is not TrackTrack and does not maintain vehicle IDs.</text>

<rect x="900" y="370" width="455" height="310" rx="18" fill="#fff7ed" stroke="#ea580c" stroke-width="3"/>
<text x="925" y="405" class="section" fill="#c2410c">EXPERIMENTAL BRANCHES</text>
<rect x="950" y="445" width="170" height="90" rx="14" fill="#ffffff" stroke="#ea580c" stroke-width="3"/>
<text x="1035" y="482" text-anchor="middle" class="node-title">D1-LL</text>
<text x="1035" y="510" text-anchor="middle" class="node-sub">low-light fine-tune</text>
<rect x="1150" y="445" width="160" height="90" rx="14" fill="#ffedd5" stroke="#ea580c" stroke-width="3"/>
<text x="1230" y="482" text-anchor="middle" class="node-title">Replace D1</text>
<text x="1230" y="510" text-anchor="middle" class="node-sub">not retained</text>
<path d="M1120 490 H1145" class="arrow dash"/>
<rect x="950" y="575" width="360" height="72" rx="14" fill="#ffffff" stroke="#ea580c" stroke-width="3"/>
<text x="1130" y="606" text-anchor="middle" class="node-title">Detection → TrackTrack</text>
<text x="1130" y="632" text-anchor="middle" class="node-sub">independent optional MOT research</text>
</svg>
"""


def write_stage_r_outputs(
    analysis: Mapping[str, Any],
    output_dir: Path,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    comparison_path = output_dir / "STAGE_R_COMPONENT_COMPARISON.csv"
    json_path = output_dir / "STAGE_R_COMPONENT_COMPARISON.json"
    delta_path = output_dir / "STAGE_R_COMPONENT_DELTAS.csv"
    transitions_path = output_dir / "STAGE_R_ERROR_TRANSITIONS.csv"
    temporal_path = output_dir / "STAGE_R_TEMPORAL_VALIDITY_AUDIT.csv"
    evidence_path = output_dir / "STAGE_R_FINAL_SYSTEM_EVIDENCE.csv"
    flow_path = output_dir / "STAGE_R_SYSTEM_FLOW.svg"

    _write_csv(
        comparison_path,
        analysis["comparison"],
        COMPARISON_FIELDS,
    )
    delta_fields = (
        "detector",
        "scope_type",
        "scope_id",
        "transition",
        "definition",
        *(f"delta_{metric}" for metric in METRIC_FIELDS),
    )
    _write_csv(delta_path, analysis["deltas"], delta_fields)
    _write_csv(
        transitions_path,
        analysis["error_transitions"],
        (
            "detector",
            "component_transition",
            "error_transition",
            "count",
        ),
    )
    _write_csv(
        temporal_path,
        _temporal_csv_rows(analysis["temporal_validity_audit"]),
        (
            "record_type",
            "sequence_id",
            "detector",
            "metric",
            "value",
            "details",
        ),
    )
    _write_csv(
        evidence_path,
        analysis["final_system_evidence"],
        ("decision_area", "evidence", "decision", "claim_boundary"),
    )
    json_path.write_text(
        json.dumps(analysis, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    flow_path.write_text(_system_flow_svg(), encoding="utf-8")
    return [
        comparison_path,
        json_path,
        delta_path,
        transitions_path,
        temporal_path,
        evidence_path,
        flow_path,
    ]


def artifact_record(
    path: Path,
    *,
    project_root: Path,
    role: str,
) -> dict[str, Any]:
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path, project_root),
        "role": role,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_stage_r_registry(
    registry_path: Path,
    *,
    project_root: Path,
    artifacts: Iterable[tuple[Path, str]],
) -> dict[str, Any]:
    records = [
        artifact_record(path, project_root=project_root, role=role)
        for path, role in artifacts
    ]
    payload = {
        "schema_version": 1,
        "protocol_id": STAGE_R_PROTOCOL_ID,
        "registry_id": "STAGE-R-ARTIFACT-REGISTRY-20260729-01",
        "status": "POSTHOC_ANALYSIS_COMPLETE_AND_HASH_VERIFIED",
        "created_on": "2026-07-29",
        "path_base": "implementation_root",
        "artifact_count": len(records),
        "registry_self_hash_included": False,
        "frozen_stage_q_v2_outputs_modified": False,
        "model_inference_run": False,
        "artifacts": records,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )
    result = verify_stage_r_registry(
        registry_path,
        project_root=project_root,
    )
    if not result["verified"]:
        raise StageRAnalysisError(f"Stage R registry verification failed: {result}")
    return result


def verify_stage_r_registry(
    registry_path: Path,
    *,
    project_root: Path,
) -> dict[str, Any]:
    payload = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("protocol_id") != STAGE_R_PROTOCOL_ID:
        errors.append("protocol_id")
    records = payload.get("artifacts", [])
    if len(records) != int(payload.get("artifact_count", -1)):
        errors.append("artifact_count")
    for record in records:
        relative = str(record["path"])
        path = (project_root / relative).resolve()
        if not path.is_file():
            errors.append(f"missing:{relative}")
        elif path.stat().st_size != int(record["bytes"]):
            errors.append(f"bytes:{relative}")
        elif sha256_file(path) != str(record["sha256"]):
            errors.append(f"sha256:{relative}")
    return {
        "protocol_id": STAGE_R_PROTOCOL_ID,
        "artifact_count": len(records),
        "verified": not errors,
        "errors": errors,
        "registry_path": str(registry_path.resolve()),
        "registry_bytes": registry_path.stat().st_size,
        "registry_sha256": sha256_file(registry_path),
    }
