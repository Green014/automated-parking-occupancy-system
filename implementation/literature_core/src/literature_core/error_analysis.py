from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Any


METHODS = ("E0", "E1", "E2", "E3")


def classify_ablation_record(
    record: Mapping[str, Any],
    parameters: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply already-frozen E0-E3 decisions to one saved probability row."""

    truth = int(record["truth"])
    p_cls = float(record["p_cls"])
    p_world = float(record["p_world"])
    p_baseline = float(record["p_baseline"])
    classifier_weight = float(parameters["e3_classifier_weight"])
    detector_weight = float(parameters["e3_detector_weight"])
    p_fusion = classifier_weight * p_cls + detector_weight * p_world
    probabilities = {
        "E0": p_baseline,
        "E1": p_cls,
        "E2": p_world,
        "E3": p_fusion,
    }
    predictions = {
        "E0": int(p_baseline > 0.0),
        "E1": int(p_cls >= float(parameters["e1_threshold"])),
        "E2": int(p_world >= float(parameters["e2_threshold"])),
        "E3": int(p_fusion >= float(parameters["e3_threshold"])),
    }
    correct = {
        method: prediction == truth
        for method, prediction in predictions.items()
    }
    errors = tuple(method for method in METHODS if not correct[method])
    return {
        **dict(record),
        "truth": truth,
        "probabilities": probabilities,
        "predictions": predictions,
        "correct": correct,
        "errors": errors,
        "error_signature": ",".join(errors) if errors else "none",
        "branch_pattern": (
            "both_correct"
            if correct["E1"] and correct["E2"]
            else "classifier_only_correct"
            if correct["E1"]
            else "world_only_correct"
            if correct["E2"]
            else "both_wrong"
        ),
        "fusion_rescued_branch_error": (
            correct["E3"] and (not correct["E1"] or not correct["E2"])
        ),
        "fusion_harmed_both_correct": (
            not correct["E3"] and correct["E1"] and correct["E2"]
        ),
    }


def summarize_ablation_records(
    records: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows = list(records)
    if not rows:
        raise ValueError("At least one classified record is required")
    error_signatures = Counter(str(row["error_signature"]) for row in rows)
    branch_patterns = Counter(str(row["branch_pattern"]) for row in rows)
    return {
        "samples": len(rows),
        "errors": {
            method: sum(not bool(row["correct"][method]) for row in rows)
            for method in METHODS
        },
        "correct": {
            method: sum(bool(row["correct"][method]) for row in rows)
            for method in METHODS
        },
        "error_signatures": dict(sorted(error_signatures.items())),
        "branch_patterns": dict(sorted(branch_patterns.items())),
        "fusion_rescued_branch_error": sum(
            bool(row["fusion_rescued_branch_error"]) for row in rows
        ),
        "fusion_harmed_both_correct": sum(
            bool(row["fusion_harmed_both_correct"]) for row in rows
        ),
        "any_method_error": sum(row["error_signature"] != "none" for row in rows),
        "all_methods_wrong": sum(
            len(tuple(row["errors"])) == len(METHODS) for row in rows
        ),
    }
