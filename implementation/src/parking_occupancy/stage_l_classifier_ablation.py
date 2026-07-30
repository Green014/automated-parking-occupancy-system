from __future__ import annotations

import csv
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

from .detector_comparison import sha256_file
from .evaluate import binary_metrics
from .image_io import read_image
from .stage_j_occupancy import _load_annotations, _slots_and_truth
from .stage_j_posthoc_analysis import paired_bootstrap_mean_difference
from .stage_l_integrated import load_stage_l_protocol


def run_classifier_only_ablation(
    *,
    config_path: Path,
    annotations_path: Path,
    source_root: Path,
    p3_predictions_path: Path,
    classifier_checkpoint: Path,
    output_root: Path,
    device: str,
) -> dict[str, Any]:
    """Evaluate frozen E1b alone by filling P3's uncomputed positive slots."""

    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite classifier ablation: {output_root}"
        )
    protocol = load_stage_l_protocol(config_path)
    e1b = protocol["models"]["E1b"]
    if sha256_file(classifier_checkpoint) != str(e1b["sha256"]):
        raise ValueError("E1b checkpoint SHA-256 mismatch")
    records = _load_annotations(annotations_path)
    with p3_predictions_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        p3_rows = list(csv.DictReader(handle))
    rows_by_sample: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in p3_rows:
        rows_by_sample[row["sample_id"]].append(row)
    if set(rows_by_sample) != {
        str(record["sample_id"]) for record in records
    }:
        raise ValueError("Annotation/P3 sample membership mismatch")

    from literature_core.classifier import MobileNetSlotClassifier
    from literature_core.patches import extract_slot_patch

    classifier = MobileNetSlotClassifier(
        classifier_checkpoint,
        device=device,
    )
    threshold = float(e1b["occupied_threshold"])
    output_root.mkdir(parents=True)
    output_rows = []
    inference_times = []
    for record in records:
        sample_id = str(record["sample_id"])
        image = read_image(source_root / record["local_path"])
        height, width = image.shape[:2]
        slots, _truth = _slots_and_truth(record, width, height)
        slot_by_id = {slot.slot_id: slot for slot in slots}
        missing_rows = [
            row
            for row in rows_by_sample[sample_id]
            if str(row["classifier_probability"]).strip() == ""
        ]
        start = time.perf_counter()
        missing_probabilities = classifier.predict_patches(
            [
                extract_slot_patch(
                    image,
                    slot_by_id[row["slot_id"]].points,
                    output_size=tuple(e1b["patch_size"]),
                    perspective_warp=bool(e1b["perspective_warp"]),
                )
                for row in missing_rows
            ]
        )
        inference_times.append((time.perf_counter() - start) * 1000.0)
        filled = {
            row["slot_id"]: probability
            for row, probability in zip(
                missing_rows,
                missing_probabilities,
                strict=True,
            )
        }
        for row in rows_by_sample[sample_id]:
            probability_text = str(row["classifier_probability"]).strip()
            probability = (
                float(probability_text)
                if probability_text
                else float(filled[row["slot_id"]])
            )
            output_rows.append(
                {
                    **row,
                    "classifier_probability": f"{probability:.8f}",
                    "e1b_prediction": int(probability >= threshold),
                }
            )

    predictions_path = output_root / "predictions.csv"
    with predictions_path.open(
        "x",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)

    def metrics(field: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
        return binary_metrics(
            [int(row["truth"]) for row in rows],
            [int(row[field]) for row in rows],
        )

    overall = {
        "P1_D1_B1": metrics("p1_prediction", output_rows),
        "E1b_classifier_only": metrics("e1b_prediction", output_rows),
        "P3_static_gate": metrics("p3_prediction", output_rows),
    }
    camera_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    sample_rows: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in output_rows:
        camera_rows[row["camera"]].append(row)
        sample_rows[row["sample_id"]].append(row)
    by_camera = {
        camera: {
            "P1_D1_B1": metrics("p1_prediction", rows),
            "E1b_classifier_only": metrics("e1b_prediction", rows),
            "P3_static_gate": metrics("p3_prediction", rows),
        }
        for camera, rows in sorted(camera_rows.items())
    }
    camera_macro = {
        method: statistics.fmean(
            camera_metrics[method]["macro_f1"]
            for camera_metrics in by_camera.values()
        )
        for method in overall
    }
    differences = {}
    outcomes = {"win": 0, "tie": 0, "loss": 0}
    for sample_id, rows in sample_rows.items():
        e1b_score = metrics("e1b_prediction", rows)["macro_f1"]
        p3_score = metrics("p3_prediction", rows)["macro_f1"]
        difference = p3_score - e1b_score
        differences[sample_id] = difference
        outcome = (
            "win"
            if difference > 1e-12
            else "loss"
            if difference < -1e-12
            else "tie"
        )
        outcomes[outcome] += 1
    report = {
        "schema_version": 1,
        "analysis": "fixed_E1b_classifier_only_ablation",
        "threshold": threshold,
        "threshold_source": e1b["threshold_source"],
        "images": len(records),
        "slot_rows": len(output_rows),
        "overall": overall,
        "by_camera": by_camera,
        "camera_macro_f1": camera_macro,
        "P3_minus_E1b_paired": {
            "outcomes": outcomes,
            "bootstrap": paired_bootstrap_mean_difference(
                differences,
                seed=20260728,
                resamples=2000,
                confidence_level=0.95,
            ),
        },
        "runtime": {
            "additional_classifier_slots": sum(
                str(row["classifier_probability"]).strip() == ""
                for row in p3_rows
            ),
            "mean_incremental_ms_per_image": statistics.fmean(
                inference_times
            ),
            "classifier": classifier.metadata(),
        },
        "parameter_selection_performed": False,
        "claim_boundary": (
            "Retrospective branch ablation on previously consumed Stage K "
            "images; no threshold or method selection is permitted."
        ),
    }
    (output_root / "metrics.json").write_text(
        json.dumps(report, indent=2) + "\n",
        encoding="utf-8",
    )
    return report
