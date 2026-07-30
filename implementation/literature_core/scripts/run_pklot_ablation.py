from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.classifier import MobileNetSlotClassifier  # noqa: E402
from literature_core.config import load_yaml  # noqa: E402
from literature_core.data import (  # noqa: E402
    SlotSample,
    load_pklot_slot_samples,
    read_image,
)
from literature_core.detector import (  # noqa: E402
    ClosedSetYOLODetector,
    YOLOWorldDetector,
)
from literature_core.mapping import map_detections_to_slots  # noqa: E402
from literature_core.metrics import (  # noqa: E402
    evaluate_probabilities,
    select_threshold,
)
from literature_core.models import ParkingSlot  # noqa: E402
from literature_core.patches import extract_slot_patch  # noqa: E402


def grouped_samples(
    samples: list[SlotSample],
) -> dict[tuple[str, str], list[SlotSample]]:
    result: dict[tuple[str, str], list[SlotSample]] = defaultdict(list)
    for sample in samples:
        if sample.split in {"development", "test"}:
            result[(sample.split, sample.sample_id)].append(sample)
    return result


def probability_metrics(
    records: list[dict[str, Any]],
    key: str,
    threshold: float,
) -> dict[str, Any]:
    return evaluate_probabilities(
        [record["truth"] for record in records],
        [record[key] for record in records],
        threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Frozen PKLot E0-E3 ablation")
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--classifier-checkpoint", required=True)
    parser.add_argument("--world-weights", required=True)
    parser.add_argument("--baseline-weights", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = load_yaml(args.config)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    samples = load_pklot_slot_samples(
        args.annotations,
        args.project_root,
        args.split_config,
    )
    groups = grouped_samples(samples)
    classifier = MobileNetSlotClassifier(
        args.classifier_checkpoint,
        device=args.device,
    )
    world_config = config["yolo_world"]
    world = YOLOWorldDetector(
        args.world_weights,
        prompts=world_config["prompts"],
        confidence=float(world_config["confidence"]),
        image_size=int(world_config["image_size"]),
        device=args.device,
    )
    baseline_config = config["baseline_yolo"]
    baseline = ClosedSetYOLODetector(
        args.baseline_weights,
        class_ids=baseline_config["class_ids"],
        confidence=float(baseline_config["confidence"]),
        image_size=int(baseline_config["image_size"]),
        device=args.device,
    )
    mapping_config = config["mapping"]
    records: list[dict[str, Any]] = []
    timings = defaultdict(float)

    with (output_dir / "raw_detections.jsonl").open(
        "w", encoding="utf-8"
    ) as raw_log:
        for (split, sample_id), image_samples in sorted(groups.items()):
            image = read_image(image_samples[0].image_path)
            slots = tuple(
                ParkingSlot(sample.slot_id, sample.points)
                for sample in image_samples
            )
            patches = [
                extract_slot_patch(image, sample.points)
                for sample in image_samples
            ]
            start = time.perf_counter()
            p_cls = classifier.predict_patches(patches)
            timings["classifier_s"] += time.perf_counter() - start

            start = time.perf_counter()
            world_detections = world.detect(image)
            timings["world_s"] += time.perf_counter() - start
            world_evidence = map_detections_to_slots(
                world_detections,
                slots,
                minimum_slot_coverage=float(
                    mapping_config["minimum_slot_coverage"]
                ),
                one_to_one=bool(mapping_config["one_to_one"]),
            )

            start = time.perf_counter()
            baseline_detections = baseline.detect(image)
            timings["baseline_s"] += time.perf_counter() - start
            baseline_evidence = map_detections_to_slots(
                baseline_detections,
                slots,
                minimum_slot_coverage=0.40,
                one_to_one=True,
            )
            raw_log.write(
                json.dumps(
                    {
                        "split": split,
                        "sample_id": sample_id,
                        "image_path": str(image_samples[0].image_path),
                        "yolo_world": [
                            asdict(detection) for detection in world_detections
                        ],
                        "baseline_yolov8": [
                            asdict(detection) for detection in baseline_detections
                        ],
                    }
                )
                + "\n"
            )
            for sample, cls_value, world_value, baseline_value in zip(
                image_samples,
                p_cls,
                world_evidence,
                baseline_evidence,
                strict=True,
            ):
                records.append(
                    {
                        "split": split,
                        "sample_id": sample.sample_id,
                        "source": sample.source,
                        "group_id": sample.group_id,
                        "slot_id": sample.slot_id,
                        "truth": sample.label,
                        "p_cls": cls_value,
                        "p_world": world_value.probability,
                        "p_baseline": baseline_value.probability,
                    }
                )

    development = [record for record in records if record["split"] == "development"]
    test = [record for record in records if record["split"] == "test"]
    development_sources = sorted(
        {str(record["source"]) for record in development}
    )
    test_sources = sorted({str(record["source"]) for record in test})
    if len(development_sources) != 1 or len(test_sources) != 1:
        raise ValueError(
            "Expected one camera per development/test split; "
            f"development={development_sources}, test={test_sources}"
        )
    development_camera = development_sources[0]
    test_camera = test_sources[0]
    e1_threshold, e1_rows = select_threshold(
        [record["truth"] for record in development],
        [record["p_cls"] for record in development],
    )
    e2_threshold, e2_rows = select_threshold(
        [record["truth"] for record in development],
        [record["p_world"] for record in development],
    )

    fusion_sensitivity: list[dict[str, Any]] = []
    selected_fusion: dict[str, Any] | None = None
    for classifier_weight_index in range(0, 21):
        classifier_weight = classifier_weight_index / 20
        detector_weight = 1.0 - classifier_weight
        probabilities = [
            classifier_weight * record["p_cls"]
            + detector_weight * record["p_world"]
            for record in development
        ]
        threshold, rows = select_threshold(
            [record["truth"] for record in development],
            probabilities,
        )
        selected_row = next(row for row in rows if row["threshold"] == threshold)
        candidate = {
            "classifier_weight": classifier_weight,
            "detector_weight": detector_weight,
            **selected_row,
        }
        fusion_sensitivity.append(candidate)
        if selected_fusion is None or (
            candidate["macro_f1"],
            -candidate["false_free_rate"],
            -abs(candidate["classifier_weight"] - 0.5),
        ) > (
            selected_fusion["macro_f1"],
            -selected_fusion["false_free_rate"],
            -abs(selected_fusion["classifier_weight"] - 0.5),
        ):
            selected_fusion = candidate
    assert selected_fusion is not None

    for record in records:
        record["p_fusion"] = (
            selected_fusion["classifier_weight"] * record["p_cls"]
            + selected_fusion["detector_weight"] * record["p_world"]
        )

    selected_parameters = {
        "selection_split": f"development/{development_camera}",
        "e0_baseline_rule": "coverage >= 0.40 and mapped evidence > 0",
        "e1_threshold": e1_threshold,
        "e2_threshold": e2_threshold,
        "e3_classifier_weight": selected_fusion["classifier_weight"],
        "e3_detector_weight": selected_fusion["detector_weight"],
        "e3_threshold": selected_fusion["threshold"],
        "test_used_for_selection": False,
    }
    metrics = {
        "split": {
            "development_samples": len(development),
            "test_samples": len(test),
            "development_camera": development_camera,
            "test_camera": test_camera,
            "pilot_holdout_note": (
                "Camera-disjoint within this fold; source images were "
                "previously used by the broader baseline study."
            ),
        },
        "selected_parameters": selected_parameters,
        "development": {
            "E0_yolov8_polygon": probability_metrics(
                development, "p_baseline", 1e-12
            ),
            "E1_mobilenet": probability_metrics(
                development, "p_cls", e1_threshold
            ),
            "E2_yolo_world_polygon": probability_metrics(
                development, "p_world", e2_threshold
            ),
            "E3_fusion": probability_metrics(
                development,
                "p_fusion",
                selected_fusion["threshold"],
            ),
        },
        "test": {
            "E0_yolov8_polygon": probability_metrics(
                test, "p_baseline", 1e-12
            ),
            "E1_mobilenet": probability_metrics(test, "p_cls", e1_threshold),
            "E2_yolo_world_polygon": probability_metrics(
                test, "p_world", e2_threshold
            ),
            "E3_fusion": probability_metrics(
                test,
                "p_fusion",
                selected_fusion["threshold"],
            ),
        },
        "timing_s": dict(timings),
        "classifier": classifier.metadata(),
        "yolo_world": world.metadata(),
        "baseline_yolo": baseline.metadata(),
        "temporal_experiments": (
            "Not run on PKLot because captures are not frame-contiguous."
        ),
    }

    with (output_dir / "branch_probabilities.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    with (output_dir / "development_sensitivity.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = list(fusion_sensitivity[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(fusion_sensitivity)
    with (output_dir / "selected_parameters.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(selected_parameters, handle, indent=2)
        handle.write("\n")
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    with (output_dir / "branch_threshold_sensitivity.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump({"E1": e1_rows, "E2": e2_rows}, handle, indent=2)
        handle.write("\n")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
