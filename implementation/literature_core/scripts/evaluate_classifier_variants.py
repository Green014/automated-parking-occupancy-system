"""Evaluate standard and paper-inspired MobileNetV3 variants consistently."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.classifier import MobileNetSlotClassifier  # noqa: E402
from literature_core.data import (  # noqa: E402
    SlotSample,
    load_pklot_slot_samples,
    read_image,
)
from literature_core.metrics import (  # noqa: E402
    evaluate_probabilities,
    select_threshold,
)
from literature_core.patches import extract_slot_patch  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--split-config", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        action="append",
        required=True,
        help="Variant specification LABEL=path/to/best.pt",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    return parser.parse_args()


def parse_checkpoints(specifications: list[str]) -> list[tuple[str, Path]]:
    checkpoints: list[tuple[str, Path]] = []
    for specification in specifications:
        if "=" not in specification:
            raise ValueError("--checkpoint must use LABEL=PATH")
        label, path = specification.split("=", 1)
        checkpoints.append((label, Path(path)))
    if len({label for label, _ in checkpoints}) != len(checkpoints):
        raise ValueError("checkpoint labels must be unique")
    return checkpoints


def extract_patches(
    samples: list[SlotSample],
) -> tuple[list[Any], float]:
    grouped: dict[Path, list[tuple[int, SlotSample]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[sample.image_path].append((index, sample))
    patches: list[Any] = [None] * len(samples)
    start = time.perf_counter()
    for image_path, image_samples in grouped.items():
        image = read_image(image_path)
        for index, sample in image_samples:
            patches[index] = extract_slot_patch(image, sample.points)
    return patches, time.perf_counter() - start


def synchronize(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints = parse_checkpoints(args.checkpoint)
    samples = load_pklot_slot_samples(
        args.annotations,
        args.project_root,
        args.split_config,
    )
    development_samples = [
        sample for sample in samples if sample.split == "development"
    ]
    evaluation_samples = [
        sample for sample in samples if sample.split == "test"
    ]
    if len(development_samples) < 100 or len(evaluation_samples) < 100:
        raise ValueError("variant benchmark requires at least 100 slot patches")
    development_patches, development_extraction_s = extract_patches(
        development_samples
    )
    evaluation_patches, evaluation_extraction_s = extract_patches(
        evaluation_samples
    )
    development_truth = [sample.label for sample in development_samples]
    evaluation_truth = [sample.label for sample in evaluation_samples]

    reports: list[dict[str, Any]] = []
    probability_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    for label, checkpoint in checkpoints:
        classifier = MobileNetSlotClassifier(
            checkpoint,
            device=args.device,
        )
        warmup = evaluation_patches[: min(64, len(evaluation_patches))]
        classifier.predict_patches(warmup)
        synchronize(classifier.device)

        start = time.perf_counter()
        development_probabilities = classifier.predict_patches(
            development_patches
        )
        synchronize(classifier.device)
        development_inference_s = time.perf_counter() - start
        threshold, sensitivity = select_threshold(
            development_truth,
            development_probabilities,
        )
        start = time.perf_counter()
        evaluation_probabilities = classifier.predict_patches(
            evaluation_patches
        )
        synchronize(classifier.device)
        evaluation_inference_s = time.perf_counter() - start
        selected_development = evaluate_probabilities(
            development_truth,
            development_probabilities,
            threshold,
        )
        evaluation = evaluate_probabilities(
            evaluation_truth,
            evaluation_probabilities,
            threshold,
        )
        payload = torch.load(
            checkpoint,
            map_location="cpu",
            weights_only=False,
        )
        parameter_count = sum(
            parameter.numel() for parameter in classifier.model.parameters()
        )
        trainable_parameter_count = int(
            payload.get(
                "trainable_parameter_count",
                sum(
                    parameter.numel()
                    for parameter in classifier.model.parameters()
                    if parameter.requires_grad
                ),
            )
        )
        training_summary_path = checkpoint.parent / "training_summary.json"
        training_summary = (
            json.loads(training_summary_path.read_text(encoding="utf-8"))
            if training_summary_path.is_file()
            else {}
        )
        reports.append(
            {
                "label": label,
                "implementation_label": payload.get("implementation_label"),
                "variant": payload.get("variant", "standard"),
                "paper_exact_reproduction": False,
                "components": payload.get("components", {}),
                "parameter_count": parameter_count,
                "trainable_parameter_count": trainable_parameter_count,
                "training_elapsed_s": training_summary.get("elapsed_s"),
                "best_epoch": payload.get("best_epoch"),
                "threshold_selected_on": "internal development camera",
                "threshold": threshold,
                "development": selected_development,
                "internal_evaluation": evaluation,
                "timing": {
                    "warmup_patches": len(warmup),
                    "timed_development_patches": len(development_patches),
                    "timed_evaluation_patches": len(evaluation_patches),
                    "development_inference_s": development_inference_s,
                    "evaluation_inference_s": evaluation_inference_s,
                    "evaluation_model_ms_per_patch": (
                        evaluation_inference_s
                        / len(evaluation_patches)
                        * 1000
                    ),
                    "evaluation_model_patches_per_s": (
                        len(evaluation_patches) / evaluation_inference_s
                    ),
                    "shared_development_extraction_s": development_extraction_s,
                    "shared_evaluation_extraction_s": evaluation_extraction_s,
                    "evaluation_extract_plus_model_ms_per_patch": (
                        (evaluation_extraction_s + evaluation_inference_s)
                        / len(evaluation_patches)
                        * 1000
                    ),
                },
            }
        )
        for split, split_samples, probabilities in (
            ("development", development_samples, development_probabilities),
            ("internal_evaluation", evaluation_samples, evaluation_probabilities),
        ):
            for sample, probability in zip(
                split_samples,
                probabilities,
                strict=True,
            ):
                probability_rows.append(
                    {
                        "variant": label,
                        "split": split,
                        "sample_id": sample.sample_id,
                        "source": sample.source,
                        "group_id": sample.group_id,
                        "slot_id": sample.slot_id,
                        "truth": sample.label,
                        "probability": probability,
                        "threshold": threshold,
                        "prediction": int(probability >= threshold),
                    }
                )
        for row in sensitivity:
            sensitivity_rows.append({"variant": label, **row})

    report = {
        "protocol": {
            "role": "internal_architecture_development_ablation",
            "split_config": str(args.split_config.resolve()),
            "external_holdout_used": False,
            "slot_level_random_split": False,
            "paper_exact_reproduction": False,
            "bsconv_implemented": False,
        },
        "development_samples": len(development_samples),
        "internal_evaluation_samples": len(evaluation_samples),
        "variants": reports,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "probabilities.csv", probability_rows)
    write_csv(args.output_dir / "threshold_sensitivity.csv", sensitivity_rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
