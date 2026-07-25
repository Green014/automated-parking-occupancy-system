from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.classifier import (  # noqa: E402
    SlotPatchDataset,
    build_mobilenet_classifier,
    resolve_device,
)
from literature_core.config import load_yaml  # noqa: E402
from literature_core.data import load_pklot_slot_samples  # noqa: E402
from literature_core.metrics import select_threshold  # noqa: E402


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def predict(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[list[int], list[float]]:
    model.eval()
    labels: list[int] = []
    probabilities: list[float] = []
    for images, target in loader:
        logits = model(images.to(device))
        occupied = torch.softmax(logits, dim=1)[:, 1].cpu()
        labels.extend(int(value) for value in target)
        probabilities.extend(float(value) for value in occupied)
    return labels, probabilities


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the adapted MobileNetV3-Small slot classifier"
    )
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--no-pretrained", action="store_true")
    args = parser.parse_args()

    config = load_yaml(args.config)
    classifier_config = config["classifier"]
    patch_config = config["patch"]
    seed = int(config["seed"])
    set_seed(seed)
    device = resolve_device(args.device)
    epochs = int(args.epochs or classifier_config["epochs"])
    batch_size = int(args.batch_size or classifier_config["batch_size"])
    pretrained = bool(classifier_config["pretrained"]) and not args.no_pretrained
    freeze_backbone = bool(classifier_config["freeze_backbone"])
    patch_size = (int(patch_config["width"]), int(patch_config["height"]))

    samples = load_pklot_slot_samples(
        args.annotations,
        args.project_root,
        args.split_config,
    )
    train_samples = [sample for sample in samples if sample.split == "train"]
    development_samples = [
        sample for sample in samples if sample.split == "development"
    ]
    if not train_samples or not development_samples:
        raise ValueError("Train and development splits must both be non-empty")
    train_dataset = SlotPatchDataset(
        train_samples,
        output_size=patch_size,
        augment=True,
    )
    development_dataset = SlotPatchDataset(
        development_samples,
        output_size=patch_size,
        augment=False,
    )
    generator = torch.Generator().manual_seed(seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=int(classifier_config["num_workers"]),
        generator=generator,
        pin_memory=device.type == "cuda",
    )
    development_loader = DataLoader(
        development_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.type == "cuda",
    )

    model = build_mobilenet_classifier(
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
    ).to(device)
    label_counts = Counter(sample.label for sample in train_samples)
    class_weights = torch.tensor(
        [
            len(train_samples) / (2 * label_counts[0]),
            len(train_samples) / (2 * label_counts[1]),
        ],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(classifier_config["learning_rate"]),
        weight_decay=float(classifier_config["weight_decay"]),
    )
    amp_enabled = (
        bool(classifier_config["amp"])
        and device.type == "cuda"
        and torch.cuda.is_available()
    )
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "training_log.jsonl"
    best_macro_f1 = -1.0
    best_epoch = 0
    best_threshold = 0.5
    start_time = time.perf_counter()

    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(1, epochs + 1):
            model.train()
            if freeze_backbone:
                model.features.eval()
            running_loss = 0.0
            seen = 0
            for images, labels in train_loader:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(
                    device_type=device.type,
                    dtype=torch.float16,
                    enabled=amp_enabled,
                ):
                    logits = model(images)
                    loss = criterion(logits, labels)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += float(loss.detach()) * len(labels)
                seen += len(labels)

            development_truth, development_probabilities = predict(
                model,
                development_loader,
                device,
            )
            threshold, sensitivity = select_threshold(
                development_truth,
                development_probabilities,
            )
            selected_metrics = next(
                row for row in sensitivity if row["threshold"] == threshold
            )
            record = {
                "epoch": epoch,
                "train_loss": running_loss / seen,
                "development_threshold": threshold,
                "development_metrics": selected_metrics,
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(json.dumps(record))

            if selected_metrics["macro_f1"] > best_macro_f1:
                best_macro_f1 = float(selected_metrics["macro_f1"])
                best_epoch = epoch
                best_threshold = threshold
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "architecture": "mobilenet_v3_small",
                        "implementation_label": (
                            "adapted_standard_mobilenet_v3_small"
                        ),
                        "pretrained_imagenet": pretrained,
                        "freeze_backbone": freeze_backbone,
                        "patch_size": patch_size,
                        "seed": seed,
                        "best_epoch": best_epoch,
                        "best_development_macro_f1": best_macro_f1,
                        "development_threshold": best_threshold,
                        "class_names": ["vacant", "occupied"],
                    },
                    output_dir / "best.pt",
                )

    summary = {
        "implementation_label": "adapted_standard_mobilenet_v3_small",
        "paper_exact_reproduction": False,
        "seed": seed,
        "device": str(device),
        "gpu": (
            torch.cuda.get_device_name(0)
            if device.type == "cuda" and torch.cuda.is_available()
            else None
        ),
        "epochs": epochs,
        "batch_size": batch_size,
        "amp": amp_enabled,
        "pretrained_imagenet": pretrained,
        "freeze_backbone": freeze_backbone,
        "patch_size": list(patch_size),
        "train_samples": len(train_samples),
        "development_samples": len(development_samples),
        "test_samples_not_used": sum(
            sample.split == "test" for sample in samples
        ),
        "train_class_counts": dict(label_counts),
        "best_epoch": best_epoch,
        "best_development_macro_f1": best_macro_f1,
        "development_threshold": best_threshold,
        "elapsed_s": time.perf_counter() - start_time,
        "checkpoint": str((output_dir / "best.pt").resolve()),
        "command_note": "test split was not loaded by the training loop",
    }
    with (output_dir / "training_summary.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

