from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("YOLO_CONFIG_DIR", str(PROJECT_ROOT / ".ultralytics"))


def _float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def evaluate_detection(
    data_yaml: Path,
    output_dir: Path,
    weights: str,
    image_size: int,
    device: str,
    split: str,
    classes: list[int] | None = None,
    single_class: bool = False,
) -> dict[str, Any]:
    """Run Ultralytics box evaluation on data with vehicle-box ground truth."""

    from ultralytics import YOLO

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    results = model.val(
        data=str(data_yaml.resolve()),
        split=split,
        imgsz=image_size,
        device=device,
        plots=True,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        verbose=False,
        classes=classes,
        single_cls=single_class,
    )
    target_counts = _float_list(results.nt_per_class)
    ground_truth_boxes = int(sum(target_counts))
    if ground_truth_boxes == 0:
        raise RuntimeError(
            "Detection evaluation loaded zero ground-truth boxes. Check the "
            "dataset class indices, class filter, and stale label cache."
        )
    class_ids = [int(value) for value in _float_list(results.box.ap_class_index)]
    per_class_ap50_95 = _float_list(results.box.maps)
    names = results.names
    report = {
        "task": "vehicle_detection",
        "note": (
            "These are detector-level box metrics, not parking-slot "
            "occupancy metrics."
        ),
        "data_yaml": str(data_yaml.resolve()),
        "weights": weights,
        "split": split,
        "image_size": image_size,
        "device": device,
        "model_class_filter": classes,
        "single_class_evaluation": single_class,
        "ground_truth_boxes": ground_truth_boxes,
        "precision": float(results.box.mp),
        "recall": float(results.box.mr),
        "map_50": float(results.box.map50),
        "map_50_95": float(results.box.map),
        "per_class_map_50_95": {
            str(names[class_id]): per_class_ap50_95[class_id]
            for class_id in class_ids
            if class_id < len(per_class_ap50_95)
        },
        "speed_ms": {
            str(key): float(value) for key, value in results.speed.items()
        },
        "ultralytics_output_dir": str(output_dir),
    }
    with (output_dir / "detection_metrics.json").open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate vehicle bounding boxes; requires a YOLO data YAML and "
            "box ground truth."
        )
    )
    parser.add_argument("--data", required=True, help="Ultralytics data YAML")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument(
        "--classes",
        help="Optional comma-separated model class IDs, e.g. 2 for COCO car",
    )
    parser.add_argument(
        "--single-cls",
        action="store_true",
        help="Evaluate the selected predictions as one ground-truth class",
    )
    args = parser.parse_args()
    device = args.device
    if device == "auto":
        import torch

        device = "0" if torch.cuda.is_available() else "cpu"
    classes = (
        [int(value) for value in args.classes.split(",")]
        if args.classes
        else None
    )
    report = evaluate_detection(
        data_yaml=Path(args.data),
        output_dir=Path(args.output_dir),
        weights=args.weights,
        image_size=args.imgsz,
        device=device,
        split=args.split,
        classes=classes,
        single_class=args.single_cls,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
