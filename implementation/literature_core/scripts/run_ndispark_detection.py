from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT.parent / ".ultralytics"))


def _float_list(values: Any) -> list[float]:
    if values is None:
        return []
    if hasattr(values, "tolist"):
        values = values.tolist()
    return [float(value) for value in values]


def evaluate_model(
    model: Any,
    *,
    data: Path,
    output_dir: Path,
    image_size: int,
    device: str,
    class_filter: list[int] | None,
    method: str,
) -> dict[str, Any]:
    """Evaluate prompted/closed-set detections as one vehicle class."""

    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    result = model.val(
        data=str(data.resolve()),
        split="val",
        imgsz=image_size,
        device=device,
        project=str(output_dir.parent),
        name=output_dir.name,
        exist_ok=True,
        plots=True,
        verbose=False,
        classes=class_filter,
        single_cls=True,
    )
    ground_truth_boxes = int(sum(_float_list(result.nt_per_class)))
    if ground_truth_boxes == 0:
        raise RuntimeError("NDISPark evaluation loaded zero ground-truth boxes")
    return {
        "method": method,
        "task": "single-class vehicle box detection",
        "data": str(data.resolve()),
        "split": "validation",
        "images": 30,
        "ground_truth_boxes": ground_truth_boxes,
        "image_size": image_size,
        "device": device,
        "class_filter": class_filter,
        "single_class_evaluation": True,
        "precision": float(result.box.mp),
        "recall": float(result.box.mr),
        "map_50": float(result.box.map50),
        "map_50_95": float(result.box.map),
        "speed_ms_per_image": {
            str(key): float(value) for key, value in result.speed.items()
        },
        "ultralytics_output_dir": str(output_dir),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare YOLOv8 and YOLO-World on NDISPark manual vehicle boxes"
        )
    )
    parser.add_argument("--data", required=True)
    parser.add_argument("--world-weights", required=True)
    parser.add_argument("--baseline-weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    import torch
    import ultralytics
    from ultralytics import YOLO, YOLOWorld

    device = (
        "0"
        if args.device == "auto" and torch.cuda.is_available()
        else "cpu"
        if args.device == "auto"
        else args.device
    )
    prompts = ["car", "truck", "bus", "motorcycle"]
    output_dir = Path(args.output_dir)

    baseline = YOLO(args.baseline_weights)
    baseline_report = evaluate_model(
        baseline,
        data=Path(args.data),
        output_dir=output_dir / "yolov8_vehicle",
        image_size=args.imgsz,
        device=device,
        class_filter=[2, 3, 5, 7],
        method="YOLOv8 closed-set vehicle classes",
    )

    world = YOLOWorld(args.world_weights)
    world.set_classes(prompts)
    world_report = evaluate_model(
        world,
        data=Path(args.data),
        output_dir=output_dir / "yolo_world_vehicle",
        image_size=args.imgsz,
        device=device,
        class_filter=None,
        method="YOLO-World prompted vehicle classes",
    )

    report = {
        "protocol": {
            "task": "detector-level box evaluation, not slot occupancy",
            "dataset": "NDISPark validation",
            "truth": "725 manual vehicle boxes in 30 night-domain images",
            "single_class_reason": (
                "The truth labels all vehicles as COCO car; both methods are "
                "collapsed to one vehicle class for a common comparison."
            ),
            "prompts": prompts,
            "ultralytics_version": ultralytics.__version__,
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "gpu": (
                torch.cuda.get_device_name(0)
                if torch.cuda.is_available()
                else None
            ),
        },
        "baseline": baseline_report,
        "yolo_world": world_report,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "comparison.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
