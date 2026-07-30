from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.data import read_image, write_image  # noqa: E402
from literature_core.detector import YOLOWorldDetector  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="One-image YOLO-World check")
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["car", "truck", "bus", "motorcycle"],
    )
    parser.add_argument("--conf", type=float, default=0.025)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    image = read_image(args.image)
    detector = YOLOWorldDetector(
        args.weights,
        prompts=args.prompts,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
    )
    start = time.perf_counter()
    detections = detector.detect(image)
    elapsed_s = time.perf_counter() - start
    canvas = image.copy()
    for detection in detections:
        x1, y1, x2, y2 = (round(value) for value in detection.bbox)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 210, 255), 2)
        cv2.putText(
            canvas,
            f"{detection.label} {detection.confidence:.2f}",
            (x1, max(16, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 210, 255),
            1,
            cv2.LINE_AA,
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    write_image(output_dir / "annotated.jpg", canvas)
    report = {
        "image": str(Path(args.image).resolve()),
        "detections": [asdict(detection) for detection in detections],
        "detection_count": len(detections),
        "elapsed_s_including_model_load": elapsed_s,
        "metadata": detector.metadata(),
        "occupancy_note": (
            "These are raw object detections; slot occupancy requires polygon mapping."
        ),
    }
    with (output_dir / "detections.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

