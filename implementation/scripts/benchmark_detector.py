from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from parking_occupancy.detector import UltralyticsDetector
from parking_occupancy.image_io import read_image


def percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    index = round((len(ordered) - 1) * p)
    return ordered[index]


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def main() -> None:
    parser = argparse.ArgumentParser(description="Warm detector latency benchmark")
    parser.add_argument("--input", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--conf", type=float, default=0.10)
    parser.add_argument("--imgsz", type=int, default=1280)
    parser.add_argument("--device", default="0")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    frame = read_image(Path(args.input))
    detector = UltralyticsDetector(
        weights=args.weights,
        confidence=args.conf,
        image_size=args.imgsz,
        device=args.device,
    )
    for _ in range(args.warmup):
        detector.detect(frame)
    synchronize()

    times_ms: list[float] = []
    detection_counts: list[int] = []
    for _ in range(args.iterations):
        start = time.perf_counter()
        detections = detector.detect(frame)
        synchronize()
        times_ms.append((time.perf_counter() - start) * 1000.0)
        detection_counts.append(len(detections))

    report = {
        "input": str(Path(args.input).resolve()),
        "weights": args.weights,
        "confidence": args.conf,
        "image_size": args.imgsz,
        "warmup_iterations": args.warmup,
        "measured_iterations": args.iterations,
        "detections_per_frame": statistics.fmean(detection_counts),
        "latency_ms": {
            "mean": statistics.fmean(times_ms),
            "p50": statistics.median(times_ms),
            "p95": percentile(times_ms, 0.95),
        },
        "fps_from_mean_latency": 1000.0 / statistics.fmean(times_ms),
        "detector": detector.metadata(),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
