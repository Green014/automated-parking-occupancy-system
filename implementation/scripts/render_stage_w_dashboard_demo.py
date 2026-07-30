from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render a post-hoc Stage W dashboard UI demonstration loop."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seconds", type=float, default=10.0)
    parser.add_argument("--fps", type=float, default=5.0)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.seconds <= 0 or args.fps <= 0:
        raise ValueError("seconds and fps must be positive")
    video = args.run_dir / "annotated.mp4"
    status_path = args.run_dir / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))
    capture = cv2.VideoCapture(str(video))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(frame)
    capture.release()
    if not frames:
        raise ValueError("Stage W source run contains no decodable frame")

    width, height = 1280, 720
    writer = cv2.VideoWriter(
        str(args.output),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError("Could not create dashboard UI demo")
    output_frames = int(round(args.seconds * args.fps))
    for index in range(output_frames):
        source = frames[index % len(frames)]
        video_panel = cv2.resize(source, (960, 540))
        canvas = np.full((height, width, 3), (11, 17, 24), dtype=np.uint8)
        canvas[:540, :960] = video_panel
        cv2.rectangle(canvas, (960, 0), (1279, 539), (20, 31, 43), -1)
        cv2.putText(
            canvas,
            "STAGE W DASHBOARD",
            (985, 48),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (85, 201, 237),
            2,
            cv2.LINE_AA,
        )
        rows = [
            ("MODE", str(status.get("mode", "fusion")).upper()),
            ("OCCUPIED", str(status.get("occupied", 0))),
            ("VACANT", str(status.get("vacant", 0))),
            ("TOTAL", str(status.get("total", 0))),
            ("RENDERED", str(status.get("rendered_slots", 0))),
            (
                "ATTRIBUTED FPS",
                f"{status.get('runtime', {}).get('attributed_fps', 0.0):.1f}",
            ),
            (
                "TEMPORAL",
                "ON" if status.get("temporal_enabled") else "OFF",
            ),
            (
                "TRACKER",
                "ON" if status.get("tracker_enabled") else "OFF",
            ),
        ]
        y = 94
        for label, value in rows:
            cv2.putText(
                canvas,
                label,
                (985, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.40,
                (145, 165, 182),
                1,
                cv2.LINE_AA,
            )
            cv2.putText(
                canvas,
                value,
                (985, y + 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.72,
                (239, 246, 251),
                2,
                cv2.LINE_AA,
            )
            y += 54
        cv2.rectangle(canvas, (0, 540), (1279, 719), (9, 15, 22), -1)
        cv2.putText(
            canvas,
            "POST-HOC INTERFACE LOOP",
            (28, 590),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 216, 74),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "Repeated four-frame consumed demonstration; not a fixed-camera "
            "performance validation.",
            (28, 632),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.58,
            (225, 231, 236),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "No truth supplied. No accuracy, transition-latency or tracking "
            "improvement claim.",
            (28, 668),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.52,
            (145, 165, 182),
            1,
            cv2.LINE_AA,
        )
        writer.write(canvas)
    writer.release()
    metadata = {
        "schema_version": 1,
        "stage": "W",
        "role": "posthoc_interface_demonstration_only",
        "source_role": "repeated_four_frame_consumed_demonstration",
        "fixed_camera_performance_validation": False,
        "accuracy_claim": False,
        "model_inference_run_for_render": False,
        "input_video": {
            "filename": video.name,
            "bytes": video.stat().st_size,
            "sha256": sha256_file(video),
            "decoded_frames": len(frames),
        },
        "output_video": {
            "filename": args.output.name,
            "bytes": args.output.stat().st_size,
            "sha256": sha256_file(args.output),
            "frames": output_frames,
            "fps": args.fps,
            "duration_seconds": output_frames / args.fps,
            "width": width,
            "height": height,
        },
    }
    args.output.with_suffix(".json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
