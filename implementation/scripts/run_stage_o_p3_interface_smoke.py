from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import cv2


PROJECT_ROOT = Path(__file__).resolve().parents[1]
for source_root in (
    PROJECT_ROOT / "src",
    PROJECT_ROOT / "literature_core" / "src",
):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from parking_occupancy.detector_comparison import sha256_file
from parking_occupancy.integrated_runner import run_integrated_video
from parking_occupancy.stage_n_lmot import read_image
from parking_occupancy.stage_o_low_light import STAGE_O_PROTOCOL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the selected Stage O D1-LL checkpoint through a truth-free "
            "P3 interface smoke. This does not estimate occupancy improvement."
        )
    )
    parser.add_argument("--source-image", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--d1-ll-weights", type=Path, required=True)
    parser.add_argument("--e1b-checkpoint", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = args.source_image.resolve()
    regions = args.regions.resolve()
    weights = args.d1_ll_weights.resolve()
    classifier = args.e1b_checkpoint.resolve()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    if input_root.exists():
        raise FileExistsError(f"Refusing to overwrite {input_root}")
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    if args.frames <= 0 or args.fps <= 0:
        raise ValueError("frames and fps must be positive")
    for path in (source, regions, weights, classifier):
        if not path.is_file():
            raise FileNotFoundError(path)
    image = read_image(source)
    if image is None:
        raise RuntimeError(f"Could not decode {source}")
    height, width = image.shape[:2]
    raw_regions = json.loads(regions.read_text(encoding="utf-8"))
    slots = {
        "schema_version": 1,
        "source_width": width,
        "source_height": height,
        "coordinate_system": "pixel",
        "slots": [
            {
                "id": str(
                    row.get("slot_id", row.get("id", f"slot-{index + 1}"))
                ),
                "points": row["points"],
            }
            for index, row in enumerate(raw_regions)
        ],
    }
    input_root.mkdir(parents=True)
    slots_path = input_root / "slots.json"
    slots_path.write_text(
        json.dumps(slots, indent=2) + "\n", encoding="utf-8"
    )
    video_path = input_root / "input.mp4"
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        args.fps,
        (width, height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Could not create {video_path}")
    try:
        for _ in range(args.frames):
            writer.write(image)
    finally:
        writer.release()
    input_manifest = {
        "schema_version": 1,
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "claim_scope": "interface_smoke_only",
        "accuracy_or_occupancy_improvement_claim": False,
        "source_image": {
            "path": str(source),
            "bytes": source.stat().st_size,
            "sha256": sha256_file(source),
            "role": "consumed_development_image_repeated_for_interface_smoke",
        },
        "source_regions": {
            "path": str(regions),
            "bytes": regions.stat().st_size,
            "sha256": sha256_file(regions),
        },
        "generated_video": {
            "path": str(video_path),
            "frames": args.frames,
            "fps": args.fps,
            "bytes": video_path.stat().st_size,
            "sha256": sha256_file(video_path),
        },
        "generated_slots": {
            "path": str(slots_path),
            "slots": len(slots["slots"]),
            "bytes": slots_path.stat().st_size,
            "sha256": sha256_file(slots_path),
        },
        "D1_LL": {
            "path": str(weights),
            "bytes": weights.stat().st_size,
            "sha256": sha256_file(weights),
        },
        "truth": None,
    }
    (input_root / "input_manifest.json").write_text(
        json.dumps(input_manifest, indent=2) + "\n",
        encoding="utf-8",
    )
    # Keep Ultralytics' mutable user settings inside this additive smoke input.
    # This avoids dependence on a writable roaming-profile directory.
    os.environ.setdefault(
        "YOLO_CONFIG_DIR", str(input_root / "ultralytics_config")
    )
    summary = run_integrated_video(
        input_path=video_path,
        slots_path=slots_path,
        detector_weights=weights,
        classifier_checkpoint=classifier,
        output_root=output_root,
        device=args.device,
        source_id="stage_o_selected_D1_LL_interface_smoke",
        truth_path=None,
        temporal_enabled=True,
        tracker_backend="none",
    )
    if summary["truth_supplied"] or summary["parameter_selection_from_run"]:
        raise RuntimeError("P3 Stage O smoke crossed a frozen boundary")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
