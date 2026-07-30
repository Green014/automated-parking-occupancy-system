from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
LITERATURE_CORE_SRC = PROJECT_ROOT / "literature_core" / "src"
for source_root in (PROJECT_SRC, LITERATURE_CORE_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import cv2
import yaml

from parking_occupancy.stage_m_data_gate import validate_formal_parking_gate
from parking_occupancy.stage_m_evaluation import (
    export_stage_m_run,
    run_os0_sequence,
    run_t0_t3_sequence,
)
from parking_occupancy.stage_m_tracking import (
    InferenceSettings,
    OS0ParkingAdapter,
    StageMProtocolError,
    UltralyticsSequenceAdapter,
    load_parking_regions,
    load_stage_m_protocol,
    sha256_file,
    verify_ultralytics_runtime,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_m_open_source_tracking_frozen_20260728.yaml"
)


class FrozenE1bProvider:
    def __init__(
        self,
        checkpoint: Path,
        *,
        device: str,
        patch_size: tuple[int, int],
        perspective_warp: bool,
    ) -> None:
        from literature_core.classifier import MobileNetSlotClassifier

        self.classifier = MobileNetSlotClassifier(checkpoint, device=device)
        self.patch_size = patch_size
        self.perspective_warp = perspective_warp

    def __call__(self, frame, slots: Sequence[Any]) -> dict[str, float]:
        from literature_core.patches import extract_slot_patch

        patches = [
            extract_slot_patch(
                frame,
                slot.points,
                output_size=self.patch_size,
                perspective_warp=self.perspective_warp,
            )
            for slot in slots
        ]
        scores = self.classifier.predict_patches(patches)
        return {
            slot.slot_id: float(score)
            for slot, score in zip(slots, scores, strict=True)
        }

    def metadata(self) -> dict[str, Any]:
        return self.classifier.metadata()


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def _settings(
    protocol: dict[str, Any],
    config_path: Path,
    device: str,
) -> InferenceSettings:
    shared = protocol["shared_inference"]
    return InferenceSettings(
        weights=str(_resolve(config_path, shared["weights_path"])),
        confidence=float(shared["confidence"]),
        nms_iou=float(shared["nms_iou"]),
        image_size=int(shared["imgsz"]),
        class_ids=tuple(int(value) for value in shared["source_class_ids"]),
        max_detections=int(shared["max_detections"]),
        device=device,
        agnostic_nms=bool(shared["agnostic_nms"]),
    )


def _read_video(path: Path) -> tuple[list[Any], float]:
    capture = cv2.VideoCapture(str(path.resolve()))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    if not frames or fps <= 0:
        raise RuntimeError("Video contains no decodable frames or valid FPS")
    return frames, fps


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run frozen Stage M methods")
    parser.add_argument("--mode", choices=("os0", "ablation"), required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--truth", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--source-id")
    parser.add_argument("--device", default="0")
    parser.add_argument(
        "--claim-scope",
        choices=("smoke_test", "retrospective_diagnostic", "formal_test"),
        default="smoke_test",
    )
    parser.add_argument(
        "--formal-gate",
        type=Path,
        help="Required machine-readable gate for claim-scope=formal_test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    protocol = load_stage_m_protocol(config_path, verify_files=True)
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage M output: {args.output_root}"
        )
    if args.claim_scope == "formal_test":
        if args.formal_gate is None or args.truth is None:
            raise StageMProtocolError(
                "formal_test requires --formal-gate and --truth"
            )
        gate_payload = yaml.safe_load(
            args.formal_gate.read_text(encoding="utf-8")
        )
        decision = validate_formal_parking_gate(
            gate_payload,
            base_dir=args.formal_gate.resolve().parent,
            verify_files=True,
        )
        if decision.status != "eligible":
            raise StageMProtocolError(
                "Formal Stage M gate is blocked: "
                + ", ".join(decision.reasons)
            )

    args.output_root.mkdir(parents=True)
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str((args.output_root / "_ultralytics_config").resolve()),
    )
    runtime_audit = verify_ultralytics_runtime(protocol)
    frames, fps = _read_video(args.video)
    truth = (
        None
        if args.truth is None
        else yaml.safe_load(args.truth.read_text(encoding="utf-8"))
    )
    source_id = args.source_id or args.video.stem
    settings = _settings(protocol, config_path, args.device)

    if args.mode == "os0":
        tracker = _resolve(
            config_path, protocol["trackers"]["tracktrack"]["config_path"]
        )
        adapter = OS0ParkingAdapter(
            settings,
            region_json=args.regions,
            tracker_config=tracker,
        )
        result = run_os0_sequence(
            frames=frames,
            fps=fps,
            source_id=source_id,
            adapter=adapter,
            truth=truth,
            continuous=True,
            claim_scope=args.claim_scope,
        )
    else:
        byte_config = _resolve(
            config_path, protocol["trackers"]["bytetrack"]["config_path"]
        )
        track_config = _resolve(
            config_path, protocol["trackers"]["tracktrack"]["config_path"]
        )
        provider = FrozenE1bProvider(
            _resolve(
                config_path,
                protocol["classifier"]["checkpoint_path"],
            ),
            device=args.device,
            patch_size=tuple(protocol["classifier"]["patch_size"]),
            perspective_warp=bool(
                protocol["classifier"]["perspective_warp"]
            ),
        )
        result = run_t0_t3_sequence(
            frames=frames,
            fps=fps,
            source_id=source_id,
            slots=load_parking_regions(args.regions),
            plain_adapter=UltralyticsSequenceAdapter(
                settings, tracker_config=None
            ),
            bytetrack_adapter=UltralyticsSequenceAdapter(
                settings, tracker_config=byte_config
            ),
            tracktrack_adapter=UltralyticsSequenceAdapter(
                settings, tracker_config=track_config
            ),
            classifier_scores=provider,
            mapping_coverage=float(
                protocol["mapping"]["minimum_slot_coverage"]
            ),
            classifier_threshold=float(
                protocol["classifier"]["occupied_threshold"]
            ),
            temporal_config=protocol["temporal"],
            truth=truth,
            claim_scope=args.claim_scope,
        )
        result.runtime_metadata["classifier"] = provider.metadata()

    result.runtime_metadata["runtime_audit"] = runtime_audit
    result.runtime_metadata["input"] = {
        "video": str(args.video.resolve()),
        "video_bytes": args.video.stat().st_size,
        "video_sha256": sha256_file(args.video.resolve()),
        "regions": str(args.regions.resolve()),
        "regions_bytes": args.regions.stat().st_size,
        "regions_sha256": sha256_file(args.regions.resolve()),
        "truth": None if args.truth is None else str(args.truth.resolve()),
        "truth_sha256": (
            None if args.truth is None else sha256_file(args.truth.resolve())
        ),
    }
    export_stage_m_run(result, output_root=args.output_root, fps=fps)
    print(
        json.dumps(
            {
                "status": "ok",
                "mode": args.mode,
                "claim_scope": args.claim_scope,
                "frames": len(frames),
                "output_root": str(args.output_root.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
