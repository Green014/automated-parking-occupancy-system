from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
LITERATURE_CORE_SRC = PROJECT_ROOT / "literature_core" / "src"
for source_root in (PROJECT_SRC, LITERATURE_CORE_SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

import cv2

from parking_occupancy.stage_m_evaluation import (
    export_stage_m_run,
    run_os0_sequence,
    run_t0_t3_sequence,
)
from parking_occupancy.stage_m_tracking import (
    InferenceSettings,
    OS0ParkingAdapter,
    UltralyticsSequenceAdapter,
    load_parking_regions,
    load_stage_m_protocol,
    sha256_file,
    verify_ultralytics_runtime,
)
from run_stage_m import FrozenE1bProvider


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_m_open_source_tracking_frozen_20260728.yaml"
)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible Stage M OS0 and T0-T3 smoke tests"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    protocol = load_stage_m_protocol(config_path, verify_files=True)
    if args.output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage M smoke output: {args.output_root}"
        )
    args.output_root.mkdir(parents=True)
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str((args.output_root / "_ultralytics_config").resolve()),
    )
    runtime_audit = verify_ultralytics_runtime(protocol)

    smoke = protocol["smoke"]
    source_path = _resolve(config_path, smoke["source_image"]["path"])
    region_path = _resolve(config_path, smoke["regions"]["path"])
    source = cv2.imread(str(source_path))
    if source is None:
        raise RuntimeError(f"Could not read smoke source: {source_path}")
    frame_count = int(smoke["frames"])
    fps = float(smoke["fps"])
    frames = [source.copy() for _ in range(frame_count)]

    shared = protocol["shared_inference"]
    settings = InferenceSettings(
        weights=str(_resolve(config_path, shared["weights_path"])),
        confidence=float(shared["confidence"]),
        nms_iou=float(shared["nms_iou"]),
        image_size=int(shared["imgsz"]),
        class_ids=tuple(int(value) for value in shared["source_class_ids"]),
        max_detections=int(shared["max_detections"]),
        device=args.device,
        agnostic_nms=bool(shared["agnostic_nms"]),
    )
    byte_config = _resolve(
        config_path, protocol["trackers"]["bytetrack"]["config_path"]
    )
    track_config = _resolve(
        config_path, protocol["trackers"]["tracktrack"]["config_path"]
    )

    os0 = run_os0_sequence(
        frames=[frame.copy() for frame in frames],
        fps=fps,
        source_id="stage_m_pklot_repeated_smoke",
        adapter=OS0ParkingAdapter(
            settings,
            region_json=region_path,
            tracker_config=track_config,
        ),
        continuous=True,
        claim_scope="smoke_test",
    )
    os0.runtime_metadata["runtime_audit"] = runtime_audit
    export_stage_m_run(
        os0,
        output_root=args.output_root / "OS0-Controlled",
        fps=fps,
    )

    os0_static = run_os0_sequence(
        frames=[source.copy(), source.copy()],
        fps=fps,
        source_id="stage_m_pklot_static_smoke",
        adapter=OS0ParkingAdapter(
            settings,
            region_json=region_path,
            tracker_config=track_config,
        ),
        continuous=False,
        claim_scope="smoke_test",
    )
    os0_static.summary["diagnostic_label"] = (
        "OS0 static centre-point diagnostic"
    )
    os0_static.summary["temporal_claim"] = False
    os0_static.runtime_metadata["runtime_audit"] = runtime_audit
    export_stage_m_run(
        os0_static,
        output_root=args.output_root / "OS0-Static-Diagnostic",
        fps=fps,
    )

    provider = FrozenE1bProvider(
        _resolve(config_path, protocol["classifier"]["checkpoint_path"]),
        device=args.device,
        patch_size=tuple(protocol["classifier"]["patch_size"]),
        perspective_warp=bool(protocol["classifier"]["perspective_warp"]),
    )
    ablation = run_t0_t3_sequence(
        frames=[frame.copy() for frame in frames],
        fps=fps,
        source_id="stage_m_pklot_repeated_smoke",
        slots=load_parking_regions(region_path),
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
        mapping_coverage=float(protocol["mapping"]["minimum_slot_coverage"]),
        classifier_threshold=float(
            protocol["classifier"]["occupied_threshold"]
        ),
        temporal_config=protocol["temporal"],
        claim_scope="smoke_test",
    )
    ablation.runtime_metadata["classifier"] = provider.metadata()
    ablation.runtime_metadata["runtime_audit"] = runtime_audit
    export_stage_m_run(
        ablation,
        output_root=args.output_root / "T0-T3",
        fps=fps,
    )

    summary = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "status": "executed_reproducible_smoke",
        "claim_scope": "smoke_test_only",
        "accuracy_or_temporal_claim": False,
        "source": {
            "path": str(source_path),
            "bytes": source_path.stat().st_size,
            "sha256": sha256_file(source_path),
            "repeat_frames": frame_count,
        },
        "regions": {
            "path": str(region_path),
            "bytes": region_path.stat().st_size,
            "sha256": sha256_file(region_path),
        },
        "runs": {
            "OS0-Controlled": os0.summary,
            "OS0-Static-Diagnostic": os0_static.summary,
            "T0-T3": ablation.summary,
        },
    }
    (args.output_root / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
