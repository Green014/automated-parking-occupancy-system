from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

import yaml

from parking_occupancy.detector_comparison import sha256_file


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = (
    "configs/stage_l_integrated_workflow_frozen_20260728.yaml",
    "data/STAGE_L_INTEGRATED_WORKFLOW_REPORT.md",
    "outputs/stage_l_p3_static_development_20260728_v1/preflight.json",
    "outputs/stage_l_p3_static_development_20260728_v1/annotated.mp4",
    "outputs/stage_l_p3_static_development_20260728_v1/predictions.csv",
    "outputs/stage_l_p3_static_development_20260728_v1/events.csv",
    "outputs/stage_l_p3_static_development_20260728_v1/metrics.json",
    "outputs/stage_l_p3_static_development_20260728_v1/paired_analysis.json",
    "outputs/stage_l_p3_static_development_20260728_v1/summary.json",
    "outputs/stage_l_p3_static_development_20260728_v1/runtime_metadata.json",
    "outputs/stage_l_p3_static_retrospective_20260728_v1/preflight.json",
    "outputs/stage_l_p3_static_retrospective_20260728_v1/annotated.mp4",
    "outputs/stage_l_p3_static_retrospective_20260728_v1/predictions.csv",
    "outputs/stage_l_p3_static_retrospective_20260728_v1/events.csv",
    "outputs/stage_l_p3_static_retrospective_20260728_v1/metrics.json",
    "outputs/stage_l_p3_static_retrospective_20260728_v1/paired_analysis.json",
    "outputs/stage_l_p3_static_retrospective_20260728_v1/summary.json",
    "outputs/stage_l_p3_static_retrospective_20260728_v1/runtime_metadata.json",
    (
        "outputs/stage_l_e1b_classifier_ablation_retrospective_20260728_v1/"
        "predictions.csv"
    ),
    (
        "outputs/stage_l_e1b_classifier_ablation_retrospective_20260728_v1/"
        "metrics.json"
    ),
    "outputs/stage_l_p3_video_virat0502_20260728_v1/preflight.json",
    "outputs/stage_l_p3_video_virat0502_20260728_v1/annotated.mp4",
    "outputs/stage_l_p3_video_virat0502_20260728_v1/occupancy.csv",
    "outputs/stage_l_p3_video_virat0502_20260728_v1/events.csv",
    "outputs/stage_l_p3_video_virat0502_20260728_v1/detections.jsonl",
    "outputs/stage_l_p3_video_virat0502_20260728_v1/metrics.json",
    (
        "outputs/stage_l_p3_video_virat0502_20260728_v1/"
        "metrics_v2_absolute_frames.json"
    ),
    "outputs/stage_l_p3_video_virat0502_20260728_v1/summary.json",
    "outputs/stage_l_p3_video_virat0502_20260728_v1/runtime_metadata.json",
    (
        "outputs/stage_l_p3_video_virat0502_20260728_v1/"
        "annotated_frame_1700_review.jpg"
    ),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze hashes for the executed Stage L evidence."
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite record: {args.output}")

    artifacts = []
    for relative in ARTIFACTS:
        path = IMPLEMENTATION_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        artifacts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    payload = {
        "schema_version": 1,
        "record_id": "P3-INTEGRATED-LITERATURE-WORKFLOW-RECORD-20260728-01",
        "recorded_at": datetime.now(timezone.utc).isoformat(),
        "status": "executed_and_hash_frozen",
        "historical_stage_j_k_modified": False,
        "model_training_or_retraining_performed": False,
        "static_development_claim": "functional_integration_check_only",
        "static_retrospective_claim": (
            "previously_consumed_test_extension_not_untouched_for_P3"
        ),
        "continuous_claim": "single_slot_single_departure_case_study_only",
        "artifacts": artifacts,
        "artifact_count": len(artifacts),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False),
        encoding="utf-8",
    )
    print(
        yaml.safe_dump(
            {
                "record_id": payload["record_id"],
                "artifact_count": len(artifacts),
                "output": str(args.output.resolve()),
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
