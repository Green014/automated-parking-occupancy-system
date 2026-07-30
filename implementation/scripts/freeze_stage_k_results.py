from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from parking_occupancy.detector_comparison import sha256_file
from parking_occupancy.stage_k_occupancy import (
    STAGE_K_PROTOCOL_ID,
    STAGE_K_RECORD_ID,
)


SOURCE_ARTIFACTS = (
    (
        "protocol",
        "configs/stage_k_p0_p1_p2_pklot_test_frozen_20260727.yaml",
    ),
    (
        "annotations",
        "data/annotations/pklot_stage_k_candidate_20260727_v2.jsonl",
    ),
    (
        "membership_manifest",
        "data/manifests/pklot_stage_k_candidate_20260727_v2.csv",
    ),
    (
        "candidate_audit",
        "data/preprocessing/pklot_stage_k_candidate_audit_20260727_v2.json",
    ),
    (
        "manual_visual_review",
        "data/preprocessing/pklot_stage_k_manual_visual_review_20260727_v2.json",
    ),
    (
        "prior_development_manifest",
        "data/manifests/stage_j_pklot_development_20260727.csv",
    ),
)
EXTERNAL_RUN_ARTIFACTS = (
    ("preflight_only", "outputs/P0_P1_P2_stage_k_20260727_preflight_v1.json"),
    (
        "candidate_contact_sheet",
        "outputs/stage_k_pklot_candidate_contact_sheet_20260727_v2.png",
    ),
    (
        "truth_contact_sheet",
        "outputs/stage_k_pklot_truth_contact_sheet_20260727_v2.png",
    ),
    (
        "run_preflight",
        "outputs/P0_P1_P2_stage_k_20260727_v1/preflight.json",
    ),
    (
        "comparison",
        "outputs/P0_P1_P2_stage_k_20260727_v1/comparison.json",
    ),
    (
        "camera_metrics",
        "outputs/P0_P1_P2_stage_k_20260727_v1/camera_metrics.csv",
    ),
    (
        "paired_image_differences",
        (
            "outputs/P0_P1_P2_stage_k_20260727_v1/"
            "paired_image_differences.csv"
        ),
    ),
)
METHOD_ARTIFACTS = (
    ("annotated_video", "annotated.mp4"),
    ("occupancy", "occupancy.csv"),
    ("events", "events.csv"),
    ("detections", "detections.jsonl"),
    ("summary", "summary.json"),
    ("metrics", "metrics.json"),
    ("runtime", "runtime_metadata.json"),
    ("errors", "errors.csv"),
    ("confusion_matrix", "confusion_matrix.png"),
    ("pr_curve", "pr_curve.png"),
)


def _artifact(root_name: str, root: Path, role: str, relative: str) -> dict:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(f"Missing Stage K artifact: {path}")
    return {
        "role": role,
        "root": root_name,
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def build_record(*, source_root: Path, external_root: Path) -> dict[str, Any]:
    source_root = source_root.resolve()
    external_root = external_root.resolve()
    artifacts = [
        _artifact("source", source_root, role, relative)
        for role, relative in SOURCE_ARTIFACTS
    ]
    artifacts.extend(
        _artifact("external", external_root, role, relative)
        for role, relative in EXTERNAL_RUN_ARTIFACTS
    )
    for method_id in ("P0", "P1", "P2"):
        for role, filename in METHOD_ARTIFACTS:
            relative = (
                f"outputs/P0_P1_P2_stage_k_20260727_v1/"
                f"{method_id}/{filename}"
            )
            artifacts.append(
                _artifact(
                    "external",
                    external_root,
                    f"{method_id}_{role}",
                    relative,
                )
            )
    comparison_path = (
        external_root
        / "outputs"
        / "P0_P1_P2_stage_k_20260727_v1"
        / "comparison.json"
    )
    comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    return {
        "schema_version": 1,
        "record_id": STAGE_K_RECORD_ID,
        "protocol_id": STAGE_K_PROTOCOL_ID,
        "status": "frozen_after_single_test_execution",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "untouched_test",
        "predictions_before_protocol_freeze": False,
        "parameters_selected_from_test": False,
        "detector_reselected_from_test": False,
        "D1_retrained": False,
        "remote_or_paid_GPU_used": False,
        "negative_results_retained": True,
        "artifact_count": len(artifacts),
        "summary": {
            "metrics": comparison["metrics"],
            "paired_comparisons": comparison["paired_comparisons"],
            "selected_detector_before_slot_evaluation": comparison[
                "selected_detector_before_slot_evaluation"
            ],
        },
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the single-execution Stage K artifact record."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage K result record: {output}"
        )
    record = build_record(
        source_root=args.source_root,
        external_root=args.external_root,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(
            record,
            sort_keys=False,
            allow_unicode=True,
            width=1000,
        ),
        encoding="utf-8",
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
