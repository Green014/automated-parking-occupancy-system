from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from parking_occupancy.detector_comparison import sha256_file
from parking_occupancy.p1_temporal_case import (
    P1_TEMPORAL_PROTOCOL_ID,
    P1_TEMPORAL_RECORD_ID,
)


SOURCE_ARTIFACTS = (
    (
        "protocol",
        "configs/p1_b1_virat_0502_continuous_case_frozen_20260727.yaml",
    ),
    (
        "truth",
        "literature_core/data/annotations/virat_0502_departure_truth.yaml",
    ),
    (
        "failed_attempt_audit",
        (
            "data/comparisons/"
            "p1_b1_virat0502_continuous_failed_attempt_20260727_v1.yaml"
        ),
    ),
)
EXTERNAL_ARTIFACTS = (
    (
        "preflight_only",
        "outputs/p1_b1_virat0502_continuous_preflight_20260727_v1.json",
    ),
    (
        "failed_preflight",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v1/"
            "preflight.json"
        ),
    ),
    (
        "failed_empty_detections",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v1/"
            "detections.jsonl"
        ),
    ),
    (
        "failed_unplayable_video_header",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v1/"
            "annotated.mp4"
        ),
    ),
    (
        "run_preflight",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v2/"
            "preflight.json"
        ),
    ),
    (
        "annotated_video",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v2/"
            "annotated.mp4"
        ),
    ),
    (
        "occupancy",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v2/"
            "occupancy.csv"
        ),
    ),
    (
        "events",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v2/"
            "events.csv"
        ),
    ),
    (
        "detections",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v2/"
            "detections.jsonl"
        ),
    ),
    (
        "metrics",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v2/"
            "metrics.json"
        ),
    ),
    (
        "summary",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v2/"
            "summary.json"
        ),
    ),
    (
        "runtime",
        (
            "outputs/p1_b1_virat0502_continuous_20260727_v2/"
            "runtime_metadata.json"
        ),
    ),
    (
        "qualitative_key_frames",
        "outputs/p1_b1_virat0502_continuous_qa_20260727_v2.png",
    ),
)


def _artifact(root_name: str, root: Path, role: str, relative: str) -> dict:
    path = root / relative
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "role": role,
        "root": root_name,
        "path": relative,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite: {args.output}")
    source_root = args.source_root.resolve()
    external_root = args.external_root.resolve()
    artifacts = [
        _artifact("source", source_root, role, relative)
        for role, relative in SOURCE_ARTIFACTS
    ]
    artifacts.extend(
        _artifact("external", external_root, role, relative)
        for role, relative in EXTERNAL_ARTIFACTS
    )
    summary_path = (
        external_root
        / "outputs"
        / "p1_b1_virat0502_continuous_20260727_v2"
        / "summary.json"
    )
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    record = {
        "schema_version": 1,
        "record_id": P1_TEMPORAL_RECORD_ID,
        "protocol_id": P1_TEMPORAL_PROTOCOL_ID,
        "status": "frozen_after_single_effective_execution",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "consumed_development_case_study",
        "untouched_claim": False,
        "real_continuous_video": True,
        "static_montage": False,
        "effective_prediction_runs": 1,
        "failed_pre_prediction_attempt_retained": True,
        "parameters_selected_from_result": False,
        "D1_retrained": False,
        "tracking_branch_run": False,
        "negative_results_retained": True,
        "artifact_count": len(artifacts),
        "summary": {
            "metrics": summary["metrics"],
            "runtime": summary["runtime"],
        },
        "artifacts": artifacts,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
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
