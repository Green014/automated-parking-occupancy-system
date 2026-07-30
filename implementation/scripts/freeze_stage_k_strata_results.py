from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml

from parking_occupancy.detector_comparison import sha256_file
from parking_occupancy.stage_k_stratified_analysis import (
    PROTOCOL_ID,
    RECORD_ID,
)


SOURCE_ARTIFACTS = (
    (
        "protocol",
        "configs/stage_k_posthoc_stratified_analysis_frozen_20260728.yaml",
    ),
    (
        "stage_k_result_record",
        "data/comparisons/stage_k_p0_p1_p2_test_20260727.yaml",
    ),
)
EXTERNAL_ARTIFACTS = (
    (
        "P0_occupancy",
        "outputs/P0_P1_P2_stage_k_20260727_v1/P0/occupancy.csv",
    ),
    (
        "P1_occupancy",
        "outputs/P0_P1_P2_stage_k_20260727_v1/P1/occupancy.csv",
    ),
    (
        "P2_occupancy",
        "outputs/P0_P1_P2_stage_k_20260727_v1/P2/occupancy.csv",
    ),
    (
        "preflight",
        "outputs/stage_k_posthoc_strata_20260728_v1/preflight.json",
    ),
    (
        "analysis",
        "outputs/stage_k_posthoc_strata_20260728_v1/analysis.json",
    ),
    (
        "date_metrics",
        "outputs/stage_k_posthoc_strata_20260728_v1/date_metrics.csv",
    ),
    (
        "weather_metrics",
        "outputs/stage_k_posthoc_strata_20260728_v1/weather_metrics.csv",
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
    analysis = json.loads(
        (
            external_root
            / "outputs"
            / "stage_k_posthoc_strata_20260728_v1"
            / "analysis.json"
        ).read_text(encoding="utf-8")
    )
    record = {
        "schema_version": 1,
        "record_id": RECORD_ID,
        "protocol_id": PROTOCOL_ID,
        "status": "frozen_after_read_only_posthoc_analysis",
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "untouched_test_already_evaluated",
        "predictions_run": False,
        "parameters_selected": False,
        "detector_reselected": False,
        "D1_retrained": False,
        "artifact_count": len(artifacts),
        "summary": {
            "dates": sorted(analysis["metrics"]["P0"]["by_date"]),
            "weather": sorted(analysis["metrics"]["P0"]["by_weather"]),
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
