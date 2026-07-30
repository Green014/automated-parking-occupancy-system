from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file
from parking_occupancy.stage_q_external import (
    ManifestRecord,
    verify_manifest_records,
)
from parking_occupancy.stage_q_v2_artifacts import (
    validate_annotation_freeze,
    validate_source_archive_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify all Stage Q-v2 pre-model bindings without loading a model."
        )
    )
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--artifact-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.artifact_root.resolve()
    source_audit = validate_source_archive_audit(
        root / "STAGE_Q_V2_SOURCE_ARCHIVE_AUDIT_20260729.yaml"
    )
    freeze = validate_annotation_freeze(
        root / "STAGE_Q_V2_ANNOTATION_FREEZE_20260729.yaml"
    )
    gate_path = root / "STAGE_Q_V2_NIGHT_TEST_GATE_20260729.yaml"
    gate = yaml.safe_load(gate_path.read_text(encoding="utf-8"))
    manifest_path = root / "STAGE_Q_V2_TEST_IMAGE_MANIFEST_20260729.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    records = [
        ManifestRecord(
            relative_path=row["relative_path"],
            bytes=int(row["bytes"]),
            sha256=row["sha256"],
        )
        for row in manifest_rows
    ]
    manifest_result = verify_manifest_records(
        args.test_root,
        records,
        expected_manifest_sha256=gate["logical_manifest_sha256"],
    )
    if sha256_file(manifest_path) != gate["manifest_file_sha256"]:
        raise RuntimeError("Manifest CSV SHA-256 mismatch")

    truth_path = root / "STAGE_Q_V2_OCCUPANCY_TRUTH_20260729.csv"
    with truth_path.open("r", encoding="utf-8", newline="") as handle:
        truth_rows = list(csv.DictReader(handle))
    expected_truth_rows = len(manifest_rows) * 21
    if len(truth_rows) != expected_truth_rows:
        raise RuntimeError(
            f"Truth row count mismatch: {len(truth_rows)} != "
            f"{expected_truth_rows}"
        )
    if sha256_file(truth_path) != freeze["occupancy_truth_sha256"]:
        raise RuntimeError("Occupancy truth SHA-256 mismatch")
    if sha256_file(
        root / "STAGE_Q_V2_SLOT_POLYGONS_20260729.json"
    ) != freeze["polygon_sha256"]:
        raise RuntimeError("Polygon SHA-256 mismatch")
    if sha256_file(
        root / "STAGE_Q_V2_POLYGON_VALIDATION_20260729.png"
    ) != freeze["polygon_validation_sha256"]:
        raise RuntimeError("Polygon validation SHA-256 mismatch")

    print(
        json.dumps(
            {
                "status": freeze["status"],
                "archive_bytes": source_audit["archive"]["archive_bytes"],
                "archive_sha256": source_audit["archive"]["archive_sha256"],
                "manifest": manifest_result,
                "truth_rows": len(truth_rows),
                "polygon_confirmation": False,
                "formal_inference_authorized": False,
                "model_loaded": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
