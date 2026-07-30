from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from parking_occupancy.detector_comparison import sha256_file
from parking_occupancy.stage_k_data_gate import (
    STAGE_K_CANDIDATE_ID,
    STAGE_K_GATE_V2_RECORD_ID,
)


SOURCE_ARTIFACTS = (
    (
        "superseded_local_inventory_gate_v1",
        "data/comparisons/stage_k_slot_occupancy_data_gate_20260727.yaml",
    ),
    (
        "candidate_audit",
        "data/preprocessing/pklot_stage_k_candidate_audit_20260727_v2.json",
    ),
    (
        "manual_visual_review",
        (
            "data/preprocessing/"
            "pklot_stage_k_manual_visual_review_20260727_v2.json"
        ),
    ),
    (
        "membership_manifest",
        "data/manifests/pklot_stage_k_candidate_20260727_v2.csv",
    ),
    (
        "annotations",
        "data/annotations/pklot_stage_k_candidate_20260727_v2.jsonl",
    ),
    (
        "prior_development_manifest",
        "data/manifests/stage_j_pklot_development_20260727.csv",
    ),
    (
        "frozen_test_protocol",
        "configs/stage_k_p0_p1_p2_pklot_test_frozen_20260727.yaml",
    ),
    (
        "frozen_test_result_record",
        "data/comparisons/stage_k_p0_p1_p2_test_20260727.yaml",
    ),
)
EXTERNAL_ARTIFACTS = (
    (
        "candidate_contact_sheet",
        "outputs/stage_k_pklot_candidate_contact_sheet_20260727_v2.png",
    ),
    (
        "truth_contact_sheet",
        "outputs/stage_k_pklot_truth_contact_sheet_20260727_v2.png",
    ),
    (
        "pre_prediction_preflight",
        "outputs/P0_P1_P2_stage_k_20260727_preflight_v1.json",
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


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def build_record(
    *,
    source_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    source_root = source_root.resolve()
    external_root = external_root.resolve()
    audit = _read_json(
        source_root
        / "data"
        / "preprocessing"
        / "pklot_stage_k_candidate_audit_20260727_v2.json"
    )
    review = _read_json(
        source_root
        / "data"
        / "preprocessing"
        / "pklot_stage_k_manual_visual_review_20260727_v2.json"
    )
    protocol = yaml.safe_load(
        (
            source_root
            / "configs"
            / "stage_k_p0_p1_p2_pklot_test_frozen_20260727.yaml"
        ).read_text(encoding="utf-8")
    )
    result = yaml.safe_load(
        (
            source_root
            / "data"
            / "comparisons"
            / "stage_k_p0_p1_p2_test_20260727.yaml"
        ).read_text(encoding="utf-8")
    )
    preflight = _read_json(
        external_root
        / "outputs"
        / "P0_P1_P2_stage_k_20260727_preflight_v1.json"
    )

    manifest_path = (
        source_root
        / "data"
        / "manifests"
        / "pklot_stage_k_candidate_20260727_v2.csv"
    )
    prior_manifest_path = (
        source_root
        / "data"
        / "manifests"
        / "stage_j_pklot_development_20260727.csv"
    )
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with prior_manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        prior_rows = list(csv.DictReader(handle))

    selected_hashes = {row["image_sha256"] for row in rows}
    prior_hashes = {row["image_sha256"] for row in prior_rows}
    totals = {
        "images": len(rows),
        "known_slots": sum(int(row["known_slots"]) for row in rows),
        "occupied": sum(int(row["occupied"]) for row in rows),
        "vacant": sum(int(row["vacant"]) for row in rows),
        "unknown": sum(int(row["unknown"]) for row in rows),
    }
    groups = sorted(
        {
            (row["source"], row["date"], row["weather"])
            for row in rows
        }
    )

    if (
        audit.get("candidate_id") != STAGE_K_CANDIDATE_ID
        or audit.get("prediction_count") != 0
        or audit["overlap"]["prior_image_sha256_overlap"] != 0
        or audit.get("gate")
        != "pending_manual_visual_review_and_protocol_freeze"
    ):
        raise ValueError("Candidate audit does not preserve the pre-run gate")
    if (
        review.get("candidate_id") != STAGE_K_CANDIDATE_ID
        or review.get("decision")
        != "pass_for_protocol_freeze_before_predictions"
        or review["selection_disclosure"]["model_predictions_or_outputs_viewed"]
        is not False
        or review["selection_disclosure"][
            "threshold_or_mapping_selection_from_candidate"
        ]
        is not False
    ):
        raise ValueError("Manual review does not establish a clean data gate")
    if (
        protocol.get("status") != "frozen_before_predictions"
        or protocol["gates"]["stage_K_data_gate"]
        != "passed_before_predictions"
        or protocol["scope"]["parameter_selection_from_this_run"]
        != "prohibited"
    ):
        raise ValueError("Stage K protocol was not frozen behind the gate")
    if (
        result.get("status") != "frozen_after_single_test_execution"
        or result.get("predictions_before_protocol_freeze") is not False
        or result.get("parameters_selected_from_test") is not False
        or result.get("detector_reselected_from_test") is not False
    ):
        raise ValueError("Stage K result violates the frozen test boundary")
    if (
        preflight.get("execution_gate") != "open"
        or preflight.get("predictions_run") is not False
        or preflight.get("parameters_selected_from_test") is not False
    ):
        raise ValueError("Stage K preflight is not prediction-free")
    if (
        totals
        != {
            "images": 90,
            "known_slots": 5034,
            "occupied": 1943,
            "vacant": 3091,
            "unknown": 6,
        }
        or len(selected_hashes) != 90
        or selected_hashes.intersection(prior_hashes)
        or len(groups) != 3
    ):
        raise ValueError("Stage K manifest does not satisfy the v2 data gate")

    artifacts = [
        _artifact("source", source_root, role, relative)
        for role, relative in SOURCE_ARTIFACTS
    ]
    artifacts.extend(
        _artifact("external", external_root, role, relative)
        for role, relative in EXTERNAL_ARTIFACTS
    )
    return {
        "schema_version": 2,
        "record_id": STAGE_K_GATE_V2_RECORD_ID,
        "gate_id": "STAGE-K-PKLOT-DATA-GATE-20260728-02",
        "status": "closed_passed_before_predictions",
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "supersedes": {
            "gate_id": "STAGE-K-PKLOT-DATA-GATE-20260727-01",
            "reason": (
                "The v1 local-inventory blocker was valid when written; "
                "additional complete PKLot JPG/XML pairs were later recovered "
                "from the partial official archive."
            ),
        },
        "decision_basis": "pre_prediction_evidence_only",
        "recorded_after_single_execution": True,
        "data_role": "untouched_test",
        "candidate_id": STAGE_K_CANDIDATE_ID,
        "protocol_id": protocol["protocol_id"],
        "result_record_id": result["record_id"],
        "selection_disclosure": {
            "camera_date_groups_selected_using_truth_distribution": True,
            "within_group_membership_selected_using_truth": False,
            "within_group_rule": (
                "timestamp-sorted evenly spaced sampling, 30 images per group"
            ),
            "model_outputs_viewed_before_freeze": False,
            "parameters_selected_from_candidate": False,
        },
        "summary": {
            **totals,
            "unknown_excluded_from_metrics": totals["unknown"],
            "cameras": sorted({group[0] for group in groups}),
            "dates": sorted({group[1] for group in groups}),
            "weather": sorted({group[2] for group in groups}),
            "camera_date_weather_groups": [
                {
                    "camera": camera,
                    "date": date,
                    "weather": weather,
                    "images": sum(
                        row["source"] == camera
                        and row["date"] == date
                        and row["weather"] == weather
                        for row in rows
                    ),
                }
                for camera, date, weather in groups
            ],
            "selected_unique_image_hashes": len(selected_hashes),
            "prior_development_image_sha256_overlap": 0,
        },
        "source_boundary": {
            "dataset": "PKLot",
            "license": protocol["source"]["license"],
            "official_archive_url": protocol["source"]["official_archive_url"],
            "local_archive_complete": False,
            "eligible_members": (
                "Only complete, individually hashed JPG/XML pairs before the "
                "truncated archive boundary."
            ),
        },
        "chronology": {
            "candidate_prepared_without_predictions": True,
            "manual_review_passed_before_protocol_freeze": True,
            "protocol_frozen_before_predictions": True,
            "preflight_opened_gate_without_predictions": True,
            "single_test_execution_frozen_afterward": True,
        },
        "restrictions": {
            "test_parameter_selection": False,
            "post_result_detector_reselection": False,
            "D1_retraining": False,
            "temporal_claim_from_sampled_images": False,
        },
        "limitations": [
            (
                "The local official archive is truncated; only complete pairs "
                "before the stream boundary were eligible."
            ),
            (
                "Each camera contributes one date, so camera and date effects "
                "are confounded."
            ),
            (
                "Cloudy weather occurs only for UFPR04, so weather and camera "
                "effects are also confounded."
            ),
        ],
        "artifact_count": len(artifacts),
        "artifacts": artifacts,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze the additive Stage K data-gate v2 record."
    )
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite: {output}")
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
