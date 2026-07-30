from __future__ import annotations

import argparse
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
    manifest_fingerprint,
)
from parking_occupancy.stage_q_v2_upm import (
    NIGHT_AUXILIARY_LUMINANCE_THRESHOLD,
    STAGE_Q_V2_PROTOCOL_ID,
    build_night_test_manifest,
    inspect_test_split,
    write_night_test_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze the pre-model UPM-GTI night test data gate."
    )
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--archive-audit", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    return parser.parse_args()


def _write_yaml(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    test_root = args.test_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    inventory_path = (
        output_root / "STAGE_Q_V2_SEQUENCE_INVENTORY_20260729.csv"
    )
    if not inventory_path.is_file():
        raise FileNotFoundError(inventory_path)
    if not args.archive_audit.is_file() or not args.contact_sheet.is_file():
        raise FileNotFoundError("Archive audit and contact sheet are required")

    inventories = inspect_test_split(test_root)
    manifest_rows, decisions = build_night_test_manifest(
        test_root,
        inventories,
    )
    qualified = [
        row["sequence_id"] for row in decisions if row["qualified"]
    ]
    status = "PASS" if qualified and manifest_rows else (
        "BLOCKED_NO_QUALIFYING_NIGHT_TEST"
    )
    manifest_path = (
        output_root / "STAGE_Q_V2_TEST_IMAGE_MANIFEST_20260729.csv"
    )
    if status == "PASS":
        write_night_test_manifest(manifest_path, manifest_rows)
        records = [
            ManifestRecord(
                relative_path=row.relative_path,
                bytes=row.bytes,
                sha256=row.sha256,
            )
            for row in manifest_rows
        ]
        logical_manifest_sha256 = manifest_fingerprint(records)
        manifest_file_sha256 = sha256_file(manifest_path)
    else:
        logical_manifest_sha256 = None
        manifest_file_sha256 = None

    selected_transitions = sum(
        int(row["selected_transition_frames"])
        for row in decisions
        if row["qualified"]
    )
    selected_slot_changes = sum(
        int(row["selected_slot_state_changes"])
        for row in decisions
        if row["qualified"]
    )
    gate = {
        "schema_version": 1,
        "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
        "gate_id": "STAGE-Q-V2-NIGHT-TEST-GATE-20260729-01",
        "status": status,
        "formal_inference_authorized": False,
        "next_gate": (
            "BLOCKED_PENDING_HUMAN_POLYGON_CONFIRMATION"
            if status == "PASS"
            else "PROJECT_CLOSURE"
        ),
        "selection_before_model_inference": True,
        "model_loaded": False,
        "model_predictions_viewed": False,
        "source_split": "official_test",
        "real_parking_lot": True,
        "previous_project_use": False,
        "night_definition": {
            "primary_evidence": (
                "visual_review_of_fixed_pre_model_sequence_contact_sheet"
            ),
            "auxiliary_rule": "mean_grayscale_luminance_at_or_below_threshold",
            "threshold": NIGHT_AUXILIARY_LUMINANCE_THRESHOLD,
            "threshold_frozen_before_full_per_image_selection": True,
            "model_output_used": False,
        },
        "camera_geometry": {
            "resolution": "800x600",
            "contact_sheet_review": (
                "stable_shared_fixed_view_across_qualified_sequences"
            ),
            "contact_sheet_sha256": sha256_file(args.contact_sheet),
        },
        "truth": {
            "vector_length": 21,
            "binary_only": True,
            "source_1": "available_vacant",
            "source_0": "not_available_occupied",
            "unknown_policy": "excluded",
        },
        "qualified_sequences": qualified,
        "qualified_sequence_count": len(qualified),
        "selected_image_count": len(manifest_rows),
        "selected_transition_frames": selected_transitions,
        "selected_slot_state_changes": selected_slot_changes,
        "timestamp_available": False,
        "reliable_fps_available": False,
        "seconds_level_transition_latency_prohibited": True,
        "manifest_path": str(manifest_path) if status == "PASS" else None,
        "manifest_file_sha256": manifest_file_sha256,
        "logical_manifest_sha256": logical_manifest_sha256,
        "archive_audit_sha256": sha256_file(args.archive_audit),
        "raw_sequence_exception": {
            "sequence_id": "gopro10",
            "images_without_truth": 32,
            "policy": "exclude_entire_sequence_from_formal_test",
        },
    }
    selection = {
        "schema_version": 1,
        "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
        "status": status,
        "deterministic_rule": (
            "all official-test sequences with complete raw image/truth "
            "bijection, shared stable 800x600 geometry, at least one "
            "truth-labelled image at frozen luminance <=70, and both "
            "occupied and vacant labels in the selected low-light frames"
        ),
        "selection_not_based_on_model_results": True,
        "qualified_sequences": qualified,
        "sequence_decisions": decisions,
        "polygon_policy": (
            "one shared geometry may be used only after official slot-order "
            "mapping is drawn and human-confirmed before inference"
        ),
        "formal_inference_authorized": False,
    }
    _write_yaml(
        output_root / "STAGE_Q_V2_NIGHT_TEST_GATE_20260729.yaml",
        gate,
    )
    _write_yaml(
        output_root / "STAGE_Q_V2_TEST_SCENE_SELECTION_20260729.yaml",
        selection,
    )
    print(
        yaml.safe_dump(
            {
                "status": status,
                "qualified_sequence_count": len(qualified),
                "qualified_sequences": qualified,
                "selected_image_count": len(manifest_rows),
                "selected_transition_frames": selected_transitions,
                "selected_slot_state_changes": selected_slot_changes,
                "logical_manifest_sha256": logical_manifest_sha256,
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
