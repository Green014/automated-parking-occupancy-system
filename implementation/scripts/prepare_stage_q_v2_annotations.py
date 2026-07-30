from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file
from parking_occupancy.stage_q_v2_upm import (
    STAGE_Q_V2_PROTOCOL_ID,
    render_polygon_validation,
    validate_upm_slot_map,
    write_occupancy_truth_from_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Validate pre-model UPM polygons and prepare truth while keeping "
            "formal inference blocked pending human confirmation."
        )
    )
    parser.add_argument("--polygons", type=Path, required=True)
    parser.add_argument("--reference-image", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    payload = json.loads(args.polygons.read_text(encoding="utf-8"))
    if payload.get("model_predictions_viewed_before_annotation") is not False:
        raise ValueError("Polygon annotation must precede model predictions")
    slot_map = validate_upm_slot_map(payload)
    if sha256_file(args.reference_image) != payload["reference_image_sha256"]:
        raise ValueError("Polygon reference-image hash mismatch")
    validation_path = (
        output_root / "STAGE_Q_V2_POLYGON_VALIDATION_20260729.png"
    )
    render_polygon_validation(
        image_path=args.reference_image,
        slot_map=slot_map,
        output_path=validation_path,
    )
    truth_path = output_root / "STAGE_Q_V2_OCCUPANCY_TRUTH_20260729.csv"
    truth_summary = write_occupancy_truth_from_manifest(
        manifest_path=args.manifest,
        output_path=truth_path,
    )
    freeze = {
        "schema_version": 1,
        "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
        "status": "BLOCKED_PENDING_HUMAN_POLYGON_CONFIRMATION",
        "formal_inference_authorized": False,
        "polygon_confirmation": False,
        "model_loaded": False,
        "model_predictions_viewed": False,
        "slot_id_order": [f"slot_{index:02d}" for index in range(21)],
        "official_numbering_source": (
            "Sensors 2023 23(6) 3329 Figure 4(a)"
        ),
        "official_figure_url": (
            "https://pub.mdpi-res.com/sensors/sensors-23-03329/"
            "article_deploy/html/images/sensors-23-03329-g004.png"
        ),
        "official_figure_sha256": (
            "78c7a4332713df7ef3460bf039ed3d5ad0c63c14952127829ef05483bf942c71"
        ),
        "reference_image": str(args.reference_image.resolve()),
        "reference_image_sha256": sha256_file(args.reference_image),
        "polygon_path": str(args.polygons.resolve()),
        "polygon_sha256": sha256_file(args.polygons),
        "polygon_validation_path": str(validation_path),
        "polygon_validation_sha256": sha256_file(validation_path),
        "occupancy_truth_path": str(truth_path),
        "occupancy_truth_sha256": sha256_file(truth_path),
        "truth_summary": truth_summary,
        "source_mapping": {
            "source_1": "available_vacant",
            "source_0": "not_available_occupied",
            "project_occupied": 1,
            "project_vacant": 0,
            "unknown": "excluded",
        },
        "timestamp_s": "blank_no_source_timestamp",
        "seconds_level_transition_latency": "prohibited",
        "required_user_action": (
            "Inspect STAGE_Q_V2_POLYGON_VALIDATION_20260729.png and "
            "explicitly confirm that polygon labels 00-20 match the "
            "official Figure 4(a) numbering and physical parking spaces."
        ),
    }
    freeze_path = (
        output_root / "STAGE_Q_V2_ANNOTATION_FREEZE_20260729.yaml"
    )
    if freeze_path.exists():
        raise FileExistsError(f"Refusing to overwrite {freeze_path}")
    freeze_path.write_text(
        yaml.safe_dump(freeze, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(yaml.safe_dump(freeze, sort_keys=False, allow_unicode=True))


if __name__ == "__main__":
    main()
