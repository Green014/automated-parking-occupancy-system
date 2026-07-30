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

from parking_occupancy.stage_n_lmot import (
    VerifiedLmotClassMap,
    audit_lmot_sequence,
    parse_lmot_gt,
    sha256_file,
    split_motor_vehicle_truth,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert gated LMOT validation GT to unified motor_vehicle"
    )
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    return parser.parse_args()


def _mot_line(row) -> str:
    return (
        f"{row.frame_number},{row.track_id},{row.x:.6f},{row.y:.6f},"
        f"{row.width:.6f},{row.height:.6f},1,1,{row.visibility:.6f}"
    )


def main() -> None:
    args = parse_args()
    validation_root = args.validation_root.resolve()
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")
    mapping_payload = yaml.safe_load(
        args.class_map.read_text(encoding="utf-8")
    )
    mapping = VerifiedLmotClassMap(
        id_to_name={
            int(key): value
            for key, value in mapping_payload["id_to_name"].items()
        },
        verification_status=mapping_payload["verification_status"],
        evidence=mapping_payload["evidence"],
        evidence_sha256=mapping_payload.get("evidence_sha256"),
    )
    if mapping.verification_status != "official_verified":
        raise ValueError("Synthetic class maps cannot convert LMOT")
    evaluated_values = frozenset(
        int(value) for value in mapping_payload["evaluated_ignore_values"]
    )
    sequence_roots = sorted(
        path
        for path in validation_root.iterdir()
        if path.is_dir()
    )
    if not sequence_roots:
        raise ValueError("No LMOT validation sequences found")
    output_root.mkdir(parents=True)
    manifest = {
        "schema_version": 1,
        "class_map": {
            "path": str(args.class_map.resolve()),
            "sha256": sha256_file(args.class_map),
        },
        "sequences": {},
    }
    for sequence_root in sequence_roots:
        audit = audit_lmot_sequence(sequence_root)
        if not audit["passed"]:
            raise ValueError(
                f"LMOT sequence failed audit: {sequence_root.name}"
            )
        rows = parse_lmot_gt(sequence_root / "gt" / "gt.txt")
        evaluated, suppression = split_motor_vehicle_truth(
            rows,
            class_map=mapping,
            evaluated_ignore_values=evaluated_values,
        )
        destination = output_root / sequence_root.name
        destination.mkdir()
        evaluated_path = destination / "gt_motor_vehicle.txt"
        suppression_path = destination / "gt_prediction_suppression.txt"
        evaluated_path.write_text(
            "\n".join(_mot_line(row) for row in evaluated) + "\n",
            encoding="utf-8",
        )
        suppression_path.write_text(
            "\n".join(_mot_line(row) for row in suppression) + "\n",
            encoding="utf-8",
        )
        audit_path = destination / "sequence_audit.json"
        audit_path.write_text(
            json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        manifest["sequences"][sequence_root.name] = {
            "source_gt_sha256": sha256_file(sequence_root / "gt" / "gt.txt"),
            "evaluated_boxes": len(evaluated),
            "suppression_boxes": len(suppression),
            "gt_motor_vehicle_sha256": sha256_file(evaluated_path),
            "gt_prediction_suppression_sha256": sha256_file(suppression_path),
            "sequence_audit_sha256": sha256_file(audit_path),
        }
    manifest_path = output_root / "conversion_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
