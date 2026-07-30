from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file


PROJECT_ARTIFACTS = (
    "README.md",
    "PLAN.md",
    "configs/stage_n_v2_lmot_tracking_diagnostic_frozen_20260729.yaml",
    "data/STAGE_N_V2_LMOT_TRACKING_REPORT.md",
    "data/stage_n_v2/STAGE_N_V2_ACQUISITION_INVENTORY_20260729.yaml",
    "data/stage_n_v2/STAGE_N_V2_EXTRACTED_FILE_MANIFEST_20260729.json",
    "data/stage_n_v2/LMOT_CLASS_MAP_FROZEN_20260729.yaml",
    "data/stage_n_v2/LMOT_CLASS_MAPPING_EVIDENCE_20260729.yaml",
    "data/stage_n_v2/LMOT_CLASS_MAPPING_CONTACT_SHEET_20260729.jpg",
    "data/stage_n_v2/LMOT_CLASS_ID6_EXTENDED_CONTACT_SHEET_20260729.jpg",
    "data/stage_n_v2/LMOT_CLASS_ID6_EXTENDED_CONTACT_SHEET_20260729.json",
    "data/stage_n_v2/LMOT_CLASS_ID6_TRACK_TIMELINE_20260729.jpg",
    "src/parking_occupancy/stage_n_lmot_v2.py",
    "scripts/prepare_stage_n_lmot_v2.py",
    "scripts/render_stage_n_lmot_v2_mapping_evidence.py",
    "scripts/run_stage_n_lmot_v2.py",
    "scripts/verify_stage_n_lmot_v2_output.py",
    "scripts/freeze_stage_n_v2_artifacts.py",
    "tests/test_stage_n_lmot_v2.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the Stage N-v2 artifact registry"
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--verify", action="store_true")
    return parser.parse_args()


def _record(path: Path, *, label: str, group: str) -> dict[str, Any]:
    return {
        "label": label,
        "group": group,
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _verify_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors = []
    for record in payload["records"]:
        artifact = Path(record["path"])
        if not artifact.is_file():
            errors.append(f"missing:{record['label']}")
            continue
        if artifact.stat().st_size != int(record["bytes"]):
            errors.append(f"bytes:{record['label']}")
            continue
        if sha256_file(artifact) != record["sha256"]:
            errors.append(f"sha256:{record['label']}")
    if len(payload["records"]) != int(payload["record_count"]):
        errors.append("record_count")
    if errors:
        raise ValueError(f"Stage N-v2 registry verification failed: {errors}")
    return {
        "status": "passed",
        "record_count": len(payload["records"]),
        "registry_sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    registry = args.registry.resolve()
    if args.verify:
        print(json.dumps(_verify_registry(registry), indent=2))
        return
    if registry.exists():
        raise FileExistsError(f"Refusing to overwrite {registry}")

    records = []
    for relative in PROJECT_ARTIFACTS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(_record(path, label=relative, group="project"))
    for path in sorted(
        row for row in output_root.rglob("*") if row.is_file()
    ):
        relative = path.relative_to(output_root).as_posix()
        records.append(
            _record(path, label=f"output/{relative}", group="formal_output")
        )
    aggregate = json.loads(
        (output_root / "aggregate_metrics.json").read_text(encoding="utf-8")
    )
    payload = {
        "schema_version": 1,
        "protocol_id": "STAGE-N-V2-LMOT-TRACKING-DIAGNOSTIC-20260729-01",
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "complete_actual_lmot_validation",
        "claim_scope": aggregate["claim_scope"],
        "output_root": str(output_root),
        "record_count": len(records),
        "summary": {
            method: {
                "HOTA": aggregate["methods"][method]["tracking"]["HOTA"],
                "IDF1": aggregate["methods"][method]["tracking"]["IDF1"],
                "MOTA": aggregate["methods"][method]["tracking"]["MOTA"],
                "ID_switches": aggregate["methods"][method]["tracking"][
                    "ID_switches"
                ],
            }
            for method in ("L0", "L1", "L2", "L3")
        },
        "records": records,
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        yaml.safe_dump(
            payload, sort_keys=False, allow_unicode=True, width=120
        ),
        encoding="utf-8",
    )
    print(json.dumps(_verify_registry(registry), indent=2))


if __name__ == "__main__":
    main()
