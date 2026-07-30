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
from parking_occupancy.stage_n_v3_correction import (
    STAGE_N_V3_CORRECTION_ID,
)


PROJECT_ARTIFACTS = (
    "src/parking_occupancy/stage_n_lmot.py",
    "scripts/run_stage_n_lmot.py",
    "src/parking_occupancy/stage_n_v3_correction.py",
    "scripts/recompute_stage_n_v3_emitted_box_metrics.py",
    "scripts/freeze_stage_n_v3_correction.py",
    "tests/test_stage_n_lmot.py",
    "tests/test_stage_n_v3_correction.py",
    "data/STAGE_N_V3_EMITTED_BOX_CORRECTION_REPORT.md",
)
OUTPUT_ARTIFACTS = (
    "emitted_box_metrics.json",
    "input_manifest.json",
    "runtime_metadata.json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze or verify the additive Stage N-v3 correction"
    )
    parser.add_argument("--output-root", type=Path)
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


def verify_registry(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    if payload.get("correction_id") != STAGE_N_V3_CORRECTION_ID:
        errors.append("correction_id")
    records = payload.get("records", [])
    if len(records) != int(payload.get("record_count", -1)):
        errors.append("record_count")
    for record in records:
        artifact = Path(record["path"])
        if not artifact.is_file():
            errors.append(f"missing:{record['label']}")
        elif artifact.stat().st_size != int(record["bytes"]):
            errors.append(f"bytes:{record['label']}")
        elif sha256_file(artifact) != record["sha256"]:
            errors.append(f"sha256:{record['label']}")
    if errors:
        raise ValueError(f"Stage N-v3 registry verification failed: {errors}")
    return {
        "status": "passed",
        "record_count": len(records),
        "registry_sha256": sha256_file(path),
    }


def main() -> None:
    args = parse_args()
    registry = args.registry.resolve()
    if args.verify:
        print(json.dumps(verify_registry(registry), indent=2))
        return
    if args.output_root is None:
        raise ValueError("--output-root is required unless --verify is used")
    if registry.exists():
        raise FileExistsError(f"Refusing to overwrite {registry}")
    output_root = args.output_root.resolve()
    input_manifest_path = output_root / "input_manifest.json"
    input_manifest = json.loads(
        input_manifest_path.read_text(encoding="utf-8")
    )

    records: list[dict[str, Any]] = []
    for source_record in input_manifest["inputs"]:
        path = Path(source_record["path"])
        if path.stat().st_size != int(source_record["bytes"]):
            raise ValueError(f"Input bytes changed: {path}")
        if sha256_file(path) != source_record["sha256"]:
            raise ValueError(f"Input SHA-256 changed: {path}")
        records.append(
            _record(
                path,
                label=f"input/{source_record['role']}/{path.name}",
                group="offline_input",
            )
        )
    for relative in PROJECT_ARTIFACTS:
        path = PROJECT_ROOT / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(_record(path, label=relative, group="project"))
    for name in OUTPUT_ARTIFACTS:
        path = output_root / name
        if not path.is_file():
            raise FileNotFoundError(path)
        records.append(
            _record(path, label=f"output/{name}", group="correction_output")
        )

    metrics = json.loads(
        (output_root / "emitted_box_metrics.json").read_text(
            encoding="utf-8"
        )
    )
    payload = {
        "schema_version": 1,
        "correction_id": STAGE_N_V3_CORRECTION_ID,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "complete_offline_emitted_box_metric_correction",
        "preserves_stage_n_v2": True,
        "inference_performed": False,
        "model_loaded": False,
        "model_track_called": False,
        "trackeval_called": False,
        "metric_scope": metrics["metric_scope"],
        "primary_table_definition": metrics["primary_table_definition"],
        "output_root": str(output_root),
        "record_count": len(records),
        "summary": {
            method: values["aggregate"]
            for method, values in metrics["methods"].items()
        },
        "records": records,
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        ),
        encoding="utf-8",
    )
    print(json.dumps(verify_registry(registry), indent=2))


if __name__ == "__main__":
    main()
