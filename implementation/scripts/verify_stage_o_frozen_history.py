from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file
from parking_occupancy.stage_o_low_light import STAGE_O_PROTOCOL_ID


PATH_MARKERS = (
    "configs/",
    "data/",
    "outputs/",
    "src/",
    "scripts/",
    "tests/",
    "literature_core/",
    "README.md",
    "PLAN.md",
)


def _candidate_paths(
    raw_path: str,
    *,
    project_root: Path,
    external_root: Path,
) -> list[Path]:
    candidates = [Path(raw_path)]
    normalized = raw_path.replace("\\", "/")
    if not Path(raw_path).is_absolute():
        candidates.extend(
            (project_root / normalized, external_root / normalized)
        )
    for marker in PATH_MARKERS:
        index = normalized.find(marker)
        if index < 0:
            continue
        relative = normalized[index:]
        candidates.extend(
            (project_root / relative, external_root / relative)
        )
    unique: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved not in unique:
            unique.append(resolved)
    return unique


def _verify_record(
    record: Mapping[str, Any],
    *,
    project_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    expected_bytes = int(record["bytes"])
    expected_hash = str(record["sha256"])
    candidates = _candidate_paths(
        str(record["path"]),
        project_root=project_root,
        external_root=external_root,
    )
    for path in candidates:
        if not path.is_file() or path.stat().st_size != expected_bytes:
            continue
        actual_hash = sha256_file(path)
        if actual_hash == expected_hash:
            return {
                "label": str(
                    record.get("label", record.get("kind", record["path"]))
                ),
                "path": str(path),
                "bytes": expected_bytes,
                "sha256": expected_hash,
                "passed": True,
            }
    return {
        "label": str(
            record.get("label", record.get("kind", record["path"]))
        ),
        "stored_path": str(record["path"]),
        "expected_bytes": expected_bytes,
        "expected_sha256": expected_hash,
        "candidate_paths": [str(path) for path in candidates],
        "passed": False,
    }


def _verify_many(
    records: Iterable[Mapping[str, Any]],
    *,
    project_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    checks = [
        _verify_record(
            record,
            project_root=project_root,
            external_root=external_root,
        )
        for record in records
    ]
    return {
        "artifact_count": len(checks),
        "passed_count": sum(bool(row["passed"]) for row in checks),
        "passed": all(bool(row["passed"]) for row in checks),
        "checks": checks,
    }


def _read_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify frozen Stage L, M, and N-v2 evidence for Stage O without "
            "rewriting historical registries or outputs."
        )
    )
    parser.add_argument("--stage-l-record", type=Path, required=True)
    parser.add_argument("--stage-m-registry", type=Path, required=True)
    parser.add_argument("--stage-n-v2-registry", type=Path, required=True)
    parser.add_argument("--external-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    external_root = args.external_root.resolve()
    l_path = args.stage_l_record.resolve()
    m_path = args.stage_m_registry.resolve()
    n_path = args.stage_n_v2_registry.resolve()
    stage_l = _read_yaml(l_path)
    stage_m = _read_yaml(m_path)
    stage_n = _read_yaml(n_path)

    stage_m_records = [
        *[
            row
            for row in stage_m["inputs"]
            if row.get("kind")
            not in {"documentation", "method_provenance"}
        ],
        *stage_m["implementation"],
        *stage_m["smoke_outputs"],
        *stage_m["preserved_stage_l_references"],
    ]
    stage_n_records = [
        row
        for row in stage_n["records"]
        if (
            row["group"] == "formal_output"
            or str(row["label"]).startswith("configs/")
            or str(row["label"]).startswith("data/STAGE_N_V2_")
            or str(row["label"]).startswith("data/stage_n_v2/")
        )
    ]
    result = {
        "schema_version": 1,
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "operation": "read_only_historical_hash_verification",
        "historical_files_modified": False,
        "registries": {
            "Stage_L": {
                "path": str(l_path),
                "sha256": sha256_file(l_path),
            },
            "Stage_M": {
                "path": str(m_path),
                "sha256": sha256_file(m_path),
            },
            "Stage_N_v2": {
                "path": str(n_path),
                "sha256": sha256_file(n_path),
            },
        },
        "Stage_L": _verify_many(
            stage_l["artifacts"],
            project_root=PROJECT_ROOT,
            external_root=external_root,
        ),
        "Stage_M": _verify_many(
            stage_m_records,
            project_root=PROJECT_ROOT,
            external_root=external_root,
        ),
        "Stage_N_v2_frozen_control_and_outputs": _verify_many(
            stage_n_records,
            project_root=PROJECT_ROOT,
            external_root=external_root,
        ),
    }
    result["passed"] = all(
        result[key]["passed"]
        for key in (
            "Stage_L",
            "Stage_M",
            "Stage_N_v2_frozen_control_and_outputs",
        )
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "passed": result["passed"],
                "Stage_L": result["Stage_L"]["passed_count"],
                "Stage_M": result["Stage_M"]["passed_count"],
                "Stage_N_v2": result[
                    "Stage_N_v2_frozen_control_and_outputs"
                ]["passed_count"],
                "output": str(output),
            },
            indent=2,
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
