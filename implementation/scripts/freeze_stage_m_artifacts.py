from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

import yaml

from parking_occupancy.stage_m_tracking import (
    load_stage_m_protocol,
    sha256_file,
    verify_ultralytics_runtime,
)


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_m_open_source_tracking_frozen_20260728.yaml"
)


def _resolve(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path.resolve()
    return (config_path.parent / path).resolve()


def _record(path: Path, kind: str) -> dict[str, Any]:
    path = path.resolve()
    try:
        display_path = path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        display_path = path.as_posix()
    return {
        "kind": kind,
        "path": display_path,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the non-overwriting Stage M artifact registry"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--smoke-output-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    protocol = load_stage_m_protocol(config_path, verify_files=True)
    runtime_audit = verify_ultralytics_runtime(protocol)
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite registry: {output}")
    smoke_root = args.smoke_output_root.resolve()
    if not smoke_root.is_dir():
        raise FileNotFoundError(f"Missing smoke output root: {smoke_root}")

    shared = protocol["shared_inference"]
    classifier = protocol["classifier"]
    smoke = protocol["smoke"]
    inputs = [
        _record(config_path, "stage_m_freeze"),
        _record(
            _resolve(config_path, shared["weights_path"]),
            "D1_weights",
        ),
        _record(
            _resolve(config_path, classifier["checkpoint_path"]),
            "E1b_checkpoint",
        ),
        _record(
            _resolve(
                config_path,
                protocol["trackers"]["bytetrack"]["config_path"],
            ),
            "ByteTrack_config",
        ),
        _record(
            _resolve(
                config_path,
                protocol["trackers"]["tracktrack"]["config_path"],
            ),
            "TrackTrack_config",
        ),
        _record(
            _resolve(config_path, smoke["source_image"]["path"]),
            "smoke_source",
        ),
        _record(
            _resolve(config_path, smoke["regions"]["path"]),
            "smoke_polygons",
        ),
        _record(
            PROJECT_ROOT
            / "data"
            / "stage_m"
            / "STAGE_M_DATA_GATES_20260728.yaml",
            "data_gate",
        ),
        _record(
            PROJECT_ROOT
            / "data"
            / "stage_m"
            / "STAGE_M_FEASIBILITY_ADDENDUM_20260728.md",
            "feasibility_addendum",
        ),
        _record(
            PROJECT_ROOT
            / "data"
            / "STAGE_M_OPEN_SOURCE_TRACKING_ROBUSTNESS_REPORT.md",
            "stage_m_report",
        ),
        _record(PROJECT_ROOT / "README.md", "documentation"),
        _record(PROJECT_ROOT / "PLAN.md", "documentation"),
        _record(
            PROJECT_ROOT
            / "literature_core"
            / "METHOD_PROVENANCE.md",
            "method_provenance",
        ),
    ]
    for key, record in runtime_audit["source_files"].items():
        inputs.append(_record(Path(record["path"]), f"runtime_source_{key}"))
    for key, record in runtime_audit["licenses"].items():
        inputs.append(_record(Path(record["path"]), f"runtime_license_{key}"))
    source_files = [
        PROJECT_ROOT
        / "src"
        / "parking_occupancy"
        / "stage_m_tracking.py",
        PROJECT_ROOT
        / "src"
        / "parking_occupancy"
        / "stage_m_evaluation.py",
        PROJECT_ROOT
        / "src"
        / "parking_occupancy"
        / "stage_m_data_gate.py",
        PROJECT_ROOT / "scripts" / "run_stage_m.py",
        PROJECT_ROOT / "scripts" / "run_stage_m_smoke.py",
        PROJECT_ROOT / "scripts" / "check_stage_m_data_gates.py",
        PROJECT_ROOT / "scripts" / "freeze_stage_m_artifacts.py",
        PROJECT_ROOT / "tests" / "test_stage_m_tracking.py",
        PROJECT_ROOT / "tests" / "test_stage_m_evaluation.py",
        PROJECT_ROOT / "tests" / "test_stage_m_data_gate.py",
    ]
    implementation = [_record(path, "implementation") for path in source_files]
    outputs = [
        _record(path, "smoke_output")
        for path in sorted(smoke_root.rglob("*"))
        if path.is_file() and "_ultralytics_config" not in path.parts
    ]
    stage_l = [
        _record(
            _resolve(config_path, item["path"]),
            f"preserved_stage_l_{key}",
        )
        for key, item in protocol["stage_l_preservation"].items()
    ]
    registry = {
        "schema_version": 1,
        "registry_id": "STAGE-M-ARTIFACT-REGISTRY-20260728-01",
        "recorded_at": "2026-07-28T20:00:00+08:00",
        "protocol_id": protocol["protocol_id"],
        "status": "smoke_executed_formal_experiments_gated",
        "stage_l_artifacts_modified": False,
        "claim_scope": "smoke_test_only",
        "truth": {
            "status": "absent_by_design",
            "reason": "repeated_consumed_development_image_interface_smoke",
            "accuracy_or_temporal_metrics": "not_computed_no_truth",
        },
        "runtime_versions": runtime_audit["package_versions"],
        "inputs": inputs,
        "implementation": implementation,
        "smoke_outputs": outputs,
        "preserved_stage_l_references": stage_l,
        "counts": {
            "inputs": len(inputs),
            "implementation": len(implementation),
            "smoke_outputs": len(outputs),
            "preserved_stage_l_references": len(stage_l),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        yaml.safe_dump(registry, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    print(
        yaml.safe_dump(
            {
                "status": "ok",
                "output": str(output),
                "counts": registry["counts"],
            },
            sort_keys=False,
        )
    )


if __name__ == "__main__":
    main()
