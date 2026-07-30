from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_q_v2_artifacts import (
    REQUIRED_LICENSE_BOUNDARY,
    artifact_record,
    verify_stage_q_v2_registry,
)
from parking_occupancy.stage_q_v2_evaluation import (
    load_frozen_stage_q_v2_config,
)
from parking_occupancy.stage_q_v2_upm import STAGE_Q_V2_PROTOCOL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Freeze and verify the completed Stage Q-v2 registry."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            PROJECT_ROOT
            / "configs"
            / "stage_q_v2_external_night_occupancy_frozen_20260729_v2.yaml"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            PROJECT_ROOT
            / "outputs"
            / "stage_q_v2_upm_gti_external_20260729_v2"
        ),
    )
    parser.add_argument(
        "--registry",
        type=Path,
        default=(
            PROJECT_ROOT
            / "data"
            / "stage_q_v2"
            / "STAGE_Q_V2_ARTIFACT_REGISTRY_20260729.yaml"
        ),
    )
    return parser.parse_args()


def _files(paths: Iterable[Path]) -> list[Path]:
    return sorted(
        (path.resolve() for path in paths if path.is_file()),
        key=lambda path: str(path).casefold(),
    )


def _role(path: Path, output_root: Path) -> str:
    if output_root in path.parents:
        return "completed_formal_output"
    failed_root = PROJECT_ROOT / (
        "outputs/stage_q_v2_upm_gti_external_20260729_v1"
    )
    if failed_root.resolve() in path.parents:
        return "preserved_failed_pre_prediction_output"
    external_root = PROJECT_ROOT / "data/external"
    if external_root.resolve() in path.parents:
        return "external_input_not_for_submission_or_redistribution"
    if path.suffix.lower() in {".pt", ".pth"}:
        return "frozen_model_input"
    if "tests" in path.parts:
        return "verification_test"
    if "configs" in path.parts:
        return "frozen_config_or_additive_runtime_correction"
    if path.suffix.lower() in {".py"}:
        return "implementation_or_verification_code"
    return "audit_report_manifest_or_annotation"


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = load_frozen_stage_q_v2_config(config_path)
    output_root = args.output_root.resolve()
    registry_path = args.registry.resolve()
    if registry_path.exists():
        raise FileExistsError(f"Refusing to overwrite {registry_path}")
    if not output_root.is_dir():
        raise FileNotFoundError(output_root)
    configured_output = (
        config_path.parent / config["formal_runs"]["output_root"]
    ).resolve()
    if output_root != configured_output:
        raise ValueError("Registry output root differs from frozen config")

    failed_root = (
        PROJECT_ROOT
        / "outputs"
        / "stage_q_v2_upm_gti_external_20260729_v1"
    )
    external_root = (
        PROJECT_ROOT / "data/external/stage_q_upm_gti_20260729"
    )
    explicit = [
        PROJECT_ROOT
        / "configs/p3_integrated_runtime_defaults_20260729.yaml",
        PROJECT_ROOT
        / "configs/stage_q_v2_external_night_occupancy_frozen_20260729.yaml",
        config_path,
        PROJECT_ROOT / "data/STAGE_Q_V2_UPM_GTI_EXTERNAL_EVALUATION_REPORT.md",
        PROJECT_ROOT / "literature_core/METHOD_PROVENANCE.md",
        PROJECT_ROOT / "literature_core/RESULTS.md",
        PROJECT_ROOT / "src/parking_occupancy/integrated_runner.py",
        PROJECT_ROOT / "src/parking_occupancy/stage_q_v2_artifacts.py",
        PROJECT_ROOT / "src/parking_occupancy/stage_q_v2_evaluation.py",
        PROJECT_ROOT / "src/parking_occupancy/stage_q_v2_upm.py",
        PROJECT_ROOT / "scripts/freeze_stage_q_v2_artifacts.py",
        PROJECT_ROOT / "scripts/freeze_stage_q_v2_night_gate.py",
        PROJECT_ROOT / "scripts/prepare_stage_q_v2_annotations.py",
        PROJECT_ROOT / "scripts/run_stage_q_v2_data_audit.py",
        PROJECT_ROOT / "scripts/run_stage_q_v2_formal.py",
        PROJECT_ROOT / "scripts/verify_stage_q_v2_premodel.py",
        PROJECT_ROOT / "tests/test_stage_q_v2_artifacts.py",
        PROJECT_ROOT / "tests/test_stage_q_v2_evaluation.py",
        PROJECT_ROOT / "tests/test_stage_q_v2_upm.py",
        external_root / "source/test.zip",
        external_root / "reference/sensors-23-03329-g004.png",
        external_root
        / "extracted/test/gopro32/images/106GOPRO-GOPR1524.JPG",
        Path(config["models"]["D1"]["path"]),
        (
            config_path.parent / config["models"]["D1_LL"]["path"]
        ).resolve(),
        Path(config["models"]["E1b"]["path"]),
    ]
    data_files = [
        path
        for path in (PROJECT_ROOT / "data/stage_q_v2").iterdir()
        if path.is_file() and path.resolve() != registry_path
    ]
    all_paths = _files(
        [
            *explicit,
            *data_files,
            *failed_root.rglob("*"),
            *output_root.rglob("*"),
        ]
    )
    unique: list[Path] = []
    seen: set[str] = set()
    for path in all_paths:
        key = str(path).casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(path)
    missing_explicit = [
        str(path)
        for path in explicit
        if not path.resolve().is_file()
    ]
    if missing_explicit:
        raise FileNotFoundError(
            f"Missing required Stage Q-v2 artifacts: {missing_explicit}"
        )

    records = [
        artifact_record(
            label=f"stage_q_v2:{index:04d}:{path.name}",
            path=path,
            role=_role(path, output_root),
        )
        for index, path in enumerate(unique, start=1)
    ]
    registry = {
        "schema_version": 1,
        "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
        "registry_id": "STAGE-Q-V2-ARTIFACT-REGISTRY-20260729-01",
        "status": "FORMAL_RUNS_COMPLETE_AND_HASH_VERIFIED",
        "created_on": "2026-07-29",
        "artifact_count": len(records),
        "license_boundary": REQUIRED_LICENSE_BOUNDARY,
        "formal_output_root": str(output_root),
        "failed_attempt_output_root": str(failed_root.resolve()),
        "raw_dataset_submission_policy": "excluded",
        "manifest_binds_selected_raw_images": True,
        "archive_sha256": (
            "92d61d8f87fe3e7068d8c42ce8dc2c415c08071c92eeddfd4d47260e8922efdc"
        ),
        "formal_config_sha256": (
            next(
                record["sha256"]
                for record in records
                if Path(record["path"]) == config_path
            )
        ),
        "completed_method_runs": {
            "QV2-0": 1,
            "QV2-1": 1,
        },
        "model_track_called": False,
        "seconds_level_transition_latency_computed": False,
        "D1_remains_project_default": True,
        "D1_LL_role": "secondary_frozen_comparison",
        "stage_p2_decision_remains": "FAIL",
        "artifacts": records,
    }
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    with registry_path.open("x", encoding="utf-8") as handle:
        yaml.safe_dump(
            registry,
            handle,
            sort_keys=False,
            allow_unicode=True,
            width=120,
        )
    verification = verify_stage_q_v2_registry(registry_path)
    if not verification["verified"]:
        raise RuntimeError(
            f"Stage Q-v2 registry verification failed: {verification}"
        )
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
