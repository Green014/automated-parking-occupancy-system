from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_o_artifacts import (
    artifact_record,
    verify_artifact_records,
)
from parking_occupancy.stage_o_low_light import STAGE_O_PROTOCOL_ID


SOURCE_FILES = (
    "src/parking_occupancy/stage_o_low_light.py",
    "src/parking_occupancy/stage_o_training.py",
    "src/parking_occupancy/stage_o_enhancement.py",
    "src/parking_occupancy/stage_o_artifacts.py",
    "scripts/run_stage_o_detector_eval.py",
    "scripts/select_stage_o_o1.py",
    "scripts/prepare_stage_o_training.py",
    "scripts/run_stage_o_training.py",
    "scripts/record_stage_o_o2_blocked.py",
    "scripts/run_stage_o_p3_interface_smoke.py",
    "scripts/select_stage_o_results.py",
    "scripts/freeze_stage_o_artifacts.py",
    "scripts/verify_stage_o_frozen_history.py",
    "scripts/verify_stage_o_outputs.py",
    "tests/test_stage_o_low_light.py",
    "tests/test_stage_o_training.py",
    "tests/test_stage_o_artifacts.py",
)

CONTROL_FILES = (
    "configs/stage_o_low_light_adaptation_frozen_20260729.yaml",
    "data/stage_o/STAGE_O_PROTOCOL_20260729.md",
    "data/stage_o/STAGE_O_DATA_MANIFEST_20260729.yaml",
    "data/stage_o/STAGE_O_O2_RUNTIME_DECISION_20260729.yaml",
    "data/stage_o/STAGE_O_SELECTION_20260729.json",
    "data/STAGE_O_LOW_LIGHT_ADAPTATION_REPORT.md",
    "literature_core/METHOD_PROVENANCE.md",
    "literature_core/RESULTS.md",
)

TRAINING_CONTROL_NAMES = {
    "training_manifest.yaml",
    "dataset.yaml",
    "dataset_smoke.yaml",
    "dataset_runtime_path_correction_20260729.yaml",
    "dataset_smoke_runtime_path_correction_20260729.yaml",
    "dataset_smoke_runtime_path_correction_v2_20260729.yaml",
    "train.txt",
    "val.txt",
    "smoke_train.txt",
    "smoke_val.txt",
    "train_runtime_v2.txt",
    "val_runtime_v2.txt",
    "smoke_train_runtime_v2.txt",
    "smoke_val_runtime_v2.txt",
}


def _iter_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if "_ultralytics_config" in path.parts or "__pycache__" in path.parts:
            continue
        yield path


def _append_unique(
    records: list[dict],
    seen: set[Path],
    *,
    label: str,
    path: Path,
    role: str,
) -> None:
    path = path.resolve()
    if path in seen:
        return
    records.append(artifact_record(label=label, path=path, role=role))
    seen.add(path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a non-overwriting Stage O artifact registry."
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--d1-weights", type=Path, required=True)
    parser.add_argument("--e1b-checkpoint", type=Path, required=True)
    parser.add_argument("--retinex-repository", type=Path, required=True)
    parser.add_argument("--retinex-weights", type=Path, required=True)
    parser.add_argument("--training-root", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    records: list[dict] = []
    seen: set[Path] = set()

    for relative in (*SOURCE_FILES, *CONTROL_FILES):
        _append_unique(
            records,
            seen,
            label=relative,
            path=PROJECT_ROOT / relative,
            role=(
                "implementation"
                if relative in SOURCE_FILES
                else "protocol_report_or_provenance"
            ),
        )

    class_map = (
        PROJECT_ROOT
        / "data"
        / "stage_n_v2"
        / "LMOT_CLASS_MAP_FROZEN_20260729.yaml"
    )
    _append_unique(
        records,
        seen,
        label="LMOT_class_map",
        path=class_map,
        role="truth_policy_input",
    )
    validation_root = args.validation_root.resolve()
    for sequence in ("LMOT-05", "LMOT-13", "LMOT-14", "LMOT-25"):
        gt = validation_root / sequence / "gt" / "gt.txt"
        _append_unique(
            records,
            seen,
            label=f"{sequence}_GT",
            path=gt,
            role="formal_detector_only_truth_input",
        )
    for label, path, role in (
        ("D1_best", args.d1_weights, "model_input"),
        ("E1b_best", args.e1b_checkpoint, "P3_smoke_model_input"),
        (
            "Retinexformer_LICENSE",
            args.retinex_repository.resolve() / "LICENSE.txt",
            "O2_license",
        ),
        ("Retinexformer_LOL_v2_real", args.retinex_weights, "O2_model_input"),
    ):
        _append_unique(
            records,
            seen,
            label=label,
            path=path,
            role=role,
        )

    training_root = args.training_root.resolve()
    for path in _iter_files(training_root):
        if path.name not in TRAINING_CONTROL_NAMES:
            continue
        _append_unique(
            records,
            seen,
            label=f"training_control:{path.relative_to(training_root)}",
            path=path,
            role="O3_training_manifest_or_runtime_path_correction",
        )

    output_root = PROJECT_ROOT / "outputs"
    for root in sorted(output_root.glob("stage_o_*")):
        if not root.is_dir():
            continue
        for path in _iter_files(root):
            _append_unique(
                records,
                seen,
                label=f"output:{path.relative_to(output_root)}",
                path=path,
                role="stage_o_output_or_retained_attempt_evidence",
            )
    data_root = PROJECT_ROOT / "data" / "stage_o"
    for path in _iter_files(data_root):
        if path.resolve() == output:
            continue
        _append_unique(
            records,
            seen,
            label=f"stage_o_data:{path.relative_to(data_root)}",
            path=path,
            role="stage_o_control_log_or_verification",
        )

    verification = verify_artifact_records(records)
    if not verification["verified"]:
        raise RuntimeError(verification)
    registry = {
        "schema_version": 1,
        "registry_id": "STAGE-O-ARTIFACT-REGISTRY-20260729-01",
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "status": "complete_additive_stage_o_record",
        "historical_stage_l_m_n_modified": False,
        "raw_detector_only": True,
        "tracker_emitted_boxes": False,
        "parking_occupancy_claim": False,
        "artifact_count": len(records),
        "artifacts": records,
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
                "artifact_count": len(records),
                "output": str(output),
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )


if __name__ == "__main__":
    main()
