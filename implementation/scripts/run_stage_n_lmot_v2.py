from __future__ import annotations

import argparse
import json
import platform
import shutil
import statistics
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
SCRIPT_ROOT = PROJECT_ROOT / "scripts"
for path in (PROJECT_SRC, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from parking_occupancy.stage_n_lmot import (
    OfficialTrackEvalAdapter,
    StageNInferenceSettings,
    audit_lmot_sequence,
    parse_lmot_gt,
    sha256_file,
    split_motor_vehicle_truth,
)
from parking_occupancy.stage_n_lmot_v2 import (
    EXPECTED_VALIDATION_SEQUENCES,
    load_lmot_class_map_v2,
    load_stage_n_v2_protocol,
    verify_file_manifest,
)
from run_stage_n_lmot import _aggregate, _run_one


DEFAULT_CONFIG = (
    PROJECT_ROOT
    / "configs"
    / "stage_n_v2_lmot_tracking_diagnostic_frozen_20260729.yaml"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run frozen Stage N-v2 LMOT L0-L3 diagnostics"
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="0")
    return parser.parse_args()


def _verify_artifact(
    config_path: Path, record: dict[str, Any]
) -> Path:
    path = (config_path.parent / record["path"]).resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    if path.stat().st_size != int(record["bytes"]):
        raise ValueError(f"Input artifact byte mismatch: {path}")
    if sha256_file(path) != record["sha256"]:
        raise ValueError(f"Input artifact SHA-256 mismatch: {path}")
    return path


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    protocol = load_stage_n_v2_protocol(
        config_path, verify_preserved_files=True
    )
    output_root = args.output_root.resolve()
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite {output_root}")

    artifacts = protocol["verified_input_artifacts"]
    inventory_path = _verify_artifact(
        config_path, artifacts["acquisition_inventory"]
    )
    manifest_path = _verify_artifact(
        config_path, artifacts["extracted_file_manifest"]
    )
    class_map_path = _verify_artifact(config_path, artifacts["class_map"])
    validation_root = Path(artifacts["validation_root"]).resolve()
    verified_manifest = verify_file_manifest(
        manifest_path, expected_root=validation_root
    )
    manifest_record = artifacts["extracted_file_manifest"]
    if (
        verified_manifest["file_count"] != int(manifest_record["file_count"])
        or verified_manifest["total_bytes"]
        != int(manifest_record["total_bytes"])
        or verified_manifest["manifest_sha256"]
        != manifest_record["canonical_records_sha256"]
    ):
        raise ValueError("Extracted LMOT manifest differs from frozen config")

    mapping = load_lmot_class_map_v2(class_map_path)
    shared = protocol["shared_inference"]
    weights = Path(shared["weights_path"]).resolve()
    if weights.stat().st_size != int(shared["weights_bytes"]):
        raise ValueError("D1 weights byte count mismatch")
    if sha256_file(weights) != shared["weights_sha256"]:
        raise ValueError("D1 weights SHA-256 mismatch")
    settings = StageNInferenceSettings(
        weights=str(weights), device=args.device
    )
    tracker_paths = {
        "ByteTrack": (
            config_path.parent
            / protocol["trackers"]["bytetrack"]["config_path"]
        ).resolve(),
        "TrackTrack": (
            config_path.parent
            / protocol["trackers"]["tracktrack"]["config_path"]
        ).resolve(),
    }
    for name, path in tracker_paths.items():
        record = protocol["trackers"][name.lower()]
        if path.stat().st_size != int(record["config_bytes"]):
            raise ValueError(f"{name} config byte count mismatch")
        if sha256_file(path) != record["config_sha256"]:
            raise ValueError(f"{name} config SHA-256 mismatch")

    sequences = [
        validation_root / name for name in EXPECTED_VALIDATION_SEQUENCES
    ]
    observed = sorted(
        path.name for path in validation_root.iterdir() if path.is_dir()
    )
    if observed != sorted(EXPECTED_VALIDATION_SEQUENCES):
        raise ValueError(f"Unexpected validation sequences: {observed}")
    sequence_audits = {}
    for sequence in sequences:
        audit = audit_lmot_sequence(sequence)
        if not audit["passed"]:
            raise ValueError(f"Sequence audit failed: {sequence.name}")
        sequence_audits[sequence.name] = audit

    output_root.mkdir(parents=True)
    shutil.copyfile(config_path, output_root / "configuration_snapshot.yaml")
    shutil.copyfile(class_map_path, output_root / "class_map_snapshot.yaml")
    sequence_metrics: dict[str, dict[str, dict[str, Any]]] = {
        method_id: {} for method_id in protocol["methods"]
    }
    trackeval_inputs: dict[str, dict[str, tuple[int, Any, Any]]] = {
        method_id: {} for method_id in protocol["methods"]
    }
    for sequence in sequences:
        annotations = parse_lmot_gt(sequence / "gt" / "gt.txt")
        evaluated_gt, suppression_gt = split_motor_vehicle_truth(
            annotations,
            class_map=mapping,
            evaluated_ignore_values=mapping.evaluated_mark_values,
        )
        for method_id, method in protocol["methods"].items():
            metrics, tracks, _detections = _run_one(
                method_id=method_id,
                sequence_root=sequence,
                image_directory=method["image_directory"],
                tracker_path=tracker_paths[method["tracker"]],
                settings=settings,
                evaluated_gt=evaluated_gt,
                suppression_gt=suppression_gt,
                output_root=output_root,
            )
            sequence_metrics[method_id][sequence.name] = metrics
            trackeval_inputs[method_id][sequence.name] = (
                int(metrics["runtime"]["frames"]),
                evaluated_gt,
                tracks,
            )

    official_tracking_aggregates = {}
    for method_id, method_inputs in trackeval_inputs.items():
        _per_sequence, official_tracking_aggregates[method_id] = (
            OfficialTrackEvalAdapter().evaluate_many(method_inputs)
        )
    aggregate = _aggregate(
        sequence_metrics, official_tracking_aggregates
    )
    aggregate["claim_scope"] = protocol["claim_scope"]["allowed"]
    aggregate["parameter_tuning_from_lmot"] = False
    aggregate["ground_truth_policy"] = {
        "evaluated": "car + motorcycle + bus + truck",
        "suppression_only": "person + bicycle",
        "active_mark_values": sorted(mapping.evaluated_mark_values),
    }
    (output_root / "sequence_metrics.json").write_text(
        json.dumps(sequence_metrics, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "aggregate_metrics.json").write_text(
        json.dumps(aggregate, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (output_root / "sequence_audits.json").write_text(
        json.dumps(sequence_audits, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    runtime_rows = [
        metrics["runtime"]
        for methods in sequence_metrics.values()
        for metrics in methods.values()
    ]
    runtime = {
        "claim_scope": protocol["claim_scope"]["allowed"],
        "python": platform.python_version(),
        "trackeval": OfficialTrackEvalAdapter.runtime_metadata(),
        "config_sha256": sha256_file(config_path),
        "class_map_sha256": sha256_file(class_map_path),
        "acquisition_inventory_sha256": sha256_file(inventory_path),
        "extracted_file_manifest_sha256": sha256_file(manifest_path),
        "extracted_records_sha256": verified_manifest["manifest_sha256"],
        "D1_weights_sha256": sha256_file(weights),
        "actual_lmot_run": True,
        "parameter_tuning_from_results": False,
        "sequence_method_runs": len(runtime_rows),
        "processed_frames": sum(row["frames"] for row in runtime_rows),
        "mean_steady_state_latency_ms_across_runs": statistics.fmean(
            row["steady_state_latency_ms"] for row in runtime_rows
        ),
    }
    (output_root / "runtime_metadata.json").write_text(
        json.dumps(runtime, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
