from __future__ import annotations

import csv
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .detector_comparison import DetectorSpec, sha256_file
from .evaluate import binary_metrics
from .image_io import read_image
from .stage_j_occupancy import (
    PIPELINE_METHOD_IDS,
    _detector_specs,
    _load_annotations,
    _run_method,
)
from .stage_j_posthoc_analysis import (
    CAMERA_METRIC_FIELDS,
    equal_group_macro,
    grouped_metrics,
    load_occupancy_rows,
    paired_bootstrap_mean_difference,
    per_sample_metrics,
)


STAGE_K_PROTOCOL_ID = "P-COMP-PKLOT-TEST-STAGEK-20260727-01"
STAGE_K_RECORD_ID = "P-COMP-PKLOT-TEST-STAGEK-RECORD-20260727-01"


class StageKProtocolError(ValueError):
    """Raised when Stage K inputs differ from the frozen test protocol."""


def _resolve_from_config(config_path: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (config_path.parent / path).resolve()


def _verified_binding(
    *,
    config_path: Path,
    binding: dict[str, Any],
    label: str,
) -> Path:
    path = _resolve_from_config(config_path, str(binding["path"]))
    if not path.is_file():
        raise StageKProtocolError(f"Missing {label}: {path}")
    if path.stat().st_size != int(binding["bytes"]):
        raise StageKProtocolError(f"{label} byte size mismatch")
    if sha256_file(path) != str(binding["sha256"]):
        raise StageKProtocolError(f"{label} SHA-256 mismatch")
    return path


def load_stage_k_protocol(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_K_PROTOCOL_ID:
        raise StageKProtocolError("Unexpected Stage K protocol ID")
    if payload.get("status") != "frozen_before_predictions":
        raise StageKProtocolError("Stage K protocol is not frozen")
    scope = payload.get("scope", {})
    if (
        scope.get("data_role") != "untouched_test"
        or scope.get("parameter_selection_from_this_run") != "prohibited"
        or scope.get("post_result_reselection") != "prohibited"
    ):
        raise StageKProtocolError("Invalid Stage K test boundary")
    if tuple(payload.get("methods", {})) != PIPELINE_METHOD_IDS:
        raise StageKProtocolError("Stage K methods must be P0/P1/P2")
    if [
        payload["methods"][method_id]["detector_id"]
        for method_id in PIPELINE_METHOD_IDS
    ] != ["D0", "D1", "D2"]:
        raise StageKProtocolError("Invalid Stage K detector mapping")
    common = payload.get("common_inference", {})
    if (
        common.get("agnostic_nms") is not True
        or int(common.get("max_detections", -1)) != 300
        or int(common.get("imgsz", -1)) != 640
        or common.get("threshold_source")
        != "Stage_I_v2_consumed_development_calibration"
    ):
        raise StageKProtocolError("Invalid Stage K inference settings")
    mapping = payload.get("common_mapping", {})
    if (
        mapping.get("algorithm") != "slot_polygon_coverage"
        or mapping.get("one_to_one") is not True
        or float(mapping.get("minimum_slot_coverage", -1)) != 0.40
        or mapping.get("temporal_stabilization") is not False
    ):
        raise StageKProtocolError("Invalid Stage K B1 mapping")
    if payload.get("gates", {}).get("stage_K_data_gate") != (
        "passed_before_predictions"
    ):
        raise StageKProtocolError("Stage K data gate is not open")
    if payload["data"]["manual_visual_review"]["decision"] != (
        "pass_for_protocol_freeze_before_predictions"
    ):
        raise StageKProtocolError("Stage K visual review did not pass")
    for key, label in (
        ("annotations", "annotations"),
        ("membership_manifest", "membership manifest"),
        ("candidate_audit", "candidate audit"),
        ("manual_visual_review", "manual visual review"),
        ("prior_development_manifest", "prior development manifest"),
    ):
        _verified_binding(
            config_path=config_path,
            binding=payload["data"][key],
            label=f"Stage K {label}",
        )
    return payload


def stage_k_preflight(
    *,
    config_path: Path,
    source_root: Path,
    weight_paths: dict[str, Path],
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, DetectorSpec]]:
    config_path = config_path.resolve()
    source_root = source_root.resolve()
    protocol = load_stage_k_protocol(config_path)
    annotations_path = _verified_binding(
        config_path=config_path,
        binding=protocol["data"]["annotations"],
        label="Stage K annotations",
    )
    manifest_path = _verified_binding(
        config_path=config_path,
        binding=protocol["data"]["membership_manifest"],
        label="Stage K membership manifest",
    )
    prior_manifest_path = _verified_binding(
        config_path=config_path,
        binding=protocol["data"]["prior_development_manifest"],
        label="Stage K prior development manifest",
    )
    records = _load_annotations(annotations_path)
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        manifest = list(csv.DictReader(handle))
    with prior_manifest_path.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        prior_rows = list(csv.DictReader(handle))
    expected = protocol["data"]["expected"]
    if len(records) != int(expected["images"]) or len(manifest) != len(records):
        raise StageKProtocolError("Stage K membership count mismatch")
    if [row["sample_id"] for row in manifest] != [
        str(record["sample_id"]) for record in records
    ]:
        raise StageKProtocolError("Stage K annotation/manifest order mismatch")

    prior_hashes = {str(row["image_sha256"]) for row in prior_rows}
    selected_hashes: set[str] = set()
    totals = {
        "slot_labels_known": 0,
        "occupied": 0,
        "vacant": 0,
        "unknown_excluded": 0,
    }
    image_checks = []
    for row, record in zip(manifest, records, strict=True):
        for key in ("source", "date", "weather", "local_path"):
            if str(row[key]) != str(record[key]):
                raise StageKProtocolError(
                    f"Stage K manifest mismatch for {row['sample_id']}:{key}"
                )
        if row["role"] != "stage_k_candidate_no_predictions":
            raise StageKProtocolError("Unexpected Stage K preparation role")
        image_path = source_root / row["local_path"]
        if not image_path.is_file():
            raise StageKProtocolError(f"Missing Stage K image: {image_path}")
        image_hash = sha256_file(image_path)
        if (
            image_path.stat().st_size != int(row["image_bytes"])
            or image_hash != row["image_sha256"]
        ):
            raise StageKProtocolError(
                f"Stage K image binding mismatch: {row['sample_id']}"
            )
        if image_hash in selected_hashes:
            raise StageKProtocolError("Duplicate Stage K image SHA-256")
        selected_hashes.add(image_hash)
        if image_hash in prior_hashes:
            raise StageKProtocolError("Stage K overlaps development images")
        image = read_image(image_path)
        if image.shape[:2] != (
            int(row["height"]),
            int(row["width"]),
        ):
            raise StageKProtocolError(
                f"Unexpected Stage K image size: {row['sample_id']}"
            )
        totals["slot_labels_known"] += int(row["known_slots"])
        totals["occupied"] += int(row["occupied"])
        totals["vacant"] += int(row["vacant"])
        totals["unknown_excluded"] += int(row["unknown"])
        image_checks.append(
            {
                "sample_id": row["sample_id"],
                "sha256": image_hash,
                "verified": True,
            }
        )
    if totals != {
        key: int(expected[key])
        for key in (
            "slot_labels_known",
            "occupied",
            "vacant",
            "unknown_excluded",
        )
    }:
        raise StageKProtocolError("Stage K slot-label totals mismatch")
    if (
        len(selected_hashes)
        != int(expected["selected_image_sha256_unique"])
        or selected_hashes.intersection(prior_hashes)
    ):
        raise StageKProtocolError("Stage K image isolation check failed")

    specs = _detector_specs(protocol)
    model_checks = {}
    for pipeline_id in PIPELINE_METHOD_IDS:
        path = weight_paths[pipeline_id].resolve()
        actual_hash = sha256_file(path) if path.is_file() else None
        if actual_hash != specs[pipeline_id].weights_sha256:
            raise StageKProtocolError(
                f"{pipeline_id} weights SHA-256 mismatch"
            )
        model_checks[pipeline_id] = {
            "detector_id": specs[pipeline_id].method_id,
            "weights_name": path.name,
            "weights_sha256": actual_hash,
            "ready": True,
        }
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "data_role": "untouched_test",
        "images": len(records),
        **totals,
        "selected_unique_image_hashes": len(selected_hashes),
        "prior_development_image_sha256_overlap": 0,
        "models": model_checks,
        "image_bindings_verified": len(image_checks),
        "manual_visual_review": "pass",
        "parameters_selected_from_test": False,
        "predictions_run": False,
        "execution_gate": "open",
    }
    return report, records, specs


def _classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise StageKProtocolError("Cannot score an empty Stage K group")
    return binary_metrics(
        [int(row["truth"]) for row in rows],
        [int(row["prediction"]) for row in rows],
    )


def _validate_cross_method_rows(
    rows_by_method: dict[str, list[dict[str, Any]]],
) -> None:
    reference = {
        (row["sample_id"], row["slot_id"]): (
            row["camera"],
            row["date"],
            row["weather"],
            row["truth"],
        )
        for row in rows_by_method["P0"]
    }
    for method_id in ("P1", "P2"):
        candidate = {
            (row["sample_id"], row["slot_id"]): (
                row["camera"],
                row["date"],
                row["weather"],
                row["truth"],
            )
            for row in rows_by_method[method_id]
        }
        if candidate != reference:
            raise StageKProtocolError(
                f"{method_id} occupancy membership/truth differs from P0"
            )


def _paired_comparison(
    *,
    candidate_id: str,
    metrics_by_method: dict[str, dict[str, Any]],
    per_sample_by_method: dict[str, dict[str, dict[str, Any]]],
    sample_metadata: dict[str, dict[str, str]],
    seed: int,
    resamples: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_samples = per_sample_by_method["P0"]
    candidate_samples = per_sample_by_method[candidate_id]
    if set(candidate_samples) != set(baseline_samples):
        raise StageKProtocolError("Paired Stage K sample membership differs")
    differences = {}
    counts = {"win": 0, "tie": 0, "loss": 0}
    rows = []
    for sample_id in sorted(baseline_samples):
        baseline_score = float(baseline_samples[sample_id]["macro_f1"])
        candidate_score = float(candidate_samples[sample_id]["macro_f1"])
        difference = candidate_score - baseline_score
        differences[sample_id] = difference
        outcome = (
            "win"
            if difference > 1e-12
            else "loss"
            if difference < -1e-12
            else "tie"
        )
        counts[outcome] += 1
        rows.append(
            {
                "comparison": f"{candidate_id}-P0",
                "sample_id": sample_id,
                **sample_metadata[sample_id],
                "baseline_macro_f1": baseline_score,
                "candidate_macro_f1": candidate_score,
                "difference": difference,
                "outcome": outcome,
            }
        )
    return (
        {
            "baseline": "P0",
            "candidate": candidate_id,
            "difference_definition": f"{candidate_id}_minus_P0",
            "pooled_macro_f1_difference": (
                float(metrics_by_method[candidate_id]["overall"]["macro_f1"])
                - float(metrics_by_method["P0"]["overall"]["macro_f1"])
            ),
            "camera_macro_f1_difference": (
                float(
                    metrics_by_method[candidate_id]["camera_macro"][
                        "macro_f1"
                    ]
                )
                - float(metrics_by_method["P0"]["camera_macro"]["macro_f1"])
            ),
            "per_camera_macro_f1_difference": {
                camera: (
                    float(
                        metrics_by_method[candidate_id]["by_camera"][camera][
                            "macro_f1"
                        ]
                    )
                    - float(
                        metrics_by_method["P0"]["by_camera"][camera][
                            "macro_f1"
                        ]
                    )
                )
                for camera in metrics_by_method["P0"]["by_camera"]
            },
            "win_tie_loss": counts,
            "paired_bootstrap": paired_bootstrap_mean_difference(
                differences,
                seed=seed,
                resamples=resamples,
                confidence_level=0.95,
            ),
        },
        rows,
    )


def _layered_comparison(
    *,
    protocol: dict[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    expected = protocol["data"]["expected"]
    rows_by_method = {}
    unknown_by_method = {}
    for method_id in PIPELINE_METHOD_IDS:
        all_rows, known_rows, unknown = load_occupancy_rows(
            output_root / method_id / "occupancy.csv"
        )
        if (
            len(all_rows)
            != int(expected["slot_labels_known"])
            + int(expected["unknown_excluded"])
            or len(known_rows) != int(expected["slot_labels_known"])
            or unknown != int(expected["unknown_excluded"])
        ):
            raise StageKProtocolError(
                f"{method_id} Stage K occupancy row totals differ"
            )
        rows_by_method[method_id] = known_rows
        unknown_by_method[method_id] = unknown
    _validate_cross_method_rows(rows_by_method)

    metrics_by_method = {}
    per_sample_by_method = {}
    sample_metadata: dict[str, dict[str, str]] = {}
    for method_id in PIPELINE_METHOD_IDS:
        rows = rows_by_method[method_id]
        by_camera = grouped_metrics(rows, "camera")
        metrics_by_method[method_id] = {
            "overall": _classification(rows),
            "by_camera": by_camera,
            "camera_macro": equal_group_macro(by_camera),
        }
        per_sample_by_method[method_id] = per_sample_metrics(rows)
        if method_id == "P0":
            for row in rows:
                sample_metadata.setdefault(
                    row["sample_id"],
                    {
                        "camera": row["camera"],
                        "date": row["date"],
                        "weather": row["weather"],
                    },
                )

    bootstrap = protocol["metric_contract"]["paired_per_image_differences"]
    paired = {}
    paired_rows = []
    for candidate_id in ("P1", "P2"):
        result, rows = _paired_comparison(
            candidate_id=candidate_id,
            metrics_by_method=metrics_by_method,
            per_sample_by_method=per_sample_by_method,
            sample_metadata=sample_metadata,
            seed=int(bootstrap["bootstrap_seed"]),
            resamples=int(bootstrap["bootstrap_iterations"]),
        )
        paired[f"{candidate_id}-P0"] = result
        paired_rows.extend(rows)

    with (output_root / "camera_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fields = ["method_id", "camera", "samples", *CAMERA_METRIC_FIELDS]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for method_id in PIPELINE_METHOD_IDS:
            for camera, metrics in metrics_by_method[method_id][
                "by_camera"
            ].items():
                writer.writerow(
                    {
                        "method_id": method_id,
                        "camera": camera,
                        "samples": metrics["samples"],
                        **{
                            field: f"{float(metrics[field]):.12f}"
                            for field in CAMERA_METRIC_FIELDS
                        },
                    }
                )
            macro = metrics_by_method[method_id]["camera_macro"]
            writer.writerow(
                {
                    "method_id": method_id,
                    "camera": "CAMERA_MACRO",
                    "samples": "",
                    **{
                        field: f"{float(macro[field]):.12f}"
                        for field in CAMERA_METRIC_FIELDS
                    },
                }
            )
    with (output_root / "paired_image_differences.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fields = [
            "comparison",
            "sample_id",
            "camera",
            "date",
            "weather",
            "baseline_macro_f1",
            "candidate_macro_f1",
            "difference",
            "outcome",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(paired_rows)
    return {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "untouched_test",
        "parameters_selected_from_test": False,
        "detector_reselected": False,
        "unknown_rows_excluded_per_method": unknown_by_method,
        "metrics": metrics_by_method,
        "paired_comparisons": paired,
        "claim_boundary": (
            "This test isolates detector choice under the frozen B1 mapping. "
            "Slot occupancy metrics are not detector mAP."
        ),
    }


def run_stage_k_comparison(
    *,
    config_path: Path,
    source_root: Path,
    output_root: Path,
    weight_paths: dict[str, Path],
    device: str,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite Stage K output: {output_root}"
        )
    preflight, records, specs = stage_k_preflight(
        config_path=config_path,
        source_root=source_root,
        weight_paths=weight_paths,
    )
    protocol = load_stage_k_protocol(config_path)
    output_root.mkdir(parents=True)
    ultralytics_config = output_root / "_ultralytics_config"
    ultralytics_config.mkdir()
    os.environ["YOLO_CONFIG_DIR"] = str(ultralytics_config.resolve())
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )
    method_reports = {}
    for pipeline_id in PIPELINE_METHOD_IDS:
        method_reports[pipeline_id] = _run_method(
            pipeline_id=pipeline_id,
            protocol=protocol,
            records=records,
            source_root=source_root.resolve(),
            output_root=output_root,
            spec=specs[pipeline_id],
            weights_path=weight_paths[pipeline_id],
            device=device,
            dataset_role="untouched_test",
            video_id="pklot_stage_k_test",
            montage_label="NON-CONTIGUOUS TEST MONTAGE",
            claim_boundary=(
                "Stage K uses previously unused PKLot camera/date groups "
                "frozen before P0/P1/P2 predictions."
            ),
        )
    comparison = _layered_comparison(
        protocol=protocol,
        output_root=output_root,
    )
    comparison["method_outputs"] = method_reports
    comparison["selected_detector_before_slot_evaluation"] = "D1"
    comparison["pipeline_reselection_from_this_result"] = False
    comparison["negative_results_retained"] = True
    (output_root / "comparison.json").write_text(
        json.dumps(comparison, indent=2) + "\n",
        encoding="utf-8",
    )
    return comparison


def verify_stage_k_record(
    *,
    record_path: Path,
    source_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    if record.get("record_id") != STAGE_K_RECORD_ID:
        raise StageKProtocolError("Unexpected Stage K result record ID")
    roots = {
        "source": source_root.resolve(),
        "external": external_root.resolve(),
    }
    checks = []
    for artifact in record["artifacts"]:
        root_name = str(artifact["root"])
        if root_name not in roots:
            raise StageKProtocolError(
                f"Unexpected Stage K artifact root: {root_name}"
            )
        path = roots[root_name] / str(artifact["path"])
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha256 = sha256_file(path) if path.is_file() else None
        passed = (
            actual_bytes == int(artifact["bytes"])
            and actual_sha256 == str(artifact["sha256"])
        )
        checks.append(
            {
                "role": artifact["role"],
                "root": root_name,
                "path": artifact["path"],
                "expected_bytes": artifact["bytes"],
                "actual_bytes": actual_bytes,
                "expected_sha256": artifact["sha256"],
                "actual_sha256": actual_sha256,
                "passed": passed,
            }
        )
    return {
        "schema_version": 1,
        "record_id": record["record_id"],
        "artifact_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
