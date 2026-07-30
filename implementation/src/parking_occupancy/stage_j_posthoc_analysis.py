from __future__ import annotations

import csv
import json
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .detector_comparison import sha256_file
from .evaluate import binary_metrics


STAGE_J_POSTHOC_PROTOCOL_ID = (
    "P-COMP-PKLOT-DEV-STAGEJ-POSTHOC-20260727-01"
)
STAGE_J_POSTHOC_RECORD_ID = (
    "P-COMP-PKLOT-DEV-STAGEJ-POSTHOC-RECORD-20260727-01"
)
METHOD_IDS = ("P0", "P1", "P2")
CAMERA_METRIC_FIELDS = (
    "macro_f1",
    "occupied_recall",
    "vacant_recall",
    "false_free_rate",
    "false_occupied_rate",
)


class StageJPosthocError(ValueError):
    """Raised when a post-hoc input differs from the frozen protocol."""


def _verified_file(
    *,
    path: Path,
    expected_bytes: int,
    expected_sha256: str,
    label: str,
) -> dict[str, Any]:
    if not path.is_file():
        raise StageJPosthocError(f"Missing {label}: {path}")
    actual_bytes = path.stat().st_size
    actual_sha256 = sha256_file(path)
    if actual_bytes != int(expected_bytes):
        raise StageJPosthocError(f"{label} byte size mismatch")
    if actual_sha256 != str(expected_sha256):
        raise StageJPosthocError(f"{label} SHA-256 mismatch")
    return {
        "path": str(path),
        "bytes": actual_bytes,
        "sha256": actual_sha256,
        "verified": True,
    }


def load_stage_j_posthoc_protocol(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_J_POSTHOC_PROTOCOL_ID:
        raise StageJPosthocError("Unexpected Stage J post-hoc protocol ID")
    if payload.get("status") != "frozen_posthoc_before_analysis":
        raise StageJPosthocError("Stage J post-hoc protocol is not frozen")
    scope = payload.get("scope", {})
    if (
        scope.get("data_role") != "consumed_development"
        or scope.get("prediction_allowed") is not False
        or scope.get("parameter_selection_allowed") is not False
        or scope.get("detector_reselection_allowed") is not False
    ):
        raise StageJPosthocError("Invalid post-hoc scope boundary")
    expected = payload.get("expected", {})
    if tuple(expected.get("methods", [])) != METHOD_IDS:
        raise StageJPosthocError("Expected P0/P1/P2 inputs")
    paired = payload.get("metrics", {}).get("paired", {})
    if (
        paired.get("baseline") != "P0"
        or tuple(paired.get("candidates", [])) != ("P1", "P2")
        or paired.get("unit") != "sample_id"
        or paired.get("score") != "per_sample_macro_f1"
    ):
        raise StageJPosthocError("Invalid paired-analysis contract")
    bootstrap = payload.get("bootstrap", {})
    if (
        bootstrap.get("unit") != "sample_id"
        or bootstrap.get("slot_level_resampling") != "prohibited"
        or int(bootstrap.get("resamples", 0)) <= 0
    ):
        raise StageJPosthocError("Invalid grouped bootstrap contract")

    for label, binding in (
        ("source Stage J protocol", payload["source_stage_j"]["protocol"]),
        (
            "source Stage J result record",
            payload["source_stage_j"]["result_record"],
        ),
    ):
        path = (config_path.parent / str(binding["path"])).resolve()
        _verified_file(
            path=path,
            expected_bytes=int(binding["bytes"]),
            expected_sha256=str(binding["sha256"]),
            label=label,
        )
    return payload


def load_occupancy_rows(
    path: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
    """Load one frozen occupancy CSV and exclude blank truth from metrics."""

    required = {
        "sample_id",
        "slot_id",
        "camera",
        "date",
        "weather",
        "truth",
        "state",
    }
    all_rows: list[dict[str, Any]] = []
    known_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = required - set(reader.fieldnames or ())
        if missing:
            raise StageJPosthocError(
                f"Missing occupancy columns: {sorted(missing)}"
            )
        for line_number, raw in enumerate(reader, start=2):
            key = (str(raw["sample_id"]), str(raw["slot_id"]))
            if key in seen:
                raise StageJPosthocError(
                    f"Duplicate occupancy key at line {line_number}: {key}"
                )
            seen.add(key)
            truth_text = str(raw["truth"]).strip()
            if truth_text not in {"", "0", "1"}:
                raise StageJPosthocError(
                    f"Invalid truth at line {line_number}: {truth_text}"
                )
            prediction_text = str(raw["state"]).strip()
            if prediction_text not in {"0", "1"}:
                raise StageJPosthocError(
                    f"Invalid prediction at line {line_number}"
                )
            row = {
                "sample_id": key[0],
                "slot_id": key[1],
                "camera": str(raw["camera"]),
                "date": str(raw["date"]),
                "weather": str(raw["weather"]),
                "truth": None if truth_text == "" else int(truth_text),
                "prediction": int(prediction_text),
            }
            all_rows.append(row)
            if row["truth"] is not None:
                known_rows.append(row)
    return all_rows, known_rows, len(all_rows) - len(known_rows)


def _classification(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise StageJPosthocError("Cannot score an empty group")
    return binary_metrics(
        [int(row["truth"]) for row in rows],
        [int(row["prediction"]) for row in rows],
    )


def grouped_metrics(
    rows: list[dict[str, Any]],
    field: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[field])].append(row)
    return {
        group: _classification(group_rows)
        for group, group_rows in sorted(grouped.items())
    }


def equal_group_macro(
    metrics_by_group: dict[str, dict[str, Any]],
    fields: tuple[str, ...] = CAMERA_METRIC_FIELDS,
) -> dict[str, float | int]:
    if not metrics_by_group:
        raise StageJPosthocError("Cannot average zero groups")
    return {
        "groups": len(metrics_by_group),
        **{
            field: statistics.fmean(
                float(metrics[field])
                for metrics in metrics_by_group.values()
            )
            for field in fields
        },
    }


def per_sample_metrics(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    return grouped_metrics(rows, "sample_id")


def paired_bootstrap_mean_difference(
    differences_by_sample: dict[str, float],
    *,
    seed: int,
    resamples: int,
    confidence_level: float,
) -> dict[str, Any]:
    """Bootstrap paired sample-level differences, never individual slots."""

    if not differences_by_sample:
        raise StageJPosthocError("No paired sample differences")
    if resamples <= 0:
        raise StageJPosthocError("Bootstrap resamples must be positive")
    if not 0.0 < confidence_level < 1.0:
        raise StageJPosthocError("Confidence level must be in (0, 1)")
    sample_ids = sorted(differences_by_sample)
    values = np.asarray(
        [float(differences_by_sample[sample_id]) for sample_id in sample_ids],
        dtype=np.float64,
    )
    rng = np.random.default_rng(seed)
    draws = rng.integers(
        0,
        len(values),
        size=(int(resamples), len(values)),
    )
    bootstrap_means = values[draws].mean(axis=1)
    tail = (1.0 - confidence_level) / 2.0
    lower, upper = np.quantile(
        bootstrap_means,
        [tail, 1.0 - tail],
        method="linear",
    )
    return {
        "unit": "sample_id",
        "sample_count": len(sample_ids),
        "resamples": int(resamples),
        "seed": int(seed),
        "confidence_level": float(confidence_level),
        "interval": "percentile",
        "statistic": "mean_paired_per_sample_macro_f1_difference",
        "estimate": float(values.mean()),
        "ci_lower": float(lower),
        "ci_upper": float(upper),
        "contains_zero": bool(lower <= 0.0 <= upper),
    }


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
            raise StageJPosthocError(
                f"{method_id} occupancy membership/truth differs from P0"
            )


def stage_j_posthoc_preflight(
    *,
    config_path: Path,
    stage_j_root: Path,
) -> tuple[dict[str, Any], dict[str, list[dict[str, Any]]]]:
    config_path = config_path.resolve()
    stage_j_root = stage_j_root.resolve()
    protocol = load_stage_j_posthoc_protocol(config_path)
    expected = protocol["expected"]
    all_rows_by_method: dict[str, list[dict[str, Any]]] = {}
    known_rows_by_method: dict[str, list[dict[str, Any]]] = {}
    input_checks = {}
    for method_id in METHOD_IDS:
        binding = protocol["source_stage_j"]["methods"][method_id]
        path = stage_j_root / str(binding["relative_path"])
        input_checks[method_id] = _verified_file(
            path=path,
            expected_bytes=int(binding["bytes"]),
            expected_sha256=str(binding["sha256"]),
            label=f"{method_id} occupancy",
        )
        all_rows, known_rows, unknown_count = load_occupancy_rows(path)
        if (
            len(all_rows) != int(expected["rows_per_method"])
            or len(known_rows) != int(expected["known_rows_per_method"])
            or unknown_count
            != int(expected["unknown_rows_excluded_per_method"])
        ):
            raise StageJPosthocError(
                f"{method_id} occupancy row totals differ from protocol"
            )
        if len({row["sample_id"] for row in known_rows}) != int(
            expected["paired_samples"]
        ):
            raise StageJPosthocError(
                f"{method_id} paired sample count differs from protocol"
            )
        cameras = sorted({row["camera"] for row in known_rows})
        if cameras != sorted(str(value) for value in expected["cameras"]):
            raise StageJPosthocError(
                f"{method_id} camera membership differs from protocol"
            )
        all_rows_by_method[method_id] = all_rows
        known_rows_by_method[method_id] = known_rows
    _validate_cross_method_rows(all_rows_by_method)
    report = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "data_role": "consumed_development",
        "read_only_posthoc": True,
        "predictions_run": False,
        "parameters_selected": False,
        "inputs": input_checks,
        "rows_per_method": int(expected["rows_per_method"]),
        "known_rows_per_method": int(expected["known_rows_per_method"]),
        "unknown_rows_excluded_per_method": int(
            expected["unknown_rows_excluded_per_method"]
        ),
        "paired_samples": int(expected["paired_samples"]),
        "execution_gate": "open_for_read_only_analysis",
    }
    return report, known_rows_by_method


def _paired_comparison(
    *,
    baseline_id: str,
    candidate_id: str,
    metrics_by_method: dict[str, dict[str, Any]],
    per_sample_by_method: dict[str, dict[str, dict[str, Any]]],
    protocol: dict[str, Any],
    sample_metadata: dict[str, dict[str, str]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    baseline_samples = per_sample_by_method[baseline_id]
    candidate_samples = per_sample_by_method[candidate_id]
    if set(candidate_samples) != set(baseline_samples):
        raise StageJPosthocError("Paired sample membership differs")
    tolerance = float(protocol["metrics"]["paired"]["tie_tolerance"])
    differences: dict[str, float] = {}
    rows = []
    counts = {"win": 0, "tie": 0, "loss": 0}
    for sample_id in sorted(baseline_samples):
        baseline_score = float(baseline_samples[sample_id]["macro_f1"])
        candidate_score = float(candidate_samples[sample_id]["macro_f1"])
        difference = candidate_score - baseline_score
        differences[sample_id] = difference
        outcome = (
            "win"
            if difference > tolerance
            else "loss"
            if difference < -tolerance
            else "tie"
        )
        counts[outcome] += 1
        rows.append(
            {
                "comparison": f"{candidate_id}-{baseline_id}",
                "sample_id": sample_id,
                **sample_metadata[sample_id],
                "baseline_macro_f1": baseline_score,
                "candidate_macro_f1": candidate_score,
                "difference": difference,
                "outcome": outcome,
            }
        )
    bootstrap = protocol["bootstrap"]
    result = {
        "baseline": baseline_id,
        "candidate": candidate_id,
        "difference_definition": f"{candidate_id}_minus_{baseline_id}",
        "pooled_macro_f1_difference": (
            float(metrics_by_method[candidate_id]["overall"]["macro_f1"])
            - float(metrics_by_method[baseline_id]["overall"]["macro_f1"])
        ),
        "camera_macro_f1_difference": (
            float(
                metrics_by_method[candidate_id]["camera_macro"]["macro_f1"]
            )
            - float(
                metrics_by_method[baseline_id]["camera_macro"]["macro_f1"]
            )
        ),
        "per_camera_macro_f1_difference": {
            camera: (
                float(
                    metrics_by_method[candidate_id]["by_camera"][camera][
                        "macro_f1"
                    ]
                )
                - float(
                    metrics_by_method[baseline_id]["by_camera"][camera][
                        "macro_f1"
                    ]
                )
            )
            for camera in metrics_by_method[baseline_id]["by_camera"]
        },
        "win_tie_loss": counts,
        "paired_bootstrap": paired_bootstrap_mean_difference(
            differences,
            seed=int(bootstrap["seed"]),
            resamples=int(bootstrap["resamples"]),
            confidence_level=float(bootstrap["confidence_level"]),
        ),
    }
    return result, rows


def run_stage_j_posthoc_analysis(
    *,
    config_path: Path,
    stage_j_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Run the frozen read-only Stage J layered and paired analysis."""

    if output_root.exists():
        raise FileExistsError(
            f"Refusing to overwrite post-hoc output: {output_root}"
        )
    preflight, rows_by_method = stage_j_posthoc_preflight(
        config_path=config_path,
        stage_j_root=stage_j_root,
    )
    protocol = load_stage_j_posthoc_protocol(config_path)
    metrics_by_method = {}
    per_sample_by_method = {}
    sample_metadata: dict[str, dict[str, str]] = {}
    for method_id in METHOD_IDS:
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

    paired = {}
    paired_rows = []
    for candidate_id in ("P1", "P2"):
        result, rows = _paired_comparison(
            baseline_id="P0",
            candidate_id=candidate_id,
            metrics_by_method=metrics_by_method,
            per_sample_by_method=per_sample_by_method,
            protocol=protocol,
            sample_metadata=sample_metadata,
        )
        paired[f"{candidate_id}-P0"] = result
        paired_rows.extend(rows)

    analysis = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "consumed_development",
        "read_only_posthoc": True,
        "predictions_run": False,
        "parameters_selected": False,
        "detector_reselected": False,
        "unknown_rows_excluded_per_method": preflight[
            "unknown_rows_excluded_per_method"
        ],
        "metrics": metrics_by_method,
        "paired_comparisons": paired,
        "required_conclusion": protocol["reporting"][
            "required_conclusion"
        ],
        "claim_boundary": (
            "These are consumed-development diagnostics, not an untouched "
            "test and not a basis for model, threshold, or mapping selection."
        ),
    }

    output_root.mkdir(parents=True)
    (output_root / "preflight.json").write_text(
        json.dumps(preflight, indent=2) + "\n",
        encoding="utf-8",
    )
    (output_root / "analysis.json").write_text(
        json.dumps(analysis, indent=2) + "\n",
        encoding="utf-8",
    )
    with (output_root / "camera_metrics.csv").open(
        "w",
        newline="",
        encoding="utf-8",
    ) as handle:
        fields = [
            "method_id",
            "camera",
            "samples",
            *CAMERA_METRIC_FIELDS,
        ]
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for method_id in METHOD_IDS:
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
        writer = csv.DictWriter(
            handle,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(paired_rows)
    return analysis


def verify_stage_j_posthoc_record(
    *,
    record_path: Path,
    source_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    """Verify the independent post-hoc analysis hash record."""

    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    if record.get("record_id") != STAGE_J_POSTHOC_RECORD_ID:
        raise StageJPosthocError("Unexpected Stage J post-hoc record ID")
    roots = {
        "source": source_root.resolve(),
        "external": external_root.resolve(),
    }
    checks = []
    for artifact in record["artifacts"]:
        root_name = str(artifact["root"])
        if root_name not in roots:
            raise StageJPosthocError(
                f"Unexpected post-hoc artifact root: {root_name}"
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
