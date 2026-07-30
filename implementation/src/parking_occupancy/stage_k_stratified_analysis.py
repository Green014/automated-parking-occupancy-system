from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from .detector_comparison import sha256_file
from .stage_j_posthoc_analysis import grouped_metrics, load_occupancy_rows


PROTOCOL_ID = "P-COMP-PKLOT-TEST-STAGEK-STRATA-POSTHOC-20260728-01"
RECORD_ID = (
    "P-COMP-PKLOT-TEST-STAGEK-STRATA-POSTHOC-RECORD-20260728-01"
)
METHOD_IDS = ("P0", "P1", "P2")


class StageKStrataError(ValueError):
    pass


def _verify(path: Path, binding: dict[str, Any], label: str) -> None:
    if (
        not path.is_file()
        or path.stat().st_size != int(binding["bytes"])
        or sha256_file(path) != str(binding["sha256"])
    ):
        raise StageKStrataError(f"{label} binding mismatch")


def load_protocol(config_path: Path) -> dict[str, Any]:
    config_path = config_path.resolve()
    payload = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if (
        payload.get("protocol_id") != PROTOCOL_ID
        or payload.get("status") != "frozen_posthoc_before_analysis"
    ):
        raise StageKStrataError("Unexpected strata protocol")
    scope = payload["scope"]
    if (
        scope.get("read_only_posthoc") is not True
        or scope.get("prediction_allowed") is not False
        or scope.get("parameter_selection_allowed") is not False
        or scope.get("detector_reselection_allowed") is not False
    ):
        raise StageKStrataError("Invalid strata scope")
    if tuple(payload["expected"]["methods"]) != METHOD_IDS:
        raise StageKStrataError("Expected P0/P1/P2")
    binding = payload["source"]["result_record"]
    path = (config_path.parent / str(binding["path"])).resolve()
    _verify(path, binding, "Stage K result record")
    return payload


def run_analysis(
    *,
    config_path: Path,
    stage_k_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    if output_root.exists():
        raise FileExistsError(f"Refusing to overwrite: {output_root}")
    protocol = load_protocol(config_path)
    expected = protocol["expected"]
    rows_by_method = {}
    input_checks = {}
    for method_id in METHOD_IDS:
        binding = protocol["source"]["methods"][method_id]
        path = stage_k_root / str(binding["relative_path"])
        _verify(path, binding, f"{method_id} occupancy")
        all_rows, known_rows, unknown = load_occupancy_rows(path)
        if (
            len(all_rows) != int(expected["rows_per_method"])
            or len(known_rows) != int(expected["known_rows_per_method"])
            or unknown != int(expected["unknown_rows_excluded_per_method"])
            or len({row["sample_id"] for row in known_rows})
            != int(expected["samples"])
        ):
            raise StageKStrataError(f"{method_id} row totals differ")
        rows_by_method[method_id] = known_rows
        input_checks[method_id] = {
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "verified": True,
        }
    reference = {
        (row["sample_id"], row["slot_id"], row["truth"])
        for row in rows_by_method["P0"]
    }
    if any(
        {
            (row["sample_id"], row["slot_id"], row["truth"])
            for row in rows_by_method[method_id]
        }
        != reference
        for method_id in ("P1", "P2")
    ):
        raise StageKStrataError("Cross-method membership differs")

    metrics = {
        method_id: {
            "by_date": grouped_metrics(rows_by_method[method_id], "date"),
            "by_weather": grouped_metrics(
                rows_by_method[method_id],
                "weather",
            ),
        }
        for method_id in METHOD_IDS
    }
    dates = sorted(metrics["P0"]["by_date"])
    weather = sorted(metrics["P0"]["by_weather"])
    if dates != sorted(str(x) for x in expected["dates"]):
        raise StageKStrataError("Date membership differs")
    if weather != sorted(str(x) for x in expected["weather"]):
        raise StageKStrataError("Weather membership differs")

    preflight = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "read_only_posthoc": True,
        "predictions_run": False,
        "parameters_selected": False,
        "inputs": input_checks,
        "execution_gate": "open_for_read_only_analysis",
    }
    analysis = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "analysis_time_utc": datetime.now(timezone.utc).isoformat(),
        "data_role": "untouched_test_already_evaluated",
        "read_only_posthoc": True,
        "predictions_run": False,
        "parameters_selected": False,
        "metrics": metrics,
        "note": (
            "Each selected camera has one selected date, so date strata are "
            "numerically identical to camera strata; the explicit layer is "
            "retained for reporting completeness."
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
    fields = ["method_id", "group", *protocol["metrics"]["fields"]]
    for field, filename in (
        ("by_date", "date_metrics.csv"),
        ("by_weather", "weather_metrics.csv"),
    ):
        with (output_root / filename).open(
            "w",
            encoding="utf-8",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields,
                lineterminator="\n",
            )
            writer.writeheader()
            for method_id in METHOD_IDS:
                for group, values in metrics[method_id][field].items():
                    writer.writerow(
                        {
                            "method_id": method_id,
                            "group": group,
                            **{
                                metric: f"{float(values[metric]):.12f}"
                                for metric in protocol["metrics"]["fields"]
                            },
                        }
                    )
    return analysis


def verify_record(
    *,
    record_path: Path,
    source_root: Path,
    external_root: Path,
) -> dict[str, Any]:
    record = yaml.safe_load(record_path.read_text(encoding="utf-8"))
    if record.get("record_id") != RECORD_ID:
        raise StageKStrataError("Unexpected strata record ID")
    roots = {
        "source": source_root.resolve(),
        "external": external_root.resolve(),
    }
    checks = []
    for artifact in record["artifacts"]:
        path = roots[str(artifact["root"])] / str(artifact["path"])
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha256 = sha256_file(path) if path.is_file() else None
        passed = (
            actual_bytes == int(artifact["bytes"])
            and actual_sha256 == str(artifact["sha256"])
        )
        checks.append(
            {
                "role": artifact["role"],
                "passed": passed,
                "actual_bytes": actual_bytes,
                "actual_sha256": actual_sha256,
            }
        )
    return {
        "record_id": record["record_id"],
        "artifact_count": len(checks),
        "passed_count": sum(check["passed"] for check in checks),
        "passed": all(check["passed"] for check in checks),
        "checks": checks,
    }
