"""Verify hashes and result counts for a frozen experiment manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def artifact_entries(payload: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    entries: list[tuple[str, dict[str, Any]]] = []
    for section_name in (
        "configurations",
        "model_artifacts",
        "data_artifacts",
        "result_artifacts",
    ):
        section = payload.get(section_name, {})
        if not isinstance(section, dict):
            raise ValueError(f"{section_name} must be a mapping")
        for name, value in section.items():
            if isinstance(value, dict) and "path" in value and "sha256" in value:
                entries.append((f"{section_name}.{name}", value))
    if not entries:
        raise ValueError("manifest contains no path/hash artifact entries")
    return entries


def resolve_artifact_path(manifest: Path, value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    # Manifests are stored below data/manifests, while paths are relative to
    # the literature-core project root.
    project_root = manifest.resolve().parents[2]
    return (project_root / path).resolve()


def verify_manifest(manifest: Path) -> dict[str, Any]:
    with manifest.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a mapping")

    rows: list[dict[str, Any]] = []
    for name, entry in artifact_entries(payload):
        path = resolve_artifact_path(manifest, str(entry["path"]))
        expected = str(entry["sha256"]).lower()
        exists = path.is_file()
        actual = sha256_file(path) if exists else None
        rows.append(
            {
                "name": name,
                "path": str(path),
                "exists": exists,
                "expected_sha256": expected,
                "actual_sha256": actual,
                "sha256_matches": actual == expected,
                "bytes": path.stat().st_size if exists else None,
            }
        )

    result_entry = payload.get("result_artifacts", {})
    metrics_entry = (
        result_entry.get("metrics", {})
        if isinstance(result_entry, dict)
        else {}
    )
    count_checks: dict[str, Any] = {}
    if isinstance(metrics_entry, dict) and "path" in metrics_entry:
        metrics_path = resolve_artifact_path(
            manifest,
            str(metrics_entry["path"]),
        )
        if metrics_path.is_file():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            integrity = metrics.get("integrity", {})
            for key in ("frames", "slot_records"):
                if key in result_entry:
                    expected_count = int(result_entry[key])
                    actual_count = int(integrity.get(key, -1))
                    count_checks[key] = {
                        "expected": expected_count,
                        "actual": actual_count,
                        "matches": expected_count == actual_count,
                    }

    passed = all(row["sha256_matches"] for row in rows) and all(
        check["matches"] for check in count_checks.values()
    )
    return {
        "schema_version": 1,
        "verified_at": datetime.now().astimezone().isoformat(),
        "manifest": str(manifest.resolve()),
        "artifact_count": len(rows),
        "artifacts": rows,
        "count_checks": count_checks,
        "passed": passed,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--manifest",
        type=Path,
        default=ROOT / "data" / "manifests" / "frozen_artifacts_20260725.yaml",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists():
        raise FileExistsError(
            "Verification output already exists; choose a new path so prior "
            "audit evidence is not overwritten."
        )
    report = verify_manifest(args.manifest)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
