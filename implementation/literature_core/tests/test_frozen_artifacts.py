import hashlib
import json
from pathlib import Path

import yaml

from scripts.verify_frozen_artifacts import verify_manifest


def test_verify_manifest_checks_hashes_and_result_counts(tmp_path: Path) -> None:
    project = tmp_path / "project"
    manifest_dir = project / "data" / "manifests"
    artifact = project / "artifact.bin"
    metrics = project / "metrics.json"
    manifest_dir.mkdir(parents=True)
    artifact.write_bytes(b"frozen")
    metrics.write_text(
        json.dumps({"integrity": {"frames": 7, "slot_records": 21}}),
        encoding="utf-8",
    )
    payload = {
        "configurations": {
            "example": {
                "path": "artifact.bin",
                "sha256": hashlib.sha256(b"frozen").hexdigest(),
            }
        },
        "result_artifacts": {
            "metrics": {
                "path": "metrics.json",
                "sha256": hashlib.sha256(metrics.read_bytes()).hexdigest(),
            },
            "frames": 7,
            "slot_records": 21,
        },
    }
    manifest = manifest_dir / "manifest.yaml"
    manifest.write_text(yaml.safe_dump(payload), encoding="utf-8")

    report = verify_manifest(manifest)

    assert report["passed"] is True
    assert report["artifact_count"] == 2
    assert report["count_checks"]["frames"]["matches"] is True


def test_verify_manifest_fails_when_hash_changes(tmp_path: Path) -> None:
    project = tmp_path / "project"
    manifest_dir = project / "data" / "manifests"
    manifest_dir.mkdir(parents=True)
    artifact = project / "artifact.bin"
    artifact.write_bytes(b"changed")
    manifest = manifest_dir / "manifest.yaml"
    manifest.write_text(
        yaml.safe_dump(
            {
                "model_artifacts": {
                    "example": {
                        "path": "artifact.bin",
                        "sha256": "0" * 64,
                    }
                }
            }
        ),
        encoding="utf-8",
    )

    report = verify_manifest(manifest)

    assert report["passed"] is False
    assert report["artifacts"][0]["sha256_matches"] is False
