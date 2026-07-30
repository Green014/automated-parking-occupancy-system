from __future__ import annotations

import argparse
import json
import os
import shutil
import stat
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import yaml


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.artifact_registry import (
    STAGE_W_2_PRE_W3_REGISTRY_SHA256,
    artifact_record,
    verify_artifact_registry,
    verify_historical_artifact_registry,
)
from parking_occupancy.stage_w_3_release import (
    FORMAL_CONFIG_RELATIVE,
    FORMAL_CONFIG_SHA256,
    SOURCE_MANIFEST_RELATIVE,
    copy_public_source_candidate,
    resolve_public_source_manifest,
    sha256_file,
)


REGISTRY_RELATIVE = (
    "implementation/data/STAGE_W_3_ARTIFACT_REGISTRY.yaml"
)
MODEL_MANIFEST_RELATIVE = (
    "implementation/data/STAGE_W_3_MODEL_RELEASE_MANIFEST.yaml"
)
MODEL_ASSET_DIRECTORY = (
    "implementation/outputs/stage_w_3_model_release_assets"
)
REGISTRY_ARTIFACTS = (
    (".gitattributes", "cross_platform_attributes", "required"),
    ("LICENSE", "project_agpl_3_0_license", "required"),
    ("THIRD_PARTY_NOTICES.md", "third_party_license_notices", "required"),
    ("README.md", "repository_readme", "required"),
    ("FINAL_RELEASE_INDEX.md", "repository_release_index", "required"),
    ("implementation/README.md", "implementation_readme", "required"),
    ("implementation/pyproject.toml", "dependency_and_cli_metadata", "required"),
    (
        "implementation/stage_w_requirements.txt",
        "dashboard_dependency_compatibility_entry",
        "required",
    ),
    (
        "implementation/configs/p3_stage_r_recommended_default_20260729.yaml",
        "frozen_config_identity",
        "required",
    ),
    (
        "implementation/data/PUBLIC_PERMISSION_AND_PROVENANCE.md",
        "anonymous_public_permission_record",
        "required",
    ),
    (
        "implementation/data/MODEL_CARD_D1.md",
        "d1_model_card",
        "required",
    ),
    (
        "implementation/data/MODEL_CARD_E1B.md",
        "e1b_model_card",
        "required",
    ),
    (
        MODEL_MANIFEST_RELATIVE,
        "model_release_manifest",
        "required",
    ),
    (
        SOURCE_MANIFEST_RELATIVE,
        "privacy_safe_public_source_manifest",
        "required",
    ),
    (
        "implementation/data/STAGE_W_2_ARTIFACT_REGISTRY.yaml",
        "pre_w3_historical_source_snapshot",
        "required",
    ),
    (
        "implementation/data/STAGE_W_3_PRIVACY_AND_MODEL_RELEASE_REPORT.md",
        "privacy_and_model_release_report",
        "required",
    ),
    (
        "implementation/data/STAGE_W_3_RELEASE_INDEX.md",
        "w3_release_index",
        "required",
    ),
    (
        "implementation/data/STAGE_U_1_MODEL_ASSETS.md",
        "model_asset_user_guide",
        "required",
    ),
    (
        "implementation/data/STAGE_W_REPRODUCTION_GUIDE.md",
        "reproduction_guide",
        "required",
    ),
    (
        "implementation/scripts/verify_stage_v_w_registries.py",
        "historical_and_current_registry_verifier",
        "required",
    ),
    (
        "implementation/scripts/verify_stage_w_3_release.py",
        "w3_release_verifier",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/artifact_registry.py",
        "registry_library",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_w_3_release.py",
        "privacy_safe_source_library",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/integrated_cli.py",
        "default_runtime_cli",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_v_runner.py",
        "controlled_comparison_cli",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_w_cli.py",
        "dashboard_cli",
        "required",
    ),
    (
        "implementation/scripts/run_p3_tt.py",
        "optional_tracking_cli",
        "required",
    ),
    (
        "implementation/tests/test_stage_w_2_cross_platform_release.py",
        "pre_w3_history_regression",
        "required",
    ),
    (
        "implementation/tests/test_stage_w_3_privacy_and_model_release.py",
        "w3_release_regression",
        "required",
    ),
    (
        f"{MODEL_ASSET_DIRECTORY}/D1_NDISPark_best.pt",
        "d1_future_github_release_asset",
        "local_ignored_optional",
    ),
    (
        f"{MODEL_ASSET_DIRECTORY}/E1b_CBAM_best.pt",
        "e1b_future_github_release_asset",
        "local_ignored_optional",
    ),
    (
        f"{MODEL_ASSET_DIRECTORY}/SHA256SUMS.txt",
        "model_release_checksums",
        "local_ignored_optional",
    ),
    (
        f"{MODEL_ASSET_DIRECTORY}/MODEL_RELEASE_METADATA.yaml",
        "model_release_metadata",
        "local_ignored_optional",
    ),
)
MANAGED_ARTIFACT_GLOBS = (
    ".gitattributes",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "README.md",
    "FINAL_RELEASE_INDEX.md",
    "implementation/README.md",
    "implementation/pyproject.toml",
    "implementation/stage_w_requirements.txt",
    "implementation/configs/p3_stage_r_recommended_default_20260729.yaml",
    "implementation/data/PUBLIC_PERMISSION_AND_PROVENANCE.md",
    "implementation/data/MODEL_CARD_*.md",
    "implementation/data/STAGE_U_1_MODEL_ASSETS.md",
    "implementation/data/STAGE_W_REPRODUCTION_GUIDE.md",
    "implementation/data/STAGE_W_2_ARTIFACT_REGISTRY.yaml",
    "implementation/data/STAGE_W_3_*.md",
    "implementation/data/STAGE_W_3_*MANIFEST.yaml",
    "implementation/scripts/run_p3_tt.py",
    "implementation/scripts/verify_stage_v_w_registries.py",
    "implementation/scripts/verify_stage_w_3_release.py",
    "implementation/src/parking_occupancy/artifact_registry.py",
    "implementation/src/parking_occupancy/integrated_cli.py",
    "implementation/src/parking_occupancy/stage_v_runner.py",
    "implementation/src/parking_occupancy/stage_w_3_release.py",
    "implementation/src/parking_occupancy/stage_w_cli.py",
    "implementation/tests/test_stage_w_2_cross_platform_release.py",
    "implementation/tests/test_stage_w_3_privacy_and_model_release.py",
    f"{MODEL_ASSET_DIRECTORY}/*",
)


def _run(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str] | None = None,
    timeout: int = 300,
) -> dict[str, Any]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
        check=False,
    )
    return {
        "name": name,
        "command": command,
        "exit_code": result.returncode,
        "passed": result.returncode == 0,
        "stdout": result.stdout,
        "stderr": result.stderr,
    }


def verify_model_release_assets(
    repository_root: Path,
    *,
    require_present: bool,
) -> dict[str, Any]:
    manifest_path = repository_root / MODEL_MANIFEST_RELATIVE
    payload = yaml.safe_load(manifest_path.read_text(encoding="utf-8"))
    errors: list[str] = []
    verified_assets = 0
    unavailable_assets = 0
    for asset in payload["assets"]:
        path = repository_root / str(asset["local_ignored_path"])
        if not path.is_file():
            unavailable_assets += 1
            if require_present:
                errors.append(f"missing:{asset['model_id']}")
            continue
        if path.stat().st_size != int(asset["bytes"]):
            errors.append(f"bytes:{asset['model_id']}")
        elif sha256_file(path) != str(asset["sha256"]):
            errors.append(f"sha256:{asset['model_id']}")
        else:
            verified_assets += 1

    asset_root = repository_root / MODEL_ASSET_DIRECTORY
    checksum_path = asset_root / "SHA256SUMS.txt"
    metadata_path = asset_root / "MODEL_RELEASE_METADATA.yaml"
    if checksum_path.is_file():
        checksum_lines = {
            line.strip()
            for line in checksum_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        }
        expected_lines = {
            f"{asset['sha256']}  {asset['release_filename']}"
            for asset in payload["assets"]
        }
        if checksum_lines != expected_lines:
            errors.append("checksum_file")
    elif require_present:
        errors.append("missing:SHA256SUMS")
    if metadata_path.is_file():
        metadata = yaml.safe_load(metadata_path.read_text(encoding="utf-8"))
        indexed = {
            item["filename"]: item for item in metadata.get("assets", [])
        }
        for asset in payload["assets"]:
            item = indexed.get(asset["release_filename"])
            if (
                item is None
                or int(item.get("bytes", -1)) != int(asset["bytes"])
                or str(item.get("sha256")) != str(asset["sha256"])
            ):
                errors.append(f"metadata:{asset['model_id']}")
    elif require_present:
        errors.append("missing:MODEL_RELEASE_METADATA")
    return {
        "verified": not errors,
        "model_assets_ready_for_github_release": (
            not errors and verified_assets == len(payload["assets"])
        ),
        "verified_assets": verified_assets,
        "unavailable_assets": unavailable_assets,
        "errors": errors,
        "model_training_run": False,
        "model_inference_run": False,
        "public_release_published": False,
    }


def write_w3_registry(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    records = [
        artifact_record(
            repository_root,
            path,
            role=role,
            availability=availability,
        )
        for path, role, availability in REGISTRY_ARTIFACTS
    ]
    payload = {
        "schema_version": 1,
        "registry_id": "STAGE-W-3-PRIVACY-MODEL-RELEASE-20260730-01",
        "stage": "W.3",
        "created_at": "2026-07-30",
        "status": "PUBLIC_SOURCE_AND_MODEL_ASSETS_PREPARED_NOT_PUBLISHED",
        "source_publication_ready": True,
        "model_assets_ready_for_github_release": True,
        "public_release_published": False,
        "model_training_run": False,
        "model_inference_run": False,
        "historical_registry_policy": {
            "stage_w_2": "pre_w3_historical_source_snapshot",
        },
        "artifact_count": len(records),
        "registry_self_path": REGISTRY_RELATIVE,
        "artifact_root": "repository_root",
        "local_optional_policy": (
            "model release assets are verified when present and are "
            "intentionally absent from the public source checkout"
        ),
        "managed_artifact_globs": list(MANAGED_ARTIFACT_GLOBS),
        "artifacts": records,
    }
    registry_path = repository_root / REGISTRY_RELATIVE
    registry_path.write_text(
        yaml.safe_dump(
            payload,
            sort_keys=False,
            allow_unicode=True,
            width=100,
        ),
        encoding="utf-8",
        newline="\n",
    )
    return verify_artifact_registry(
        registry_path,
        artifact_root=repository_root,
    )


def verify_current_checkout(
    repository_root: Path,
    *,
    require_model_assets: bool,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    errors: list[str] = []
    formal_config = repository_root / FORMAL_CONFIG_RELATIVE
    formal_sha256 = sha256_file(formal_config)
    formal_has_crlf = b"\r\n" in formal_config.read_bytes()
    if formal_sha256 != FORMAL_CONFIG_SHA256:
        errors.append("formal_config_sha256")
    if formal_has_crlf:
        errors.append("formal_config_crlf")

    attributes = _run(
        "git_check_attr",
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            "LICENSE",
            FORMAL_CONFIG_RELATIVE,
        ],
        cwd=repository_root,
    )
    normalized_attributes = attributes["stdout"].replace("\\", "/")
    for relative in ("LICENSE", FORMAL_CONFIG_RELATIVE):
        if (
            f"{relative}: text: set" not in normalized_attributes
            or f"{relative}: eol: lf" not in normalized_attributes
        ):
            errors.append(f"git_attributes:{relative}")

    source = resolve_public_source_manifest(repository_root)
    if not source["verified"]:
        errors.extend(f"source:{item}" for item in source["violations"])

    registry_path = repository_root / REGISTRY_RELATIVE
    if registry_path.is_file():
        registry = verify_artifact_registry(
            registry_path,
            artifact_root=repository_root,
        )
        if not registry["verified"]:
            errors.extend(f"registry:{item}" for item in registry["errors"])
    else:
        registry = {"verified": False, "errors": ["missing"]}
        errors.append("registry:missing")

    w2_history = verify_historical_artifact_registry(
        repository_root
        / "implementation"
        / "data"
        / "STAGE_W_2_ARTIFACT_REGISTRY.yaml",
        artifact_root=repository_root,
        expected_registry_sha256=STAGE_W_2_PRE_W3_REGISTRY_SHA256,
        immutable_path_prefixes=(),
        classification="pre_w3_historical_source_snapshot",
    )
    if not w2_history["verified"]:
        errors.extend(f"w2_history:{item}" for item in w2_history["errors"])

    models = verify_model_release_assets(
        repository_root,
        require_present=require_model_assets,
    )
    if not models["verified"]:
        errors.extend(f"models:{item}" for item in models["errors"])

    staged = _run(
        "real_repository_index_unchanged",
        ["git", "diff", "--cached", "--quiet"],
        cwd=repository_root,
    )
    if not staged["passed"]:
        errors.append("real_repository_has_staged_changes")
    return {
        "schema_version": 1,
        "stage": "W.3",
        "verified": not errors,
        "errors": errors,
        "formal_config_sha256": formal_sha256,
        "formal_config_has_crlf": formal_has_crlf,
        "git_attributes": attributes,
        "source_manifest": {
            key: value
            for key, value in source.items()
            if key
            not in {
                "candidate_files",
                "included_by_category",
                "excluded_by_category",
                "privacy_exclusion_reasons",
            }
        },
        "registry": registry,
        "w2_history": w2_history,
        "model_assets": models,
        "real_repository_index_unchanged": staged["passed"],
        "source_publication_ready": (
            source["source_publication_ready"] and not errors
        ),
        "model_assets_ready_for_github_release": models[
            "model_assets_ready_for_github_release"
        ],
        "public_release_published": False,
        "model_training_run": False,
        "model_inference_run": False,
    }


def _remove_readonly(function, path: str, _error: object) -> None:
    os.chmod(path, stat.S_IWRITE)
    function(path)


def simulate_windows_checkout(
    repository_root: Path,
    *,
    keep: bool = False,
) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    inventory = resolve_public_source_manifest(repository_root)
    if not inventory["verified"]:
        raise RuntimeError(inventory["violations"])

    temp_parent = Path(tempfile.mkdtemp(prefix="stage_w_3_release_"))
    source = temp_parent / "stage_w_3_public_source_candidate"
    clone = temp_parent / "stage_w_3_autocrlf_clone"
    copied = copy_public_source_candidate(
        repository_root=repository_root,
        destination=source,
        candidates=inventory["candidate_files"],
    )
    setup = [
        _run("git_init", ["git", "init", "-q"], cwd=source),
        _run(
            "git_identity_name",
            ["git", "config", "user.name", "Stage W.3 Verification"],
            cwd=source,
        ),
        _run(
            "git_identity_email",
            ["git", "config", "user.email", "stage-w3-verification.invalid"],
            cwd=source,
        ),
        _run("git_add_temp_only", ["git", "add", "-A"], cwd=source),
        _run(
            "git_commit_temp_only",
            ["git", "commit", "-qm", "temporary W.3 public source"],
            cwd=source,
        ),
    ]
    if not all(item["passed"] for item in setup):
        raise RuntimeError(f"Temporary source Git setup failed: {setup}")

    clone_result = _run(
        "clone_core_autocrlf_true",
        [
            "git",
            "-c",
            "core.autocrlf=true",
            "clone",
            "-q",
            "--no-hardlinks",
            str(source),
            str(clone),
        ],
        cwd=temp_parent,
    )
    if not clone_result["passed"]:
        raise RuntimeError(f"Temporary clone failed: {clone_result}")
    persist_autocrlf = _run(
        "persist_clone_core_autocrlf_true",
        ["git", "config", "core.autocrlf", "true"],
        cwd=clone,
    )

    python = str(Path(sys.executable).resolve())
    clone_env = os.environ.copy()
    clone_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(clone / "implementation" / "src"),
            str(clone / "implementation" / "literature_core" / "src"),
        ]
    )
    checks = [
        _run(
            "w3_source_registry_privacy",
            [
                python,
                "implementation/scripts/verify_stage_w_3_release.py",
                "--current-only",
                "--source-only",
            ],
            cwd=clone,
            env=clone_env,
        ),
        _run(
            "parking_run_final_help",
            [python, "-m", "parking_occupancy.integrated_cli", "--help"],
            cwd=clone,
            env=clone_env,
        ),
        _run(
            "stage_v_compare_help",
            [python, "-m", "parking_occupancy.stage_v_runner", "--help"],
            cwd=clone,
            env=clone_env,
        ),
        _run(
            "stage_w_dashboard_help",
            [python, "-m", "parking_occupancy.stage_w_cli", "--help"],
            cwd=clone,
            env=clone_env,
        ),
        _run(
            "p3_tt_help",
            [python, "implementation/scripts/run_p3_tt.py", "--help"],
            cwd=clone,
            env=clone_env,
        ),
        _run(
            "compileall",
            [
                python,
                "-m",
                "compileall",
                "-q",
                "implementation/src",
                "implementation/scripts",
                "implementation/tests",
                "implementation/literature_core/src",
            ],
            cwd=clone,
            env=clone_env,
        ),
    ]
    clone_formal = clone / FORMAL_CONFIG_RELATIVE
    clone_sha256 = sha256_file(clone_formal)
    clone_has_crlf = b"\r\n" in clone_formal.read_bytes()
    forbidden = [
        relative
        for relative in inventory["candidate_files"]
        if Path(relative).suffix.lower()
        in {".pt", ".pth", ".ckpt", ".onnx", ".engine"}
        or any(
            part.lower()
            in {"outputs", "runs", "datasets", "vendor", "weights"}
            for part in Path(relative).parts
        )
    ]
    passed = (
        persist_autocrlf["passed"]
        and all(item["passed"] for item in checks)
        and clone_sha256 == FORMAL_CONFIG_SHA256
        and not clone_has_crlf
        and not forbidden
    )
    payload = {
        "schema_version": 1,
        "stage": "W.3",
        "verified": passed,
        "source_copy": copied,
        "setup": setup,
        "clone": clone_result,
        "persist_autocrlf": persist_autocrlf,
        "checks": checks,
        "formal_config_sha256": clone_sha256,
        "formal_config_has_crlf": clone_has_crlf,
        "forbidden_candidate_files": forbidden,
        "temporary_repository_only": True,
        "public_remote_connected": False,
        "model_training_run": False,
        "model_inference_run": False,
        "public_release_published": False,
        "temporary_directory_removed": False,
    }
    if not keep:
        try:
            shutil.rmtree(temp_parent, onerror=_remove_readonly)
            payload["temporary_directory_removed"] = True
        except OSError as exc:
            payload["cleanup_error"] = str(exc)
    else:
        payload["temporary_directory"] = str(temp_parent)
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the W.3 privacy-safe public source, model assets, "
            "registry, and optional core.autocrlf=true temporary clone."
        )
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Verify the current checkout without a temporary Git clone.",
    )
    parser.add_argument(
        "--source-only",
        action="store_true",
        help="Allow intentionally absent ignored model assets.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary source and clone for inspection.",
    )
    parser.add_argument(
        "--write-registry",
        action="store_true",
        help="Regenerate only the W.3 registry from declared artifacts.",
    )
    args = parser.parse_args()
    if args.write_registry:
        result = write_w3_registry(REPOSITORY_ROOT)
    elif args.current_only:
        result = verify_current_checkout(
            REPOSITORY_ROOT,
            require_model_assets=not args.source_only,
        )
    else:
        result = simulate_windows_checkout(
            REPOSITORY_ROOT,
            keep=args.keep_temp,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["verified"] else 1)


if __name__ == "__main__":
    main()
