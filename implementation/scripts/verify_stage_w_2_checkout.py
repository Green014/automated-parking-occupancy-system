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
    artifact_record,
    verify_artifact_registry,
)
from parking_occupancy.stage_w_2_release import (
    FORMAL_CONFIG_RELATIVE,
    FORMAL_CONFIG_SHA256,
    SOURCE_MANIFEST_RELATIVE,
    copy_source_candidate,
    resolve_source_manifest,
    sha256_file,
)


REGISTRY_RELATIVE = (
    "implementation/data/STAGE_W_2_ARTIFACT_REGISTRY.yaml"
)
REGISTRY_ARTIFACTS = (
    (".gitattributes", "cross_platform_git_attributes", "required"),
    ("FINAL_RELEASE_INDEX.md", "repository_release_index", "required"),
    ("README.md", "repository_readme", "required"),
    ("implementation/README.md", "implementation_readme", "required"),
    (
        "implementation/configs/p3_stage_r_recommended_default_20260729.yaml",
        "frozen_config_identity",
        "required",
    ),
    (
        "implementation/data/STAGE_U_1_MODEL_ASSETS.md",
        "model_asset_boundary",
        "required",
    ),
    (
        "implementation/data/STAGE_W_1_ARTIFACT_REGISTRY.yaml",
        "pre_w2_historical_source_snapshot",
        "required",
    ),
    (
        "implementation/data/STAGE_W_2_CROSS_PLATFORM_RELEASE_REPORT.md",
        "cross_platform_release_report",
        "required",
    ),
    (
        "implementation/data/STAGE_W_2_RELEASE_INDEX.md",
        "release_index",
        "required",
    ),
    (
        "implementation/data/STAGE_W_2_SOURCE_COMMIT_MANIFEST.yaml",
        "source_commit_manifest",
        "required",
    ),
    (
        "implementation/data/STAGE_W_PERMISSION_AND_PROVENANCE.md",
        "permission_and_provenance",
        "required",
    ),
    (
        "implementation/data/STAGE_W_REPRODUCTION_GUIDE.md",
        "reproduction_guide",
        "required",
    ),
    ("implementation/pyproject.toml", "dependency_and_cli_metadata", "required"),
    (
        "implementation/scripts/run_p3_tt.py",
        "optional_tracking_cli",
        "required",
    ),
    (
        "implementation/scripts/verify_stage_v_w_registries.py",
        "historical_and_current_registry_verifier",
        "required",
    ),
    (
        "implementation/scripts/verify_stage_w_2_checkout.py",
        "cross_platform_checkout_verifier",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/artifact_registry.py",
        "registry_library",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/integrated_cli.py",
        "default_runtime_cli_identity",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/p3_tt_runtime.py",
        "optional_tracking_runtime",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_t_tracktrack.py",
        "optional_tracking_contract",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_v.py",
        "unified_backend_contract",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_v_runner.py",
        "corrected_config_identity_runner",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_w_2_release.py",
        "source_manifest_library",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_w_cli.py",
        "dashboard_cli",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_w_member_reference.py",
        "external_member_adapter",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_w_server.py",
        "optional_dashboard_server",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_w_ui_adapter.py",
        "dashboard_adapter",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_w_web/static/style.css",
        "member_derived_local_ui_style",
        "required",
    ),
    (
        "implementation/src/parking_occupancy/stage_w_web/templates/dashboard.html",
        "member_derived_local_ui_template",
        "required",
    ),
    (
        "implementation/stage_w_requirements.txt",
        "compatibility_dependency_entry",
        "required",
    ),
    (
        "implementation/tests/test_stage_v_multimode.py",
        "stage_v_config_regression_tests",
        "required",
    ),
    (
        "implementation/tests/test_stage_w_1_release_hardening.py",
        "pre_w2_snapshot_boundary_tests",
        "required",
    ),
    (
        "implementation/tests/test_stage_w_2_cross_platform_release.py",
        "cross_platform_release_tests",
        "required",
    ),
    (
        "implementation/tests/test_stage_w_server.py",
        "optional_dashboard_server_tests",
        "required",
    ),
    (
        "implementation/tests/test_stage_w_ui_adapter.py",
        "dashboard_adapter_tests",
        "required",
    ),
    (
        "implementation/outputs/stage_w_dashboard_smoke_20260730_v3/annotated.mp4",
        "unchanged_local_demonstration_evidence",
        "local_ignored_optional",
    ),
    (
        "implementation/outputs/stage_w_dashboard_smoke_20260730_v3/configuration_snapshot.yaml",
        "unchanged_local_demonstration_evidence",
        "local_ignored_optional",
    ),
    (
        "implementation/outputs/stage_w_dashboard_smoke_20260730_v3/dashboard_ui_demo.json",
        "unchanged_local_demonstration_evidence",
        "local_ignored_optional",
    ),
    (
        "implementation/outputs/stage_w_dashboard_smoke_20260730_v3/dashboard_ui_demo.mp4",
        "unchanged_local_demonstration_evidence",
        "local_ignored_optional",
    ),
    (
        "implementation/outputs/stage_w_dashboard_smoke_20260730_v3/dashboard_ui_demo_preview.png",
        "unchanged_local_demonstration_evidence",
        "local_ignored_optional",
    ),
    (
        "implementation/outputs/stage_w_dashboard_smoke_20260730_v3/events.jsonl",
        "unchanged_local_demonstration_evidence",
        "local_ignored_optional",
    ),
    (
        "implementation/outputs/stage_w_dashboard_smoke_20260730_v3/status.json",
        "unchanged_local_demonstration_evidence",
        "local_ignored_optional",
    ),
    (
        "implementation/outputs/stage_w_dashboard_smoke_20260730_v3/summary.json",
        "unchanged_local_demonstration_evidence",
        "local_ignored_optional",
    ),
)
MANAGED_ARTIFACT_GLOBS = (
    ".gitattributes",
    "FINAL_RELEASE_INDEX.md",
    "README.md",
    "implementation/README.md",
    "implementation/pyproject.toml",
    "implementation/stage_w_requirements.txt",
    "implementation/configs/p3_stage_r_recommended_default_20260729.yaml",
    "implementation/data/STAGE_U_1_MODEL_ASSETS.md",
    "implementation/data/STAGE_W_1_ARTIFACT_REGISTRY.yaml",
    "implementation/data/STAGE_W_2_*.md",
    "implementation/data/STAGE_W_2_SOURCE_COMMIT_MANIFEST.yaml",
    "implementation/data/STAGE_W_PERMISSION_AND_PROVENANCE.md",
    "implementation/data/STAGE_W_REPRODUCTION_GUIDE.md",
    "implementation/scripts/run_p3_tt.py",
    "implementation/scripts/verify_stage_v_w_registries.py",
    "implementation/scripts/verify_stage_w_2_checkout.py",
    "implementation/src/parking_occupancy/artifact_registry.py",
    "implementation/src/parking_occupancy/integrated_cli.py",
    "implementation/src/parking_occupancy/p3_tt_runtime.py",
    "implementation/src/parking_occupancy/stage_t_tracktrack.py",
    "implementation/src/parking_occupancy/stage_v*.py",
    "implementation/src/parking_occupancy/stage_w*.py",
    "implementation/src/parking_occupancy/stage_w_web/templates/*.html",
    "implementation/src/parking_occupancy/stage_w_web/static/*.css",
    "implementation/tests/test_stage_v_multimode.py",
    "implementation/tests/test_stage_w*.py",
)


def write_w2_registry(repository_root: Path) -> dict[str, Any]:
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
    registry_path = repository_root / REGISTRY_RELATIVE
    payload = {
        "schema_version": 1,
        "registry_id": "STAGE-W-2-CROSS-PLATFORM-RELEASE-20260730-01",
        "stage": "W.2",
        "created_at": "2026-07-30",
        "status": "LOCAL_SOURCE_CANDIDATE_NOT_PUBLIC",
        "public_release_ready": False,
        "model_training_run": False,
        "model_inference_run": False,
        "historical_registry_policy": {
            "stage_v_1": "pre_hardening_historical_snapshot",
            "stage_w": "pre_hardening_historical_snapshot",
            "stage_w_1": "pre_w2_historical_source_snapshot",
        },
        "artifact_count": len(records),
        "registry_self_path": REGISTRY_RELATIVE,
        "artifact_root": "repository_root",
        "local_optional_policy": (
            "existing ignored demonstration artifacts are hash-verified when "
            "present and optional_unavailable in the source-only checkout"
        ),
        "managed_artifact_globs": list(MANAGED_ARTIFACT_GLOBS),
        "artifacts": records,
    }
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


def verify_current_checkout(repository_root: Path) -> dict[str, Any]:
    repository_root = repository_root.resolve()
    errors: list[str] = []
    formal_config = repository_root / FORMAL_CONFIG_RELATIVE
    formal_bytes = formal_config.read_bytes()
    formal_sha256 = sha256_file(formal_config)
    if b"\r\n" in formal_bytes:
        errors.append("formal_config_crlf")
    if formal_sha256 != FORMAL_CONFIG_SHA256:
        errors.append("formal_config_sha256")

    attributes = _run(
        "git_check_attr",
        [
            "git",
            "check-attr",
            "text",
            "eol",
            "--",
            FORMAL_CONFIG_RELATIVE,
        ],
        cwd=repository_root,
    )
    normalized_output = attributes["stdout"].replace("\\", "/")
    if (
        not attributes["passed"]
        or f"{FORMAL_CONFIG_RELATIVE}: text: set" not in normalized_output
        or f"{FORMAL_CONFIG_RELATIVE}: eol: lf" not in normalized_output
    ):
        errors.append("formal_config_git_attributes")

    manifest = resolve_source_manifest(
        repository_root,
        repository_root / SOURCE_MANIFEST_RELATIVE,
    )
    if not manifest["verified"]:
        errors.extend(
            f"source_manifest:{item}" for item in manifest["violations"]
        )

    registry_path = repository_root / REGISTRY_RELATIVE
    if not registry_path.is_file():
        errors.append("w2_registry_missing")
        registry: dict[str, Any] = {
            "verified": False,
            "errors": ["missing"],
        }
    else:
        registry = verify_artifact_registry(
            registry_path,
            artifact_root=repository_root,
        )
        if not registry["verified"]:
            errors.extend(
                f"w2_registry:{item}" for item in registry["errors"]
            )
    return {
        "schema_version": 1,
        "stage": "W.2",
        "verified": not errors,
        "errors": errors,
        "formal_config_sha256": formal_sha256,
        "formal_config_has_crlf": b"\r\n" in formal_bytes,
        "git_attributes": attributes,
        "source_manifest": {
            key: value
            for key, value in manifest.items()
            if key not in {
                "candidate_files",
                "included_by_category",
                "excluded_by_category",
            }
        },
        "registry": registry,
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
    inventory = resolve_source_manifest(repository_root)
    if not inventory["verified"]:
        raise RuntimeError(inventory["violations"])

    temp_parent = Path(
        tempfile.mkdtemp(prefix="stage_w_2_cross_platform_")
    )
    source = temp_parent / "stage_w_2_source_candidate"
    clone = temp_parent / "stage_w_2_autocrlf_clone"
    copied = copy_source_candidate(
        repository_root=repository_root,
        destination=source,
        candidates=inventory["candidate_files"],
    )
    env = os.environ.copy()
    python = str(Path(sys.executable).resolve())
    setup = [
        _run("git_init", ["git", "init", "-q"], cwd=source),
        _run(
            "git_identity_name",
            ["git", "config", "user.name", "Stage W.2 Verification"],
            cwd=source,
        ),
        _run(
            "git_identity_email",
            ["git", "config", "user.email", "stage-w2-verification.invalid"],
            cwd=source,
        ),
        _run("git_add_temp_only", ["git", "add", "-A"], cwd=source),
        _run(
            "git_commit_temp_only",
            ["git", "commit", "-qm", "temporary W.2 source candidate"],
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

    clone_env = env.copy()
    clone_env["PYTHONPATH"] = os.pathsep.join(
        [
            str(clone / "implementation" / "src"),
            str(clone / "implementation" / "literature_core" / "src"),
        ]
    )
    checks = [
        _run(
            "w2_registry_manifest_and_config",
            [
                python,
                "implementation/scripts/verify_stage_w_2_checkout.py",
                "--current-only",
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
            [
                python,
                "implementation/scripts/run_p3_tt.py",
                "--help",
            ],
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
    formal_config = clone / FORMAL_CONFIG_RELATIVE
    clone_formal_sha256 = sha256_file(formal_config)
    clone_formal_has_crlf = b"\r\n" in formal_config.read_bytes()
    forbidden = [
        relative
        for relative in inventory["candidate_files"]
        if Path(relative).suffix.lower()
        in {".pt", ".pth", ".ckpt", ".onnx", ".engine"}
        or any(
            part.lower() in {"outputs", "runs", "raw", "processed"}
            for part in Path(relative).parts
        )
    ]
    passed = (
        persist_autocrlf["passed"]
        and all(item["passed"] for item in checks)
        and clone_formal_sha256 == FORMAL_CONFIG_SHA256
        and not clone_formal_has_crlf
        and not forbidden
    )
    payload = {
        "schema_version": 1,
        "stage": "W.2",
        "verified": passed,
        "model_training_run": False,
        "model_inference_run": False,
        "public_remote_connected": False,
        "temporary_repository_only": True,
        "source_copy": copied,
        "setup": setup,
        "clone": clone_result,
        "persist_autocrlf": persist_autocrlf,
        "checks": checks,
        "formal_config_sha256": clone_formal_sha256,
        "formal_config_has_crlf": clone_formal_has_crlf,
        "forbidden_candidate_files": forbidden,
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
            "Verify the W.2 LF/hash/source boundary, optionally through a "
            "temporary core.autocrlf=true Git clone."
        )
    )
    parser.add_argument(
        "--current-only",
        action="store_true",
        help="Verify the current checkout without creating temporary Git repositories.",
    )
    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep the temporary source and clone for manual inspection.",
    )
    parser.add_argument(
        "--write-registry",
        action="store_true",
        help="Regenerate the W.2 registry from the declared local artifact set.",
    )
    args = parser.parse_args()
    if args.write_registry:
        result = write_w2_registry(REPOSITORY_ROOT)
    elif args.current_only:
        result = verify_current_checkout(REPOSITORY_ROOT)
    else:
        result = simulate_windows_checkout(
            REPOSITORY_ROOT,
            keep=args.keep_temp,
        )
    print(json.dumps(result, indent=2, ensure_ascii=False))
    raise SystemExit(0 if result["verified"] else 1)


if __name__ == "__main__":
    main()
