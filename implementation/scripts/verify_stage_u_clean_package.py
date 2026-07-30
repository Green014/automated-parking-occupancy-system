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


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_u_portable_release import (
    PORTABLE_REGISTRY_RELATIVE,
    copy_clean_package,
    submission_candidates,
)

LOCAL_ONLY_ARTIFACT_TESTS = (
    (
        "tests/test_stage_s_release.py::"
        "test_historical_registry_entry_state_is_unchanged"
    ),
    "tests/test_stage_n_lmot.py::test_stage_n_protocol_preserves_frozen_inputs",
    (
        "tests/test_stage_n_lmot_v2.py::"
        "test_stage_n_v2_protocol_preserves_original_stage_n"
    ),
    (
        "tests/test_stage_q_v2_evaluation.py::"
        "test_real_preflight_verifies_confirmation_and_all_manifest_images"
    ),
    (
        "tests/test_stage_q_v2_evaluation.py::"
        "test_preflight_refuses_existing_output_root"
    ),
    (
        "tests/test_stage_r_component_attribution.py::"
        "test_repository_stage_r_matches_independent_sanity_values"
    ),
    (
        "tests/test_stage_s_demo.py::"
        "test_demo_plan_uses_required_frozen_consecutive_segments"
    ),
    (
        "tests/test_stage_s_demo.py::"
        "test_demo_recoveries_and_failures_follow_frozen_fields"
    ),
)


def _run(
    name: str,
    command: list[str],
    *,
    cwd: Path,
    env: dict[str, str],
    timeout: int = 300,
) -> dict[str, Any]:
    started = __import__("time").perf_counter()
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
        "cwd": str(cwd),
        "exit_code": result.returncode,
        "elapsed_s": __import__("time").perf_counter() - started,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "passed": result.returncode == 0,
    }


def _remove_readonly(function, path: str, _error: object) -> None:
    """Make Git's read-only object files writable before removing a temp tree."""
    os.chmod(path, stat.S_IWRITE)
    function(path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and verify a temporary Stage U clean submission package."
    )
    parser.add_argument("--keep", action="store_true")
    parser.add_argument(
        "--no-record",
        action="store_true",
        help="Run read-only final verification without rewriting the repository audit.",
    )
    args = parser.parse_args()

    candidates = submission_candidates(REPOSITORY_ROOT)
    temp_parent = Path(tempfile.mkdtemp(prefix="stage_u_clean_parent_"))
    clean_root = temp_parent / "portable_release"
    package = copy_clean_package(
        repository_root=REPOSITORY_ROOT,
        destination=clean_root,
        candidates=candidates,
    )
    python = Path(sys.executable).resolve()
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        [
            str(clean_root / "implementation" / "src"),
            str(clean_root / "implementation" / "literature_core" / "src"),
        ]
    )
    env["STAGE_U_PORTABLE_PACKAGE"] = "1"
    env["YOLO_CONFIG_DIR"] = str(temp_parent / "ultralytics_config")

    setup = [
        _run(
            "clean_git_init",
            ["git", "init", "-q"],
            cwd=clean_root,
            env=env,
        ),
        _run(
            "clean_git_add",
            ["git", "add", "-A"],
            cwd=clean_root,
            env=env,
        ),
    ]
    if not all(row["passed"] for row in setup):
        raise RuntimeError(f"Could not initialize clean-package test Git state: {setup}")

    implementation = clean_root / "implementation"
    tests = [
        _run(
            "stage_u_targeted",
            [
                str(python),
                "-m",
                "pytest",
                "tests/test_stage_u_p3_tt_runtime.py",
                "tests/test_stage_u_portable_release.py",
                "tests/test_stage_u_1_presentation.py",
                "tests/test_stage_u_1_release.py",
                "-o",
                "addopts=",
                "-q",
            ],
            cwd=implementation,
            env=env,
        ),
        _run(
            "standard_implementation_portable",
            [
                str(python),
                "-m",
                "pytest",
                "tests",
                "-o",
                "addopts=",
                "-q",
            ],
            cwd=implementation,
            env=env,
        ),
        _run(
            "literature_core",
            [
                str(python),
                "-m",
                "pytest",
                "tests",
                "-o",
                "addopts=",
                "-q",
            ],
            cwd=implementation / "literature_core",
            env=env,
        ),
        _run(
            "compileall",
            [
                str(python),
                "-m",
                "compileall",
                "-q",
                "src",
                "scripts",
                "tests",
                "literature_core/src",
                "literature_core/scripts",
                "literature_core/tests",
            ],
            cwd=implementation,
            env=env,
        ),
        _run(
            "portable_registry",
            [
                str(python),
                "-c",
                (
                    "from pathlib import Path; "
                    "from parking_occupancy.stage_u_portable_release "
                    "import verify_portable_registry; "
                    f"r=Path(r'{PORTABLE_REGISTRY_RELATIVE}'); "
                    "x=verify_portable_registry(r, package_root=Path('.'), "
                    "require_complete_coverage=True); "
                    "print(x); raise SystemExit(0 if x['verified'] else 1)"
                ),
            ],
            cwd=clean_root,
            env=env,
        ),
    ]
    passed = all(row["passed"] for row in tests)
    payload = {
        "schema_version": 1,
        "protocol_id": "STAGE-U-PORTABLE-FINAL-RELEASE-20260730-01",
        "status": "PASS" if passed else "FAIL",
        "verification_phase": (
            "phase_2_final_no_record"
            if args.no_record
            else "phase_1_pre_final_audit_and_registry"
        ),
        "candidate_total_is_final": args.no_record,
        "final_candidate_recheck_command": (
            "python scripts/verify_stage_u_clean_package.py --no-record"
        ),
        "clean_package": package,
        "contains_git_test_harness_only": True,
        "git_test_harness_in_candidate_count": False,
        "contains_outputs_datasets_weights_or_venv": False,
        "local_only_artifact_tests_skipped": list(LOCAL_ONLY_ARTIFACT_TESTS),
        "skip_reason": (
            "The Stage U portable-package environment marks these tests skipped "
            "because they explicitly require local-only historical registries, "
            "model weights, datasets, or ignored runtime outputs."
        ),
        "official_trackeval_optional_tests_auto_skip_without_dependency": True,
        "official_trackeval_run_separately_when_installed": True,
        "setup": setup,
        "tests": tests,
        "temporary_directory_removed_after_success": False,
    }
    result_path = (
        IMPLEMENTATION_ROOT
        / "data"
        / "stage_u"
        / "STAGE_U_CLEAN_PACKAGE_VERIFICATION.json"
    )
    if passed and not args.keep:
        try:
            shutil.rmtree(temp_parent, onexc=_remove_readonly)
            payload["temporary_directory_removed_after_success"] = True
        except OSError as exc:
            payload["temporary_directory_cleanup_error"] = str(exc)
    if not args.no_record:
        result_path.write_text(
            json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
