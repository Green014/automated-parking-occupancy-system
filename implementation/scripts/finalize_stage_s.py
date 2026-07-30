from __future__ import annotations

import json
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_s_release import (
    compare_registry_snapshots,
    historical_registry_snapshot,
    submission_candidate_audit,
    write_stage_s_registry,
    write_submission_audit,
)


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    implementation_root = IMPLEMENTATION_ROOT
    repository_root = implementation_root.parent
    stage_dir = implementation_root / "data" / "stage_s"

    before_path = stage_dir / "STAGE_S_HISTORICAL_REGISTRY_PRE_SNAPSHOT.json"
    before = json.loads(before_path.read_text(encoding="utf-8"))
    after = historical_registry_snapshot(implementation_root)
    after_path = stage_dir / "STAGE_S_HISTORICAL_REGISTRY_POST_SNAPSHOT.json"
    _write_json(after_path, after)

    gate = compare_registry_snapshots(before, after)
    gate["status"] = "PASS" if gate["unchanged"] else "FAIL"
    gate["interpretation"] = (
        "All Stage L-R registry files and their entry-state binding results are unchanged."
        if gate["unchanged"]
        else "At least one Stage L-R registry file or entry-state binding result changed."
    )
    gate_path = stage_dir / "STAGE_S_HISTORICAL_REGISTRY_GATE.json"
    _write_json(gate_path, gate)
    if not gate["unchanged"]:
        raise RuntimeError(f"Historical Stage L-R registry gate failed: {gate['changes']}")

    audit = submission_candidate_audit(repository_root)
    write_submission_audit(
        audit,
        json_path=stage_dir / "STAGE_S_SUBMISSION_AUDIT.json",
        csv_path=stage_dir / "STAGE_S_SUBMISSION_CANDIDATES.csv",
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"Submission audit failed: {audit['violations']}")

    artifacts = [
        (repository_root / ".gitignore", "submission exclusion rules"),
        (implementation_root / "pyproject.toml", "final CLI registration"),
        (
            implementation_root
            / "configs"
            / "p3_stage_r_recommended_default_20260729.yaml",
            "Stage R recommended final default configuration",
        ),
        (
            implementation_root
            / "src"
            / "parking_occupancy"
            / "integrated_cli.py",
            "final default user entry",
        ),
        (
            implementation_root
            / "src"
            / "parking_occupancy"
            / "stage_s_release.py",
            "Stage S release audit implementation",
        ),
        (
            implementation_root
            / "src"
            / "parking_occupancy"
            / "stage_s_demo.py",
            "Stage S frozen-output renderer",
        ),
        (
            implementation_root / "scripts" / "audit_stage_s_release.py",
            "Stage S preflight audit command",
        ),
        (
            implementation_root / "scripts" / "render_stage_s_demo.py",
            "Stage S rendering command",
        ),
        (
            implementation_root / "scripts" / "finalize_stage_s.py",
            "Stage S finalization command",
        ),
        (
            implementation_root / "tests" / "test_stage_s_release.py",
            "Stage S release tests",
        ),
        (
            implementation_root / "tests" / "test_stage_s_demo.py",
            "Stage S demo tests",
        ),
        (
            implementation_root
            / "data"
            / "STAGE_S_FINAL_DEFAULT_AND_DEMO_REPORT.md",
            "Stage S report",
        ),
        (
            implementation_root / "data" / "SYSTEM_RELEASE_INDEX.md",
            "additive final release index",
        ),
        (
            stage_dir / "STAGE_S_FINAL_SYSTEM_EVIDENCE.csv",
            "corrected final evidence table",
        ),
        (
            stage_dir / "STAGE_S_SUBMISSION_AUDIT.json",
            "submission candidate audit summary",
        ),
        (
            stage_dir / "STAGE_S_SUBMISSION_CANDIDATES.csv",
            "submission candidate inventory",
        ),
        (before_path, "historical registry entry snapshot"),
        (after_path, "historical registry exit snapshot"),
        (gate_path, "historical registry unchanged gate"),
        (
            stage_dir / "demo" / "demo_main.mp4",
            "50-second frozen-output default demo",
        ),
        (
            stage_dir / "demo" / "demo_keyframe_default.png",
            "default pipeline demo keyframe",
        ),
        (
            stage_dir / "demo" / "demo_keyframe_d1_vs_d1ll.png",
            "D1 versus D1-LL keyframe",
        ),
        (
            stage_dir / "demo" / "demo_keyframe_f2_recovery.png",
            "B1-to-F2 recovery keyframe",
        ),
        (
            stage_dir / "demo" / "STAGE_S_DEMO_METADATA.json",
            "demo source and codec metadata",
        ),
    ]
    registry_path = stage_dir / "STAGE_S_ARTIFACT_REGISTRY_20260729.yaml"
    result = write_stage_s_registry(
        registry_path,
        implementation_root=repository_root,
        artifacts=artifacts,
    )
    print(
        json.dumps(
            {
                "historical_gate": gate["status"],
                "historical_registry_count": gate["historical_registry_count"],
                "candidate_files": audit["candidate_files"],
                "candidate_bytes": audit["candidate_bytes"],
                "stage_s_registry": result,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
