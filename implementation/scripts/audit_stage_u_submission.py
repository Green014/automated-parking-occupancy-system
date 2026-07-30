from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_s_demo import verify_demo_video
from parking_occupancy.stage_t_demo import verify_stage_t_demo
from parking_occupancy.stage_u_portable_release import (
    audit_historical_registries,
    audit_submission,
    sha256_file,
    write_historical_registry_audit,
    write_submission_audit,
)


def _ffmpeg_audit() -> dict[str, object]:
    executable = shutil.which("ffmpeg")
    if executable is None:
        return {
            "available": False,
            "h264_encoder_detected": False,
            "presentation_copy_generated": False,
            "reason": "No existing ffmpeg executable; no dependency downloaded.",
        }
    result = subprocess.run(
        [executable, "-hide_banner", "-encoders"],
        capture_output=True,
        text=True,
        check=False,
    )
    encoders = result.stdout + result.stderr
    h264 = any(
        name in encoders
        for name in ("libx264", "h264_nvenc", "h264_amf", "h264_qsv")
    )
    return {
        "available": True,
        "path": executable,
        "encoder_query_exit_code": result.returncode,
        "h264_encoder_detected": h264,
        "presentation_copy_generated": False,
        "reason": (
            "Frozen FMP4 originals retained; Stage U does not require or replace "
            "them with a transcoded copy."
        ),
    }


def _demo_audit() -> dict[str, object]:
    stage_s_root = IMPLEMENTATION_ROOT / "data" / "stage_s" / "demo"
    stage_t_root = IMPLEMENTATION_ROOT / "data" / "stage_t" / "demo"
    stage_s_metadata = json.loads(
        (stage_s_root / "STAGE_S_DEMO_METADATA.json").read_text(encoding="utf-8")
    )
    stage_t_metadata = json.loads(
        (stage_t_root / "STAGE_T_DEMO_METADATA.json").read_text(encoding="utf-8")
    )
    stage_s = verify_demo_video(stage_s_root / "demo_main.mp4")
    stage_t = verify_stage_t_demo(
        stage_t_root / "demo_tracktrack_optional.mp4"
    )
    assertions = {
        "stage_s_raw_state": stage_s_metadata["main_state_field"] == "raw_state",
        "stage_s_E4_off": (
            stage_s_metadata["E4_state_used_for_main_visualization"] is False
        ),
        "stage_s_tracker_none": stage_s_metadata["tracker_used"] is False,
        "stage_t_optional_title": (
            stage_t_metadata["title"] == "Optional TrackTrack-enhanced variant"
        ),
        "stage_t_consumed_development": (
            stage_t_metadata["source_claim"] == "consumed-development diagnostic"
        ),
        "stage_t_E4_off": stage_t_metadata["temporal_enabled"] is False,
        "stage_t_no_occupancy_improvement_claim": (
            stage_t_metadata["tracktrack_occupancy_improvement_claimed"] is False
        ),
        "stage_s_frozen_sha256": (
            stage_s["sha256"]
            == "f4e9e59b5bcef1b51f2e94b8443c5f22a69ca850bfc77f5c9b94a1bf947ac608"
        ),
        "stage_t_frozen_sha256": (
            stage_t["sha256"]
            == "b5dfdeb850acdd0a87072a9c48fda44dd5e13725fb7f0e428cfc6164b4d24c1f"
        ),
    }
    return {
        "schema_version": 1,
        "protocol_id": "STAGE-U-PORTABLE-FINAL-RELEASE-20260730-01",
        "status": "PASS" if all(assertions.values()) else "FAIL",
        "frozen_media_modified": False,
        "stage_s": {
            "video": stage_s,
            "metadata_sha256": sha256_file(
                stage_s_root / "STAGE_S_DEMO_METADATA.json"
            ),
        },
        "stage_t": {
            "video": stage_t,
            "metadata_sha256": sha256_file(
                stage_t_root / "STAGE_T_DEMO_METADATA.json"
            ),
        },
        "semantic_assertions": assertions,
        "powerpoint_compatibility": {
            "risk": (
                "FMP4 may not play in some PowerPoint/Windows codec environments."
            ),
            "frozen_originals_replaced": False,
            "h264_tooling": _ffmpeg_audit(),
        },
    }


def main() -> None:
    stage_u = IMPLEMENTATION_ROOT / "data" / "stage_u"
    stage_u.mkdir(parents=True, exist_ok=True)
    historical = audit_historical_registries(REPOSITORY_ROOT)
    write_historical_registry_audit(
        historical,
        json_path=stage_u / "STAGE_U_HISTORICAL_REGISTRY_CLASSIFICATION.json",
        csv_path=stage_u / "STAGE_U_HISTORICAL_REGISTRY_CLASSIFICATION.csv",
    )
    demo = _demo_audit()
    (stage_u / "STAGE_U_DEMO_COMPATIBILITY_AUDIT.json").write_text(
        json.dumps(demo, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    audit_json = stage_u / "STAGE_U_SUBMISSION_AUDIT.json"
    audit_csv = stage_u / "STAGE_U_SUBMISSION_CANDIDATES.csv"
    audit_json.touch(exist_ok=True)
    audit_csv.touch(exist_ok=True)
    previous_signature: tuple[int, int] | None = None
    for _ in range(8):
        audit = audit_submission(REPOSITORY_ROOT)
        write_submission_audit(
            audit,
            json_path=audit_json,
            csv_path=audit_csv,
        )
        signature = (
            audit["candidate_files"],
            audit_submission(REPOSITORY_ROOT)["candidate_bytes"],
        )
        if signature == previous_signature:
            break
        previous_signature = signature
    audit = audit_submission(REPOSITORY_ROOT)
    write_submission_audit(
        audit,
        json_path=audit_json,
        csv_path=audit_csv,
    )
    if audit["status"] != "PASS":
        raise RuntimeError(f"Stage U submission audit failed: {audit['violations']}")
    if demo["status"] != "PASS":
        raise RuntimeError(f"Stage U demo audit failed: {demo}")
    print(
        json.dumps(
            {
                "submission": {
                    "status": audit["status"],
                    "candidate_files": audit["candidate_files"],
                    "candidate_bytes": audit["candidate_bytes"],
                    "largest_file_bytes": audit["largest_file_bytes"],
                },
                "historical_registries": {
                    "count": historical["registry_count"],
                    "portable": historical["portable_count"],
                    "local_only": historical["local_only_count"],
                },
                "demos": demo,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
