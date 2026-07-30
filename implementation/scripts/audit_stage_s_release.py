from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_s_release import (
    historical_registry_snapshot,
    load_stage_s_config,
    submission_candidate_audit,
    write_final_evidence_table,
    write_submission_audit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit Stage S final defaults and submission candidates."
    )
    parser.add_argument("--repository-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument(
        "--config",
        type=Path,
        default=(
            IMPLEMENTATION_ROOT
            / "configs"
            / "p3_stage_r_recommended_default_20260729.yaml"
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=IMPLEMENTATION_ROOT / "data" / "stage_s",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    repository_root = args.repository_root.resolve()
    implementation_root = repository_root / "implementation"
    output_dir = args.output_dir.resolve()
    config = load_stage_s_config(args.config)
    audit = submission_candidate_audit(repository_root)
    if audit["status"] != "PASS":
        raise RuntimeError(f"Submission audit failed: {audit['violations']}")
    write_submission_audit(
        audit,
        json_path=output_dir / "STAGE_S_SUBMISSION_AUDIT.json",
        csv_path=output_dir / "STAGE_S_SUBMISSION_CANDIDATES.csv",
    )
    history = historical_registry_snapshot(implementation_root)
    (output_dir / "STAGE_S_HISTORICAL_REGISTRY_PRE_SNAPSHOT.json").write_text(
        json.dumps(history, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_final_evidence_table(output_dir / "STAGE_S_FINAL_SYSTEM_EVIDENCE.csv")
    print(
        json.dumps(
            {
                "config_id": config["config_id"],
                "default_temporal": config["temporal"]["default_enabled"],
                "default_tracker": config["tracking"]["default_backend"],
                "candidate_files": audit["candidate_files"],
                "candidate_bytes": audit["candidate_bytes"],
                "historical_registries_verified": history[
                    "all_registry_files_and_artifacts_verified"
                ],
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
