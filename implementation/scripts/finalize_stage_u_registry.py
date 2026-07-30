from __future__ import annotations

import json
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = IMPLEMENTATION_ROOT.parent
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.stage_u_portable_release import (
    PORTABLE_REGISTRY_RELATIVE,
    audit_submission,
    sha256_file,
    submission_candidates,
    verify_saved_submission_evidence,
    write_portable_registry,
    write_submission_audit,
)


def main() -> None:
    stage_u = IMPLEMENTATION_ROOT / "data" / "stage_u"
    audit_json = stage_u / "STAGE_U_SUBMISSION_AUDIT.json"
    audit_csv = stage_u / "STAGE_U_SUBMISSION_CANDIDATES.csv"
    registry_path = REPOSITORY_ROOT / PORTABLE_REGISTRY_RELATIVE
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.touch(exist_ok=True)

    # The audit never stores hashes for itself or the registry. Iterate until
    # the three publication files have a stable size/hash signature. This
    # produces a one-way chain: audit/CSV -> explicit exclusion marker, then
    # registry -> actual audit/CSV hashes, with registry self-hash omitted.
    previous_signature: tuple[tuple[int, str], ...] | None = None
    result: dict[str, object] | None = None
    evidence: dict[str, object] | None = None
    for _ in range(24):
        audit = audit_submission(REPOSITORY_ROOT)
        write_submission_audit(
            audit,
            json_path=audit_json,
            csv_path=audit_csv,
        )
        candidates = submission_candidates(REPOSITORY_ROOT)
        result = write_portable_registry(
            registry_path,
            repository_root=REPOSITORY_ROOT,
            candidates=candidates,
        )
        signature = tuple(
            (path.stat().st_size, sha256_file(path))
            for path in (audit_json, audit_csv, registry_path)
        )
        evidence = verify_saved_submission_evidence(
            repository_root=REPOSITORY_ROOT,
            audit_json_path=audit_json,
            audit_csv_path=audit_csv,
            registry_path=registry_path,
        )
        if signature == previous_signature and evidence["verified"]:
            break
        previous_signature = signature
    else:
        raise RuntimeError(
            "Stage U publication files did not reach a stable evidence state"
        )
    if audit["status"] != "PASS" or evidence is None or not evidence["verified"]:
        raise RuntimeError(
            f"Submission evidence failed: audit={audit['violations']} "
            f"evidence={evidence}"
        )
    print(
        json.dumps(
            {
                "audit": {
                    "candidate_files": audit["candidate_files"],
                    "candidate_bytes": audit["candidate_bytes"],
                },
                "portable_registry": result,
                "saved_evidence_recheck": evidence,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
