from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.stage_k_data_gate import (
    CandidateGroup,
    prepare_stage_k_candidate,
)


DEFAULT_GROUPS = [
    CandidateGroup("parking2", "sunny", "2012-09-15", 30),
    CandidateGroup("parking1a", "cloudy", "2013-01-16", 30),
    CandidateGroup("parking1b", "sunny", "2013-04-14", 30),
]


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare a no-prediction PKLot Stage K candidate from a local "
            "official archive or preserved partial archive."
        )
    )
    parser.add_argument("--source-archive", type=Path, required=True)
    parser.add_argument("--prior-manifest", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest-out", type=Path, required=True)
    parser.add_argument("--annotations-out", type=Path, required=True)
    parser.add_argument("--audit-out", type=Path, required=True)
    parser.add_argument("--contact-sheet-out", type=Path, required=True)
    parser.add_argument(
        "--reuse-existing",
        action="store_true",
        help=(
            "Verify and reuse a previously extracted candidate directory; "
            "never overwrite mismatching files."
        ),
    )
    args = parser.parse_args()

    audit = prepare_stage_k_candidate(
        archive_path=args.source_archive.resolve(),
        prior_manifest=args.prior_manifest.resolve(),
        output_root=args.output_root.resolve(),
        manifest_path=args.manifest_out.resolve(),
        annotations_path=args.annotations_out.resolve(),
        audit_path=args.audit_out.resolve(),
        contact_sheet_path=args.contact_sheet_out.resolve(),
        groups=DEFAULT_GROUPS,
        official_archive_bytes=3_860_376_865,
        official_archive_url=(
            "https://www.inf.ufpr.br/lesoliveira/download/PKLot.tar.gz"
        ),
        reuse_existing=args.reuse_existing,
    )
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
