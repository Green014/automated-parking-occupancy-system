from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_q_v2_upm import (
    STAGE_Q_V2_PROTOCOL_ID,
    extract_zip_safely,
    inspect_test_split,
    inspect_zip_archive,
    render_sequence_contact_sheet,
    write_sequence_inventory,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Safely extract and audit the authorized UPM-GTI test.zip."
    )
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--extract-parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--downloaded-at", required=True)
    return parser.parse_args()


def _write_yaml(path: Path, payload: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()
    output_root = args.output_root.resolve()
    archive = args.archive.resolve()
    archive_audit = inspect_zip_archive(archive)
    test_root = args.extract_parent.resolve() / "test"
    source_audit = {
        "schema_version": 1,
        "protocol_id": STAGE_Q_V2_PROTOCOL_ID,
        "status": "ARCHIVE_VALID",
        "official_dataset": "ETSIT / UPM-GTI Parking Lot Occupancy Database",
        "official_page": "https://gti.ssr.upm.es/data/parking-lot-database",
        "official_storage": "https://drive.upm.es/index.php/s/TdqfDr25NAsGIea",
        "download_url": (
            "https://drive.upm.es/public.php/dav/files/"
            "TdqfDr25NAsGIea/test.zip"
        ),
        "downloaded_at": args.downloaded_at,
        "official_public_download": True,
        "explicit_dataset_license_found": False,
        "use_scope": "local_noncommercial_course_research",
        "redistribution": "prohibited_by_project_policy",
        "attribution_required": True,
        "legal_interpretation_not_claimed": True,
        "paper_license_not_applied_to_dataset_images": True,
        "login_required": False,
        "additional_terms_prompted": False,
        "displayed_size": "239.1 MB",
        "server_content_length": 250698837,
        "server_checksums": {
            "sha1": "9e462c0720eddf92bb11b4eed7d5e0e597112a5f",
            "md5": "3157555f948f225621bc618656b60e75",
        },
        "archive": archive_audit,
        "git_ignored": True,
        "raw_archive_modified": False,
        "model_loaded": False,
        "inference_run": False,
    }
    source_audit_path = (
        output_root / "STAGE_Q_V2_SOURCE_ARCHIVE_AUDIT_20260729.yaml"
    )
    if source_audit_path.exists():
        existing = yaml.safe_load(
            source_audit_path.read_text(encoding="utf-8")
        )
        if (
            existing.get("archive", {}).get("archive_sha256")
            != archive_audit["archive_sha256"]
        ):
            raise RuntimeError("Existing source audit binds another archive")
    else:
        _write_yaml(source_audit_path, source_audit)
    if test_root.exists():
        extracted_root = test_root
    else:
        extracted_root = extract_zip_safely(
            archive,
            args.extract_parent,
        )
    inventories = inspect_test_split(extracted_root)
    write_sequence_inventory(
        output_root / "STAGE_Q_V2_SEQUENCE_INVENTORY_20260729.csv",
        inventories,
    )
    render_sequence_contact_sheet(
        test_root=extracted_root,
        inventories=inventories,
        output_path=(
            output_root / "STAGE_Q_V2_SEQUENCE_CONTACT_SHEET_20260729.png"
        ),
    )
    print(
        yaml.safe_dump(
            {
                "status": "structure_audit_complete",
                "archive": archive_audit,
                "test_root": str(extracted_root),
                "sequence_count": len(inventories),
                "image_count": sum(row.image_count for row in inventories),
                "groundtruth_file_count": len(inventories),
                "auxiliary_low_light_candidates": [
                    row.sequence_id
                    for row in inventories
                    if row.auxiliary_low_light_candidate
                ],
                "inventory_preview": [asdict(row) for row in inventories[:2]],
            },
            sort_keys=False,
            allow_unicode=True,
        )
    )


if __name__ == "__main__":
    main()
