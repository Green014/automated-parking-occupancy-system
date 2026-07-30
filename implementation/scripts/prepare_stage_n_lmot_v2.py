from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import audit_lmot_sequence
from parking_occupancy.stage_n_lmot_v2 import (
    EXPECTED_VALIDATION_SEQUENCES,
    build_file_manifest,
    class_distribution,
    discover_split_tar_parts,
    extract_validation_annotations,
    inspect_or_extract_rgb_split_tar,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit split LMOT RGB train/val tar files and extract validation "
            "RGB plus annotations without creating a joined tar."
        )
    )
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--light-parts-dir", type=Path, required=True)
    parser.add_argument("--dark-parts-dir", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--file-manifest", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_root = args.validation_root.resolve()
    partial_root = validation_root.with_name(validation_root.name + ".partial")
    inventory_path = args.inventory.resolve()
    manifest_path = args.file_manifest.resolve()
    for path in (validation_root, partial_root, inventory_path, manifest_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    light_parts = discover_split_tar_parts(
        args.light_parts_dir, "LMOT_light_rgb_trainval.tar"
    )
    dark_parts = discover_split_tar_parts(
        args.dark_parts_dir, "LMOT_dark_rgb_trainval.tar"
    )
    partial_root.mkdir(parents=True)
    status = "extracting"
    payload: dict[str, object] = {
        "schema_version": 2,
        "protocol": "STAGE-N-V2-LMOT-TRACKING-DIAGNOSTIC-20260729-01",
        "created_at": datetime.now().astimezone().isoformat(),
        "network_download_performed": False,
        "source": "user_supplied_official_baidu_netdisk_release",
        "transport_policy": {
            "train_members_allowed_in_archive": True,
            "train_members_extracted": False,
            "validation_only_extraction": True,
            "joined_tar_created": False,
            "raw_extracted": False,
            "real_extracted": False,
        },
    }
    try:
        payload["annotations"] = extract_validation_annotations(
            archive_path=args.annotations,
            extract_to=partial_root,
        )
        payload["light_rgb"] = inspect_or_extract_rgb_split_tar(
            parts=light_parts,
            archive_root="LMOT_light_rgb_trainval",
            image_directory="img_light_rgb",
            expected_extension=".jpg",
            extract_to=partial_root,
        )
        payload["dark_rgb"] = inspect_or_extract_rgb_split_tar(
            parts=dark_parts,
            archive_root="LMOT_dark_rgb_trainval",
            image_directory="img_dark_rgb",
            expected_extension=".png",
            extract_to=partial_root,
        )
        audits = {
            sequence: audit_lmot_sequence(partial_root / sequence)
            for sequence in EXPECTED_VALIDATION_SEQUENCES
        }
        failed = [
            sequence
            for sequence, audit in audits.items()
            if not audit["passed"]
        ]
        if failed:
            raise RuntimeError(f"LMOT sequence audit failed: {failed}")
        payload["sequence_audits"] = audits
        payload["class_distribution"] = class_distribution(partial_root)
        payload["status"] = "approved_validation_only_extraction"
        status = "approved_validation_only_extraction"

        manifest = build_file_manifest(partial_root)
        manifest["root"] = str(validation_root)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        payload["file_manifest"] = {
            "path": str(manifest_path),
            "file_count": manifest["file_count"],
            "total_bytes": manifest["total_bytes"],
            "manifest_sha256": manifest["manifest_sha256"],
        }
        os.replace(partial_root, validation_root)
        payload["validation_root"] = str(validation_root)
    except Exception as exc:
        payload["status"] = "blocked"
        payload["error"] = f"{type(exc).__name__}: {exc}"
        status = "blocked"
        raise
    finally:
        inventory_path.parent.mkdir(parents=True, exist_ok=True)
        inventory_path.write_text(
            yaml.safe_dump(
                payload, sort_keys=False, allow_unicode=True, width=100
            ),
            encoding="utf-8",
        )
        print(
            json.dumps(
                {
                    "status": status,
                    "inventory": str(inventory_path),
                    "partial_root": (
                        str(partial_root) if partial_root.exists() else None
                    ),
                    "validation_root": (
                        str(validation_root)
                        if validation_root.exists()
                        else None
                    ),
                },
                indent=2,
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
