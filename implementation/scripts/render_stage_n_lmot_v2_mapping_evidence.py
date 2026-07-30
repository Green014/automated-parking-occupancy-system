from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file
from parking_occupancy.stage_n_lmot_v2 import (
    EXPECTED_VALIDATION_SEQUENCES,
    class_distribution,
    render_class_mapping_contact_sheet,
)


PROPOSED_MAP = {
    1: "person",
    2: "bicycle",
    3: "car",
    4: "motorcycle",
    5: "bus",
    6: "truck",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render and freeze LMOT numeric-class visual evidence"
    )
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--contact-sheet", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_root = args.validation_root.resolve()
    contact_sheet = args.contact_sheet.resolve()
    evidence_path = args.evidence.resolve()
    for path in (contact_sheet, evidence_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite {path}")

    samples = render_class_mapping_contact_sheet(
        validation_root=validation_root,
        id_to_name=PROPOSED_MAP,
        output_path=contact_sheet,
    )
    source_records = []
    for sequence in EXPECTED_VALIDATION_SEQUENCES:
        for relative in (Path("gt") / "gt.txt", Path("seqinfo.ini")):
            path = validation_root / sequence / relative
            source_records.append(
                {
                    "path": str(path),
                    "bytes": path.stat().st_size,
                    "sha256": sha256_file(path),
                }
            )
    evidence = {
        "schema_version": 1,
        "created_at": datetime.now().astimezone().isoformat(),
        "status": "pending_visual_confirmation",
        "scope": "LMOT validation numeric class and active-mark evidence",
        "official_readme": {
            "url": "https://github.com/xinzwang/LMOT",
            "statements": [
                "LMOT is organized in MOTChallenge17 form",
                "the listed category order is person, bicycle, car, motorcycle, bus, truck",
                "validation contains 131781 boxes and 626 tracks",
            ],
        },
        "proposed_id_to_name": PROPOSED_MAP,
        "distribution": class_distribution(validation_root),
        "annotation_field": {
            "released_name": "ignore",
            "observed_values": [1],
            "interpretation": "positive active/evaluated MOT mark",
            "basis": [
                "all 131781 validation rows use value 1",
                "treating 1 as ignored would leave zero evaluable GT",
                "MOTChallenge uses positive mark/conf values for active GT",
            ],
        },
        "contact_sheet": {
            "path": str(contact_sheet),
            "bytes": contact_sheet.stat().st_size,
            "sha256": sha256_file(contact_sheet),
            "green_rectangle": "the released GT box being inspected",
        },
        "samples": samples,
        "source_records": source_records,
        "manual_confirmation": {
            "completed": False,
            "confirmed_mapping": None,
            "review_note": None,
        },
    }
    evidence_path.parent.mkdir(parents=True, exist_ok=True)
    evidence_path.write_text(
        yaml.safe_dump(
            evidence, sort_keys=False, allow_unicode=True, width=100
        ),
        encoding="utf-8",
    )
    print(evidence_path)


if __name__ == "__main__":
    main()
