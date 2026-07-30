from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_o_artifacts import verify_detector_only_output
from parking_occupancy.stage_o_low_light import STAGE_O_PROTOCOL_ID


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify formal Stage O detector-only output contracts, count "
            "identities, and explicit no-tracker runtime flags."
        )
    )
    parser.add_argument(
        "--method-root",
        action="append",
        required=True,
        help="METHOD=PATH, for example O0=implementation/outputs/stage_o_o0...",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    roots: dict[str, Path] = {}
    for value in args.method_root:
        method, separator, path = value.partition("=")
        if not separator or method not in {"O0", "O1", "O2", "O3"}:
            raise ValueError(f"Expected O0|O1|O2|O3=PATH, got {value}")
        if method in roots:
            raise ValueError(f"Duplicate method {method}")
        roots[method] = Path(path)
    results = {
        method: verify_detector_only_output(path, expected_method=method)
        for method, path in sorted(roots.items())
    }
    payload = {
        "schema_version": 1,
        "protocol_id": STAGE_O_PROTOCOL_ID,
        "operation": "formal_detector_only_output_verification",
        "inference_api_required": "ultralytics.YOLO.predict",
        "model_track_required": False,
        "tracker_loading_required": False,
        "results": results,
        "verified": all(row["verified"] for row in results.values()),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "verified": payload["verified"],
                "methods": list(results),
                "output": str(output),
            },
            indent=2,
        )
    )
    if not payload["verified"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
