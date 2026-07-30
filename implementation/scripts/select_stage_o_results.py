from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_n_lmot import sha256_file
from parking_occupancy.stage_o_low_light import (
    STAGE_O_PROTOCOL_ID,
    load_stage_o_protocol,
    select_stage_o_candidate,
)


def _load_metrics(path: Path, expected_method: str) -> dict[str, Any]:
    path = path.resolve()
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("protocol_id") != STAGE_O_PROTOCOL_ID:
        raise ValueError(f"Unexpected protocol in {path}")
    if payload.get("method_id") != expected_method:
        raise ValueError(
            f"Expected {expected_method} metrics, got "
            f"{payload.get('method_id')} in {path}"
        )
    if payload.get("tracker_emitted_boxes") is not False:
        raise ValueError(f"{path} is not raw detector-only evidence")
    return payload


def _input_record(path: Path) -> dict[str, Any]:
    path = path.resolve()
    return {
        "path": str(path),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply the frozen Stage O detector selection rule to completed "
            "raw detector-only metrics without threshold reselection."
        )
    )
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--o0", type=Path, required=True)
    parser.add_argument("--o1", type=Path, required=True)
    parser.add_argument("--o2", type=Path)
    parser.add_argument("--o3", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite {output}")
    protocol_path = args.protocol.resolve()
    protocol = load_stage_o_protocol(protocol_path)
    baseline = _load_metrics(args.o0, "O0")
    candidates = {
        "O1": _load_metrics(args.o1, "O1"),
        "O3": _load_metrics(args.o3, "O3"),
    }
    input_paths = {
        "protocol": protocol_path,
        "O0": args.o0.resolve(),
        "O1": args.o1.resolve(),
        "O3": args.o3.resolve(),
    }
    blocked = {
        "GLARE": (
            "blocked_before_download_or_build: local Python 3.8, CUDA 11.3 "
            "native toolchain, nvcc and cl were unavailable"
        )
    }
    if args.o2 is not None:
        candidates["O2"] = _load_metrics(args.o2, "O2")
        input_paths["O2"] = args.o2.resolve()
    else:
        blocked["O2"] = "formal detector-only diagnostic did not complete"

    selection = select_stage_o_candidate(
        protocol=protocol,
        baseline_metrics=baseline,
        candidate_metrics=candidates,
        blocked_methods=blocked,
    )
    selection["input_metrics"] = {
        label: _input_record(path)
        for label, path in sorted(input_paths.items())
    }
    selection["claim_scope"] = (
        "consumed-development raw detector-only LMOT diagnostic"
    )
    selection["occupancy_improvement_claim"] = False
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(selection, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(selection, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
