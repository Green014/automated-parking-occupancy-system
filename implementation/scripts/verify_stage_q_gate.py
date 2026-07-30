from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROJECT_SRC = PROJECT_ROOT / "src"
if str(PROJECT_SRC) not in sys.path:
    sys.path.insert(0, str(PROJECT_SRC))

from parking_occupancy.stage_q_external import (
    execute_formal_run_if_authorized,
    load_candidate_gate,
    validate_frozen_comparison,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Verify the frozen Stage Q candidate gate without loading a model."
        )
    )
    parser.add_argument(
        "--gate",
        type=Path,
        default=PROJECT_ROOT
        / "data/stage_q/STAGE_Q_CANDIDATE_GATE_20260729.yaml",
    )
    parser.add_argument(
        "--p3-defaults",
        type=Path,
        default=PROJECT_ROOT
        / "configs/p3_integrated_runtime_defaults_20260729.yaml",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    gate = load_candidate_gate(args.gate)
    defaults = yaml.safe_load(args.p3_defaults.read_text(encoding="utf-8"))
    comparison = validate_frozen_comparison(gate, defaults)
    callback_called = False

    def forbidden_model_callback() -> None:
        nonlocal callback_called
        callback_called = True
        raise AssertionError("Blocked Stage Q gate loaded a model")

    blocked = False
    try:
        execute_formal_run_if_authorized(gate, forbidden_model_callback)
    except ValueError:
        blocked = True
    if not blocked or callback_called:
        raise RuntimeError("Stage Q blocked gate did not stop before model load")
    print(
        json.dumps(
            {
                "status": gate["status"],
                "formal_inference_authorized": False,
                "model_callback_called": callback_called,
                "comparison": comparison,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
