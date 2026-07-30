from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
SRC = IMPLEMENTATION_ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from parking_occupancy.artifact_registry import (
    STAGE_V_1_PRE_HARDENING_REGISTRY_SHA256,
    STAGE_W_1_PRE_W2_REGISTRY_SHA256,
    STAGE_W_2_PRE_W3_REGISTRY_SHA256,
    STAGE_W_PRE_HARDENING_REGISTRY_SHA256,
    verify_artifact_registry,
    verify_historical_artifact_registry,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify historical Stage V.1/W/W.1/W.2 snapshots and current W.3 "
            "release registry."
        )
    )
    parser.add_argument(
        "registries",
        nargs="*",
        type=Path,
        help="Optional current-format registries to verify instead of defaults",
    )
    args = parser.parse_args()
    if args.registries:
        results = [
            verify_artifact_registry(path, artifact_root=IMPLEMENTATION_ROOT)
            for path in args.registries
        ]
    else:
        results = [
            verify_historical_artifact_registry(
                IMPLEMENTATION_ROOT
                / "data"
                / "STAGE_V_1_ARTIFACT_REGISTRY.yaml",
                artifact_root=IMPLEMENTATION_ROOT,
                expected_registry_sha256=(
                    STAGE_V_1_PRE_HARDENING_REGISTRY_SHA256
                ),
                immutable_path_prefixes=("outputs",),
            ),
            verify_historical_artifact_registry(
                IMPLEMENTATION_ROOT
                / "data"
                / "STAGE_W_ARTIFACT_REGISTRY.yaml",
                artifact_root=IMPLEMENTATION_ROOT,
                expected_registry_sha256=(
                    STAGE_W_PRE_HARDENING_REGISTRY_SHA256
                ),
                immutable_path_prefixes=("outputs",),
            ),
            verify_historical_artifact_registry(
                IMPLEMENTATION_ROOT
                / "data"
                / "STAGE_W_2_ARTIFACT_REGISTRY.yaml",
                artifact_root=IMPLEMENTATION_ROOT.parent,
                expected_registry_sha256=STAGE_W_2_PRE_W3_REGISTRY_SHA256,
                immutable_path_prefixes=(),
                classification="pre_w3_historical_source_snapshot",
            ),
            verify_historical_artifact_registry(
                IMPLEMENTATION_ROOT
                / "data"
                / "STAGE_W_1_ARTIFACT_REGISTRY.yaml",
                artifact_root=IMPLEMENTATION_ROOT.parent,
                expected_registry_sha256=(
                    STAGE_W_1_PRE_W2_REGISTRY_SHA256
                ),
                immutable_path_prefixes=("implementation/outputs",),
                classification="pre_w2_historical_source_snapshot",
            ),
            verify_artifact_registry(
                IMPLEMENTATION_ROOT
                / "data"
                / "STAGE_W_3_ARTIFACT_REGISTRY.yaml",
                artifact_root=IMPLEMENTATION_ROOT.parent,
            ),
        ]
    print(json.dumps(results, indent=2, ensure_ascii=False))
    raise SystemExit(0 if all(result["verified"] for result in results) else 1)


if __name__ == "__main__":
    main()
