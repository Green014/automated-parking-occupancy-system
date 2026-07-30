from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

LITERATURE_CORE_SRC = (
    Path(__file__).resolve().parents[1] / "literature_core" / "src"
)
if str(LITERATURE_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(LITERATURE_CORE_SRC))

from parking_occupancy.stage_l_classifier_ablation import (  # noqa: E402
    run_classifier_only_ablation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the fixed-threshold Stage L E1b-only ablation."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--p3-predictions", type=Path, required=True)
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()
    report = run_classifier_only_ablation(
        config_path=args.config,
        annotations_path=args.annotations,
        source_root=args.source_root,
        p3_predictions_path=args.p3_predictions,
        classifier_checkpoint=args.classifier_checkpoint,
        output_root=args.output_dir,
        device=args.device,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
