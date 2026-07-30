from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parking_occupancy.stage_o_low_light import (
    O1Parameters,
    VALIDATION_SEQUENCES,
    run_detector_only_evaluation,
)
from parking_occupancy.stage_o_enhancement import RetinexformerPreprocessor


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run Stage O raw detector-only evaluation through YOLO.predict. "
            "No tracker is loaded or called."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validation-root", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--method", choices=("O0", "O1", "O2", "O3"), required=True
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--enhancer-repository", type=Path)
    parser.add_argument("--enhancer-weights", type=Path)
    parser.add_argument("--sequences", nargs="+", default=list(VALIDATION_SEQUENCES))
    parser.add_argument(
        "--illumination",
        choices=("both", "light", "dark"),
        default="both",
    )
    parser.add_argument("--internal-o1-threshold", type=float)
    parser.add_argument("--internal-o1-gamma", type=float)
    parser.add_argument("--internal-o1-clahe-clip", type=float)
    args = parser.parse_args()

    override = None
    supplied = (
        args.internal_o1_threshold,
        args.internal_o1_gamma,
        args.internal_o1_clahe_clip,
    )
    if any(value is not None for value in supplied):
        if args.method != "O1" or not all(value is not None for value in supplied):
            parser.error(
                "all three internal O1 parameters require --method O1"
            )
        override = O1Parameters(
            threshold=args.internal_o1_threshold,
            gamma=args.internal_o1_gamma,
            clahe_clip_limit=args.internal_o1_clahe_clip,
        )
    directories = {
        "both": ("img_light_rgb", "img_dark_rgb"),
        "light": ("img_light_rgb",),
        "dark": ("img_dark_rgb",),
    }[args.illumination]
    preprocessor = None
    if args.method == "O2":
        if (
            args.enhancer_repository is None
            or args.enhancer_weights is None
        ):
            parser.error(
                "O2 requires --enhancer-repository and --enhancer-weights"
            )
        preprocessor = RetinexformerPreprocessor(
            repository=args.enhancer_repository,
            weights=args.enhancer_weights,
            device=args.device,
        )
    elif (
        args.enhancer_repository is not None
        or args.enhancer_weights is not None
    ):
        parser.error("Enhancer arguments are valid only for O2")
    metrics = run_detector_only_evaluation(
        protocol_path=args.config,
        validation_root=args.validation_root,
        class_map_path=args.class_map,
        weights_path=args.weights,
        output_dir=args.output_dir,
        method_id=args.method,
        sequences=tuple(args.sequences),
        illumination_directories=directories,
        device=args.device,
        o1_parameters_override=override,
        image_preprocessor=preprocessor,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
