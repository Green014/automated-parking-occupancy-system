from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from parking_occupancy.stage_o_low_light import (
    O1Parameters,
    load_stage_o_protocol,
    run_detector_only_evaluation,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Select the fixed O1 sequence gate and Gamma/CLAHE tuple using "
            "only the frozen LMOT-train internal-development sequences."
        )
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--development-root", type=Path, required=True)
    parser.add_argument("--class-map", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite {args.output_dir}")
    protocol = load_stage_o_protocol(args.config)
    method = protocol["methods"]["O1"]
    grid = method["internal_development_grid"]
    sequences = tuple(protocol["data"]["internal_development_sequences"])
    args.output_dir.mkdir(parents=True)
    candidates = []
    for index, (threshold, gamma, clip) in enumerate(
        itertools.product(
            grid["thresholds"],
            grid["gamma"],
            grid["clahe_clip_limit"],
        ),
        start=1,
    ):
        parameters = O1Parameters(
            threshold=float(threshold),
            gamma=float(gamma),
            clahe_clip_limit=float(clip),
            clahe_tile_grid=tuple(grid["clahe_tile_grid"]),
            calibration_frames=int(method["gate_calibration_frames"]),
        )
        candidate_root = args.output_dir / f"candidate_{index:02d}"
        metrics = run_detector_only_evaluation(
            protocol_path=args.config,
            validation_root=args.development_root,
            class_map_path=args.class_map,
            weights_path=args.weights,
            output_dir=candidate_root,
            method_id="O1",
            sequences=sequences,
            illumination_directories=("img_dark_rgb",),
            device=args.device,
            o1_parameters_override=parameters,
        )
        dark = metrics["illumination"]["dark"]["pooled_micro"]
        latency = metrics["runtime"]["wall_ms_per_frame"]
        candidates.append(
            {
                "parameters": {
                    "threshold": parameters.threshold,
                    "gamma": parameters.gamma,
                    "clahe_clip_limit": parameters.clahe_clip_limit,
                    "clahe_tile_grid": list(parameters.clahe_tile_grid),
                },
                "dark": dark,
                "wall_ms_per_frame": latency,
                "artifact_dir": candidate_root.name,
            }
        )
    candidates.sort(
        key=lambda row: (
            -float(row["dark"]["AP50"]),
            -float(row["dark"]["recall"]),
            -float(row["dark"]["AP50-95"]),
            -float(row["dark"]["precision"]),
            float(row["wall_ms_per_frame"]),
            tuple(row["parameters"].values()),
        )
    )
    result = {
        "schema_version": 1,
        "protocol_id": protocol["protocol_id"],
        "dataset_role": "LMOT_train_internal_development_only",
        "sequences": list(sequences),
        "validation_accessed": False,
        "selection_order": method["internal_selection_order"],
        "selected": candidates[0],
        "candidates": candidates,
    }
    (args.output_dir / "selection.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.output_dir / "config_snapshot.yaml").write_text(
        yaml.safe_dump(protocol, sort_keys=False, allow_unicode=True, width=100),
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
