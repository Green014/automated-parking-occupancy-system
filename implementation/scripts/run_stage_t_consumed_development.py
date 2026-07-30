from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


IMPLEMENTATION_ROOT = Path(__file__).resolve().parents[1]
SRC = IMPLEMENTATION_ROOT / "src"
LITERATURE_CORE_SRC = IMPLEMENTATION_ROOT / "literature_core" / "src"
for source_root in (LITERATURE_CORE_SRC, SRC):
    if str(source_root) not in sys.path:
        sys.path.insert(0, str(source_root))

from parking_occupancy.stage_t_tracktrack import (
    P3_TT_CONFIG_NAME,
    analyze_stage_t_outputs,
    prepare_consumed_development_inputs,
    run_stage_t_variant,
    write_stage_t_comparison,
)


DOCUMENTS_IMPLEMENTATION = (
    Path.home() / "Documents" / "停车场识别系统项目" / "implementation"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run TT0/TT1 on already-consumed VIRAT 0502 development data. "
            "This is not an untouched test."
        )
    )
    parser.add_argument(
        "--video",
        type=Path,
        default=(
            DOCUMENTS_IMPLEMENTATION
            / "literature_core"
            / "datasets"
            / "virat"
            / "screening"
            / "videos_round4"
            / "VIRAT_S_050202_10_002159_002233.mp4"
        ),
    )
    parser.add_argument(
        "--d1-weights",
        type=Path,
        default=(
            DOCUMENTS_IMPLEMENTATION
            / "outputs"
            / "d1_ndispark_formal_20260727_v1"
            / "weights"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--e1b-checkpoint",
        type=Path,
        default=(
            DOCUMENTS_IMPLEMENTATION
            / "literature_core"
            / "outputs"
            / "mobilenet_variant_ablation"
            / "cbam_supplement"
            / "best.pt"
        ),
    )
    parser.add_argument(
        "--truth-yaml",
        type=Path,
        default=(
            IMPLEMENTATION_ROOT
            / "literature_core"
            / "data"
            / "annotations"
            / "virat_0502_departure_truth.yaml"
        ),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=(
            IMPLEMENTATION_ROOT
            / "outputs"
            / "stage_t_tracktrack_consumed_dev_20260729"
        ),
    )
    parser.add_argument("--device", default="0")
    parser.add_argument("--classifier-batch-size", type=int, default=64)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    os.environ.setdefault(
        "YOLO_CONFIG_DIR",
        str(IMPLEMENTATION_ROOT / "outputs" / ".ultralytics_config"),
    )
    data_dir = IMPLEMENTATION_ROOT / "data" / "stage_t"
    inputs = prepare_consumed_development_inputs(
        truth_yaml=args.truth_yaml,
        output_dir=data_dir,
    )
    slots_path = Path(inputs["slot_map"]["path"])
    truth_path = Path(inputs["truth_csv"]["path"])
    config_path = IMPLEMENTATION_ROOT / "configs" / P3_TT_CONFIG_NAME
    source_id = str(inputs["source_video_id"])

    summaries: dict[str, object] = {}
    for variant_id, backend in (("TT0", "none"), ("TT1", "tracktrack")):
        output = args.output_root / variant_id.lower()
        summaries[variant_id] = run_stage_t_variant(
            variant_id=variant_id,
            tracker_backend=backend,
            input_path=args.video,
            slots_path=slots_path,
            detector_weights=args.d1_weights,
            classifier_checkpoint=args.e1b_checkpoint,
            output_root=output,
            config_path=config_path,
            truth_path=truth_path,
            source_id=source_id,
            device=args.device,
            classifier_batch_size=args.classifier_batch_size,
        )

    comparison = analyze_stage_t_outputs(
        truth_path=truth_path,
        tt0_root=args.output_root / "tt0",
        tt1_root=args.output_root / "tt1",
    )
    write_stage_t_comparison(
        comparison,
        json_path=data_dir / "STAGE_T_TT0_TT1_COMPARISON.json",
        csv_path=data_dir / "STAGE_T_TT0_TT1_COMPARISON.csv",
    )
    print(
        json.dumps(
            {
                "protocol_id": comparison["protocol_id"],
                "claim_class": comparison["claim_class"],
                "summaries": summaries,
                "comparison": comparison,
            },
            indent=2,
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
