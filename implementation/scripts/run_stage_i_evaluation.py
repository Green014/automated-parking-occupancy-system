from __future__ import annotations

import argparse
import json
from pathlib import Path

from parking_occupancy.stage_i_evaluation import (
    analyze_stage_i_v2_max_det_sensitivity,
    count_test_preflight,
    export_development_qualitative_evidence,
    run_count_test,
    run_stage_i_v2_posthoc_count_sensitivity,
    select_detector_and_count_rule,
    select_stage_i_v2_operating_points,
    stage_i_v2_posthoc_count_preflight,
)


def _weights(args: argparse.Namespace) -> dict[str, Path]:
    return {
        "D0": Path(args.d0_weights),
        "D1": Path(args.d1_weights),
        "D2": Path(args.d2_weights),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Freeze Stage I development selection, preflight the count-only "
            "test, or execute the frozen D0/D1/D2 count comparison."
        )
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    select = subparsers.add_parser("select")
    select.add_argument("--comparison-root", required=True)
    select.add_argument("--output", required=True)

    select_v2 = subparsers.add_parser("select-v2")
    select_v2.add_argument("--comparison-root", required=True)
    select_v2.add_argument("--comparison-config", required=True)
    select_v2.add_argument("--output", required=True)

    maxdet = subparsers.add_parser("maxdet-decision")
    maxdet.add_argument("--max-det-300-root", required=True)
    maxdet.add_argument("--max-det-1000-root", required=True)
    maxdet.add_argument("--output", required=True)

    qualitative = subparsers.add_parser("qualitative")
    qualitative.add_argument("--comparison-root", required=True)
    qualitative.add_argument("--data", required=True)
    qualitative.add_argument("--confidence-threshold", type=float, required=True)
    qualitative.add_argument("--output-dir", required=True)

    for command in ("preflight-count", "run-count"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--selection-record", required=True)
        subparser.add_argument("--comparison", required=True)
        subparser.add_argument("--comparison-config", required=True)
        subparser.add_argument("--truth-manifest", required=True)
        subparser.add_argument("--test-images", required=True)
        subparser.add_argument("--d0-weights", required=True)
        subparser.add_argument("--d1-weights", required=True)
        subparser.add_argument("--d2-weights", required=True)
        subparser.add_argument("--device", default="auto")
        if command == "run-count":
            subparser.add_argument("--output-dir", required=True)

    for command in ("preflight-posthoc-v2", "run-posthoc-v2"):
        subparser = subparsers.add_parser(command)
        subparser.add_argument("--config", required=True)
        subparser.add_argument("--operating-points", required=True)
        subparser.add_argument("--max-det-decision", required=True)
        subparser.add_argument("--comparison", required=True)
        subparser.add_argument("--comparison-config", required=True)
        subparser.add_argument("--truth-manifest", required=True)
        subparser.add_argument("--test-images", required=True)
        subparser.add_argument("--d0-weights", required=True)
        subparser.add_argument("--d1-weights", required=True)
        subparser.add_argument("--d2-weights", required=True)
        subparser.add_argument("--device", default="auto")
        if command == "run-posthoc-v2":
            subparser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    if args.command == "select":
        report = select_detector_and_count_rule(
            comparison_root=Path(args.comparison_root),
            output_path=Path(args.output),
        )
    elif args.command == "select-v2":
        report = select_stage_i_v2_operating_points(
            comparison_root=Path(args.comparison_root),
            comparison_config=Path(args.comparison_config),
            output_path=Path(args.output),
        )
    elif args.command == "maxdet-decision":
        report = analyze_stage_i_v2_max_det_sensitivity(
            max_det_300_root=Path(args.max_det_300_root),
            max_det_1000_root=Path(args.max_det_1000_root),
            output_path=Path(args.output),
        )
    elif args.command == "qualitative":
        report = export_development_qualitative_evidence(
            comparison_root=Path(args.comparison_root),
            data_yaml=Path(args.data),
            output_root=Path(args.output_dir),
            confidence_threshold=args.confidence_threshold,
        )
    elif args.command == "preflight-count":
        report = count_test_preflight(
            config_path=Path(args.config),
            selection_record=Path(args.selection_record),
            comparison_path=Path(args.comparison),
            comparison_config=Path(args.comparison_config),
            truth_manifest=Path(args.truth_manifest),
            test_images_root=Path(args.test_images),
            weight_paths=_weights(args),
        )
    elif args.command == "run-count":
        report = run_count_test(
            config_path=Path(args.config),
            selection_record=Path(args.selection_record),
            comparison_path=Path(args.comparison),
            comparison_config=Path(args.comparison_config),
            truth_manifest=Path(args.truth_manifest),
            test_images_root=Path(args.test_images),
            output_root=Path(args.output_dir),
            weight_paths=_weights(args),
            device=args.device,
        )
    elif args.command == "preflight-posthoc-v2":
        report = stage_i_v2_posthoc_count_preflight(
            config_path=Path(args.config),
            operating_points_record=Path(args.operating_points),
            max_det_decision=Path(args.max_det_decision),
            comparison_path=Path(args.comparison),
            comparison_config=Path(args.comparison_config),
            truth_manifest=Path(args.truth_manifest),
            test_images_root=Path(args.test_images),
            weight_paths=_weights(args),
        )
    else:
        report = run_stage_i_v2_posthoc_count_sensitivity(
            config_path=Path(args.config),
            operating_points_record=Path(args.operating_points),
            max_det_decision=Path(args.max_det_decision),
            comparison_path=Path(args.comparison),
            comparison_config=Path(args.comparison_config),
            truth_manifest=Path(args.truth_manifest),
            test_images_root=Path(args.test_images),
            output_root=Path(args.output_dir),
            weight_paths=_weights(args),
            device=args.device,
        )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
