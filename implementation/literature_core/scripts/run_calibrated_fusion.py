"""Fit and audit unified calibrated fusion on camera-grouped OOF predictions."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.calibration import (  # noqa: E402
    CalibratedFusionModel,
    calibration_metrics,
    reliability_bins,
)
from literature_core.metrics import (  # noqa: E402
    evaluate_probabilities,
    select_threshold,
)


METRIC_NAMES = (
    "macro_f1",
    "occupied_recall",
    "vacant_recall",
    "false_free_rate",
    "false_occupied_rate",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Camera-grouped internal development of E3b calibrated fusion "
            "from saved out-of-fold branch predictions"
        )
    )
    parser.add_argument(
        "--fold",
        action="append",
        required=True,
        help="Fold specification NAME=path/to/branch_probabilities.csv",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--frozen-config", type=Path, required=True)
    parser.add_argument("--calibration-bins", type=int, default=10)
    return parser.parse_args()


def load_fold(specification: str) -> tuple[str, list[dict[str, Any]]]:
    if "=" not in specification:
        raise ValueError("--fold must use NAME=PATH")
    name, path_text = specification.split("=", 1)
    path = Path(path_text)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        source_rows = list(csv.DictReader(handle))
    rows: list[dict[str, Any]] = []
    for row in source_rows:
        if row["split"] != "test":
            continue
        rows.append(
            {
                "origin_fold": name,
                "sample_id": row["sample_id"],
                "source": row["source"].lower(),
                "group_id": row["group_id"],
                "slot_id": row["slot_id"],
                "truth": int(row["truth"]),
                "p_cls": float(row["p_cls"]),
                "p_det": float(row["p_world"]),
                "p_baseline": float(row["p_baseline"]),
            }
        )
    if not rows:
        raise ValueError(f"{path} has no historical test-partition rows")
    sources = {row["source"] for row in rows}
    if len(sources) != 1:
        raise ValueError(f"fold {name} has multiple evaluation cameras: {sources}")
    return name, rows


def columns(
    rows: list[dict[str, Any]],
) -> tuple[list[int], list[float], list[float]]:
    return (
        [row["truth"] for row in rows],
        [row["p_cls"] for row in rows],
        [row["p_det"] for row in rows],
    )


def choose_weighted_fusion(
    rows: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    truth, p_cls, p_det = columns(rows)
    candidates: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for index in range(21):
        classifier_weight = index / 20
        detector_weight = 1.0 - classifier_weight
        probabilities = [
            classifier_weight * cls_value + detector_weight * det_value
            for cls_value, det_value in zip(p_cls, p_det, strict=True)
        ]
        threshold, sensitivity = select_threshold(truth, probabilities)
        metrics = next(
            row for row in sensitivity if row["threshold"] == threshold
        )
        candidate = {
            "classifier_weight": classifier_weight,
            "detector_weight": detector_weight,
            **metrics,
        }
        candidates.append(candidate)
        if selected is None or (
            candidate["macro_f1"],
            -candidate["false_free_rate"],
            -abs(candidate["classifier_weight"] - 0.5),
        ) > (
            selected["macro_f1"],
            -selected["false_free_rate"],
            -abs(selected["classifier_weight"] - 0.5),
        ):
            selected = candidate
    assert selected is not None
    return selected, candidates


def error_overlap(
    truth: list[int],
    classifier_state: list[int],
    detector_state: list[int],
    fusion_state: list[int],
) -> dict[str, int]:
    cls_correct = [
        prediction == target
        for prediction, target in zip(classifier_state, truth, strict=True)
    ]
    det_correct = [
        prediction == target
        for prediction, target in zip(detector_state, truth, strict=True)
    ]
    fusion_correct = [
        prediction == target
        for prediction, target in zip(fusion_state, truth, strict=True)
    ]
    return {
        "both_branches_correct": sum(
            left and right for left, right in zip(cls_correct, det_correct, strict=True)
        ),
        "classifier_only_correct": sum(
            left and not right
            for left, right in zip(cls_correct, det_correct, strict=True)
        ),
        "detector_only_correct": sum(
            right and not left
            for left, right in zip(cls_correct, det_correct, strict=True)
        ),
        "both_branches_wrong": sum(
            not left and not right
            for left, right in zip(cls_correct, det_correct, strict=True)
        ),
        "fusion_rescues_branch_error": sum(
            fused and not (left and right)
            for fused, left, right in zip(
                fusion_correct,
                cls_correct,
                det_correct,
                strict=True,
            )
        ),
        "fusion_wrong_when_both_branches_correct": sum(
            not fused and left and right
            for fused, left, right in zip(
                fusion_correct,
                cls_correct,
                det_correct,
                strict=True,
            )
        ),
    }


def aggregate_folds(folds: list[dict[str, Any]]) -> dict[str, Any]:
    methods = sorted(folds[0]["evaluation"])
    aggregate: dict[str, Any] = {}
    for method in methods:
        aggregate[method] = {}
        for metric in METRIC_NAMES:
            values = [
                float(fold["evaluation"][method][metric]) for fold in folds
            ]
            aggregate[method][metric] = {
                "camera_equal_mean": statistics.fmean(values),
                "population_std": statistics.pstdev(values),
                "values": values,
            }
    return aggregate


def main() -> None:
    args = parse_args()
    if args.calibration_bins <= 0:
        raise ValueError("--calibration-bins must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    loaded = [load_fold(specification) for specification in args.fold]
    all_rows = [row for _, rows in loaded for row in rows]
    cameras = sorted({row["source"] for row in all_rows})
    if len(cameras) != len(loaded):
        raise ValueError(
            "each fold must contribute exactly one different OOF camera"
        )

    fold_reports: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    reliability_rows: list[dict[str, Any]] = []
    threshold_rows: list[dict[str, Any]] = []
    weighted_sensitivity_rows: list[dict[str, Any]] = []

    for held_camera in cameras:
        train_rows = [row for row in all_rows if row["source"] != held_camera]
        evaluation_rows = [
            row for row in all_rows if row["source"] == held_camera
        ]
        train_truth, train_cls, train_det = columns(train_rows)
        eval_truth, eval_cls, eval_det = columns(evaluation_rows)

        calibrated = CalibratedFusionModel.fit(
            train_cls,
            train_det,
            train_truth,
        )
        train_cls_cal, train_det_cal = calibrated.predict_branches(
            train_cls,
            train_det,
        )
        eval_cls_cal, eval_det_cal = calibrated.predict_branches(
            eval_cls,
            eval_det,
        )
        train_fusion = calibrated.fusion.predict(
            train_cls_cal,
            train_det_cal,
        )
        eval_fusion = calibrated.fusion.predict(
            eval_cls_cal,
            eval_det_cal,
        )

        e1_raw_threshold, _ = select_threshold(train_truth, train_cls)
        e2_raw_threshold, _ = select_threshold(train_truth, train_det)
        e1_cal_threshold, _ = select_threshold(train_truth, train_cls_cal)
        e2_cal_threshold, _ = select_threshold(train_truth, train_det_cal)
        e3b_threshold, e3b_sensitivity = select_threshold(
            train_truth,
            train_fusion,
        )
        e3a_selected, e3a_sensitivity = choose_weighted_fusion(train_rows)
        eval_e3a = [
            e3a_selected["classifier_weight"] * cls_value
            + e3a_selected["detector_weight"] * det_value
            for cls_value, det_value in zip(eval_cls, eval_det, strict=True)
        ]

        evaluation = {
            "E1a_raw": evaluate_probabilities(
                eval_truth,
                eval_cls,
                e1_raw_threshold,
            ),
            "E1a_calibrated": evaluate_probabilities(
                eval_truth,
                eval_cls_cal,
                e1_cal_threshold,
            ),
            "E2_raw_evidence": evaluate_probabilities(
                eval_truth,
                eval_det,
                e2_raw_threshold,
            ),
            "E2_calibrated": evaluate_probabilities(
                eval_truth,
                eval_det_cal,
                e2_cal_threshold,
            ),
            "E3a_weighted": evaluate_probabilities(
                eval_truth,
                eval_e3a,
                float(e3a_selected["threshold"]),
            ),
            "E3b_calibrated_logistic": evaluate_probabilities(
                eval_truth,
                eval_fusion,
                e3b_threshold,
            ),
        }
        calibration = {
            "evaluation": {
                "p_cls_raw": calibration_metrics(
                    eval_truth,
                    eval_cls,
                    n_bins=args.calibration_bins,
                ),
                "p_cls_calibrated": calibration_metrics(
                    eval_truth,
                    eval_cls_cal,
                    n_bins=args.calibration_bins,
                ),
                "p_det_raw_evidence_as_score": calibration_metrics(
                    eval_truth,
                    eval_det,
                    n_bins=args.calibration_bins,
                ),
                "p_det_calibrated": calibration_metrics(
                    eval_truth,
                    eval_det_cal,
                    n_bins=args.calibration_bins,
                ),
                "p_fusion_e3b": calibration_metrics(
                    eval_truth,
                    eval_fusion,
                    n_bins=args.calibration_bins,
                ),
            }
        }
        fold_reports.append(
            {
                "held_camera": held_camera,
                "fit_cameras": sorted(
                    {row["source"] for row in train_rows}
                ),
                "fit_samples": len(train_rows),
                "evaluation_samples": len(evaluation_rows),
                "parameters": {
                    "E3a": {
                        key: e3a_selected[key]
                        for key in (
                            "classifier_weight",
                            "detector_weight",
                            "threshold",
                        )
                    },
                    "E3b": {
                        **calibrated.to_dict(),
                        "occupied_threshold": e3b_threshold,
                    },
                    "branch_thresholds": {
                        "e1_raw": e1_raw_threshold,
                        "e1_calibrated": e1_cal_threshold,
                        "e2_raw": e2_raw_threshold,
                        "e2_calibrated": e2_cal_threshold,
                    },
                },
                "evaluation": evaluation,
                "calibration": calibration,
                "error_overlap": error_overlap(
                    eval_truth,
                    [
                        int(value >= e1_cal_threshold)
                        for value in eval_cls_cal
                    ],
                    [
                        int(value >= e2_cal_threshold)
                        for value in eval_det_cal
                    ],
                    [
                        int(value >= e3b_threshold)
                        for value in eval_fusion
                    ],
                ),
            }
        )

        for row in e3b_sensitivity:
            threshold_rows.append(
                {
                    "held_camera": held_camera,
                    "method": "E3b_calibrated_logistic",
                    **row,
                }
            )
        for row in e3a_sensitivity:
            weighted_sensitivity_rows.append(
                {
                    "held_camera": held_camera,
                    **row,
                }
            )
        curves = {
            "p_cls_raw": eval_cls,
            "p_cls_calibrated": eval_cls_cal,
            "p_det_raw_evidence_as_score": eval_det,
            "p_det_calibrated": eval_det_cal,
            "p_fusion_e3b": eval_fusion,
        }
        for branch, values in curves.items():
            for bin_row in reliability_bins(
                eval_truth,
                values,
                n_bins=args.calibration_bins,
            ):
                reliability_rows.append(
                    {
                        "held_camera": held_camera,
                        "partition": "camera_grouped_internal_evaluation",
                        "branch": branch,
                        **bin_row,
                    }
                )

        for source, p_cls_cal, p_det_cal, p_fused, p_e3a in zip(
            evaluation_rows,
            eval_cls_cal,
            eval_det_cal,
            eval_fusion,
            eval_e3a,
            strict=True,
        ):
            prediction_rows.append(
                {
                    **source,
                    "held_camera": held_camera,
                    "p_cls_calibrated": p_cls_cal,
                    "p_det_calibrated": p_det_cal,
                    "p_e3a": p_e3a,
                    "p_e3b": p_fused,
                    "e3a_state": int(
                        p_e3a >= float(e3a_selected["threshold"])
                    ),
                    "e3b_state": int(p_fused >= e3b_threshold),
                }
            )

    final_truth, final_cls, final_det = columns(all_rows)
    final_model = CalibratedFusionModel.fit(
        final_cls,
        final_det,
        final_truth,
    )
    final_probabilities = final_model.predict(final_cls, final_det)
    final_threshold, _ = select_threshold(final_truth, final_probabilities)
    frozen_config = {
        "schema_version": 1,
        "method_id": "E3b",
        "implementation_label": "calibrated_nonnegative_logistic_fusion",
        "paper_exact_reproduction": False,
        "development_scope": {
            "dataset": "PKLot selected 27-image method-development set",
            "selection": (
                "camera-grouped OOF predictions; final fit uses all three "
                "development cameras"
            ),
            "external_holdout_used": False,
            "samples": len(all_rows),
            "cameras": cameras,
        },
        "evidence_semantics": {
            "p_cls": "MobileNetV3 occupied-class softmax score",
            "p_det": (
                "YOLO-World confidence x slot-coverage evidence; not a "
                "native probability"
            ),
        },
        "calibration": {
            "classifier": final_model.classifier_calibrator.to_dict(),
            "detector": final_model.detector_calibrator.to_dict(),
        },
        "fusion": {
            "type": "nonnegative_logistic_on_calibrated_log_odds",
            **final_model.fusion.to_dict(),
            "occupied_threshold": final_threshold,
        },
        "temporal": {
            "retune_on_grand_bassin": False,
            "status": "not selected without mixed continuous truth",
        },
    }
    args.frozen_config.parent.mkdir(parents=True, exist_ok=True)
    args.frozen_config.write_text(
        yaml.safe_dump(frozen_config, sort_keys=False),
        encoding="utf-8",
    )

    report = {
        "protocol": {
            "role": "internal_method_development_only",
            "source_rows": (
                "historical held-camera rows, one OOF prediction set per "
                "PKLot camera"
            ),
            "grouping": "leave_one_camera_out",
            "slot_level_random_split": False,
            "external_holdout_used": False,
            "p_det_is_probability": False,
        },
        "samples": len(all_rows),
        "cameras": cameras,
        "folds": fold_reports,
        "aggregate": aggregate_folds(fold_reports),
        "frozen_config": frozen_config,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    write_csv(args.output_dir / "predictions.csv", prediction_rows)
    write_csv(args.output_dir / "reliability_curves.csv", reliability_rows)
    write_csv(args.output_dir / "threshold_sensitivity.csv", threshold_rows)
    write_csv(
        args.output_dir / "weighted_fusion_sensitivity.csv",
        weighted_sensitivity_rows,
    )
    print(
        json.dumps(
            {
                "samples": len(all_rows),
                "cameras": cameras,
                "aggregate": report["aggregate"],
                "frozen_config": str(args.frozen_config.resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
