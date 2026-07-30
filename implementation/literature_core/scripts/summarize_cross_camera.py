from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.cross_validation import (  # noqa: E402
    METHODS,
    summarize_cross_camera_folds,
)


def _markdown(report: dict) -> str:
    rows = []
    for fold in report["folds"]:
        rows.append(
            "| {fold} | {train} | {dev} | {test} | {e0:.6f} | "
            "{e1:.6f} | {e2:.6f} | "
            "{e3:.6f} |".format(
                fold=fold["fold"],
                train=fold["train_camera"],
                dev=fold["development_camera"],
                test=fold["test_camera"],
                **{
                    method.lower(): fold["methods"][method]["macro_f1"]
                    for method in METHODS
                },
            )
        )
    aggregate = report["aggregate"]
    rows.append(
        "| Mean | - | - | three cameras | {e0:.6f} | {e1:.6f} | "
        "{e2:.6f} | "
        "{e3:.6f} |".format(
            **{
                method.lower(): aggregate[method]["macro_f1"]["mean"]
                for method in METHODS
            }
        )
    )
    rows.append(
        "| Population std | - | - | three cameras | {e0:.6f} | "
        "{e1:.6f} | "
        "{e2:.6f} | {e3:.6f} |".format(
            **{
                method.lower(): aggregate[method]["macro_f1"][
                    "population_std"
                ]
                for method in METHODS
            }
        )
    )
    comparisons = "\n".join(
        (
            f"- {method} vs E0: "
            f"{aggregate[method]['macro_f1_vs_E0']['wins']} wins, "
            f"{aggregate[method]['macro_f1_vs_E0']['ties']} ties, "
            f"{aggregate[method]['macro_f1_vs_E0']['losses']} losses; "
            f"mean delta "
            f"{aggregate[method]['macro_f1_vs_E0']['mean_delta']:.6f}"
        )
        for method in ("E1", "E2", "E3")
    )
    selected_rows = "\n".join(
        (
            f"| {fold['fold']} | "
            f"{fold['selected_parameters']['e1_threshold']:.2f} | "
            f"{fold['selected_parameters']['e2_threshold']:.2f} | "
            f"{fold['selected_parameters']['e3_classifier_weight']:.2f} | "
            f"{fold['selected_parameters']['e3_detector_weight']:.2f} | "
            f"{fold['selected_parameters']['e3_threshold']:.2f} |"
        )
        for fold in report["folds"]
    )
    return f"""# Three-Camera Rotation Robustness Study

This is a post-hoc robustness study added after the original UFPR05 pilot
holdout was observed. Within every fold, training, development selection, and
test cameras remain disjoint. Results are not used to rewrite the original
confirmatory interpretation or default parameters.

| Fold | Train | Development | Test | E0 macro F1 | E1 macro F1 | E2 macro F1 | E3 macro F1 |
|---|---|---|---|---:|---:|---:|---:|
{chr(10).join(rows)}

## Wins versus E0

{comparisons}

## Development-selected parameters

| Fold | E1 threshold | E2 threshold | E3 w_cls | E3 w_det | E3 threshold |
|---|---:|---:|---:|---:|---:|
{selected_rows}
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Aggregate frozen E0-E3 metrics across camera folds"
    )
    parser.add_argument(
        "--fold",
        action="append",
        required=True,
        help="Fold specification NAME=path/to/metrics.json",
    )
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    folds = []
    for specification in args.fold:
        if "=" not in specification:
            raise ValueError("--fold must use NAME=PATH")
        name, path = specification.split("=", 1)
        folds.append(
            (
                name,
                json.loads(Path(path).read_text(encoding="utf-8")),
            )
        )
    report = summarize_cross_camera_folds(folds)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    fields = [
        "fold",
        "train_camera",
        "development_camera",
        "test_camera",
        "method",
        "macro_f1",
        "occupied_recall",
        "vacant_recall",
        "false_free_rate",
        "false_occupied_rate",
        "macro_f1_delta_vs_E0",
    ]
    with (output_dir / "fold_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for fold in report["folds"]:
            for method, metrics in fold["methods"].items():
                writer.writerow(
                    {
                        "fold": fold["fold"],
                        "train_camera": fold["train_camera"],
                        "development_camera": fold["development_camera"],
                        "test_camera": fold["test_camera"],
                        "method": method,
                        **{
                            field: metrics[field]
                            for field in fields
                            if field in metrics
                        },
                    }
                )
    (output_dir / "REPORT.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
