from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.data import (  # noqa: E402
    load_pklot_slot_samples,
    read_image,
    write_image,
)
from literature_core.error_analysis import (  # noqa: E402
    METHODS,
    classify_ablation_record,
    summarize_ablation_records,
)
from literature_core.patches import extract_slot_patch  # noqa: E402


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _tile(patch: np.ndarray, row: dict[str, Any]) -> np.ndarray:
    width, height = 360, 330
    tile = np.full((height, width, 3), 245, dtype=np.uint8)
    resized = cv2.resize(patch, (224, 224), interpolation=cv2.INTER_AREA)
    tile[100:324, 68:292] = resized
    lines = [
        f"{row['sample_id']}  slot {row['slot_id']}  truth={row['truth']}",
        (
            f"E0 {row['probabilities']['E0']:.3f}"
            f"->{row['predictions']['E0']}  "
            f"E1 {row['probabilities']['E1']:.3f}"
            f"->{row['predictions']['E1']}"
        ),
        (
            f"E2 {row['probabilities']['E2']:.3f}"
            f"->{row['predictions']['E2']}  "
            f"E3 {row['probabilities']['E3']:.3f}"
            f"->{row['predictions']['E3']}"
        ),
        f"errors: {row['error_signature']}",
    ]
    for index, text in enumerate(lines):
        cv2.putText(
            tile,
            text,
            (8, 20 + index * 23),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.46,
            (20, 20, 20),
            1,
            cv2.LINE_AA,
        )
    border = (
        (35, 170, 35)
        if row["correct"]["E3"]
        else (40, 40, 220)
    )
    cv2.rectangle(tile, (1, 1), (width - 2, height - 2), border, 4)
    return tile


def _montage(tiles: list[np.ndarray], columns: int = 4) -> np.ndarray:
    if not tiles:
        return np.full((120, 480, 3), 255, dtype=np.uint8)
    rows = math.ceil(len(tiles) / columns)
    height, width = tiles[0].shape[:2]
    canvas = np.full((rows * height, columns * width, 3), 230, dtype=np.uint8)
    for index, tile in enumerate(tiles):
        row, column = divmod(index, columns)
        canvas[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = tile
    return canvas


def _markdown(summary: dict[str, Any]) -> str:
    development = summary["development"]
    test = summary["test"]
    return f"""# Frozen Ablation Error Analysis

This is a post-hoc explanation of saved E0-E3 probabilities. It does not
select or change any threshold, fusion weight, model, or test prediction.

| Split | Samples | E0 errors | E1 errors | E2 errors | E3 errors | Any error |
|---|---:|---:|---:|---:|---:|---:|
| UFPR04 development | {development['samples']} | {development['errors']['E0']} | {development['errors']['E1']} | {development['errors']['E2']} | {development['errors']['E3']} | {development['any_method_error']} |
| UFPR05 pilot holdout | {test['samples']} | {test['errors']['E0']} | {test['errors']['E1']} | {test['errors']['E2']} | {test['errors']['E3']} | {test['any_method_error']} |

## UFPR05 branch interaction

- both E1/E2 correct: {test['branch_patterns'].get('both_correct', 0)}
- classifier only correct: {test['branch_patterns'].get('classifier_only_correct', 0)}
- YOLO-World only correct: {test['branch_patterns'].get('world_only_correct', 0)}
- both branches wrong: {test['branch_patterns'].get('both_wrong', 0)}
- fusion correct while at least one branch is wrong:
  {test['fusion_rescued_branch_error']}
- fusion wrong despite both branches being correct:
  {test['fusion_harmed_both_correct']}
- all four methods wrong: {test['all_methods_wrong']}

The montage includes every UFPR05 slot for which at least one method is
wrong. A green border means E3 is correct; a red border means E3 is wrong.
The purpose is attribution, not test-set model selection.
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explain frozen E0-E3 slot errors without retuning"
    )
    parser.add_argument("--probabilities", required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--split-config", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    parameters = json.loads(Path(args.parameters).read_text(encoding="utf-8"))
    classified = [
        classify_ablation_record(row, parameters)
        for row in _read_csv(Path(args.probabilities))
    ]
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in classified:
        by_split[str(row["split"])].append(row)
    summary = {
        "protocol": {
            "analysis_only": True,
            "parameters_changed": False,
            "parameters": parameters,
            "test_used_for_selection": False,
        },
        "development": summarize_ablation_records(by_split["development"]),
        "test": summarize_ablation_records(by_split["test"]),
    }

    samples = load_pklot_slot_samples(
        args.annotations,
        args.project_root,
        args.split_config,
    )
    sample_lookup = {
        (sample.split, sample.sample_id, sample.slot_id): sample
        for sample in samples
    }
    image_cache: dict[Path, np.ndarray] = {}
    error_rows = [
        row
        for row in by_split["test"]
        if row["error_signature"] != "none"
    ]
    tiles = []
    for row in sorted(
        error_rows,
        key=lambda item: (str(item["sample_id"]), int(item["slot_id"])),
    ):
        sample = sample_lookup[
            (str(row["split"]), str(row["sample_id"]), str(row["slot_id"]))
        ]
        if sample.image_path not in image_cache:
            image_cache[sample.image_path] = read_image(sample.image_path)
        patch = extract_slot_patch(
            image_cache[sample.image_path],
            sample.points,
        )
        tiles.append(_tile(patch, row))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    flat_fields = [
        "split",
        "sample_id",
        "source",
        "group_id",
        "slot_id",
        "truth",
        "p_baseline",
        "p_cls",
        "p_world",
        "p_fusion",
        "pred_E0",
        "pred_E1",
        "pred_E2",
        "pred_E3",
        "error_signature",
        "branch_pattern",
        "fusion_rescued_branch_error",
        "fusion_harmed_both_correct",
    ]
    with (output_dir / "error_rows.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for row in classified:
            writer.writerow(
                {
                    "split": row["split"],
                    "sample_id": row["sample_id"],
                    "source": row["source"],
                    "group_id": row["group_id"],
                    "slot_id": row["slot_id"],
                    "truth": row["truth"],
                    "p_baseline": row["probabilities"]["E0"],
                    "p_cls": row["probabilities"]["E1"],
                    "p_world": row["probabilities"]["E2"],
                    "p_fusion": row["probabilities"]["E3"],
                    "pred_E0": row["predictions"]["E0"],
                    "pred_E1": row["predictions"]["E1"],
                    "pred_E2": row["predictions"]["E2"],
                    "pred_E3": row["predictions"]["E3"],
                    "error_signature": row["error_signature"],
                    "branch_pattern": row["branch_pattern"],
                    "fusion_rescued_branch_error": row[
                        "fusion_rescued_branch_error"
                    ],
                    "fusion_harmed_both_correct": row[
                        "fusion_harmed_both_correct"
                    ],
                }
            )
    write_image(output_dir / "test_error_montage.jpg", _montage(tiles))
    (output_dir / "ERROR_ANALYSIS.md").write_text(
        _markdown(summary),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
