from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from literature_core.stability import (  # noqa: E402
    positive_only_stability_metrics,
    probability_summary,
)

Key = tuple[str, int, str]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _key(row: dict[str, str]) -> Key:
    return row["video_id"], int(row["frame_index"]), row["slot_id"]


def _markdown(report: dict[str, Any]) -> str:
    classifier = report["classifier_only"]
    detector = report["detector_only"]
    raw = report["raw_fusion_state"]
    final = report["temporal_state"]
    return f"""# Grand Bassin Positive-Only Stability

The seven polygons are manually checked as continuously occupied for all 793
frames. There are no verified vacant slots or transitions. The table therefore
supports occupied recall, false-free, and stability claims only.

| Output | Occupied recall | Positive-only F1 | False-free | Post-warm-up changes | Flicker / slot-minute |
|---|---:|---:|---:|---:|---:|
| Frozen E1 classifier | {classifier['occupied_recall']:.6f} | {classifier['positive_only_f1']:.6f} | {classifier['false_free_rate']:.6f} | {classifier['post_warmup_unsupported_changes']} | {classifier['post_warmup_flicker_per_slot_minute']:.6f} |
| Frozen E2 detector | {detector['occupied_recall']:.6f} | {detector['positive_only_f1']:.6f} | {detector['false_free_rate']:.6f} | {detector['post_warmup_unsupported_changes']} | {detector['post_warmup_flicker_per_slot_minute']:.6f} |
| Raw frozen E3 | {raw['occupied_recall']:.6f} | {raw['positive_only_f1']:.6f} | {raw['false_free_rate']:.6f} | {raw['post_warmup_unsupported_changes']} | {raw['post_warmup_flicker_per_slot_minute']:.6f} |
| E3 + generic hysteresis | {final['occupied_recall']:.6f} | {final['positive_only_f1']:.6f} | {final['false_free_rate']:.6f} | {final['post_warmup_unsupported_changes']} | {final['post_warmup_flicker_per_slot_minute']:.6f} |

Fusion weights/threshold were frozen on PKLot UFPR04. The temporal parameters
were generic pre-registered defaults and were not selected on Grand Bassin.
Initialization frames are retained in recall and excluded only from flicker.

Vacant recall, false-occupied rate, transition latency, detector mAP, and
tracking metrics are deliberately not reported.
"""


def _branch_diagnostics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    fields = ("p_cls", "p_det", "p_occ", "p_occ_filtered")
    slot_ids = sorted({str(row["slot_id"]) for row in rows})
    return {
        "overall": {
            field: probability_summary(float(row[field]) for row in rows)
            for field in fields
        },
        "per_slot": {
            slot_id: {
                field: probability_summary(
                    float(row[field])
                    for row in rows
                    if str(row["slot_id"]) == slot_id
                )
                for field in fields
            }
            for slot_id in slot_ids
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate positive-only raw/final stability"
    )
    parser.add_argument("--ground-truth", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--summary", required=True)
    parser.add_argument("--parameters", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--warmup-frames", type=int, default=6)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    truth = {_key(row): row for row in _read_csv(Path(args.ground_truth))}
    prediction = {
        _key(row): row for row in _read_csv(Path(args.predictions))
    }
    if set(truth) != set(prediction):
        raise ValueError(
            "Truth/prediction keys differ: "
            f"missing={len(set(truth) - set(prediction))}, "
            f"extra={len(set(prediction) - set(truth))}"
        )
    rows = [
        {
            "video_id": key[0],
            "frame_index": key[1],
            "slot_id": key[2],
            "truth": int(truth[key]["state"]),
            "raw_state": int(prediction[key]["raw_state"]),
            "state": int(prediction[key]["state"]),
            "p_cls": float(prediction[key]["p_cls"]),
            "p_det": float(prediction[key]["p_det"]),
            "p_occ": float(prediction[key]["p_occ"]),
            "p_occ_filtered": float(prediction[key]["p_occ_filtered"]),
        }
        for key in sorted(truth)
    ]
    run_summary = json.loads(Path(args.summary).read_text(encoding="utf-8"))
    parameters = json.loads(Path(args.parameters).read_text(encoding="utf-8"))
    classifier_rows = [
        {
            **row,
            "branch_state": int(
                row["p_cls"] >= float(parameters["e1_threshold"])
            ),
        }
        for row in rows
    ]
    detector_rows = [
        {
            **row,
            "branch_state": int(
                row["p_det"] >= float(parameters["e2_threshold"])
            ),
        }
        for row in rows
    ]
    report = {
        "protocol": {
            "ground_truth": str(Path(args.ground_truth).resolve()),
            "predictions": str(Path(args.predictions).resolve()),
            "truth_scope": "seven continuously occupied bus bays only",
            "selection": (
                "fusion frozen on UFPR04; temporal generic defaults; "
                "Grand Bassin not used for selection"
            ),
            "frames": run_summary["frames"],
            "processing_fps": run_summary["processing_fps"],
            "frozen_parameters": parameters,
        },
        "classifier_only": positive_only_stability_metrics(
            classifier_rows,
            state_key="branch_state",
            fps=args.fps,
            warmup_frames=args.warmup_frames,
        ),
        "detector_only": positive_only_stability_metrics(
            detector_rows,
            state_key="branch_state",
            fps=args.fps,
            warmup_frames=args.warmup_frames,
        ),
        "raw_fusion_state": positive_only_stability_metrics(
            rows,
            state_key="raw_state",
            fps=args.fps,
            warmup_frames=args.warmup_frames,
        ),
        "temporal_state": positive_only_stability_metrics(
            rows,
            state_key="state",
            fps=args.fps,
            warmup_frames=args.warmup_frames,
        ),
        "branch_diagnostics": _branch_diagnostics(rows),
    }
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.write("\n")
    (output_dir / "REPORT.md").write_text(
        _markdown(report),
        encoding="utf-8",
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
