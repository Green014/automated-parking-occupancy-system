from __future__ import annotations

import csv
import sys
from pathlib import Path

LITERATURE_CORE_SRC = (
    Path(__file__).resolve().parents[1] / "literature_core" / "src"
)
if str(LITERATURE_CORE_SRC) not in sys.path:
    sys.path.insert(0, str(LITERATURE_CORE_SRC))

from parking_occupancy.stage_l_analysis import (
    analyze_static_predictions,
    analyze_video_predictions,
)


def _write_csv(
    path: Path,
    fields: list[str],
    rows: list[dict[str, object]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_static_analysis_uses_image_paired_differences(tmp_path: Path) -> None:
    path = tmp_path / "predictions.csv"
    rows = [
        {
            "sample_id": sample,
            "camera": camera,
            "truth": truth,
            "p1_prediction": p1,
            "p3_prediction": p3,
        }
        for sample, camera, truth, p1, p3 in (
            ("a", "c1", 1, 0, 1),
            ("a", "c1", 0, 0, 0),
            ("b", "c2", 1, 1, 1),
            ("b", "c2", 0, 0, 0),
        )
    ]
    _write_csv(path, list(rows[0]), rows)

    report = analyze_static_predictions(path, resamples=20)

    assert report["outcomes"] == {"win": 1, "tie": 1, "loss": 0}
    assert report["paired_bootstrap"]["unit"] == "sample_id"
    assert report["model_prediction_rerun"] is False


def test_video_analysis_restores_absolute_transition_frame(
    tmp_path: Path,
) -> None:
    path = tmp_path / "occupancy.csv"
    rows = []
    for frame in range(8):
        truth = int(frame < 5)
        prediction = int(frame < 6)
        rows.append(
            {
                "frame_index": frame,
                "truth": truth,
                "p1_raw": prediction,
                "p3_gate": prediction,
                "p3_temporal": prediction,
                "p3_full_tracking_temporal": prediction,
                "gate_branch": "detector_confirmed",
                "full_branch": "detector_confirmed",
                "p_cls": "",
            }
        )
    _write_csv(path, list(rows[0]), rows)

    report = analyze_video_predictions(
        path,
        fps=1.0,
        warmup_frames=2,
        stable_frames=1,
    )

    event = report["methods"]["p1_raw"]["temporal"]["transition_events"][0]
    assert event["truth_transition_frame_absolute"] == 5
    assert event["predicted_transition_frame_absolute"] == 6
    assert report["diagnostics"]["post_transition_detector_positive_frames"] == 1
