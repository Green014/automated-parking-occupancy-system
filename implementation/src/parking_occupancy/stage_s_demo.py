from __future__ import annotations

import csv
import hashlib
import json
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .stage_r_component_attribution import validate_frozen_inputs
from .stage_s_release import STAGE_S_PROTOCOL_ID


DEMO_FPS = 10.0
DEMO_SIZE = (1280, 720)
DEMO_FRAMES = 500
CONTINUOUS_SEGMENTS = (
    ("gopro1", 92, 116),
    ("gopro4", 95, 123),
    ("gopro26", 109, 136),
)


class StageSDemoError(RuntimeError):
    """Raised when a frozen-only demo cannot be rendered or verified."""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows: dict[tuple[str, int], dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            row = json.loads(line)
            key = (str(row["video_id"]), int(row["frame_index"]))
            if key in rows:
                raise StageSDemoError(f"Duplicate detection frame: {key}")
            rows[key] = row
    return rows


def _truth_by_key(
    truth: Mapping[tuple[str, int, str], Mapping[str, str]],
) -> dict[tuple[str, int, str], int]:
    return {key: int(row["state"]) for key, row in truth.items()}


def load_demo_assets(implementation_root: Path) -> dict[str, Any]:
    implementation_root = implementation_root.resolve()
    frozen = validate_frozen_inputs(implementation_root)
    output_root = frozen["output_root"]
    detections = {
        "D1": _read_jsonl(output_root / "QV2-0" / "detections.jsonl"),
        "D1-LL": _read_jsonl(output_root / "QV2-1" / "detections.jsonl"),
    }
    polygon_path = (
        implementation_root
        / "data"
        / "stage_q_v2"
        / "STAGE_Q_V2_SLOT_POLYGONS_20260729.json"
    )
    polygon_payload = json.loads(polygon_path.read_text(encoding="utf-8"))
    polygons = {
        str(row["id"]): np.asarray(row["points"], dtype=np.int32)
        for row in polygon_payload["slots"]
    }
    if sorted(polygons) != [f"slot_{index:02d}" for index in range(21)]:
        raise StageSDemoError("Unexpected frozen Stage Q-v2 polygon IDs")
    frame_keys = set(frozen["manifest"])
    if set(detections["D1"]) != frame_keys:
        raise StageSDemoError("D1 detection frames differ from frozen manifest")
    if set(detections["D1-LL"]) != frame_keys:
        raise StageSDemoError("D1-LL detection frames differ from frozen manifest")
    return {
        **frozen,
        "detections": detections,
        "polygons": polygons,
        "truth_values": _truth_by_key(frozen["truth"]),
        "source_root": (
            implementation_root
            / "data"
            / "external"
            / "stage_q_upm_gti_20260729"
            / "extracted"
            / "test"
        ),
    }


def _rows_by_frame(
    predictions: Mapping[
        str,
        Mapping[tuple[str, int, str], Mapping[str, str]],
    ],
) -> dict[str, dict[tuple[str, int], dict[str, Mapping[str, str]]]]:
    grouped: dict[
        str,
        dict[tuple[str, int], dict[str, Mapping[str, str]]],
    ] = {}
    for detector, rows in predictions.items():
        detector_rows: dict[
            tuple[str, int],
            dict[str, Mapping[str, str]],
        ] = defaultdict(dict)
        for (sequence_id, frame_index, slot_id), row in rows.items():
            detector_rows[(sequence_id, frame_index)][slot_id] = row
        grouped[detector] = dict(detector_rows)
    return grouped


def build_demo_plan(assets: Mapping[str, Any]) -> dict[str, Any]:
    frame_rows = _rows_by_frame(assets["predictions"])
    truth = assets["truth_values"]

    continuous: list[dict[str, Any]] = []
    for sequence_id, start, end in CONTINUOUS_SEGMENTS:
        for frame_index in range(start, end + 1):
            key = (sequence_id, frame_index)
            if key not in assets["manifest"]:
                raise StageSDemoError(f"Missing continuous demo frame: {key}")
            continuous.append(
                {"sequence_id": sequence_id, "frame_index": frame_index}
            )
    if len(continuous) != 82:
        raise StageSDemoError("Continuous demo selection must have 82 frames")

    comparison_scores: list[tuple[int, str, int]] = []
    for sequence_id, frame_index in sorted(assets["manifest"]):
        d1 = frame_rows["D1"][(sequence_id, frame_index)]
        d1_ll = frame_rows["D1-LL"][(sequence_id, frame_index)]
        state_differences = sum(
            int(d1[slot_id]["raw_state"]) != int(d1_ll[slot_id]["raw_state"])
            for slot_id in d1
        )
        detection_difference = abs(
            len(assets["detections"]["D1"][(sequence_id, frame_index)][
                "detections"
            ])
            - len(assets["detections"]["D1-LL"][(sequence_id, frame_index)][
                "detections"
            ])
        )
        comparison_scores.append(
            (
                state_differences * 100 + detection_difference,
                sequence_id,
                frame_index,
            )
        )
    comparison = [
        {"sequence_id": sequence_id, "frame_index": frame_index}
        for _score, sequence_id, frame_index in sorted(
            comparison_scores,
            reverse=True,
        )[:13]
    ]

    recoveries: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    used_recovery_frames: set[tuple[str, int]] = set()
    used_failure_frames: set[tuple[str, int]] = set()
    for key in sorted(assets["predictions"]["D1"]):
        sequence_id, frame_index, slot_id = key
        row = assets["predictions"]["D1"][key]
        expected = truth[key]
        frame_key = (sequence_id, frame_index)
        if (
            expected == 1
            and int(row["detector_occupied"]) == 0
            and int(row["raw_state"]) == 1
            and frame_key not in used_recovery_frames
            and len(recoveries) < 10
        ):
            recoveries.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "slot_id": slot_id,
                    "gate_branch": row["gate_branch"],
                }
            )
            used_recovery_frames.add(frame_key)
        if (
            expected == 0
            and int(row["detector_occupied"]) == 1
            and int(row["raw_state"]) == 1
            and frame_key not in used_failure_frames
            and len(failures) < 7
        ):
            failures.append(
                {
                    "sequence_id": sequence_id,
                    "frame_index": frame_index,
                    "slot_id": slot_id,
                    "failure_type": "B1_geometry_false_occupied",
                }
            )
            used_failure_frames.add(frame_key)
    if len(recoveries) != 10 or len(failures) != 7:
        raise StageSDemoError("Insufficient frozen recovery/failure demo cases")
    return {
        "continuous": continuous,
        "comparison": comparison,
        "recoveries": recoveries,
        "failures": failures,
        "timeline": {
            "continuous": [0, 200],
            "detector_comparison": [200, 330],
            "F2_recovery": [330, 430],
            "failure_cases": [430, 500],
        },
    }


def _load_source_frame(
    assets: Mapping[str, Any],
    sequence_id: str,
    frame_index: int,
) -> np.ndarray:
    row = assets["manifest"][(sequence_id, frame_index)]
    path = assets["source_root"] / str(row["relative_path"])
    if not path.is_file():
        raise StageSDemoError(f"Missing frozen source image: {path}")
    if path.stat().st_size != int(row["bytes"]):
        raise StageSDemoError(f"Frozen source image byte count changed: {path}")
    if _sha256(path) != str(row["sha256"]):
        raise StageSDemoError(f"Frozen source image SHA-256 changed: {path}")
    # cv2.imread can fail on Windows when the absolute path contains non-ASCII
    # characters. Decode the already-verified frozen bytes instead.
    frame = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    if frame is None or frame.shape[:2] != (600, 800):
        raise StageSDemoError(f"Could not decode frozen source image: {path}")
    return frame


def _put_text(
    image: np.ndarray,
    text: str,
    origin: tuple[int, int],
    *,
    scale: float = 0.6,
    color: tuple[int, int, int] = (240, 245, 255),
    thickness: int = 1,
) -> None:
    cv2.putText(
        image,
        text,
        origin,
        cv2.FONT_HERSHEY_SIMPLEX,
        scale,
        color,
        thickness,
        cv2.LINE_AA,
    )


def _render_method_overlay(
    frame: np.ndarray,
    *,
    polygons: Mapping[str, np.ndarray],
    slot_rows: Mapping[str, Mapping[str, str]],
    detections: Sequence[Mapping[str, Any]],
    highlight_slot: str | None = None,
    detection_label: str = "D1",
) -> np.ndarray:
    rendered = frame.copy()
    fill = frame.copy()
    for slot_id, points in polygons.items():
        occupied = bool(int(slot_rows[slot_id]["raw_state"]))
        color = (55, 75, 235) if occupied else (65, 190, 85)
        cv2.fillPoly(fill, [points], color)
        thickness = 4 if slot_id == highlight_slot else 1
        outline = (0, 230, 255) if slot_id == highlight_slot else color
        cv2.polylines(rendered, [points], True, outline, thickness, cv2.LINE_AA)
        centre = tuple(np.mean(points, axis=0).astype(int))
        _put_text(
            rendered,
            slot_id.split("_")[-1],
            (centre[0] - 8, centre[1] + 5),
            scale=0.38,
            color=(255, 255, 255),
            thickness=1,
        )
    rendered = cv2.addWeighted(fill, 0.17, rendered, 0.83, 0)
    for detection in detections:
        x1, y1, x2, y2 = (int(round(value)) for value in detection["bbox"])
        cv2.rectangle(rendered, (x1, y1), (x2, y2), (255, 205, 55), 2)
        _put_text(
            rendered,
            f"{detection_label} {float(detection['confidence']):.2f}",
            (x1, max(18, y1 - 5)),
            scale=0.45,
            color=(255, 225, 100),
        )
    return rendered


def _base_canvas(title: str, subtitle: str) -> np.ndarray:
    canvas = np.full((DEMO_SIZE[1], DEMO_SIZE[0], 3), (22, 29, 42), np.uint8)
    cv2.rectangle(canvas, (0, 0), (1280, 72), (10, 16, 28), -1)
    _put_text(canvas, title, (28, 34), scale=0.82, thickness=2)
    _put_text(canvas, subtitle, (28, 60), scale=0.48, color=(165, 210, 255))
    return canvas


def _default_frame(
    assets: Mapping[str, Any],
    frame_rows: Mapping[str, Any],
    case: Mapping[str, Any],
    *,
    section: str,
    highlight_slot: str | None = None,
    detail_lines: Sequence[str] = (),
) -> np.ndarray:
    sequence_id = str(case["sequence_id"])
    frame_index = int(case["frame_index"])
    raw = _load_source_frame(assets, sequence_id, frame_index)
    rows = frame_rows["D1"][(sequence_id, frame_index)]
    rendered = _render_method_overlay(
        raw,
        polygons=assets["polygons"],
        slot_rows=rows,
        detections=assets["detections"]["D1"][(sequence_id, frame_index)][
            "detections"
        ],
        highlight_slot=highlight_slot,
    )
    rendered = cv2.resize(rendered, (864, 648), interpolation=cv2.INTER_AREA)
    canvas = _base_canvas(
        "Default system: D1 -> B1 -> F2 -> Occupancy Output",
        "R1 raw_state | E4 OFF | tracker none",
    )
    canvas[72:720, :864] = rendered
    cv2.rectangle(canvas, (864, 72), (1279, 719), (18, 27, 40), -1)
    _put_text(canvas, section, (888, 112), scale=0.56, color=(80, 220, 255), thickness=2)
    _put_text(
        canvas,
        f"{sequence_id}  source frame {frame_index}",
        (888, 150),
        scale=0.48,
    )
    predicted_count = sum(int(row["raw_state"]) for row in rows.values())
    truth_count = sum(
        assets["truth_values"][(sequence_id, frame_index, slot_id)]
        for slot_id in rows
    )
    _put_text(
        canvas,
        f"R1 occupied count: {predicted_count}",
        (888, 194),
        scale=0.5,
    )
    _put_text(canvas, f"Truth occupied count: {truth_count}", (888, 224), scale=0.5)
    _put_text(canvas, "Green = vacant", (888, 276), scale=0.46, color=(100, 225, 120))
    _put_text(canvas, "Red = occupied", (888, 304), scale=0.46, color=(100, 130, 255))
    y = 370
    for line in detail_lines:
        _put_text(canvas, line, (888, y), scale=0.46, color=(225, 230, 240))
        y += 30
    _put_text(
        canvas,
        "Post-hoc rendering",
        (888, 650),
        scale=0.44,
        color=(180, 185, 195),
    )
    _put_text(
        canvas,
        "No model inference",
        (888, 678),
        scale=0.44,
        color=(180, 185, 195),
    )
    return canvas


def _comparison_frame(
    assets: Mapping[str, Any],
    frame_rows: Mapping[str, Any],
    case: Mapping[str, Any],
) -> np.ndarray:
    sequence_id = str(case["sequence_id"])
    frame_index = int(case["frame_index"])
    raw = _load_source_frame(assets, sequence_id, frame_index)
    panels = []
    for detector in ("D1", "D1-LL"):
        panels.append(
            _render_method_overlay(
                raw,
                polygons=assets["polygons"],
                slot_rows=frame_rows[detector][(sequence_id, frame_index)],
                detections=assets["detections"][detector][
                    (sequence_id, frame_index)
                ]["detections"],
                detection_label=detector,
            )
        )
    canvas = _base_canvas(
        "Frozen same-frame detector comparison",
        "D1 remains default | D1-LL is a retained negative experiment",
    )
    for index, panel in enumerate(panels):
        resized = cv2.resize(panel, (620, 465), interpolation=cv2.INTER_AREA)
        x = 20 + 640 * index
        canvas[110:575, x : x + 620] = resized
        label = "D1 + B1 + F2" if index == 0 else "D1-LL + B1 + F2"
        _put_text(canvas, label, (x, 98), scale=0.58, thickness=2)
    d1_count = sum(
        int(row["raw_state"])
        for row in frame_rows["D1"][(sequence_id, frame_index)].values()
    )
    ll_count = sum(
        int(row["raw_state"])
        for row in frame_rows["D1-LL"][(sequence_id, frame_index)].values()
    )
    _put_text(
        canvas,
        f"{sequence_id} source frame {frame_index} | occupied: D1={d1_count}, D1-LL={ll_count}",
        (30, 625),
        scale=0.6,
    )
    _put_text(
        canvas,
        "Same frozen truth, polygons, thresholds and F2 gate",
        (30, 666),
        scale=0.5,
        color=(180, 210, 245),
    )
    return canvas


def _resampled_case(cases: Sequence[Mapping[str, Any]], index: int, count: int):
    case_index = min(len(cases) - 1, int(index * len(cases) / count))
    return cases[case_index]


def verify_demo_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise StageSDemoError(f"Could not decode demo video: {path}")
    frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc_value = int(capture.get(cv2.CAP_PROP_FOURCC))
    decoded = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if frame.shape[:2] != (DEMO_SIZE[1], DEMO_SIZE[0]):
            capture.release()
            raise StageSDemoError("Decoded demo frame dimensions changed")
        decoded += 1
    capture.release()
    fourcc = "".join(chr((fourcc_value >> (8 * index)) & 0xFF) for index in range(4))
    duration = frames / fps if fps > 0 else 0.0
    if (frames, decoded, width, height) != (
        DEMO_FRAMES,
        DEMO_FRAMES,
        DEMO_SIZE[0],
        DEMO_SIZE[1],
    ):
        raise StageSDemoError("Demo video frame/decode/dimension validation failed")
    if abs(fps - DEMO_FPS) > 0.01 or not 45.0 <= duration <= 60.0:
        raise StageSDemoError("Demo video FPS or duration validation failed")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
        "decodable": True,
        "frames": frames,
        "decoded_frames": decoded,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": duration,
        "requested_codec": "mp4v",
        "decoded_fourcc": fourcc,
    }


def render_stage_s_demo(
    implementation_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    assets = load_demo_assets(implementation_root)
    plan = build_demo_plan(assets)
    frame_rows = _rows_by_frame(assets["predictions"])
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / "demo_main.mp4"
    if video_path.exists():
        raise FileExistsError(f"Refusing to overwrite Stage S demo: {video_path}")
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        DEMO_FPS,
        DEMO_SIZE,
    )
    if not writer.isOpened():
        raise StageSDemoError(f"Could not create Stage S demo: {video_path}")
    keyframes: dict[str, np.ndarray] = {}
    try:
        for output_index in range(DEMO_FRAMES):
            if output_index < 200:
                case = _resampled_case(plan["continuous"], output_index, 200)
                frame = _default_frame(
                    assets,
                    frame_rows,
                    case,
                    section="Consecutive selected frames",
                    detail_lines=(
                        "selected consecutive",
                        "source frames, slowed",
                        "for visualization",
                    ),
                )
                if "default" not in keyframes and output_index >= 80:
                    keyframes["default"] = frame.copy()
            elif output_index < 330:
                case = _resampled_case(
                    plan["comparison"],
                    output_index - 200,
                    130,
                )
                frame = _comparison_frame(assets, frame_rows, case)
                if "comparison" not in keyframes:
                    keyframes["comparison"] = frame.copy()
            elif output_index < 430:
                case = _resampled_case(
                    plan["recoveries"],
                    output_index - 330,
                    100,
                )
                slot_id = str(case["slot_id"])
                frame = _default_frame(
                    assets,
                    frame_rows,
                    case,
                    section="B1 -> F2 successful recovery",
                    highlight_slot=slot_id,
                    detail_lines=(
                        f"highlight: {slot_id}",
                        "B1: vacant",
                        "F2: occupied",
                        "Truth: occupied",
                    ),
                )
                if "recovery" not in keyframes:
                    keyframes["recovery"] = frame.copy()
            else:
                case = _resampled_case(
                    plan["failures"],
                    output_index - 430,
                    70,
                )
                slot_id = str(case["slot_id"])
                frame = _default_frame(
                    assets,
                    frame_rows,
                    case,
                    section="Frozen geometry failure",
                    highlight_slot=slot_id,
                    detail_lines=(
                        f"highlight: {slot_id}",
                        "B1: occupied",
                        "F2 retains detector",
                        "Truth: vacant",
                    ),
                )
            writer.write(frame)
    finally:
        writer.release()

    image_paths = {
        "default": output_dir / "demo_keyframe_default.png",
        "comparison": output_dir / "demo_keyframe_d1_vs_d1ll.png",
        "recovery": output_dir / "demo_keyframe_f2_recovery.png",
    }
    for key, path in image_paths.items():
        encoded, buffer = cv2.imencode(".png", keyframes[key])
        if not encoded:
            raise StageSDemoError(f"Could not write Stage S keyframe: {path}")
        buffer.tofile(path)
    validation = verify_demo_video(video_path)
    metadata = {
        "schema_version": 1,
        "protocol_id": STAGE_S_PROTOCOL_ID,
        "status": "FROZEN_OUTPUT_POSTHOC_RENDER_COMPLETE",
        "model_inference_run": False,
        "training_or_tuning_run": False,
        "main_state_field": "raw_state",
        "E4_state_used_for_main_visualization": False,
        "tracker_used": False,
        "source_claim": (
            "selected consecutive source frames, slowed for visualization"
        ),
        "timeline": plan["timeline"],
        "case_counts": {
            "continuous_source_frames": len(plan["continuous"]),
            "D1_D1_LL_comparison_frames": len(plan["comparison"]),
            "F2_recovery_cases": len(plan["recoveries"]),
            "geometry_failure_cases": len(plan["failures"]),
        },
        "cases": plan,
        "video_validation": validation,
        "images": {
            key: {
                "path": str(path.resolve()),
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            for key, path in image_paths.items()
        },
    }
    metadata_path = output_dir / "STAGE_S_DEMO_METADATA.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata
