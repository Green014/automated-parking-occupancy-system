from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .detector_comparison import sha256_file
from .stage_t_tracktrack import STAGE_T_PROTOCOL_ID


DEMO_START_FRAME = 1450
DEMO_FRAME_COUNT = 450


class StageTDemoError(ValueError):
    """Raised when the optional Stage T demo contract is violated."""


def verify_stage_t_demo(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path.resolve()))
    if not capture.isOpened():
        raise StageTDemoError(f"Could not decode Stage T demo: {path}")
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    decoded_fourcc = "".join(chr((fourcc >> (8 * index)) & 0xFF) for index in range(4))
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if frames != DEMO_FRAME_COUNT or decoded != DEMO_FRAME_COUNT:
        raise StageTDemoError(
            f"Stage T demo frame count mismatch metadata={frames} decoded={decoded}"
        )
    if fps <= 0 or width <= 0 or height <= 0:
        raise StageTDemoError("Stage T demo has invalid video metadata")
    return {
        "path": str(path.resolve()),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "decodable": True,
        "frames": frames,
        "decoded_frames": decoded,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": frames / fps,
        "requested_codec": "mp4v",
        "decoded_fourcc": decoded_fourcc,
    }


def _banner(frame: np.ndarray, source_frame: int) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 82), (12, 8, 5), -1)
    cv2.rectangle(overlay, (0, height - 44), (width, height), (12, 8, 5), -1)
    cv2.addWeighted(overlay, 0.92, canvas, 0.08, 0, canvas)
    cv2.putText(
        canvas,
        "Optional TrackTrack-enhanced variant",
        (24, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.92,
        (70, 230, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        "D1 -> TrackTrack IDs -> B1 -> F2 | E4 OFF",
        (25, 67),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        canvas,
        (
            f"VIRAT 0502 consumed-development frame {source_frame} | "
            "not Stage S default | no occupancy improvement claim"
        ),
        (24, height - 15),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (230, 230, 230),
        1,
        cv2.LINE_AA,
    )
    return canvas


def render_stage_t_demo(
    *,
    tt1_root: Path,
    output_dir: Path,
) -> dict[str, Any]:
    tt1_root = tt1_root.resolve()
    summary = json.loads(
        (tt1_root / "summary.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        (tt1_root / "runtime_metadata.json").read_text(encoding="utf-8")
    )
    if summary.get("variant_id") != "TT1":
        raise StageTDemoError("Stage T demo requires TT1 output")
    if summary.get("tracker_backend") != "tracktrack":
        raise StageTDemoError("Stage T demo requires TrackTrack output")
    if summary.get("temporal_enabled") is not False:
        raise StageTDemoError("Stage T demo must keep E4 disabled")
    if summary.get("data_role") != "consumed-development diagnostic":
        raise StageTDemoError("Stage T demo claim class mismatch")
    if int(runtime["track_output"]["unique_source_track_ids"]) <= 0:
        raise StageTDemoError("Stage T demo has no TrackTrack identities to display")

    output_dir = output_dir.resolve()
    video_path = output_dir / "demo_tracktrack_optional.mp4"
    keyframe_path = output_dir / "demo_tracktrack_optional_keyframe.png"
    metadata_path = output_dir / "STAGE_T_DEMO_METADATA.json"
    for path in (video_path, keyframe_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite Stage T demo: {path}")
    output_dir.mkdir(parents=True, exist_ok=True)

    source_path = tt1_root / "annotated.mp4"
    capture = cv2.VideoCapture(str(source_path))
    if not capture.isOpened():
        raise StageTDemoError(f"Could not open TT1 annotated video: {source_path}")
    source_frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if DEMO_START_FRAME + DEMO_FRAME_COUNT > source_frames:
        capture.release()
        raise StageTDemoError("TT1 annotated video is shorter than demo selection")
    capture.set(cv2.CAP_PROP_POS_FRAMES, DEMO_START_FRAME)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        capture.release()
        raise StageTDemoError(f"Could not create Stage T demo: {video_path}")

    keyframe: np.ndarray | None = None
    try:
        for offset in range(DEMO_FRAME_COUNT):
            ok, frame = capture.read()
            if not ok:
                raise StageTDemoError(
                    f"TT1 annotated video ended at demo offset {offset}"
                )
            source_frame = DEMO_START_FRAME + offset
            rendered = _banner(frame, source_frame)
            writer.write(rendered)
            if source_frame == 1660:
                keyframe = rendered.copy()
    finally:
        capture.release()
        writer.release()
    if keyframe is None:
        raise StageTDemoError("Departure boundary keyframe was not selected")
    encoded, buffer = cv2.imencode(".png", keyframe)
    if not encoded:
        raise StageTDemoError("Could not encode Stage T keyframe")
    buffer.tofile(keyframe_path)

    validation = verify_stage_t_demo(video_path)
    metadata = {
        "schema_version": 1,
        "protocol_id": STAGE_T_PROTOCOL_ID,
        "status": "OPTIONAL_TRACKTRACK_DEMO_COMPLETE",
        "title": "Optional TrackTrack-enhanced variant",
        "source_claim": "consumed-development diagnostic",
        "untouched_test": False,
        "stage_s_default_demo": False,
        "model_inference_run_for_demo_render": False,
        "source_runtime_inference": "Stage T TT1 consumed-development diagnostic",
        "tracker_backend": "tracktrack",
        "temporal_enabled": False,
        "source_frame_range_inclusive": [
            DEMO_START_FRAME,
            DEMO_START_FRAME + DEMO_FRAME_COUNT - 1,
        ],
        "identity_ground_truth_available": False,
        "tracktrack_occupancy_improvement_claimed": False,
        "unique_source_track_ids_in_full_TT1_output": int(
            runtime["track_output"]["unique_source_track_ids"]
        ),
        "video": validation,
        "keyframe": {
            "path": str(keyframe_path),
            "bytes": keyframe_path.stat().st_size,
            "sha256": sha256_file(keyframe_path),
            "source_frame": 1660,
        },
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata
