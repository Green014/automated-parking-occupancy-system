from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .stage_u_portable_release import sha256_file


FROZEN_STAGE_T_DEMO_SHA256 = (
    "b5dfdeb850acdd0a87072a9c48fda44dd5e13725fb7f0e428cfc6164b4d24c1f"
)
SOURCE_START_FRAME = 1450
EXPECTED_FRAMES = 450
KNOWN_FAILURE_START_FRAME = 1660
PRESENTATION_FILENAME = "demo_tracktrack_identity_diagnostic_presentation.mp4"
PRESENTATION_KEYFRAME_FILENAME = (
    "demo_tracktrack_identity_diagnostic_presentation_keyframe.png"
)
PRESENTATION_METADATA_FILENAME = (
    "STAGE_U_1_TRACKTRACK_PRESENTATION_METADATA.json"
)


class StageU1PresentationError(ValueError):
    """Raised when the Stage U.1 presentation-copy contract is violated."""


def overlay_presentation_labels(
    frame: np.ndarray,
    *,
    source_frame: int,
) -> np.ndarray:
    canvas = frame.copy()
    height, width = canvas.shape[:2]
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, 0), (width, 178), (8, 8, 8), -1)
    cv2.addWeighted(overlay, 0.88, canvas, 0.12, 0, canvas)
    lines = (
        "TrackTrack identity-output diagnostic (not a full parking-lot evaluation)",
        "Yellow boxes = vehicle detections with TrackTrack IDs",
        "Red/green polygon = predicted state of 1 evaluated parking slot",
        "Evaluated slots in this diagnostic: 1",
        "Other visible parking positions are not evaluated",
    )
    colors = (
        (70, 230, 255),
        (0, 255, 255),
        (235, 235, 235),
        (235, 235, 235),
        (235, 235, 235),
    )
    for index, (line, color) in enumerate(zip(lines, colors, strict=True)):
        cv2.putText(
            canvas,
            line,
            (24, 31 + index * 32),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.67 if index else 0.74,
            color,
            2 if index == 0 else 1,
            cv2.LINE_AA,
        )
    if source_frame >= KNOWN_FAILURE_START_FRAME:
        failure = canvas.copy()
        cv2.rectangle(
            failure,
            (0, height - 126),
            (width, height),
            (20, 20, 155),
            -1,
        )
        cv2.addWeighted(failure, 0.92, canvas, 0.08, 0, canvas)
        cv2.putText(
            canvas,
            "truth = vacant, prediction = occupied",
            (24, height - 73),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            "known false-occupied failure",
            (24, height - 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (180, 220, 255),
            2,
            cv2.LINE_AA,
        )
    return canvas


def _open_writer(
    output_path: Path,
    *,
    fps: float,
    size: tuple[int, int],
) -> tuple[cv2.VideoWriter, str]:
    for codec in ("avc1", "H264", "mp4v"):
        writer = cv2.VideoWriter(
            str(output_path),
            cv2.VideoWriter_fourcc(*codec),
            fps,
            size,
        )
        if writer.isOpened():
            return writer, codec
        writer.release()
        output_path.unlink(missing_ok=True)
    raise StageU1PresentationError("No existing local MP4 encoder is available")


def _verify_video(path: Path) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise StageU1PresentationError(f"Could not decode presentation copy: {path}")
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = int(capture.get(cv2.CAP_PROP_FOURCC))
    decoded_fourcc = "".join(
        chr((fourcc >> (8 * index)) & 0xFF) for index in range(4)
    )
    decoded = 0
    while True:
        ok, _ = capture.read()
        if not ok:
            break
        decoded += 1
    capture.release()
    if frames != EXPECTED_FRAMES or decoded != EXPECTED_FRAMES:
        raise StageU1PresentationError(
            f"Presentation frame mismatch metadata={frames} decoded={decoded}"
        )
    return {
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
        "frames": frames,
        "decoded_frames": decoded,
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": frames / fps,
        "decoded_fourcc": decoded_fourcc,
    }


def render_stage_u_1_presentation_copy(
    *,
    frozen_demo_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    frozen_demo_path = frozen_demo_path.resolve()
    if sha256_file(frozen_demo_path) != FROZEN_STAGE_T_DEMO_SHA256:
        raise StageU1PresentationError("Frozen Stage T demo SHA-256 changed")
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    video_path = output_dir / PRESENTATION_FILENAME
    keyframe_path = output_dir / PRESENTATION_KEYFRAME_FILENAME
    metadata_path = output_dir / PRESENTATION_METADATA_FILENAME
    for path in (video_path, keyframe_path, metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite presentation artifact: {path}")

    capture = cv2.VideoCapture(str(frozen_demo_path))
    if not capture.isOpened():
        raise StageU1PresentationError("Could not decode frozen Stage T demo")
    frames = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if frames != EXPECTED_FRAMES or min(fps, width, height) <= 0:
        capture.release()
        raise StageU1PresentationError("Unexpected frozen Stage T video metadata")
    writer, requested_codec = _open_writer(
        video_path,
        fps=fps,
        size=(width, height),
    )
    keyframe: np.ndarray | None = None
    try:
        for offset in range(EXPECTED_FRAMES):
            ok, frame = capture.read()
            if not ok:
                raise StageU1PresentationError(
                    f"Frozen Stage T demo ended at offset {offset}"
                )
            source_frame = SOURCE_START_FRAME + offset
            rendered = overlay_presentation_labels(
                frame,
                source_frame=source_frame,
            )
            writer.write(rendered)
            if source_frame == KNOWN_FAILURE_START_FRAME:
                keyframe = rendered.copy()
    finally:
        capture.release()
        writer.release()
    if keyframe is None:
        raise StageU1PresentationError("Known failure keyframe was not rendered")
    encoded, buffer = cv2.imencode(".png", keyframe)
    if not encoded:
        raise StageU1PresentationError("Could not encode presentation keyframe")
    buffer.tofile(keyframe_path)

    validation = _verify_video(video_path)
    source_hash_after = sha256_file(frozen_demo_path)
    if source_hash_after != FROZEN_STAGE_T_DEMO_SHA256:
        raise StageU1PresentationError("Frozen Stage T demo changed during rendering")
    metadata = {
        "schema_version": 1,
        "protocol_id": "STAGE-U.1-FINAL-RELEASE-CORRECTION-20260730-01",
        "status": "POST_HOC_PRESENTATION_COPY_COMPLETE",
        "claim_class": "TrackTrack identity-output diagnostic",
        "new_experiment": False,
        "model_inference_run": False,
        "source_demo_modified": False,
        "source_demo_sha256_before_and_after": source_hash_after,
        "evaluated_slots": 1,
        "legend": [
            "Yellow boxes = vehicle detections with TrackTrack IDs",
            "Red/green polygon = predicted state of 1 evaluated parking slot",
            "Evaluated slots in this diagnostic: 1",
            "Other visible parking positions are not evaluated",
        ],
        "known_failure": {
            "from_source_frame_inclusive": KNOWN_FAILURE_START_FRAME,
            "truth": "vacant",
            "prediction": "occupied",
            "description": "known false-occupied failure",
        },
        "presentation_video": {
            "path": (
                "implementation/data/stage_u_1/demo/" + PRESENTATION_FILENAME
            ),
            "requested_codec": requested_codec,
            **validation,
        },
        "presentation_keyframe": {
            "path": (
                "implementation/data/stage_u_1/demo/"
                + PRESENTATION_KEYFRAME_FILENAME
            ),
            "source_frame": KNOWN_FAILURE_START_FRAME,
            "bytes": keyframe_path.stat().st_size,
            "sha256": sha256_file(keyframe_path),
        },
        "h264_presentation_copy": (
            "this file" if requested_codec in {"avc1", "H264"} else None
        ),
        "powerpoint_compatibility_note": (
            "No local H.264 writer was available; FMP4 retained."
            if requested_codec == "mp4v"
            else "Existing local H.264 writer used; no dependency downloaded."
        ),
    }
    metadata_path.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return metadata
