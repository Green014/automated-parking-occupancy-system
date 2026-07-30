from __future__ import annotations

import copy
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import yaml

try:
    from flask import Flask, Response, jsonify, render_template
except ModuleNotFoundError as exc:
    if exc.name != "flask":
        raise
    Flask = None
    Response = None
    jsonify = None
    render_template = None

from .models import ParkingSlot
from .stage_v import FrameOccupancyResult, OccupancyBackend, draw_stage_v_frame
from .stage_v_runner import load_stage_v_slot_map
from .stage_w_ui_adapter import (
    StateEventTracker,
    error_payload,
    frame_result_to_ui_payload,
    redact_source,
    sanitize_for_api,
)


def _require_flask() -> None:
    if Flask is None:
        raise RuntimeError(
            "Stage W dashboard requires Flask; install "
            'the "dashboard" optional dependency with '
            'pip install -e ".[integrated,dashboard,dev]"'
        )


class StageWRunRecorder:
    def __init__(
        self,
        output_dir: Path,
        *,
        fps: float,
        frame_size: tuple[int, int],
        source: str,
        mode: str,
        config_snapshot: Mapping[str, Any],
    ) -> None:
        self.output_dir = output_dir.resolve()
        self.output_dir.mkdir(parents=True, exist_ok=False)
        self.video_path = self.output_dir / "annotated.mp4"
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            frame_size,
        )
        if not self.writer.isOpened():
            raise RuntimeError("Could not create Stage W annotated video")
        self.events_handle = (self.output_dir / "events.jsonl").open(
            "x", encoding="utf-8"
        )
        self.source = source
        self.mode = mode
        self.config_snapshot = sanitize_for_api(dict(config_snapshot))
        self.frames = 0
        self.events = 0
        self.closed = False

    def write(
        self,
        annotated: np.ndarray,
        emitted_events: Sequence[Mapping[str, Any]],
    ) -> None:
        self.writer.write(annotated)
        self.frames += 1
        for event in emitted_events:
            self.events_handle.write(
                json.dumps(sanitize_for_api(dict(event)), sort_keys=True) + "\n"
            )
            self.events += 1

    def close(
        self,
        *,
        status: Mapping[str, Any],
        events_temporally_valid: bool,
        event_semantics: str,
    ) -> None:
        if self.closed:
            return
        self.closed = True
        self.writer.release()
        self.events_handle.close()
        safe_status = sanitize_for_api(dict(status))
        (self.output_dir / "status.json").write_text(
            json.dumps(safe_status, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        summary = {
            "schema_version": 1,
            "stage": "W",
            "status": safe_status.get("health"),
            "mode": self.mode,
            "source": self.source,
            "frames": self.frames,
            "slots": safe_status.get("total", 0),
            "events": self.events,
            "events_temporally_valid": events_temporally_valid,
            "event_semantics": event_semantics,
            "initial_frame_emits_event": False,
            "accuracy_status": "not_computed_no_truth",
            "claim_boundary": "consumed demonstration only; no accuracy claim",
            "output_files": [
                "annotated.mp4",
                "events.jsonl",
                "status.json",
                "summary.json",
                "configuration_snapshot.yaml",
            ],
        }
        (self.output_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        (self.output_dir / "configuration_snapshot.yaml").write_text(
            yaml.safe_dump(
                {
                    "schema_version": 1,
                    "stage": "W",
                    "mode": self.mode,
                    "source": self.source,
                    **self.config_snapshot,
                },
                sort_keys=False,
                allow_unicode=True,
            ),
            encoding="utf-8",
        )


class StageWProcessor:
    """One synchronized inference loop shared by every Flask client."""

    def __init__(
        self,
        *,
        source: str | int,
        slots_path: Path,
        backend: OccupancyBackend,
        mode: str,
        config_snapshot: Mapping[str, Any],
        output_dir: Path | None = None,
        max_frames: int | None = None,
        warmup: bool = True,
        capture_factory: Callable[[Any], Any] = cv2.VideoCapture,
    ) -> None:
        if output_dir is not None and output_dir.exists():
            raise FileExistsError(
                f"Refusing to overwrite Stage W output: {output_dir.resolve()}"
            )
        if max_frames is not None and max_frames <= 0:
            raise ValueError("max_frames must be positive")
        self.source = source
        self.public_source = redact_source(source)
        self.slots_path = slots_path.resolve()
        self.backend = backend
        self.mode = mode
        self.config_snapshot = dict(config_snapshot)
        self.output_dir = output_dir
        self.max_frames = max_frames
        self.warmup = bool(warmup)
        self.capture_factory = capture_factory
        self.latest_frame: bytes | None = None
        self.slots: tuple[ParkingSlot, ...] = ()
        self._lock = threading.RLock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._event_tracker = StateEventTracker(
            continuous=True,
            temporal_enabled=self.temporal_enabled,
        )
        self._state = {
            "schema_version": 1,
            "stage": "W",
            "health": "starting",
            "running": False,
            "message": "Starting video processor",
            "mode": mode,
            "source": self.public_source,
            "frame_index": None,
            "timestamp_s": None,
            "occupied": 0,
            "vacant": 0,
            "total": 0,
            "rendered_slots": 0,
            "temporal_enabled": self.temporal_enabled,
            "tracker_enabled": self.tracker_enabled,
            "slots": [],
            "detections": [],
            "runtime": {
                "attributed_processing_ms": None,
                "attributed_fps": None,
                "cache": "not-used",
                "timing_ms": {},
            },
            "warnings": [],
            "events_temporally_valid": True,
            "event_semantics": self._event_tracker.semantics,
        }

    @property
    def temporal_enabled(self) -> bool:
        return bool(
            self.config_snapshot.get("temporal", {}).get("enabled", False)
        )

    @property
    def tracker_enabled(self) -> bool:
        return (
            self.config_snapshot.get("tracking", {}).get("backend", "none")
            != "none"
        )

    @staticmethod
    def _capture_source(value: str | int) -> str | int:
        if isinstance(value, int):
            return value
        text = str(value)
        return int(text) if text.isdigit() else text

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop.clear()
            self._thread = threading.Thread(
                target=self._run,
                name="stage-w-video-processor",
                daemon=True,
            )
            self._thread.start()

    def run_blocking(self) -> None:
        self._stop.clear()
        self._run()

    def stop(self) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=5.0)

    def _publish(self, frame: np.ndarray, state: Mapping[str, Any]) -> None:
        ok, encoded = cv2.imencode(
            ".jpg",
            frame,
            [cv2.IMWRITE_JPEG_QUALITY, 84],
        )
        if not ok:
            raise RuntimeError("Could not encode dashboard frame")
        with self._lock:
            self.latest_frame = encoded.tobytes()
            self._state = sanitize_for_api(dict(state))

    def _publish_error(self, message: str, *, total: int = 0) -> None:
        safe = error_payload(
            message,
            mode=self.mode,
            total=total,
            source=self.source,
        )
        image = np.full((720, 1280, 3), (22, 26, 32), dtype=np.uint8)
        cv2.putText(
            image,
            str(safe["message"])[:110],
            (40, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 180, 255),
            2,
            cv2.LINE_AA,
        )
        self._publish(image, safe)

    def _initialize_backend(self, first_frame: np.ndarray) -> None:
        prepare = getattr(self.backend, "prepare_slots", None)
        if prepare is not None:
            prepare(self.slots)
        cache = getattr(self.backend, "cache", None)
        if cache is not None:
            cache.begin_source(self.public_source, continuous=True)
        reset = getattr(self.backend, "reset_state", None)
        if reset is not None:
            reset(self.slots)
        if cache is not None and self.warmup:
            self.backend.process_frame(first_frame, self.slots, -1, 0.0)
            cache.begin_source(self.public_source, continuous=True)
            if reset is not None:
                reset(self.slots)

    def _run(self) -> None:
        capture = None
        recorder: StageWRunRecorder | None = None
        final_status: Mapping[str, Any] = self._state
        try:
            capture = self.capture_factory(self._capture_source(self.source))
            if not capture.isOpened():
                raise RuntimeError("Unable to open video source")
            fps_value = float(capture.get(cv2.CAP_PROP_FPS))
            fps = fps_value if fps_value > 1.0 else 25.0
            ok, first = capture.read()
            if not ok or first is None:
                raise RuntimeError("Video source contains no decodable frame")
            height, width = first.shape[:2]
            if not self.slots_path.is_file():
                raise FileNotFoundError("Parking-space configuration is missing")
            self.slots = load_stage_v_slot_map(
                self.slots_path,
                frame_size=(width, height),
            ).slots
            self._state["total"] = len(self.slots)
            self._state["vacant"] = len(self.slots)
            self._initialize_backend(first)
            if self.output_dir is not None:
                recorder = StageWRunRecorder(
                    self.output_dir,
                    fps=fps,
                    frame_size=(width, height),
                    source=self.public_source,
                    mode=self.mode,
                    config_snapshot=self.config_snapshot,
                )
            frame = first
            frame_index = 0
            while not self._stop.is_set():
                if self.max_frames is not None and frame_index >= self.max_frames:
                    break
                timestamp_s = frame_index / fps
                result = self.backend.process_frame(
                    frame,
                    self.slots,
                    frame_index,
                    timestamp_s,
                )
                emitted = self._event_tracker.update(result)
                cache_value = result.timing_ms.get("cache_hit")
                cache_status = (
                    "not-used"
                    if cache_value is None
                    else ("hit" if bool(cache_value) else "miss")
                )
                payload = frame_result_to_ui_payload(
                    result,
                    self.slots,
                    mode=self.mode,
                    temporal_enabled=self.temporal_enabled,
                    tracker_enabled=self.tracker_enabled,
                    cache_status=cache_status,
                    health="running",
                    message="Processing",
                    source=self.source,
                )
                payload.update(
                    {
                        "running": True,
                        "events_temporally_valid": True,
                        "event_semantics": self._event_tracker.semantics,
                        "recent_event_count": len(self._event_tracker.events()),
                    }
                )
                attributed_ms = float(
                    result.timing_ms.get(
                        "attributed_backend_total",
                        result.timing_ms.get("backend_total", 0.0),
                    )
                )
                annotated, rendered = draw_stage_v_frame(
                    frame=frame,
                    detections=result.vehicle_detections,
                    slots=self.slots,
                    states=result.state_by_slot(),
                    mode=self.mode,
                    processing_fps=1000.0 / max(attributed_ms, 1e-9),
                    cache_status=cache_status,
                    temporal_enabled=self.temporal_enabled,
                    tracker_enabled=self.tracker_enabled,
                    stage_label="STAGE W",
                )
                payload["rendered_slots"] = rendered
                self._publish(annotated, payload)
                if recorder is not None:
                    recorder.write(annotated, emitted)
                final_status = payload
                frame_index += 1
                ok, next_frame = capture.read()
                if not ok or next_frame is None:
                    break
                frame = next_frame
            completed = dict(final_status)
            completed["health"] = "completed"
            completed["running"] = False
            completed["message"] = (
                "Stopped" if self._stop.is_set() else "Input ended"
            )
            with self._lock:
                self._state = sanitize_for_api(completed)
            final_status = completed
        except Exception as exc:
            message = f"Stage W processing error: {exc}"
            self._publish_error(message, total=len(self.slots))
            final_status = self.status()
        finally:
            if capture is not None:
                capture.release()
            if recorder is not None:
                recorder.close(
                    status=final_status,
                    events_temporally_valid=True,
                    event_semantics=self._event_tracker.semantics,
                )

    def status(self) -> dict[str, Any]:
        with self._lock:
            return copy.deepcopy(self._state)

    def frame(self) -> bytes | None:
        with self._lock:
            return self.latest_frame

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        return self._event_tracker.events(limit)

    def sessions(self) -> list[dict[str, Any]]:
        return self._event_tracker.sessions()

    def mjpeg_stream(self):
        while not self._stop.is_set():
            frame = self.frame()
            if frame is not None:
                yield (
                    b"--frame\r\n"
                    b"Content-Type: image/jpeg\r\n\r\n"
                    + frame
                    + b"\r\n"
                )
            time.sleep(0.04)


class StageWErrorProcessor:
    """Serve an explicit dashboard error without attempting a model fallback."""

    def __init__(self, message: str, *, mode: str, source: str | int) -> None:
        self._state = error_payload(
            message,
            mode=mode,
            source=source,
        )
        image = np.full((720, 1280, 3), (22, 26, 32), dtype=np.uint8)
        cv2.putText(
            image,
            str(self._state["message"])[:110],
            (40, 360),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (80, 180, 255),
            2,
            cv2.LINE_AA,
        )
        ok, encoded = cv2.imencode(".jpg", image)
        self._frame = encoded.tobytes() if ok else None
        self._stop = threading.Event()

    def start(self) -> None:
        return None

    def stop(self) -> None:
        self._stop.set()

    def status(self) -> dict[str, Any]:
        return copy.deepcopy(self._state)

    def events(self, limit: int = 50) -> list[dict[str, Any]]:
        return []

    def sessions(self) -> list[dict[str, Any]]:
        return []

    def mjpeg_stream(self):
        if self._frame is not None:
            yield (
                b"--frame\r\nContent-Type: image/jpeg\r\n\r\n"
                + self._frame
                + b"\r\n"
            )


def create_stage_w_app(processor: Any) -> Flask:
    _require_flask()
    assert Flask is not None
    assert Response is not None
    assert jsonify is not None
    assert render_template is not None
    asset_root = Path(__file__).resolve().parent / "stage_w_web"
    app = Flask(
        "stage_w_dashboard",
        template_folder=str(asset_root / "templates"),
        static_folder=str(asset_root / "static"),
    )
    app.config.update(
        DEBUG=False,
        TESTING=False,
        JSON_SORT_KEYS=False,
    )

    @app.get("/")
    def dashboard():
        return render_template("dashboard.html")

    @app.get("/video_feed")
    def video_feed():
        return Response(
            processor.mjpeg_stream(),
            mimetype="multipart/x-mixed-replace; boundary=frame",
        )

    @app.get("/api/status")
    def api_status():
        payload = processor.status()
        return jsonify(payload), (503 if payload.get("health") == "error" else 200)

    @app.get("/api/events")
    def api_events():
        return jsonify(processor.events(limit=50))

    @app.get("/api/sessions")
    def api_sessions():
        return jsonify(processor.sessions())

    @app.get("/api/health")
    def api_health():
        payload = processor.status()
        return jsonify(
            {
                "health": payload.get("health"),
                "running": payload.get("running", False),
                "message": payload.get("message"),
            }
        )

    return app
