from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parking_occupancy.artifact_registry import (
    STAGE_W_PRE_HARDENING_REGISTRY_SHA256,
    verify_historical_artifact_registry,
)
from parking_occupancy.models import Detection
from parking_occupancy.stage_v import (
    DetectionCache,
    FrameOccupancyResult,
    FusionBackend,
    SlotOccupancyState,
)

pytest.importorskip(
    "flask",
    reason='Stage W server tests require the optional "dashboard" dependency',
)

from parking_occupancy.stage_w_server import (
    StageWErrorProcessor,
    StageWProcessor,
    create_stage_w_app,
)


class FakeCapture:
    def __init__(self, frames: list[np.ndarray], *, opened: bool = True) -> None:
        self.frames = list(frames)
        self.opened = opened
        self.released = False

    def isOpened(self):
        return self.opened

    def get(self, key):
        return 5.0 if key == cv2.CAP_PROP_FPS else 0.0

    def read(self):
        if not self.frames:
            return False, None
        return True, self.frames.pop(0)

    def release(self):
        self.released = True


class FakeBackend:
    mode = "fusion"

    def __init__(self, *, fail: bool = False) -> None:
        self.fail = fail
        self.prepare_calls = 0
        self.process_calls = 0

    def prepare_slots(self, slots):
        self.prepare_calls += 1

    def process_frame(self, frame, slots, frame_index, timestamp_s):
        self.process_calls += 1
        if self.fail:
            raise RuntimeError(
                "model failed at " + r"C:" + r"\Users\private\best.pt"
            )
        occupied = frame_index == 0
        return FrameOccupancyResult(
            frame_index=frame_index,
            timestamp_s=timestamp_s,
            slot_states=tuple(
                SlotOccupancyState(
                    slot_id=slot.slot_id,
                    occupied=occupied if index == 0 else False,
                    evidence_score=0.9,
                    evidence_source="fake-fusion",
                    track_id=None,
                )
                for index, slot in enumerate(slots)
            ),
            vehicle_detections=(
                Detection(
                    bbox=(2, 2, 13, 13),
                    confidence=0.9,
                    class_id=0,
                    class_name="vehicle",
                ),
            ),
            timing_ms={
                "backend_total": 5.0,
                "attributed_backend_total": 10.0,
                "cache_hit": float(frame_index > 0),
            },
        )

    def metadata(self):
        return {
            "mode": "fusion",
            "method_id": "fake",
            "model_load_count": 1,
        }


class FakeDetector:
    def __init__(self) -> None:
        self.calls = 0
        self.sources = 0

    def begin_source(self, source_id, *, continuous):
        self.sources += 1

    def detect(self, frame):
        self.calls += 1
        return [
            Detection(
                bbox=(2, 2, 13, 13),
                confidence=0.9,
                class_id=0,
                class_name="vehicle",
            )
        ]

    def metadata(self):
        return {"backend": "fake-detector"}


class FakeClassifier:
    def __init__(self) -> None:
        self.calls = 0

    def predict(self, frame, slots):
        self.calls += 1
        return {slot.slot_id: 0.9 for slot in slots}

    def metadata(self):
        return {"backend": "fake-classifier"}


def _slots(tmp_path: Path) -> Path:
    path = tmp_path / "slots.json"
    path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_width": 32,
                "source_height": 16,
                "slots": [
                    {
                        "id": "A01",
                        "points": [[1, 1], [14, 1], [14, 14], [1, 14]],
                    },
                    {
                        "id": "A02",
                        "points": [[17, 1], [30, 1], [30, 14], [17, 14]],
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return path


def _processor(
    tmp_path: Path,
    backend: FakeBackend,
    capture: FakeCapture,
    *,
    output: bool = True,
) -> StageWProcessor:
    return StageWProcessor(
        source=(
            "rtsp://"
            + "fixture-user"
            + ":"
            + "fixture-token"
            + "@"
            + "example.test:8554/live"
        ),
        slots_path=_slots(tmp_path),
        backend=backend,
        mode="fusion",
        config_snapshot={
            "temporal": {"enabled": False},
            "tracking": {"backend": "none"},
        },
        output_dir=tmp_path / "dashboard-smoke" if output else None,
        warmup=False,
        capture_factory=lambda _source: capture,
    )


def test_fake_backend_end_to_end_dashboard_output_and_release(
    tmp_path: Path,
) -> None:
    frames = [
        np.zeros((16, 32, 3), dtype=np.uint8),
        np.ones((16, 32, 3), dtype=np.uint8),
    ]
    capture = FakeCapture(frames)
    backend = FakeBackend()
    processor = _processor(tmp_path, backend, capture)
    processor.run_blocking()
    status = processor.status()
    assert status["health"] == "completed"
    assert status["frame_index"] == 1
    assert status["occupied"] == 0
    assert status["vacant"] == 2
    assert status["total"] == 2
    assert status["rendered_slots"] == 2
    assert status["source"] == "rtsp://example.test:8554/<redacted>"
    assert capture.released is True
    assert backend.process_calls == 2
    assert backend.prepare_calls == 1

    output = tmp_path / "dashboard-smoke"
    assert {
        "annotated.mp4",
        "configuration_snapshot.yaml",
        "events.jsonl",
        "status.json",
        "summary.json",
    } == {path.name for path in output.iterdir()}
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["frames"] == 2
    assert summary["events"] == 1
    assert summary["initial_frame_emits_event"] is False
    assert "secret" not in (output / "summary.json").read_text(encoding="utf-8")


def test_flask_status_and_video_stream_endpoints(tmp_path: Path) -> None:
    capture = FakeCapture([np.zeros((16, 32, 3), dtype=np.uint8)])
    backend = FakeBackend()
    processor = _processor(tmp_path, backend, capture, output=False)
    processor.run_blocking()
    app = create_stage_w_app(processor)
    app.config["TESTING"] = True
    client = app.test_client()

    dashboard = client.get("/")
    assert dashboard.status_code == 200
    assert b"Attributed FPS" in dashboard.data
    response = client.get("/api/status")
    assert response.status_code == 200
    assert response.get_json()["total"] == 2
    assert client.get("/api/events").status_code == 200
    assert client.get("/api/sessions").status_code == 200
    stream = client.get("/video_feed", buffered=False)
    first_chunk = next(stream.response)
    assert first_chunk.startswith(b"--frame\r\nContent-Type: image/jpeg")
    stream.close()


def test_backend_exception_is_visible_and_sanitized_in_api(
    tmp_path: Path,
) -> None:
    capture = FakeCapture([np.zeros((16, 32, 3), dtype=np.uint8)])
    processor = _processor(
        tmp_path,
        FakeBackend(fail=True),
        capture,
        output=False,
    )
    processor.run_blocking()
    app = create_stage_w_app(processor)
    app.config["TESTING"] = True
    response = app.test_client().get("/api/status")
    payload = response.get_json()
    assert response.status_code == 503
    assert payload["health"] == "error"
    serialized = json.dumps(payload)
    assert ("C:" + r"\\Users") not in serialized
    assert "secret" not in serialized
    assert capture.released is True


def test_model_prepare_happens_once_across_multiple_clients(
    tmp_path: Path,
) -> None:
    capture = FakeCapture([np.zeros((16, 32, 3), dtype=np.uint8)])
    backend = FakeBackend()
    processor = _processor(tmp_path, backend, capture, output=False)
    processor.run_blocking()
    app = create_stage_w_app(processor)
    app.config["TESTING"] = True
    client = app.test_client()
    for _ in range(4):
        assert client.get("/api/status").status_code == 200
        assert client.get("/").status_code == 200
    assert backend.prepare_calls == 1


def test_fake_detector_classifier_reach_flask_ui_end_to_end(
    tmp_path: Path,
) -> None:
    detector = FakeDetector()
    classifier = FakeClassifier()
    backend = FusionBackend(
        DetectionCache(detector),
        classifier,
        overlap_threshold=0.40,
        classifier_threshold=0.76,
    )
    capture = FakeCapture([np.zeros((16, 32, 3), dtype=np.uint8)])
    processor = StageWProcessor(
        source="input.mp4",
        slots_path=_slots(tmp_path),
        backend=backend,
        mode="fusion",
        config_snapshot={
            "temporal": {"enabled": False},
            "tracking": {"backend": "none"},
        },
        warmup=False,
        capture_factory=lambda _source: capture,
    )
    processor.run_blocking()
    app = create_stage_w_app(processor)
    app.config["TESTING"] = True
    payload = app.test_client().get("/api/status").get_json()
    assert payload["health"] == "completed"
    assert payload["occupied"] == 2
    assert len(payload["slots"]) == 2
    assert detector.calls == 1
    assert detector.sources == 1
    assert classifier.calls == 1


def test_explicit_error_processor_does_not_silently_switch_backend() -> None:
    processor = StageWErrorProcessor(
        "Fusion model is missing",
        mode="fusion",
        source="camera.mp4",
    )
    app = create_stage_w_app(processor)
    app.config["TESTING"] = True
    payload = app.test_client().get("/api/status").get_json()
    assert payload["health"] == "error"
    assert payload["mode"] == "fusion"
    assert "missing" in payload["message"]


def test_existing_stage_w_output_is_rejected(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    capture = FakeCapture([np.zeros((16, 32, 3), dtype=np.uint8)])
    try:
        StageWProcessor(
            source="input.mp4",
            slots_path=_slots(tmp_path),
            backend=FakeBackend(),
            mode="fusion",
            config_snapshot={},
            output_dir=output,
            capture_factory=lambda _source: capture,
        )
    except FileExistsError:
        pass
    else:
        raise AssertionError("Existing output directory must be rejected")


def test_checked_in_stage_w_registry_verifies() -> None:
    implementation_root = Path(__file__).resolve().parents[1]
    result = verify_historical_artifact_registry(
        implementation_root / "data" / "STAGE_W_ARTIFACT_REGISTRY.yaml",
        artifact_root=implementation_root,
        expected_registry_sha256=STAGE_W_PRE_HARDENING_REGISTRY_SHA256,
        immutable_path_prefixes=("outputs",),
    )
    assert result["verified"] is True, result["errors"]
    assert result["live_release_artifacts_compared"] is False
