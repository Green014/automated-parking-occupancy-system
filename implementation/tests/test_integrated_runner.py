from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from parking_occupancy.integrated_cli import main
from parking_occupancy.integrated_runner import (
    DEFAULT_INTEGRATED_CONFIG,
    IntegratedFrameProcessor,
    load_integrated_config,
)
from parking_occupancy.models import Detection, ParkingSlot


class FakeCapture:
    def __init__(self, _: str) -> None:
        self.frames = [
            np.zeros((24, 32, 3), dtype=np.uint8),
            np.zeros((24, 32, 3), dtype=np.uint8),
        ]
        self.index = 0

    def isOpened(self) -> bool:
        return True

    def get(self, property_id: int) -> float:
        return {
            cv2.CAP_PROP_FRAME_WIDTH: 32.0,
            cv2.CAP_PROP_FRAME_HEIGHT: 24.0,
            cv2.CAP_PROP_FPS: 10.0,
        }.get(property_id, 0.0)

    def read(self) -> tuple[bool, np.ndarray | None]:
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self) -> None:
        pass


class FakeWriter:
    def __init__(self, path: str, *_: object) -> None:
        self.path = Path(path)
        self.path.write_bytes(b"fake integrated video")

    def isOpened(self) -> bool:
        return True

    def write(self, _: np.ndarray) -> None:
        pass

    def release(self) -> None:
        pass


class FakeDetector:
    def __init__(self) -> None:
        self.frame = 0
        self.sources: list[tuple[str, bool]] = []
        self.current: tuple[Detection, ...] | None = None

    def begin_source(self, source_id: str, *, continuous: bool) -> None:
        self.sources.append((source_id, continuous))
        self.frame = 0

    def detect(self, _: np.ndarray):
        if self.current is not None:
            return self.current
        self.frame += 1
        if self.frame == 1:
            return (
                Detection(
                    bbox=(4.0, 4.0, 20.0, 20.0),
                    confidence=0.9,
                    class_id=0,
                    class_name="vehicle",
                    track_id=7,
                ),
            )
        return ()

    def metadata(self):
        return {
            "backend": "fake",
            "generation": len(self.sources),
            "source_switch_reset": True,
        }


class FakeClassifier:
    def __init__(self) -> None:
        self.calls: list[tuple[str, ...]] = []
        self.score = 0.1

    def predict(self, _frame, slots):
        self.calls.append(tuple(slot.slot_id for slot in slots))
        return {slot.slot_id: self.score for slot in slots}

    def metadata(self):
        return {"backend": "fake"}


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    video = tmp_path / "input.mp4"
    video.write_bytes(b"fake video input")
    slots = tmp_path / "slots.json"
    slots.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_width": 32,
                "source_height": 24,
                "coordinate_system": "pixel",
                "slots": [
                    {
                        "id": "slot-1",
                        "points": [[2, 2], [22, 2], [22, 22], [2, 22]],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    weights = tmp_path / "d1.pt"
    checkpoint = tmp_path / "e1b.pt"
    weights.write_bytes(b"fake d1")
    checkpoint.write_bytes(b"fake e1b")
    return video, slots, weights, checkpoint


def test_integrated_cli_fake_run_writes_complete_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    video, slots, weights, checkpoint = _write_inputs(tmp_path)
    detector = FakeDetector()
    classifier = FakeClassifier()
    monkeypatch.setattr(
        "parking_occupancy.integrated_runner.cv2.VideoCapture",
        FakeCapture,
    )
    monkeypatch.setattr(
        "parking_occupancy.integrated_runner.cv2.VideoWriter_fourcc",
        lambda *_: 0,
    )
    monkeypatch.setattr(
        "parking_occupancy.integrated_runner.cv2.VideoWriter",
        FakeWriter,
    )
    monkeypatch.setattr(
        "parking_occupancy.integrated_runner.create_detector",
        lambda **_: detector,
    )
    monkeypatch.setattr(
        "parking_occupancy.integrated_runner.create_classifier",
        lambda **_: classifier,
    )
    output = tmp_path / "integrated"

    main(
        [
            "--input",
            str(video),
            "--slots",
            str(slots),
            "--d1-weights",
            str(weights),
            "--e1b-checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output),
            "--tracker",
            "none",
        ]
    )

    expected = {
        "occupancy.csv",
        "events.csv",
        "detections.jsonl",
        "annotated.mp4",
        "metrics.json",
        "summary.json",
        "runtime_metadata.json",
    }
    assert {path.name for path in output.iterdir()} == expected
    assert all((output / name).stat().st_size > 0 for name in expected)
    with (output / "occupancy.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 2
    assert rows[0]["classifier_consulted"] == "0"
    assert rows[1]["classifier_consulted"] == "1"
    assert classifier.calls == [(), ("slot-1",)]
    metrics = json.loads((output / "metrics.json").read_text())
    assert metrics["status"] == "not_computed_no_truth"
    assert metrics["truth_required_for_inference"] is False

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        main(
            [
                "--input",
                str(video),
                "--slots",
                str(slots),
                "--d1-weights",
                str(weights),
                "--e1b-checkpoint",
                str(checkpoint),
                "--output-dir",
                str(output),
            ]
        )


def test_integrated_processor_resets_temporal_and_event_state_per_source() -> None:
    config = load_integrated_config(DEFAULT_INTEGRATED_CONFIG)
    detector = FakeDetector()
    classifier = FakeClassifier()
    processor = IntegratedFrameProcessor(
        slots=(
            ParkingSlot(
                slot_id="slot-1",
                points=((2, 2), (22, 2), (22, 22), (2, 22)),
            ),
        ),
        detector=detector,
        classifier=classifier,
        config=config,
        temporal_enabled=True,
    )
    frame = np.zeros((24, 32, 3), dtype=np.uint8)
    detector.current = (
        Detection(
            bbox=(4, 4, 20, 20),
            confidence=0.9,
            class_id=0,
            class_name="vehicle",
        ),
    )
    processor.begin_source("source-a")
    occupied = processor.process(frame, fps=10.0)
    assert occupied.states["slot-1"].occupied

    detector.current = ()
    classifier.score = 0.1
    processor.begin_source("source-b")
    vacant = processor.process(frame, fps=10.0)

    assert not vacant.states["slot-1"].occupied
    assert vacant.frame_index == 0
    assert vacant.events == ()
    assert detector.sources == [
        ("source-a", True),
        ("source-b", True),
    ]
