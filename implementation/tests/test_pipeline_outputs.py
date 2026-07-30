import csv
import json
from pathlib import Path

import cv2
import numpy as np

from parking_occupancy.models import Detection
from parking_occupancy.pipeline import PipelineConfig, process_video


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
        self.path.write_bytes(b"fake annotated video")

    def isOpened(self) -> bool:
        return True

    def write(self, _: np.ndarray) -> None:
        pass

    def release(self) -> None:
        pass


class FakeDetector:
    def detect(self, _: np.ndarray) -> list[Detection]:
        return [
            Detection(
                bbox=(4.0, 4.0, 20.0, 20.0),
                confidence=0.9,
                class_id=2,
                class_name="car",
            )
        ]

    def metadata(self) -> dict[str, str]:
        return {"backend": "fake"}


def test_canonical_pipeline_writes_complete_submission_artifact_set(
    tmp_path: Path,
    monkeypatch,
) -> None:
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"fake input video")
    slots_path = tmp_path / "slots.json"
    slots_path.write_text(
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
    monkeypatch.setattr(
        "parking_occupancy.pipeline.cv2.VideoCapture",
        FakeCapture,
    )
    monkeypatch.setattr(
        "parking_occupancy.pipeline.cv2.VideoWriter_fourcc",
        lambda *_: 0,
    )
    monkeypatch.setattr(
        "parking_occupancy.pipeline.cv2.VideoWriter",
        FakeWriter,
    )

    output_dir = tmp_path / "run"
    summary = process_video(
        input_path,
        slots_path,
        output_dir,
        FakeDetector(),
        PipelineConfig(
            experiment="b1",
            method_id="B1",
            method_name="YOLOv8 polygon-coverage baseline",
            data_role="development_baseline",
            overlap_threshold=0.40,
        ),
    )

    for name in (
        "annotated.mp4",
        "occupancy.csv",
        "events.csv",
        "detections.jsonl",
        "summary.json",
        "metrics.json",
        "runtime_metadata.json",
    ):
        assert (output_dir / name).is_file()
    assert summary["experiment"] == "b1"
    assert summary["method"]["id"] == "B1"
    assert summary["config"]["method_id"] == "B1"
    assert summary["mapping"]["one_to_one"] is True
    with (output_dir / "occupancy.csv").open(
        newline="",
        encoding="utf-8",
    ) as handle:
        assert len(list(csv.DictReader(handle))) == 2
    detections = [
        json.loads(line)
        for line in (output_dir / "detections.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert len(detections) == 2
    assert detections[0]["detections"][0]["class_name"] == "car"
    assert summary["outputs"]["metrics_json"].endswith("metrics.json")
    assert summary["outputs"]["runtime_metadata_json"].endswith(
        "runtime_metadata.json"
    )
