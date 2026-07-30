from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from parking_occupancy.stage_m_tracking import (
    InferenceSettings,
    OS0ParkingAdapter,
    UltralyticsSequenceAdapter,
    centre_point_slot_states,
    load_parking_regions,
    load_stage_m_protocol,
    load_tracker_config,
)


class FakeBoxes:
    def __init__(
        self,
        xyxy: list[list[float]],
        *,
        track_ids: list[int] | None,
    ) -> None:
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape((-1, 4))
        self.conf = np.asarray([0.9] * len(xyxy), dtype=np.float32)
        self.cls = np.asarray([0] * len(xyxy), dtype=np.float32)
        self.id = (
            None
            if track_ids is None
            else np.asarray(track_ids, dtype=np.float32)
        )

    def __len__(self) -> int:
        return len(self.xyxy)


class FakeModel:
    names = {0: "vehicle"}

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def predict(self, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                boxes=FakeBoxes([], track_ids=None),
                names=self.names,
            )
        ]

    def track(self, **kwargs):
        self.calls.append(kwargs)
        return [
            SimpleNamespace(
                boxes=FakeBoxes([[1, 1, 9, 9]], track_ids=[37]),
                names=self.names,
            )
        ]


def _settings() -> InferenceSettings:
    return InferenceSettings(
        weights="frozen.pt",
        confidence=0.3,
        nms_iou=0.7,
        image_size=640,
        class_ids=(0,),
        max_detections=300,
        device="cpu",
    )


def test_stage_m_protocol_and_tracker_configs_are_frozen() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    protocol = load_stage_m_protocol(
        root
        / "configs"
        / "stage_m_open_source_tracking_frozen_20260728.yaml"
    )

    assert tuple(protocol["methods"]) == (
        "OS0-Controlled",
        "T0",
        "T1",
        "T2",
        "T3",
    )
    assert protocol["scope"]["stage_l_artifact_modification"] == "prohibited"
    assert protocol["classifier"]["occupied_threshold"] == 0.76
    assert (
        load_tracker_config(
            root
            / "configs"
            / "tracktrack_stage_m_frozen_20260728.yaml"
        )["tracker_type"]
        == "tracktrack"
    )


def test_tracker_persists_within_source_and_resets_on_switch() -> None:
    models: list[FakeModel] = []

    def factory(_weights: str) -> FakeModel:
        model = FakeModel()
        models.append(model)
        return model

    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    adapter = UltralyticsSequenceAdapter(
        _settings(),
        tracker_config=(
            root
            / "configs"
            / "tracktrack_stage_m_frozen_20260728.yaml"
        ),
        model_factory=factory,
    )
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    adapter.begin_source("video-a", continuous=True)
    first = adapter.detect(frame)
    second = adapter.detect(frame)

    assert adapter.generation == 1
    assert len(models[0].calls) == 2
    assert first[0].track_id == second[0].track_id == 37
    assert all(call["persist"] is True for call in models[0].calls)

    adapter.begin_source("video-b", continuous=True)
    adapter.detect(frame)
    assert adapter.generation == 2
    assert len(models[1].calls) == 1

    adapter.begin_source("still-1", continuous=False)
    adapter.detect(frame)
    adapter.begin_source("still-1", continuous=False)
    adapter.detect(frame)
    assert adapter.generation == 4


def test_no_detection_frame_is_preserved_without_error() -> None:
    model = FakeModel()
    adapter = UltralyticsSequenceAdapter(
        _settings(),
        tracker_config=None,
        model_factory=lambda _weights: model,
    )
    adapter.begin_source("video", continuous=True)

    assert adapter.detect(np.zeros((8, 8, 3), dtype=np.uint8)) == ()


def test_centre_point_slot_logging_matches_expected_region() -> None:
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    slots = load_parking_regions(
        root
        / "data"
        / "stage_m"
        / "stage_m_smoke_regions_20260728.json"
    )
    from parking_occupancy.models import Detection

    states = centre_point_slot_states(
        [
            Detection(
                bbox=(326, 186, 344, 220),
                confidence=0.9,
                class_id=0,
                class_name="vehicle",
                track_id=9,
            )
        ],
        slots,
    )

    assert states["slot_002"] is True
    assert sum(states.values()) == 1


def test_os0_static_mode_constructs_a_fresh_official_object(tmp_path) -> None:
    region = tmp_path / "regions.json"
    region.write_text(
        '[{"slot_id":"s1","points":[[0,0],[10,0],[10,10],[0,10]]}]',
        encoding="utf-8",
    )
    root = __import__("pathlib").Path(__file__).resolve().parents[1]
    managers = []

    class Manager:
        def __init__(self, **kwargs) -> None:
            self.kwargs = kwargs
            self.boxes = [np.asarray([1, 1, 9, 9], dtype=np.float32)]
            self.clss = [0]
            self.confs = [0.8]
            self.track_ids = [5]
            self.model = SimpleNamespace(names={0: "vehicle"})

        def process(self, frame):
            return SimpleNamespace(
                plot_im=frame,
                filled_slots=1,
                available_slots=0,
            )

    def factory(**kwargs):
        manager = Manager(**kwargs)
        managers.append(manager)
        return manager

    adapter = OS0ParkingAdapter(
        _settings(),
        region_json=region,
        tracker_config=(
            root
            / "configs"
            / "tracktrack_stage_m_frozen_20260728.yaml"
        ),
        manager_factory=factory,
    )
    frame = np.zeros((12, 12, 3), dtype=np.uint8)
    adapter.begin_source("still", continuous=False)
    assert adapter.process(frame).slot_states == {"s1": True}
    adapter.begin_source("still", continuous=False)
    adapter.process(frame)

    assert adapter.generation == 2
    assert len(managers) == 2
    assert managers[0].kwargs["tracker"].endswith(
        "tracktrack_stage_m_frozen_20260728.yaml"
    )
