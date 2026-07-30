import numpy as np

from literature_core.fusion import FusionConfig
from literature_core.models import Detection, ParkingSlot
from literature_core.pipeline import LiteratureCorePipeline, PipelineConfig


class FakeClassifier:
    patch_size = (224, 224)

    def predict_patches(self, patches):
        return [0.8 for _ in patches]


class FakeDetector:
    def detect(self, frame):
        return [Detection((0, 0, 10, 10), 1.0, 0, "car")]

    def metadata(self):
        return {"backend": "fake"}


def test_pipeline_retains_branch_evidence_and_decision() -> None:
    slot = ParkingSlot("A", ((0, 0), (10, 0), (10, 10), (0, 10)))
    pipeline = LiteratureCorePipeline(
        (slot,),
        FakeClassifier(),
        FakeDetector(),
        PipelineConfig(
            minimum_slot_coverage=0.1,
            decision_threshold=0.5,
            fusion=FusionConfig(0.5, 0.5, 0.0),
            use_temporal=False,
        ),
    )
    result = pipeline.process_frame(
        np.zeros((20, 20, 3), dtype=np.uint8),
        frame_index=0,
        timestamp_s=0.0,
    )
    assert result.decisions[0].p_cls == 0.8
    assert result.decisions[0].p_det == 1.0
    assert result.decisions[0].probability == 0.9
    assert result.decisions[0].filtered_probability == 0.9
    assert result.decisions[0].occupied
    assert result.detector_evidence[0].detection_label == "car"


def test_classifier_only_ablation_is_supported() -> None:
    slot = ParkingSlot("A", ((0, 0), (10, 0), (10, 10), (0, 10)))
    pipeline = LiteratureCorePipeline(
        (slot,),
        FakeClassifier(),
        None,
        PipelineConfig(use_temporal=False),
    )
    result = pipeline.process_frame(
        np.zeros((20, 20, 3), dtype=np.uint8),
        frame_index=0,
        timestamp_s=0.0,
    )
    assert result.decisions[0].probability == 0.8
    assert result.decisions[0].filtered_probability == 0.8
    assert result.decisions[0].p_det is None


def test_temporal_pipeline_retains_raw_and_filtered_fusion_scores() -> None:
    slot = ParkingSlot("A", ((0, 0), (10, 0), (10, 10), (0, 10)))
    pipeline = LiteratureCorePipeline(
        (slot,),
        FakeClassifier(),
        None,
        PipelineConfig(use_temporal=True),
    )
    result = pipeline.process_frame(
        np.zeros((20, 20, 3), dtype=np.uint8),
        frame_index=0,
        timestamp_s=0.0,
    )
    assert result.decisions[0].probability == 0.8
    assert result.decisions[0].filtered_probability == 0.48
    assert result.decisions[0].raw_occupied
    assert not result.decisions[0].occupied
