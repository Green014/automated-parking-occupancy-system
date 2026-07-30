import numpy as np

from literature_core.calibration import (
    CalibratedFusionModel,
    NonnegativeLogisticFusion,
    PlattCalibrator,
    brier_score,
    expected_calibration_error,
    reliability_bins,
)


def test_platt_calibrator_is_monotonic_and_bounded() -> None:
    scores = [0.02, 0.08, 0.15, 0.70, 0.85, 0.95]
    truth = [0, 0, 0, 1, 1, 1]
    calibrator = PlattCalibrator.fit(scores, truth)
    predictions = calibrator.predict(scores)
    assert calibrator.slope >= 0.0
    assert all(0.0 <= value <= 1.0 for value in predictions)
    assert predictions == sorted(predictions)


def test_calibrated_fusion_round_trip_preserves_predictions() -> None:
    p_cls = [0.05, 0.15, 0.25, 0.60, 0.75, 0.92]
    p_det = [0.00, 0.03, 0.15, 0.20, 0.55, 0.70]
    truth = [0, 0, 0, 1, 1, 1]
    model = CalibratedFusionModel.fit(p_cls, p_det, truth)
    restored = CalibratedFusionModel.from_dict(model.to_dict())
    assert np.allclose(model.predict(p_cls, p_det), restored.predict(p_cls, p_det))
    assert model.fusion.classifier_coefficient >= 0.0
    assert model.fusion.detector_coefficient >= 0.0


def test_nonnegative_fusion_rejects_mismatched_inputs() -> None:
    fusion = NonnegativeLogisticFusion(
        classifier_coefficient=1.0,
        detector_coefficient=1.0,
        intercept=0.0,
        l2=0.0,
        iterations=1,
        converged=True,
    )
    try:
        fusion.predict([0.2], [0.2, 0.3])
    except ValueError as error:
        assert "equally sized" in str(error)
    else:
        raise AssertionError("mismatched branch lengths should fail")


def test_calibration_metrics_and_bins_are_grouped_by_probability() -> None:
    truth = [0, 0, 1, 1]
    probabilities = [0.1, 0.2, 0.8, 0.9]
    assert np.isclose(brier_score(truth, probabilities), 0.025)
    assert 0.0 <= expected_calibration_error(truth, probabilities) <= 1.0
    bins = reliability_bins(truth, probabilities, n_bins=5)
    assert sum(int(row["count"]) for row in bins) == 4


def test_calibration_requires_both_classes() -> None:
    try:
        PlattCalibrator.fit([0.1, 0.2], [0, 0])
    except ValueError as error:
        assert "both occupied and vacant" in str(error)
    else:
        raise AssertionError("one-class calibration should fail")
