"""Deterministic calibration and interpretable logistic evidence fusion."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Sequence

import numpy as np


EPSILON = 1e-6


def _arrays(
    y_true: Sequence[int],
    *columns: Sequence[float],
) -> tuple[np.ndarray, ...]:
    truth = np.asarray(y_true, dtype=np.float64)
    if truth.ndim != 1 or truth.size == 0:
        raise ValueError("y_true must be a non-empty one-dimensional sequence")
    if not np.all(np.isin(truth, (0.0, 1.0))):
        raise ValueError("y_true must contain only 0 and 1")
    if np.unique(truth).size != 2:
        raise ValueError("calibration needs both occupied and vacant examples")
    result = [truth]
    for column in columns:
        values = np.asarray(column, dtype=np.float64)
        if values.shape != truth.shape:
            raise ValueError("score columns must match y_true")
        if not np.all(np.isfinite(values)):
            raise ValueError("scores must be finite")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("scores must lie in [0, 1]")
        result.append(values)
    return tuple(result)


def sigmoid(values: np.ndarray | Sequence[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    output = np.empty_like(array)
    positive = array >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-array[positive]))
    exponent = np.exp(array[~positive])
    output[~positive] = exponent / (1.0 + exponent)
    return output


def logit(values: np.ndarray | Sequence[float]) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float64), EPSILON, 1 - EPSILON)
    return np.log(clipped / (1.0 - clipped))


def _logistic_objective(
    features: np.ndarray,
    truth: np.ndarray,
    parameters: np.ndarray,
    l2: float,
) -> float:
    logits = features @ parameters
    data_loss = np.mean(np.logaddexp(0.0, logits) - truth * logits)
    return float(data_loss + 0.5 * l2 * np.dot(parameters[:-1], parameters[:-1]))


def _fit_logistic(
    features: np.ndarray,
    truth: np.ndarray,
    *,
    l2: float,
    nonnegative_coefficients: bool,
    max_iterations: int,
    tolerance: float,
) -> tuple[np.ndarray, int, bool]:
    if features.ndim != 2 or features.shape[0] != truth.size:
        raise ValueError("invalid logistic feature matrix")
    if l2 < 0.0 or max_iterations <= 0 or tolerance <= 0.0:
        raise ValueError("invalid logistic optimizer settings")

    design = np.column_stack((features, np.ones(truth.size)))
    parameters = np.zeros(design.shape[1], dtype=np.float64)
    prevalence = np.clip(np.mean(truth), EPSILON, 1.0 - EPSILON)
    parameters[-1] = np.log(prevalence / (1.0 - prevalence))
    regularizer = np.diag([l2] * features.shape[1] + [0.0])
    converged = False

    for iteration in range(1, max_iterations + 1):
        probabilities = sigmoid(design @ parameters)
        gradient = design.T @ (probabilities - truth) / truth.size
        gradient[:-1] += l2 * parameters[:-1]
        weights = np.maximum(probabilities * (1.0 - probabilities), 1e-8)
        hessian = (design.T * weights) @ design / truth.size + regularizer
        try:
            step = np.linalg.solve(hessian, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(hessian) @ gradient

        objective = _logistic_objective(design, truth, parameters, l2)
        scale = 1.0
        candidate = parameters.copy()
        for _ in range(24):
            candidate = parameters - scale * step
            if nonnegative_coefficients:
                candidate[:-1] = np.maximum(candidate[:-1], 0.0)
            if _logistic_objective(design, truth, candidate, l2) <= objective:
                break
            scale *= 0.5

        delta = float(np.max(np.abs(candidate - parameters)))
        parameters = candidate
        if delta < tolerance:
            converged = True
            return parameters, iteration, converged
    return parameters, max_iterations, converged


@dataclass(frozen=True, slots=True)
class PlattCalibrator:
    """Monotonic sigmoid calibration for one score/evidence branch."""

    slope: float
    intercept: float
    l2: float
    iterations: int
    converged: bool

    @classmethod
    def fit(
        cls,
        scores: Sequence[float],
        y_true: Sequence[int],
        *,
        l2: float = 1e-3,
        max_iterations: int = 100,
        tolerance: float = 1e-9,
    ) -> "PlattCalibrator":
        truth, values = _arrays(y_true, scores)
        parameters, iterations, converged = _fit_logistic(
            values[:, None],
            truth,
            l2=l2,
            nonnegative_coefficients=True,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        return cls(
            slope=float(parameters[0]),
            intercept=float(parameters[1]),
            l2=l2,
            iterations=iterations,
            converged=converged,
        )

    def predict(self, scores: Sequence[float]) -> list[float]:
        values = np.asarray(scores, dtype=np.float64)
        if values.ndim != 1 or not np.all(np.isfinite(values)):
            raise ValueError("scores must be a finite one-dimensional sequence")
        if np.any((values < 0.0) | (values > 1.0)):
            raise ValueError("scores must lie in [0, 1]")
        return [float(value) for value in sigmoid(self.slope * values + self.intercept)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PlattCalibrator":
        return cls(
            slope=float(payload["slope"]),
            intercept=float(payload["intercept"]),
            l2=float(payload.get("l2", 0.0)),
            iterations=int(payload.get("iterations", 0)),
            converged=bool(payload.get("converged", True)),
        )


@dataclass(frozen=True, slots=True)
class NonnegativeLogisticFusion:
    """Fuse calibrated branch log-odds with non-negative coefficients."""

    classifier_coefficient: float
    detector_coefficient: float
    intercept: float
    l2: float
    iterations: int
    converged: bool

    @classmethod
    def fit(
        cls,
        p_cls_calibrated: Sequence[float],
        p_det_calibrated: Sequence[float],
        y_true: Sequence[int],
        *,
        l2: float = 1e-2,
        max_iterations: int = 100,
        tolerance: float = 1e-9,
    ) -> "NonnegativeLogisticFusion":
        truth, cls_values, det_values = _arrays(
            y_true,
            p_cls_calibrated,
            p_det_calibrated,
        )
        features = np.column_stack((logit(cls_values), logit(det_values)))
        parameters, iterations, converged = _fit_logistic(
            features,
            truth,
            l2=l2,
            nonnegative_coefficients=True,
            max_iterations=max_iterations,
            tolerance=tolerance,
        )
        return cls(
            classifier_coefficient=float(parameters[0]),
            detector_coefficient=float(parameters[1]),
            intercept=float(parameters[2]),
            l2=l2,
            iterations=iterations,
            converged=converged,
        )

    def predict(
        self,
        p_cls_calibrated: Sequence[float],
        p_det_calibrated: Sequence[float],
    ) -> list[float]:
        cls_values = np.asarray(p_cls_calibrated, dtype=np.float64)
        det_values = np.asarray(p_det_calibrated, dtype=np.float64)
        if cls_values.shape != det_values.shape or cls_values.ndim != 1:
            raise ValueError("calibrated branches must be equally sized vectors")
        if np.any((cls_values < 0.0) | (cls_values > 1.0)) or np.any(
            (det_values < 0.0) | (det_values > 1.0)
        ):
            raise ValueError("calibrated branch values must lie in [0, 1]")
        logits = (
            self.classifier_coefficient * logit(cls_values)
            + self.detector_coefficient * logit(det_values)
            + self.intercept
        )
        return [float(value) for value in sigmoid(logits)]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "NonnegativeLogisticFusion":
        return cls(
            classifier_coefficient=float(payload["classifier_coefficient"]),
            detector_coefficient=float(payload["detector_coefficient"]),
            intercept=float(payload["intercept"]),
            l2=float(payload.get("l2", 0.0)),
            iterations=int(payload.get("iterations", 0)),
            converged=bool(payload.get("converged", True)),
        )


@dataclass(frozen=True, slots=True)
class CalibratedFusionModel:
    classifier_calibrator: PlattCalibrator
    detector_calibrator: PlattCalibrator
    fusion: NonnegativeLogisticFusion

    @classmethod
    def fit(
        cls,
        p_cls: Sequence[float],
        p_det: Sequence[float],
        y_true: Sequence[int],
    ) -> "CalibratedFusionModel":
        _arrays(y_true, p_cls, p_det)
        classifier_calibrator = PlattCalibrator.fit(p_cls, y_true)
        detector_calibrator = PlattCalibrator.fit(p_det, y_true)
        p_cls_calibrated = classifier_calibrator.predict(p_cls)
        p_det_calibrated = detector_calibrator.predict(p_det)
        fusion = NonnegativeLogisticFusion.fit(
            p_cls_calibrated,
            p_det_calibrated,
            y_true,
        )
        return cls(classifier_calibrator, detector_calibrator, fusion)

    def predict_branches(
        self,
        p_cls: Sequence[float],
        p_det: Sequence[float],
    ) -> tuple[list[float], list[float]]:
        if len(p_cls) != len(p_det):
            raise ValueError("branch lengths differ")
        return (
            self.classifier_calibrator.predict(p_cls),
            self.detector_calibrator.predict(p_det),
        )

    def predict(
        self,
        p_cls: Sequence[float],
        p_det: Sequence[float],
    ) -> list[float]:
        calibrated = self.predict_branches(p_cls, p_det)
        return self.fusion.predict(*calibrated)

    def to_dict(self) -> dict[str, Any]:
        return {
            "classifier_calibrator": self.classifier_calibrator.to_dict(),
            "detector_calibrator": self.detector_calibrator.to_dict(),
            "fusion": self.fusion.to_dict(),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "CalibratedFusionModel":
        return cls(
            PlattCalibrator.from_dict(payload["classifier_calibrator"]),
            PlattCalibrator.from_dict(payload["detector_calibrator"]),
            NonnegativeLogisticFusion.from_dict(payload["fusion"]),
        )


def brier_score(y_true: Sequence[int], probabilities: Sequence[float]) -> float:
    truth, values = _arrays(y_true, probabilities)
    return float(np.mean((values - truth) ** 2))


def reliability_bins(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> list[dict[str, float | int]]:
    truth, values = _arrays(y_true, probabilities)
    if n_bins <= 0:
        raise ValueError("n_bins must be positive")
    bin_ids = np.minimum((values * n_bins).astype(int), n_bins - 1)
    rows: list[dict[str, float | int]] = []
    for index in range(n_bins):
        mask = bin_ids == index
        count = int(np.sum(mask))
        mean_score = float(np.mean(values[mask])) if count else 0.0
        observed_rate = float(np.mean(truth[mask])) if count else 0.0
        rows.append(
            {
                "bin": index,
                "lower": index / n_bins,
                "upper": (index + 1) / n_bins,
                "count": count,
                "mean_score": mean_score,
                "observed_rate": observed_rate,
                "absolute_gap": abs(mean_score - observed_rate),
            }
        )
    return rows


def expected_calibration_error(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> float:
    rows = reliability_bins(y_true, probabilities, n_bins=n_bins)
    total = sum(int(row["count"]) for row in rows)
    return sum(
        int(row["count"]) / total * float(row["absolute_gap"])
        for row in rows
    )


def calibration_metrics(
    y_true: Sequence[int],
    probabilities: Sequence[float],
    *,
    n_bins: int = 10,
) -> dict[str, float]:
    return {
        "brier_score": brier_score(y_true, probabilities),
        "ece": expected_calibration_error(
            y_true,
            probabilities,
            n_bins=n_bins,
        ),
    }
