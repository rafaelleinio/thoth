from __future__ import annotations

import datetime
from dataclasses import dataclass, replace
from statistics import fmean
from typing import Any, List, Optional, Set

from thoth.anomaly.base import Point, TimeSeries, convert_to_timeseries
from thoth.anomaly.models import (
    BaseModelFactory,
    DefaultModelFactory,
    Model,
    SimpleModelFactory,
)
from thoth.logging import get_logger
from thoth.profiler import Metric, ProfilingReport


class OptimizationFailedError(Exception):
    pass


@dataclass
class ValidationPoint:
    ts: datetime.datetime
    true_value: float
    predicted: Optional[float] = None
    error: Optional[float] = None


@dataclass
class ValidationTimeSeries:
    model_name: str
    points: List[ValidationPoint]
    mean_error: float
    threshold: float
    below_threshold_proportion: float

    def __lt__(self, other: ValidationTimeSeries) -> bool:
        return (self.threshold, self.mean_error) < (other.threshold, self.mean_error)


@dataclass
class MetricAnomalyOptimizationReport:
    metric: Metric
    best_model_name: str
    threshold: float
    validation_results: List[ValidationTimeSeries]
    window: Optional[Any] = None


@dataclass
class DatasetAnomalyOptimizationReport:
    dataset: str
    confidence: float
    metric_anomaly_optimization_reports: List[MetricAnomalyOptimizationReport]

    def get_metric_optimization(
        self, metric: Metric
    ) -> MetricAnomalyOptimizationReport:
        return [
            metric_config
            for metric_config in self.metric_anomaly_optimization_reports
            if metric_config.metric == metric
        ].pop(0)

    def get_metrics(self) -> Set[Metric]:
        return set(
            profiling_value.metric
            for profiling_value in self.metric_anomaly_optimization_reports
        )


def _find_proportion_below_threshold(errors: List[float], threshold: float) -> float:
    n_points = len(errors)
    n_points_below_threshold = len([e for e in errors if e <= threshold])
    return n_points_below_threshold / n_points


@dataclass
class _ThresholdProportion:
    threshold: float
    below_threshold_proportion: float

    def __lt__(self, other: _ThresholdProportion) -> bool:
        return self.threshold < other.threshold


def _find_best_threshold(
    validation_points: List[ValidationPoint], confidence: float
) -> _ThresholdProportion:
    errors = [p.error for p in validation_points if p.error is not None]
    thresholds_proportions = [
        _ThresholdProportion(
            threshold,
            below_threshold_proportion=_find_proportion_below_threshold(
                errors=errors, threshold=threshold
            ),
        )
        for threshold in [(t / 100) for t in range(1, 101)]
    ]
    best_threshold: _ThresholdProportion = min(
        [
            tp
            for tp in thresholds_proportions
            if tp.below_threshold_proportion >= confidence
        ]
    )
    return best_threshold


def _validate_last_ts(
    points: List[Point], model: Model, start_ts: datetime.datetime
) -> ValidationPoint:
    """Validate last ts point using all other points as train data.

    The model is reset before scoring to guarantee a fresh model training each time this
     method is called.

    """
    last_point = points[-1]
    validation_point = ValidationPoint(ts=last_point.ts, true_value=last_point.value)
    if last_point.ts < start_ts:
        return validation_point
    model.reset()
    predicted, error = model.score(points=points)
    get_logger().info(f"Finished validation for ts={points[-1].ts.isoformat()}.")
    return replace(validation_point, predicted=predicted, error=error)


def _forward_chaining_cross_validation(
    points: List[Point], model: Model, start_proportion: float, confidence: float
) -> ValidationTimeSeries:
    logger = get_logger()
    logger.info(f"Cross validation for model {model.__name__} started ...")
    start_ts = points[int(start_proportion * len(points))].ts
    logger.info(
        f"Validating {int((1 - start_proportion) * len(points))} timestamps from "
        f"{start_ts.isoformat()} to {points[-1].ts.isoformat()}."
    )
    validation_points = [
        _validate_last_ts(points=points[: i + 1], model=model, start_ts=start_ts)
        for i in range(len(points))
    ]
    mean_error = fmean([p.error for p in validation_points if p.error is not None])
    thresholds_proportion = _find_best_threshold(
        validation_points=validation_points, confidence=confidence
    )
    logger.info(
        f"Results: mean error = {mean_error}, "
        f"minimum threshold = {thresholds_proportion.threshold}, points "
        f"below threshold = {thresholds_proportion.below_threshold_proportion}"
    )
    logger.info(f"Cross validation for model {model.__name__} finished!")
    return ValidationTimeSeries(
        model_name=model.__name__,
        points=validation_points,
        mean_error=mean_error,
        threshold=thresholds_proportion.threshold,
        below_threshold_proportion=thresholds_proportion.below_threshold_proportion,
    )


@dataclass
class _ModelThreshold:
    model_name: str
    threshold: float


def _select_best_model(
    validation_time_series: List[ValidationTimeSeries],
    confidence: float,
    metric: Metric,
) -> _ModelThreshold:
    best_model = min(validation_time_series)
    if best_model.threshold == 1.0:
        raise OptimizationFailedError(
            f"Error while optimizing for metric {metric} - it was not possible to find "
            f"model and an error threshold below 1.0 (precision limit) to match given "
            f"confidence of {confidence}."
        )
    return _ModelThreshold(
        model_name=best_model.model_name, threshold=best_model.threshold
    )


def _is_time_series_constant(ts: TimeSeries) -> bool:
    return True if len({p.value for p in ts.points}) == 1 else False


def _optimize_time_series(
    ts: TimeSeries,
    confidence: float,
    model_factory: BaseModelFactory,
    start_proportion: float,
) -> MetricAnomalyOptimizationReport:
    logger = get_logger()
    logger.info(f"Optimizing for metric = {ts.metric} started...")
    if _is_time_series_constant(ts):
        logger.info("Time series is constant, using optimized model_factory...")
        model_factory = SimpleModelFactory()

    validation_time_series = [
        _forward_chaining_cross_validation(
            points=ts.points,
            model=model,
            start_proportion=start_proportion,
            confidence=confidence,
        )
        for model in model_factory.create_all_models()
    ]
    best_model_threshold = _select_best_model(
        validation_time_series, confidence, ts.metric
    )

    logger.info(f"Optimizing for metric = {ts.metric} finished!")
    return MetricAnomalyOptimizationReport(
        metric=ts.metric,
        best_model_name=best_model_threshold.model_name,
        threshold=best_model_threshold.threshold,
        validation_results=validation_time_series,
    )


def optimize(
    profiling_history: List[ProfilingReport],
    start_proportion: float = 0.5,
    confidence: float = 0.95,
    model_factory: BaseModelFactory = DefaultModelFactory(),
) -> DatasetAnomalyOptimizationReport:
    logger = get_logger()
    logger.info("📈️ Optimization started ...")
    last_profiling_report = profiling_history[-1]
    time_series = convert_to_timeseries(profiling_history)
    metric_anomaly_optimization_report = [
        _optimize_time_series(
            ts=ts,
            confidence=confidence,
            model_factory=model_factory,
            start_proportion=start_proportion,
        )
        for ts in time_series
    ]
    logger.info("📈 Optimization finished !")
    return DatasetAnomalyOptimizationReport(
        dataset=last_profiling_report.dataset,
        confidence=confidence,
        metric_anomaly_optimization_reports=metric_anomaly_optimization_report,
    )
