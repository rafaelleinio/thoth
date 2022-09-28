from __future__ import annotations

import datetime
from dataclasses import dataclass
from statistics import fmean
from typing import Any, List, Optional, Set

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import Column as SqlAlchemyColumn
from sqlmodel import Field, SQLModel

from thoth.anomaly.base import _convert_to_timeseries, _Point, _TimeSeries
from thoth.anomaly.models import (
    BaseModelFactory,
    DefaultModelFactory,
    Model,
    SimpleModelFactory,
)
from thoth.profiler import Metric, ProfilingReport
from thoth.util.custom_typing import pydantic_column_type


class OptimizationFailedError(Exception):
    """Exception defining a base error for the optimization flow."""

    pass


class ValidationPoint(BaseModel):
    """Model defining the target ts and the validation error."""

    ts: datetime.datetime
    true_value: float
    predicted: Optional[float] = None
    error: Optional[float] = None


class ValidationTimeSeries(BaseModel):
    """Model defining the full validation results for a given series and model."""

    model_name: str
    points: List[ValidationPoint]
    mean_error: float
    threshold: float
    below_threshold_proportion: float

    def __lt__(self, other: ValidationTimeSeries) -> bool:
        return (self.threshold, self.mean_error) < (other.threshold, self.mean_error)


class MetricOptimization(BaseModel):
    """Holds the optimization results for a specific metric.

    Attributes:
        metric: metric identification.
        best_model_name: name of the best performant model for this metric.
        threshold: anomaly threshold automatically found in the optimization flow.
        validation_results: all models validation results for the metrics series.

    """

    metric: Metric
    best_model_name: str
    threshold: float
    validation_results: List[ValidationTimeSeries]
    window: Optional[Any] = None


class AnomalyOptimization(SQLModel, table=True):
    """Optimization results for a given dataset.

    Attributes:
        dataset_uri: dataset URI.
        confidence: target confidence for the optimization.

    """

    dataset_uri: str = Field(primary_key=True)
    confidence: float
    metric_optimizations: List[MetricOptimization] = Field(
        sa_column=SqlAlchemyColumn(pydantic_column_type(List[MetricOptimization]))
    )

    def get_metric_optimization(self, metric: Metric) -> MetricOptimization:
        """Get one specific metric optimization from the metric_optimizations att."""
        return [
            metric_config
            for metric_config in self.metric_optimizations
            if metric_config.metric == metric
        ].pop(0)

    def get_metrics(self) -> Set[Metric]:
        """Get all metric identifications from the metric_optimizations att."""
        return set(
            profiling_value.metric for profiling_value in self.metric_optimizations
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
    points: List[_Point], model: Model, start_ts: datetime.datetime
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
    logger.debug(f"Finished validation for ts={points[-1].ts.isoformat()}.")
    return ValidationPoint(
        **validation_point.dict(exclude_unset=True), predicted=predicted, error=error
    )


def _forward_chaining_cross_validation(
    points: List[_Point], model: Model, start_proportion: float, confidence: float
) -> ValidationTimeSeries:
    logger.debug(f"Cross validation for model {type(model).__name__} started ...")
    start_ts = points[int(start_proportion * len(points))].ts
    logger.debug(
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
    logger.debug(
        f"Results: mean error = {mean_error}, "
        f"minimum threshold = {thresholds_proportion.threshold}, points "
        f"below threshold = {thresholds_proportion.below_threshold_proportion}"
    )
    logger.debug(f"Cross validation for model {type(model).__name__} finished!")
    return ValidationTimeSeries(
        model_name=type(model).__name__,
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


def _is_time_series_constant(ts: _TimeSeries) -> bool:
    return True if len({p.value for p in ts.points}) == 1 else False


def _optimize_time_series(
    ts: _TimeSeries,
    confidence: float,
    model_factory: BaseModelFactory,
    start_proportion: float,
) -> MetricOptimization:
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
    return MetricOptimization(
        metric=ts.metric,
        best_model_name=best_model_threshold.model_name,
        threshold=best_model_threshold.threshold,
        validation_results=validation_time_series,
    )


def optimize(
    profiling_history: List[ProfilingReport],
    start_proportion: Optional[float] = None,
    confidence: Optional[float] = None,
    model_factory: Optional[BaseModelFactory] = None,
) -> AnomalyOptimization:
    """Optimize the anomaly strategy for a given dataset using its profiling history."""
    logger.info("📈️ Optimization started ...")
    confidence = confidence or 0.95
    last_profiling_report = profiling_history[-1]
    time_series = _convert_to_timeseries(profiling_history)
    metric_anomaly_optimization_report = [
        _optimize_time_series(
            ts=ts,
            confidence=confidence,
            model_factory=model_factory or DefaultModelFactory(),
            start_proportion=start_proportion or 0.5,
        )
        for ts in time_series
    ]
    logger.info("📈 Optimization finished !")
    return AnomalyOptimization(
        dataset_uri=last_profiling_report.dataset_uri,
        confidence=confidence,
        metric_optimizations=metric_anomaly_optimization_report,
    )
