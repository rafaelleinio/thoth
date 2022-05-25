import datetime

import pytest

from thoth.anomaly.base import Point, TimeSeries
from thoth.anomaly.models import AutoSarimaModel, BaseModelFactory
from thoth.anomaly.optimization import (
    OptimizationFailedError,
    ValidationPoint,
    ValidationTimeSeries,
    _find_best_threshold,
    _optimize_time_series,
    _select_best_model,
    _ThresholdProportion,
)
from thoth.profiler import Metric


@pytest.mark.parametrize(
    ["input_validation_points", "confidence", "target_threshold"],
    [
        (
            [
                ValidationPoint(ts=datetime.datetime(2022, 1, 1), true_value=100),
                ValidationPoint(ts=datetime.datetime(2022, 1, 2), true_value=100),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 3),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 4),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 5),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 6),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 7),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 8),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 9),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 10),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 11),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 12),
                    true_value=100,
                    predicted=50,
                    error=0.5,
                ),
            ],
            0.9,
            _ThresholdProportion(threshold=0.1, below_threshold_proportion=0.9),
        ),
        (
            [
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 3),
                    true_value=100,
                    predicted=200,
                    error=1.0,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 4),
                    true_value=100,
                    predicted=200,
                    error=1.0,
                ),
                ValidationPoint(
                    ts=datetime.datetime(2022, 1, 5),
                    true_value=100,
                    predicted=90,
                    error=0.1,
                ),
            ],
            0.9,
            _ThresholdProportion(threshold=1.0, below_threshold_proportion=1.0),
        ),
    ],
)
def test__find_best_threshold(input_validation_points, confidence, target_threshold):
    # act
    output_threshold = _find_best_threshold(
        validation_points=input_validation_points, confidence=confidence
    )

    # assert
    assert output_threshold == target_threshold


def test__select_best_model():
    # arrange
    input_validation_time_series = [
        ValidationTimeSeries(
            model_name="BestModel",
            points=[],
            mean_error=0.01,
            threshold=0.01,
            below_threshold_proportion=0.99,
        ),
        ValidationTimeSeries(
            model_name="BadModel",
            points=[],
            mean_error=1.0,
            threshold=1.0,
            below_threshold_proportion=1.0,
        ),
    ]

    # act
    output = _select_best_model(
        input_validation_time_series,
        confidence=0.95,
        metric=Metric(entity="Column", instance="feature", name="CountDistinct"),
    )

    # assert
    assert output.model_name == "BestModel"


def test__select_best_model_exception():
    # arrange
    input_validation_time_series = [
        ValidationTimeSeries(
            model_name="BadModel",
            points=[],
            mean_error=1.0,
            threshold=1.0,
            below_threshold_proportion=1.0,
        )
    ]

    # act and assert
    with pytest.raises(OptimizationFailedError):
        _ = _select_best_model(
            input_validation_time_series,
            confidence=0.95,
            metric=Metric(entity="Column", instance="feature", name="CountDistinct"),
        )


def test__optimize_time_series_constant_flow():
    # arrange
    input_ts = TimeSeries(
        metric=None,  # not relevant for this test specifically
        points=[
            Point(ts=datetime.datetime(2022, 1, 1), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 2), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 3), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 4), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 5), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 6), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 7), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 8), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 9), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 10), value=15.0),
        ],
    )

    # act
    metric_anomaly_optimization_report = _optimize_time_series(
        ts=input_ts,
        confidence=0.95,
        model_factory=BaseModelFactory(
            models={AutoSarimaModel.__name__: AutoSarimaModel}
        ),
        start_proportion=0.5,
    )

    # assert
    assert metric_anomaly_optimization_report.best_model_name == "Simple"
    assert metric_anomaly_optimization_report.threshold == 0.01
