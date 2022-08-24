import datetime
from dataclasses import dataclass
from typing import List

from thoth.anomaly.base import TimeSeries, convert_to_timeseries
from thoth.anomaly.models import BaseModelFactory, DefaultModelFactory
from thoth.anomaly.optimization import (
    DatasetAnomalyOptimizationReport,
    MetricAnomalyOptimizationReport,
)
from thoth.profiler import Metric, ProfilingReport


@dataclass
class Score:
    metric: Metric
    value: float
    predicted: float


@dataclass
class AnomalyScoring:
    dataset: str
    ts: datetime.datetime
    scores: List[Score]


def _score_model(
    time_series: TimeSeries,
    metric_config: MetricAnomalyOptimizationReport,
    model_factory: BaseModelFactory,
) -> Score:
    model = model_factory.create_model(name=metric_config.best_model_name)
    predicted, error = model.score(points=time_series.points)
    return Score(metric=metric_config.metric, value=error, predicted=predicted)


def score(
    profiling_history: List[ProfilingReport],
    dataset_anomaly_config: DatasetAnomalyOptimizationReport,
    model_factory: BaseModelFactory = DefaultModelFactory(),
) -> AnomalyScoring:
    last_profiling_report = profiling_history[-1]
    time_series = convert_to_timeseries(profiling_history)
    scores = [
        _score_model(
            ts,
            dataset_anomaly_config.get_metric_optimization(ts.metric),
            model_factory,
        )
        for ts in time_series
    ]
    return AnomalyScoring(
        dataset=last_profiling_report.dataset,
        ts=last_profiling_report.ts,
        scores=scores,
    )
