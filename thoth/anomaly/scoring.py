from __future__ import annotations

import datetime
from hashlib import sha1
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import Column as SqlAlchemyColumn
from sqlmodel import Field, SQLModel

from thoth import anomaly, base, profiler, util


class Score(BaseModel):
    """Holds the score and predicted value for a given metric."""

    metric: profiler.Metric
    value: float
    predicted: float


class AnomalyScoring(SQLModel, table=True):
    """Represents a data quality scoring event for a given dataset and timestamp.

    This model holds scores for all the profiling metrics found in the dataset, but just
    for one specific timestamp.

    """

    id_: str = Field(default=None, primary_key=True)
    dataset_uri: str
    ts: datetime.datetime
    scores: List[Score] = Field(
        sa_column=SqlAlchemyColumn(util.custom_typing.pydantic_column_type(List[Score]))
    )

    @classmethod
    def _build_id(cls, dataset_uri: str, ts: datetime.datetime) -> str:
        return sha1(f"{dataset_uri}{ts.isoformat()}".encode("utf-8")).hexdigest()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.id_ = self._build_id(self.dataset_uri, self.ts)

    def get_metric_score(self, metric: profiler.Metric) -> Score:
        """Get the score for a given metric."""
        return [score_ for score_ in self.scores if score_.metric == metric].pop(0)

    def __lt__(self, other: AnomalyScoring) -> bool:
        return self.ts < other.ts


def _score_model(
    ts: base.TimeSeries,
    metric_optimization: anomaly.MetricOptimization,
    model_factory: anomaly.BaseModelFactory,
) -> Score:
    logger.info(
        f"Scoring for metric={metric_optimization.metric} "
        f"with model={metric_optimization.best_model_name} started..."
    )
    model = model_factory.create_model(name=metric_optimization.best_model_name)
    predicted, error = model.score(points=ts.points)
    score_value = Score(
        metric=metric_optimization.metric, value=error, predicted=predicted
    )
    logger.info(f"Metric score done! Value={score_value.value}")
    return score_value


def score(
    profiling_history: List[profiler.ProfilingReport],
    optimization: anomaly.AnomalyOptimization,
    model_factory: Optional[anomaly.BaseModelFactory] = None,
) -> AnomalyScoring:
    """Calculate the anomaly score for a target dataset timestamp batch."""
    logger.info("💯 Scoring started...")
    last_n_profiling_history = anomaly.get_last_n(
        profiling_history=profiling_history, last_n=optimization.last_n
    )
    last_profiling_report = last_n_profiling_history[-1]
    time_series = base.convert_to_timeseries(last_n_profiling_history)

    scores = [
        _score_model(
            ts=ts,
            metric_optimization=optimization.get_metric_optimization(ts.metric),
            model_factory=model_factory or anomaly.DefaultModelFactory(),
        )
        for ts in time_series
    ]
    anomaly_scoring = AnomalyScoring(
        dataset_uri=last_profiling_report.dataset_uri,
        ts=last_profiling_report.ts,
        scores=scores,
    )
    logger.info("💯 Scoring finished!")
    return anomaly_scoring
