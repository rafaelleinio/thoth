import datetime
from hashlib import sha1
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import Column as SqlAlchemyColumn
from sqlmodel import Field, SQLModel

from thoth.anomaly.models import BaseModelFactory, DefaultModelFactory
from thoth.anomaly.optimization import AnomalyOptimization, MetricOptimization
from thoth.base import TimeSeries, convert_to_timeseries
from thoth.profiler import Metric, ProfilingReport
from thoth.util.custom_typing import pydantic_column_type


class Score(BaseModel):
    """Holds the score and predicted value for a given metric."""

    metric: Metric
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
        sa_column=SqlAlchemyColumn(pydantic_column_type(List[Score]))
    )

    @classmethod
    def _build_id(cls, dataset_uri: str, ts: datetime.datetime) -> str:
        return sha1(f"{dataset_uri}{ts.isoformat()}".encode("utf-8")).hexdigest()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.id_ = self._build_id(self.dataset_uri, self.ts)

    def get_metric_score(self, metric: Metric) -> Score:
        """Get the score for a given metric."""
        return [score_ for score_ in self.scores if score_.metric == metric].pop(0)


def _score_model(
    time_series: TimeSeries,
    metric_optimization: MetricOptimization,
    model_factory: BaseModelFactory,
) -> Score:
    logger.info(
        f"Scoring for metric={metric_optimization.metric} "
        f"with model={metric_optimization.best_model_name} started..."
    )
    model = model_factory.create_model(name=metric_optimization.best_model_name)
    predicted, error = model.score(points=time_series.points)
    score_value = Score(
        metric=metric_optimization.metric, value=error, predicted=predicted
    )
    logger.info(f"Metric score done! Value={score_value.value}")
    return score_value


def score(
    profiling_history: List[ProfilingReport],
    optimization: AnomalyOptimization,
    model_factory: Optional[BaseModelFactory] = None,
) -> AnomalyScoring:
    """Calculate the anomaly score for a target dataset timestamp batch."""
    logger.info("💯 Scoring started...")
    last_profiling_report = profiling_history[-1]
    metrics_ts = convert_to_timeseries(profiling_history)
    scores = [
        _score_model(
            time_series=metric_ts,
            metric_optimization=optimization.get_metric_optimization(metric_ts.metric),
            model_factory=model_factory or DefaultModelFactory(),
        )
        for metric_ts in metrics_ts
    ]
    anomaly_scoring = AnomalyScoring(
        dataset_uri=last_profiling_report.dataset_uri,
        ts=last_profiling_report.ts,
        scores=scores,
    )
    logger.info("💯 Scoring finished!")
    return anomaly_scoring
