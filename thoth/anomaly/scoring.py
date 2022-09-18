import datetime
from hashlib import sha1
from typing import Any, List, Optional

from loguru import logger
from pydantic import BaseModel
from sqlalchemy import Column as SqlAlchemyColumn
from sqlmodel import Field, SQLModel

from thoth.anomaly.base import TimeSeries, convert_to_timeseries
from thoth.anomaly.models import BaseModelFactory, DefaultModelFactory
from thoth.anomaly.optimization import AnomalyOptimization, MetricOptimization
from thoth.profiler import Metric, ProfilingReport
from thoth.util.custom_typing import pydantic_column_type


class Score(BaseModel):
    metric: Metric
    value: float
    predicted: float


class AnomalyScoring(SQLModel, table=True):
    id: str = Field(default=None, primary_key=True)
    dataset: str
    ts: datetime.datetime
    scores: List[Score] = Field(
        sa_column=SqlAlchemyColumn(pydantic_column_type(List[Score]))
    )

    @classmethod
    def build_id(cls, dataset: str, ts: datetime.datetime) -> str:
        return sha1(f"{dataset}{ts.isoformat()}".encode("utf-8")).hexdigest()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.id = self.build_id(self.dataset, self.ts)


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
        dataset=last_profiling_report.dataset,
        ts=last_profiling_report.ts,
        scores=scores,
    )
    logger.info("💯 Scoring finished!")
    return anomaly_scoring
