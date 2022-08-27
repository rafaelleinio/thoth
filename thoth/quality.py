import abc
import dataclasses
import datetime
from typing import List, Optional

from thoth import anomaly, logging, profiler


@dataclasses.dataclass
class AnomalousMetric:
    metric: profiler.Metric
    score: float
    threshold: float


def _build_dashboard_link(dataset: str) -> str:
    return f"PLACEHOLDER/{dataset}"


class NotificationHandler(abc.ABC):
    @abc.abstractmethod
    def _notify(
        self,
        dataset: str,
        ts: datetime.datetime,
        anomalous_metrics: List[AnomalousMetric],
        dashboard_link: Optional[str] = None,
    ):
        """Child class must implement warn logic."""

    def notify(
        self,
        dataset: str,
        ts: datetime.datetime,
        anomalous_metrics: List[AnomalousMetric],
    ) -> None:
        self._notify(
            dataset=dataset,
            ts=ts,
            anomalous_metrics=anomalous_metrics,
            dashboard_link=_build_dashboard_link(dataset=dataset),
        )


class LogHandler(NotificationHandler):
    def _notify(
        self,
        dataset: str,
        ts: datetime.datetime,
        anomalous_metrics: List[AnomalousMetric],
        dashboard_link: Optional[str] = None,
    ) -> None:
        logging.get_logger().error(
            msg=f"Anomaly detected for ts={ts} on dataset={dataset}!\n"
            f"The following metrics have scores above the defined threshold by the "
            f"optimization: {anomalous_metrics}. \n"
            f"Please check the dataset dashboard for more information: "
            f"{dashboard_link}"
        )


def assess_quality(
    anomaly_optimization: anomaly.optimization.DatasetAnomalyOptimizationReport,
    anomaly_scoring: anomaly.AnomalyScoring,
    notification_handlers: List[NotificationHandler],
) -> None:
    metrics = [
        AnomalousMetric(
            metric=score.metric,
            score=score.value,
            threshold=anomaly_optimization.get_metric_optimization(
                metric=score.metric
            ).threshold,
        )
        for score in anomaly_scoring.scores
    ]
    anomalous_metrics = [
        metric for metric in metrics if metric.score > metric.threshold
    ]
    if anomalous_metrics:
        for handler in notification_handlers:
            handler.notify(
                dataset=anomaly_optimization.dataset,
                ts=anomaly_scoring.ts,
                anomalous_metrics=anomalous_metrics,
            )
