import abc
import dataclasses
import datetime
from typing import List, Optional, Sequence

from loguru import logger

from thoth import anomaly, profiler


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
    ) -> None:
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
        logger.error(
            f"Anomaly detected for ts={ts} on dataset={dataset}!\n"
            f"The following metrics have scores above the defined threshold by the "
            f"optimization: {anomalous_metrics}. \n"
            f"Please check the dataset dashboard for more information: "
            f"{dashboard_link}"
        )


def assess_quality(
    anomaly_optimization: anomaly.optimization.AnomalyOptimization,
    anomaly_scoring: anomaly.AnomalyScoring,
    notification_handlers: Optional[Sequence[NotificationHandler]] = None,
) -> bool:
    logger.info(f"🔍️ Assessing quality for ts={anomaly_scoring.ts} ...")
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
        logger.error("Anomaly detected, notifying handlers...")
        for handler in notification_handlers or [LogHandler()]:
            handler.notify(
                dataset=anomaly_optimization.dataset,
                ts=anomaly_scoring.ts,
                anomalous_metrics=anomalous_metrics,
            )
        logger.info("🔍️ Quality assessment finished, handlers notified!")
        return False
    logger.info("🔍️ Quality assessment finished, everything good! 🍰 ✨")
    return True
