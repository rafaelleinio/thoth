import abc
import dataclasses
import datetime
from typing import List, Optional, Sequence

from loguru import logger

from thoth import anomaly, profiler
from thoth.util.dashboard import SCORING_VIEW, build_dashboard_link


@dataclasses.dataclass
class AnomalousScore:
    """Holds an anomalous metric score and the reference optimized threshold."""

    metric: profiler.Metric
    score: float
    threshold: float


class NotificationHandler(abc.ABC):
    """Abstract base class for notification handlers.

    Notification handlers implement the warning logic to notify data users about
    anomalies on new dataset batches.

    """

    @abc.abstractmethod
    def _notify(
        self,
        dataset_uri: str,
        ts: datetime.datetime,
        anomalous_scores: List[AnomalousScore],
        dashboard_link: Optional[str] = None,
    ) -> None:
        """Child class must implement warn logic."""

    def notify(
        self,
        dataset_uri: str,
        ts: datetime.datetime,
        anomalous_scores: List[AnomalousScore],
    ) -> None:
        """Trigger a notification for a collection of anomalous scores.

        Args:
            dataset_uri: dataset URI.
            ts: timestamp reference for the scoring.
            anomalous_scores: collection of metrics with anomalous scores.

        """
        self._notify(
            dataset_uri=dataset_uri,
            ts=ts,
            anomalous_scores=anomalous_scores,
            dashboard_link=build_dashboard_link(
                dataset_uri=dataset_uri,
                instances=list(set(a.metric.instance for a in anomalous_scores)),
                view=SCORING_VIEW,
            ),
        )


class LogHandler(NotificationHandler):
    """Sim notification handler that outputs the anomalous scores as a log error."""

    def _notify(
        self,
        dataset_uri: str,
        ts: datetime.datetime,
        anomalous_scores: List[AnomalousScore],
        dashboard_link: Optional[str] = None,
    ) -> None:
        logger.error(
            f"Anomaly detected for ts={ts} on dataset_uri={dataset_uri}!\n"
            f"The following metrics have scores above the defined threshold by the "
            f"optimization: {anomalous_scores}. \n"
            f"Please check the dataset dashboard for more information: "
            f"{dashboard_link}"
        )


def assess_quality(
    anomaly_optimization: anomaly.AnomalyOptimization,
    anomaly_scoring: anomaly.AnomalyScoring,
    notification_handlers: Optional[Sequence[NotificationHandler]] = None,
) -> bool:
    """Perform the quality assessment for a target dataset scoring timestamp."""
    logger.info(f"🔍️ Assessing quality for ts={anomaly_scoring.ts} ...")
    metrics = [
        AnomalousScore(
            metric=score.metric,
            score=score.value,
            threshold=anomaly_optimization.get_metric_optimization(
                metric=score.metric
            ).threshold,
        )
        for score in anomaly_scoring.scores
    ]
    anomalous_scores = [metric for metric in metrics if metric.score > metric.threshold]
    if anomalous_scores:
        logger.error("🚨 ️Anomaly detected, notifying handlers...")
        for handler in notification_handlers or [LogHandler()]:
            handler.notify(
                dataset_uri=anomaly_optimization.dataset_uri,
                ts=anomaly_scoring.ts,
                anomalous_scores=anomalous_scores,
            )
        logger.info("🔍️ Quality assessment finished, handlers notified!")
        return False
    logger.info("🔍️ Quality assessment finished, everything good! 🍰 ✨")
    return True
