import datetime
from typing import List, Optional

from thoth import anomaly, profiler, quality


class MockNotificationHandler(quality.NotificationHandler):
    def __init__(self):
        self.notifications = 0

    def _notify(
        self,
        dataset: str,
        ts: datetime.datetime,
        anomalous_metrics: List[quality.AnomalousMetric],
        dashboard_link: Optional[str] = None,
    ):
        self.notifications += 1


def test_assess_quality():
    # arrange
    notification_handler = MockNotificationHandler()
    metric = profiler.Metric(entity="Column", instance="f1", name="Mean")
    anomaly_optimization = anomaly.optimization.DatasetAnomalyOptimizationReport(
        dataset="my-dataset",
        confidence=0.95,
        metric_anomaly_optimization_reports=[
            anomaly.optimization.MetricAnomalyOptimizationReport(
                metric=metric,
                best_model_name="best-model",
                threshold=0.15,
                validation_results=[],  # field not needed for this test
            ),
        ],
    )
    anomaly_scoring = anomaly.AnomalyScoring(
        dataset="best-model",
        ts=datetime.datetime.utcnow(),
        scores=[
            anomaly.Score(
                metric=metric,
                value=0.16,
                predicted=84,
            )
        ],
    )

    # act
    quality.assess_quality(
        anomaly_optimization=anomaly_optimization,
        anomaly_scoring=anomaly_scoring,
        notification_handlers=[notification_handler, notification_handler],
    )

    # assert
    assert notification_handler.notifications == 2


class TestLogHandler:
    def test__notify(self, caplog):
        # arrange
        handler = quality.LogHandler()
        ts = datetime.datetime.utcnow()

        # act
        handler._notify(
            dataset="my-dataset",
            ts=ts,
            anomalous_metrics=[],  # field not needed for this test
        )

        # assert
        assert f"Anomaly detected for ts={ts} on dataset=my-dataset!" in caplog.text
