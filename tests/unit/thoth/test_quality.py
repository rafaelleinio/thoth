import datetime
from typing import List, Optional

from thoth import anomaly, profiler, quality


class MockNotificationHandler(quality.NotificationHandler):
    def __init__(self):
        self.notifications = 0

    def _notify(
        self,
        dataset_uri: str,
        ts: datetime.datetime,
        anomalous_scores: List[quality.AnomalousScore],
        dashboard_link: Optional[str] = None,
    ):
        self.notifications += 1


def test_assess_quality_error():
    # arrange
    notification_handler = MockNotificationHandler()
    metric = profiler.Metric(entity="Column", instance="f1", name="Mean")
    anomaly_optimization = anomaly.AnomalyOptimization(
        dataset_uri="my-dataset",
        confidence=0.95,
        metric_optimizations=[
            anomaly.MetricOptimization(
                metric=metric,
                best_model_name="best-model",
                threshold=0.15,
                validation_results=[],  # field not needed for this test
            ),
        ],
    )
    anomaly_scoring = anomaly.AnomalyScoring(
        dataset_uri="best-model",
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
    success = quality.assess_quality(
        anomaly_optimization=anomaly_optimization,
        anomaly_scoring=anomaly_scoring,
        notification_handlers=[notification_handler, notification_handler],
    )

    # assert
    assert notification_handler.notifications == 2
    assert success is False


def test_assess_quality_success():
    # arrange
    metric = profiler.Metric(entity="Column", instance="f1", name="Mean")
    anomaly_optimization = anomaly.AnomalyOptimization(
        dataset_uri="my-dataset",
        confidence=0.95,
        metric_optimizations=[
            anomaly.MetricOptimization(
                metric=metric,
                best_model_name="best-model",
                threshold=0.15,
                validation_results=[],  # field not needed for this test
            ),
        ],
    )
    anomaly_scoring = anomaly.AnomalyScoring(
        dataset_uri="best-model",
        ts=datetime.datetime.utcnow(),
        scores=[
            anomaly.Score(
                metric=metric,
                value=0.14,
                predicted=84,
            )
        ],
    )

    # act
    success = quality.assess_quality(
        anomaly_optimization=anomaly_optimization,
        anomaly_scoring=anomaly_scoring,
    )

    # assert
    assert success is True


class TestLogHandler:
    def test__notify(self, caplog):
        # arrange
        handler = quality.LogHandler()
        ts = datetime.datetime.utcnow()

        # act
        handler._notify(
            dataset_uri="my-dataset",
            ts=ts,
            anomalous_scores=[],  # field not needed for this test
        )

        # assert
        assert f"Anomaly detected for ts={ts} on dataset_uri=my-dataset!" in caplog.text
