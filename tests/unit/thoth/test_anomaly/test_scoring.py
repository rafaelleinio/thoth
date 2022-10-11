import datetime

import thoth.anomaly.optimization
import thoth.base
from thoth import anomaly, profiler


class TestAnomalyScoring:
    def test_score(self, base_profiling_history):
        # arrange
        target_anomaly_scoring = anomaly.AnomalyScoring(
            dataset_uri="my_dataset",
            ts=base_profiling_history[-1].ts,
            scores=[
                anomaly.Score(
                    metric=profiler.Metric(entity="Column", instance="f1", name="Mean"),
                    value=0.15517241379310331,
                    predicted=14.700000000000001,
                )
            ],
        )

        anomaly_config = thoth.anomaly.AnomalyOptimization(
            dataset_uri="my_dataset",
            confidence=0.95,
            metric_optimizations=[
                thoth.anomaly.MetricOptimization(
                    metric=base_profiling_history[0].profiling_values[0].metric,
                    best_model_name="SimpleModel",
                    threshold=0.2,
                    validation_results=[],
                )
            ],
        )

        # act
        output_anomaly_scoring = anomaly.score(
            profiling_history=base_profiling_history,
            optimization=anomaly_config,
        )

        # assert
        assert output_anomaly_scoring == target_anomaly_scoring

    def test__convert_to_timeseries(self, base_profiling_history, json_data):
        # arrange
        points = [
            thoth.base.Point(
                ts=datetime.datetime.fromisoformat(record["ts"]), value=record["value"]
            )
            for record in json_data
        ]
        target_output_time_series = [
            thoth.base.TimeSeries(
                metric=base_profiling_history[0].profiling_values[0].metric,
                points=points,
            )
        ]

        # act
        output_time_series = thoth.base.convert_to_timeseries(
            profiling=base_profiling_history,
        )

        # assert
        assert output_time_series == target_output_time_series
