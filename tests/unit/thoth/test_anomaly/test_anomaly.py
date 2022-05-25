import datetime

import thoth.anomaly.base
import thoth.anomaly.optimization
from thoth import anomaly, profiler


class TestAnomalyScoring:
    def test_score(self, base_profiling_history):
        # arrange
        target_anomaly_scoring = anomaly.AnomalyScoring(
            dataset="my_dataset",
            ts=base_profiling_history[-1].ts,
            scores=[
                anomaly.Score(
                    metric=profiler.Metric(entity="Column", instance="f1", name="Mean"),
                    value=0.15517241379310331,
                    predicted=14.700000000000001,
                )
            ],
        )

        anomaly_config = thoth.anomaly.optimization.DatasetAnomalyOptimizationReport(
            dataset="my_dataset",
            confidence=0.95,
            metric_anomaly_optimization_reports=[
                thoth.anomaly.optimization.MetricAnomalyOptimizationReport(
                    metric=base_profiling_history[0].profiling_values[0].metric,
                    best_model_name="SimpleModel",
                    threshold=0.2,
                    validation_results=None,
                )
            ],
        )

        # act
        output_anomaly_scoring = anomaly.score(
            profiling_history=base_profiling_history,
            dataset_anomaly_config=anomaly_config,
        )

        # assert
        assert output_anomaly_scoring == target_anomaly_scoring

    def test__convert_to_timeseries(self, base_profiling_history, json_data):
        # arrange
        points = [
            anomaly.base.Point(
                ts=datetime.datetime.fromisoformat(record["ts"]), value=record["value"]
            )
            for record in json_data
        ]
        target_output_time_series = [
            thoth.anomaly.base.TimeSeries(
                metric=base_profiling_history[0].profiling_values[0].metric,
                points=points,
            )
        ]

        # act
        output_time_series = thoth.anomaly.base.convert_to_timeseries(
            history=base_profiling_history,
        )

        # assert
        assert output_time_series == target_output_time_series
