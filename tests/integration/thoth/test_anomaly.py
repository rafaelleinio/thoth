from thoth.anomaly import optimization
from thoth.anomaly.optimization import (
    DatasetAnomalyOptimizationReport,
    MetricAnomalyOptimizationReport,
)
from thoth.profiler import Metric


def test_optimize(base_profiling_history):
    """Test optimize service from optimization module.

    Running for all base models (default factory). Using 364/365 points as start
    proportion to speed up the test.

    """
    # arrange
    confidence = 0.95
    target = DatasetAnomalyOptimizationReport(
        dataset="my_dataset",
        confidence=confidence,
        metric_anomaly_optimization_reports=[
            MetricAnomalyOptimizationReport(
                metric=Metric(entity="Column", instance="f1", name="Mean"),
                best_model_name="Simple",
                threshold=0.16,
                validation_results=[],  # not relevant for this test specifically
            )
        ],
    )

    # act
    output = optimization.optimize(
        profiling_history=base_profiling_history,
        start_proportion=0.999,
        confidence=confidence,
    )
    output.metric_anomaly_optimization_reports[0].validation_results = []

    # assert
    assert target == output
