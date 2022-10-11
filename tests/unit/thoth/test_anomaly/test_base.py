from thoth import base
from thoth.profiler import Metric


def test_time_series_sortin():
    # arrange
    ts_collection = [
        base.TimeSeries(
            metric=Metric(entity="Column", instance="f2", name="Mean"),
            points=[],
        ),
        base.TimeSeries(
            metric=Metric(entity="Column", instance="f3", name="Mean"),
            points=[],
        ),
        base.TimeSeries(
            metric=Metric(entity="Column", instance="f1", name="Mean"),
            points=[],
        ),
    ]
    target = [
        base.TimeSeries(
            metric=Metric(entity="Column", instance="f1", name="Mean"),
            points=[],
        ),
        base.TimeSeries(
            metric=Metric(entity="Column", instance="f2", name="Mean"),
            points=[],
        ),
        base.TimeSeries(
            metric=Metric(entity="Column", instance="f3", name="Mean"),
            points=[],
        ),
    ]

    # act
    output = sorted(ts_collection)

    # assert
    assert output == target
