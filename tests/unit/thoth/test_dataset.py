from thoth.dataset import Dataset
from thoth.profiler import Metric


def test_get_instances():
    # arrange
    dataset = Dataset(
        uri="my-dataset",
        ts_column="ts",
        columns=["f1", "f2"],
        metrics=[
            Metric(entity="Column", instance="f1", name="Mean"),
            Metric(entity="Column", instance="f1", name="Max"),
            Metric(entity="Column", instance="f2", name="Max"),
            Metric(entity="Column", instance="f2", name="Mean"),
        ],
    )

    # act
    output = dataset.get_instances()

    # assert
    assert output == {"f1", "f2"}
