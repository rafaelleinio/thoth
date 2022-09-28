import datetime

import pytest

from thoth import repository
from thoth.dataset import Dataset
from thoth.profiler import Granularity, Metric, ProfilingReport, ProfilingValue
from thoth.repository import MetricsRepositoryError

DATASET = Dataset(
    uri="my-dataset",
    ts_column="ts",
    columns=["f1"],
    granularity=Granularity.DAY,
    metrics=[Metric(entity="Column", instance="f1", name="Mean")],
)


def test__validate_profiling_records_exception():
    # act and assert
    with pytest.raises(MetricsRepositoryError):
        repository._validate_profiling_records(
            dataset=DATASET,
            profiling_records=[
                ProfilingReport(
                    dataset_uri="my-dataset",
                    ts=datetime.datetime.utcnow(),
                    granularity=Granularity.DAY,
                    profiling_values=[
                        ProfilingValue(
                            metric=Metric(
                                entity="Column", instance="f1", name="OtherMean"
                            ),
                            value=123,
                        )
                    ],
                )
            ],
        )

    with pytest.raises(MetricsRepositoryError):
        repository._validate_profiling_records(
            dataset=DATASET,
            profiling_records=[
                ProfilingReport(
                    dataset_uri="my-dataset2",
                    ts=datetime.datetime.utcnow(),
                    granularity=Granularity.DAY,
                    profiling_values=[
                        ProfilingValue(
                            metric=Metric(entity="Column", instance="f1", name="Mean"),
                            value=123,
                        )
                    ],
                )
            ],
        )

    with pytest.raises(MetricsRepositoryError):
        repository._validate_profiling_records(
            dataset=DATASET,
            profiling_records=[
                ProfilingReport(
                    dataset_uri="my-dataset",
                    ts=datetime.datetime.utcnow(),
                    granularity="MONTH",
                    profiling_values=[
                        ProfilingValue(
                            metric=Metric(entity="Column", instance="f1", name="Mean"),
                            value=123,
                        )
                    ],
                )
            ],
        )


def test_add_profiling_exception(session):
    # arrange
    repo = repository.SqlRepository(session=session)
    repo.add_dataset(dataset=DATASET)

    # act and assert
    with pytest.raises(MetricsRepositoryError):
        repo.add_profiling(dataset_uri="wrong-uri", records=[])
