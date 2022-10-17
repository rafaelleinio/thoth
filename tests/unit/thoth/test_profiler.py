import datetime
from unittest import mock

import pytest
from pydeequ import analyzers
from pyspark.sql.types import (
    DoubleType,
    LongType,
    StringType,
    StructField,
    StructType,
    TimestampType,
)

from thoth.profiler import (
    DefaultProfilingBuilder,
    Granularity,
    Metric,
    ProfilingBuilder,
    ProfilingReport,
    ProfilingValue,
    SimpleProfilingBuilder,
    profile,
)
from thoth.testing.dataframe import create_df_from_collection


class TestProfilingBuilder:
    def test_build_analyzers(self):
        # arrange
        profiling_builder = DefaultProfilingBuilder()
        input_fields = [
            StructField("f1", LongType()),
            StructField("f2", StringType()),
        ]
        target_analyzers = [
            analyzers.Mean("f1"),
            analyzers.StandardDeviation("f1"),
            analyzers.Maximum("f1"),
            analyzers.Minimum("f1"),
            analyzers.ApproxQuantiles("f1", quantiles=[0.25, 0.5, 0.75]),
            analyzers.Completeness("f1"),
            analyzers.Completeness("f2"),
            analyzers.MinLength("f2"),
            analyzers.MaxLength("f2"),
            analyzers.CountDistinct("f2"),
            analyzers.Size(),
        ]

        # act
        output_analyzers = profiling_builder.build_analyzers(
            structured_fields=input_fields
        )

        # assert
        assert [type(a) for a in output_analyzers] == [
            type(a) for a in target_analyzers
        ]
        assert [a.__dict__ for a in output_analyzers] == [
            a.__dict__ for a in target_analyzers
        ]

    def test_value_error(self):
        with pytest.raises(ValueError):
            ProfilingBuilder()

    def test_simple_profiling_builder(self):
        assert isinstance(SimpleProfilingBuilder(), ProfilingBuilder)


def test_profile(spark_context, spark_session):
    # arrange
    profiling_builder = ProfilingBuilder(
        analyzers=[
            analyzers.Size(),
            analyzers.Minimum("f1"),
            analyzers.Maximum("f2"),
        ]
    )
    df = create_df_from_collection(
        data=[
            {"f1": 1, "f2": 1.0, "ts": "2022-04-01"},
            {"f1": 2, "f2": 2.0, "ts": "2022-04-01"},
            {"f1": 3, "f2": 3.0, "ts": "2022-04-01"},
            {"f1": 10, "f2": 10.0, "ts": "2022-04-02"},
            {"f1": 11, "f2": 11.0, "ts": "2022-04-02"},
            {"f1": 12, "f2": 12.0, "ts": "2022-04-02"},
        ],
        spark_context=spark_context,
        spark_session=spark_session,
        schema=StructType(
            [
                StructField("f1", LongType()),
                StructField("f2", DoubleType()),
                StructField("ts", TimestampType()),
            ]
        ),
    )

    # act
    with mock.patch("uuid.uuid4", side_effect=lambda: "uuid"):
        profiling_report = profile(
            profiling_builder=profiling_builder,
            dataset_uri="ds",
            df=df,
            ts_column="ts",
            spark=spark_session,
        )

    # assert
    assert profiling_report == [
        ProfilingReport(
            dataset_uri="ds",
            ts=datetime.datetime(2022, 4, 1),
            granularity=Granularity.DAY,
            profiling_values=[
                ProfilingValue(
                    metric=Metric(entity="Dataset", instance="*", name="Size"),
                    value=3.0,
                ),
                ProfilingValue(
                    metric=Metric(entity="Column", instance="f1", name="Minimum"),
                    value=1.0,
                ),
                ProfilingValue(
                    metric=Metric(entity="Column", instance="f2", name="Maximum"),
                    value=3.0,
                ),
            ],
        ),
        ProfilingReport(
            dataset_uri="ds",
            ts=datetime.datetime(2022, 4, 2),
            granularity=Granularity.DAY,
            profiling_values=[
                ProfilingValue(
                    metric=Metric(entity="Dataset", instance="*", name="Size"),
                    value=3.0,
                ),
                ProfilingValue(
                    metric=Metric(entity="Column", instance="f1", name="Minimum"),
                    value=10.0,
                ),
                ProfilingValue(
                    metric=Metric(entity="Column", instance="f2", name="Maximum"),
                    value=12.0,
                ),
            ],
        ),
    ]


class TestProfilingReport:
    def test_get_metrics(self, base_profiling_history):
        # act
        output = base_profiling_history[0].get_metrics()

        # assert
        assert output == {Metric(entity="Column", instance="f1", name="Mean")}
