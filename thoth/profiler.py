from __future__ import annotations

import datetime
import functools
from dataclasses import dataclass
from hashlib import sha1
from typing import Any, List, Optional, Set, Type, Union

from loguru import logger
from pydantic import BaseModel
from pydeequ.analyzers import (
    AnalysisRunner,
    AnalyzerContext,
    ApproxQuantiles,
    Completeness,
    CountDistinct,
    Maximum,
    MaxLength,
    Mean,
    Minimum,
    MinLength,
    Size,
    StandardDeviation,
    _AnalyzerObject,
)
from pyspark.sql import Column as SparkColumn
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, NumericType, StringType, StructField
from sqlalchemy import Column as SqlAlchemyColumn
from sqlmodel import Field, SQLModel

from thoth.util.custom_typing import pydantic_column_type


@dataclass
class Type2Analyzers:
    """Mapping between a Spark data type and a collection profiling metrics to apply."""

    type: Type[DataType]
    analyzers: List[Type[_AnalyzerObject]]


class ProfilingBuilder:
    """Build a set of analyzers to be used in a profiling pipeline."""

    def __init__(
        self,
        type_mappings: Optional[List[Type2Analyzers]] = None,
        analyzers: Optional[List[_AnalyzerObject]] = None,
    ):
        self.type_mappings = type_mappings or []
        self.analyzers = analyzers or []
        if not (type_mappings or analyzers):
            raise ValueError("At least type_mappings or analyzers must be used.")

    def build_analyzers(
        self, structured_fields: List[StructField]
    ) -> List[_AnalyzerObject]:
        """Build a set of analyzers.

        Args:
            structured_fields: spark data type fields.

        Returns:
            set of created analyzers.

        """
        analyzers = []
        for struct_field in structured_fields:
            col_name, col_type = struct_field.name, type(struct_field.dataType)
            for type_mapping in self.type_mappings:
                if issubclass(col_type, type_mapping.type):
                    analyzers += [
                        analyzer_cls(col_name)
                        for analyzer_cls in type_mapping.analyzers
                    ]
        return analyzers + self.analyzers


class DefaultProfilingBuilder(ProfilingBuilder):
    """Default setup for a profiling builder."""

    def __init__(self) -> None:
        super().__init__(
            type_mappings=[
                Type2Analyzers(
                    type=NumericType,
                    analyzers=[
                        Mean,
                        StandardDeviation,
                        Maximum,
                        Minimum,
                        functools.partial(  # type: ignore
                            ApproxQuantiles, quantiles=[0.25, 0.5, 0.75]
                        ),
                    ],
                ),
                Type2Analyzers(
                    type=DataType,
                    analyzers=[
                        Completeness,
                    ],
                ),
                Type2Analyzers(
                    type=StringType,
                    analyzers=[
                        MinLength,
                        MaxLength,
                        CountDistinct,
                    ],
                ),
            ],
            analyzers=[Size()],
        )


class Metric(BaseModel):
    entity: str
    instance: str
    name: str

    def __lt__(self, other: Metric) -> bool:
        return tuple(self.dict().values()) < tuple(other.dict().values())

    def __hash__(self) -> int:
        return hash(self.entity + self.instance + self.name)


class ProfilingValue(BaseModel):
    """Metrics schema from pydeequ."""

    metric: Metric
    value: float


class Column(BaseModel):
    """Schema of column."""

    name: str
    type: str


class ProfilingReport(SQLModel, table=True):
    """Profiling metrics aggregated for a given timestamp for a dataset."""

    id: str = Field(default=None, primary_key=True)
    dataset: str
    ts: datetime.datetime
    granularity: str
    profiling_values: List[ProfilingValue] = Field(
        sa_column=SqlAlchemyColumn(pydantic_column_type(List[ProfilingValue]))
    )

    @classmethod
    def build_id(cls, dataset: str, ts: datetime.datetime) -> str:
        return sha1(f"{dataset}{ts.isoformat()}".encode("utf-8")).hexdigest()

    def __init__(self, **data: Any):
        super().__init__(**data)
        self.id = self.build_id(self.dataset, self.ts)

    def get_profiling_value(self, metric: Metric) -> ProfilingValue:
        return [
            profiling_value
            for profiling_value in self.profiling_values
            if profiling_value.metric == metric
        ].pop(0)

    def get_metrics(self) -> Set[Metric]:
        return set(profiling_value.metric for profiling_value in self.profiling_values)


class Granularity:
    """Describe possible granularities for timestamp partitions."""

    DAY = "DAY"


def _transform_day(c: Union[SparkColumn, str]) -> SparkColumn:
    return F.to_timestamp(F.to_date(c))


_GranularityTransform = {Granularity.DAY: _transform_day}


def _transform_ts_granularity(
    df: DataFrame, ts_column: str, granularity: str
) -> DataFrame:
    return df.withColumn("ts", _GranularityTransform[granularity](ts_column))


def _build_report(
    profiling_builder: ProfilingBuilder,
    dataset: str,
    single_partition_df: DataFrame,
    ts: datetime.datetime,
    granularity: str,
    spark: SparkSession,
) -> ProfilingReport:
    analyzers = profiling_builder.build_analyzers(
        structured_fields=single_partition_df.schema.fields
    )
    analysis_runner = AnalysisRunner(spark).onData(single_partition_df)
    for analyzer in analyzers:
        analysis_runner = analysis_runner.addAnalyzer(analyzer=analyzer)
    analyzer_context: AnalyzerContext = analysis_runner.run()
    metrics = AnalyzerContext.successMetricsAsJson(
        spark_session=spark, analyzerContext=analyzer_context
    )
    logger.info(f"Finished profiling report for ts={ts.isoformat()}.")
    profiling = ProfilingReport(
        dataset=dataset,
        ts=ts,
        granularity=granularity,
        profiling_values=[
            ProfilingValue(
                metric=Metric(
                    entity=record["entity"],
                    instance=record["instance"],
                    name=record["name"],
                ),
                value=record["value"],
            )
            for record in metrics
        ],
    )
    return profiling


def profile(
    df: DataFrame,
    dataset: str,
    ts_column: str,
    profiling_builder: Optional[ProfilingBuilder] = None,
    granularity: Optional[str] = None,
    spark: Optional[SparkSession] = None,
) -> List[ProfilingReport]:
    """Run a profiling pipeline on a given dataset.

    Args:
        df: data to be processed.
        dataset: unique identification for the dataset.
        ts_column: column name that defines the timestamp.
        profiling_builder: profiling metrics configuration.
        granularity: granularity for the ts partitions.
            This granularity is going to be used to transform ts column into
            discrete partitions. By default, it uses a daily granularity.
        spark: spark context.

    Returns:
        set of profiling reports for each ts partition.

    """
    logger.info("👤 Profiling started ...")
    spark = spark or SparkSession.builder.getOrCreate()
    granularity = granularity or Granularity.DAY
    ts_transformed_df = _transform_ts_granularity(
        df=df, ts_column=ts_column, granularity=granularity
    )
    ts_values: List[datetime.datetime] = sorted(
        ts_transformed_df.select(ts_column)
        .distinct()
        .rdd.flatMap(lambda x: x)
        .collect()
    )
    logger.info(
        f"Processing {len(ts_values)} timestamps from {ts_values[0].isoformat()} to "
        f"{ts_values[-1].isoformat()}, with {granularity} granularity."
    )
    profiling = [
        _build_report(
            profiling_builder=profiling_builder or DefaultProfilingBuilder(),
            dataset=dataset,
            single_partition_df=ts_transformed_df.where(
                F.col(ts_column) == F.lit(ts_value)
            ),
            ts=ts_value,
            granularity=granularity,
            spark=spark,
        )
        for ts_value in ts_values
    ]
    logger.info("👤 Profiling done!")
    return profiling
