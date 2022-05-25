import functools
from dataclasses import dataclass
from functools import reduce
from typing import Dict, List, Optional, Type

import desert
from pydeequ.analyzers import (
    AnalysisRunner,
    AnalyzerContext,
    ApproxQuantiles,
    Completeness,
    CountDistinct,
    Distinctness,
    Maximum,
    MaxLength,
    Mean,
    Minimum,
    MinLength,
    Size,
    StandardDeviation,
    _AnalyzerObject,
)
from pyspark.sql import DataFrame, SparkSession
from pyspark.sql import functions as F
from pyspark.sql.types import DataType, NumericType, StringType, StructField


@dataclass
class Type2Analyzers:
    """Mapping between spark data types and analyzers to user."""

    type: Type[DataType]
    analyzers: List[Type[_AnalyzerObject]]


class ProfilingBuilder:
    """Build a set of analyzers to be used in a profiling pipeline."""

    def __init__(
        self,
        type_mappings: List[Type2Analyzers],
        extra_analyzers: Optional[List[_AnalyzerObject]] = None,
    ):
        self.type_mappings = type_mappings
        self.extra_analyzers = extra_analyzers or []

    def build_analyzers(
        self, structed_fields: List[StructField]
    ) -> List[_AnalyzerObject]:
        """Build a set of analyzers.

        Args:
            structed_fields: spark data type fields.

        Returns:
            set of created analyzers.

        """
        analyzers = []
        for struct_field in structed_fields:
            col_name, col_type = struct_field.name, type(struct_field.dataType)
            for type_mapping in self.type_mappings:
                if issubclass(col_type, type_mapping.type):
                    analyzers += [
                        analyzer_cls(col_name)
                        for analyzer_cls in type_mapping.analyzers
                    ]
        return analyzers + self.extra_analyzers


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
                        CountDistinct,
                        Distinctness,
                    ],
                ),
                Type2Analyzers(type=StringType, analyzers=[MinLength, MaxLength]),
            ],
            extra_analyzers=[Size()],
        )


@dataclass
class Metric:
    """Metrics schema from pydeequ."""

    entity: str
    instance: str
    name: str
    value: float


@dataclass
class Column:
    """Schema of column."""

    name: str
    type: str


@dataclass
class ProfilingReport:
    """Profiling report schema."""

    dataset: str
    ts: str
    schema: List[Column]
    metrics: List[Metric]


class Profiler:
    """Defines a profiling pipeline."""

    def __init__(self, profiling_builder: Optional[ProfilingBuilder] = None) -> None:
        self.profiling_builder = profiling_builder or DefaultProfilingBuilder()

    def _build_report(
        self,
        dataset: str,
        single_partition_df: DataFrame,
        ts: Dict[str, str],
        spark: SparkSession,
    ) -> ProfilingReport:
        analyzers = self.profiling_builder.build_analyzers(
            structed_fields=single_partition_df.schema.fields
        )
        analysis_run_builder = AnalysisRunner(spark).onData(single_partition_df)
        for analyzer in analyzers:
            analysis_run_builder = analysis_run_builder.addAnalyzer(analyzer)
        analysis_result: AnalyzerContext = analysis_run_builder.run()
        metrics_dict = AnalyzerContext.successMetricsAsJson(spark, analysis_result)
        return ProfilingReport(
            dataset=dataset,
            ts="-".join(str(value) for value in ts.values()),
            schema=[
                Column(name=c.name, type=c.dataType.typeName())
                for c in single_partition_df.schema
            ],
            metrics=desert.schema(cls=Metric).load(data=metrics_dict, many=True),
        )

    def _build_single_partition_df(
        self, df: DataFrame, ts: Dict[str, str]
    ) -> DataFrame:
        filter_col = reduce(
            lambda x, y: x & y, [F.col(k) == v for (k, v) in ts.items()]
        )
        return df.where(filter_col).drop(*ts.keys())

    def run(
        self,
        dataset: str,
        df: DataFrame,
        ts_partition_columns: List[str],
        spark: SparkSession,
    ) -> List[ProfilingReport]:
        """Run the profiling pipeline.

        Args:
            dataset: unique identification for the dataset.
            df: data to be processed.
            ts_partition_columns: columns that defines the ts partitions.
            spark: spark context.

        Returns:
            set of profiling reports for each ts partition.

        """
        ts_values = (
            df.select(*ts_partition_columns)
            .orderBy(*ts_partition_columns)
            .distinct()
            .rdd.map(
                lambda row: {key: str(value) for key, value in row.asDict().items()}
            )
            .collect()
        )
        profiling_reports = [
            self._build_report(
                dataset=dataset,
                single_partition_df=self._build_single_partition_df(df=df, ts=ts),
                ts=ts,
                spark=spark,
            )
            for ts in ts_values
        ]
        return profiling_reports
