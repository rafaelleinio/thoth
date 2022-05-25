from abc import ABC, abstractmethod
from typing import List

from pyspark.sql import DataFrame, functions

from thoth.old_profiler.profilers import Profiler


class PreProcessing(ABC):
    @abstractmethod
    def run(self, input_df: DataFrame, time_column: str) -> DataFrame:
        """."""


class BaseDatePreProcessing(ABC):
    def run(self, input_df: DataFrame, time_column: str) -> DataFrame:
        return input_df.withColumn(
            colName=time_column, col=functions.expr(f"timestamp(date({time_column}))")
        )


class ProfilingPipeline:
    def __init__(
        self,
        profilers: List[Profiler],
        pre_processing: PreProcessing = BaseDatePreProcessing(),
    ):
        self.profilers = profilers
        self.pre_processing = pre_processing

    def run(self, input_df: DataFrame, time_column: str):
        pre_processed_df = self.pre_processing.run(input_df, time_column)
        return pre_processed_df.groupBy(time_column).agg(
            *[column for profiler in self.profilers for column in profiler.profile()]
        )
