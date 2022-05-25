from abc import ABC, abstractmethod
from typing import List

from pyspark.sql import Column, functions


class Profiler(ABC):
    def __init__(self, from_column: str):
        self.from_column = from_column

    @abstractmethod
    def profile(self) -> List[Column]:
        """."""


class ColumnStatsProfiler(Profiler):
    MAPPER = {
        "max": functions.max,
        "min": functions.min,
        "sum": functions.sum,
        "avg": functions.avg,
        "count": functions.count,
        "count_distinct": functions.countDistinct,
    }

    def __init__(self, from_column: str, metrics: List[str], not_numeric: bool = False):
        super().__init__(from_column)
        self.metrics = metrics
        self.not_numeric = not_numeric

    def profile(self) -> List[Column]:
        from_column = (
            functions.length(functions.col(self.from_column).cast("string"))
            if self.not_numeric
            else functions.col(self.from_column)
        )
        return [
            self.MAPPER[metric](from_column).alias(f"{self.from_column}__{metric}")
            for metric in self.metrics
        ]


class CompletenessProfiler(Profiler):
    def __init__(self, from_column: str):
        super().__init__(from_column)

    def profile(self) -> List[Column]:
        column = functions.col(self.from_column)
        return [
            (
                functions.count(
                    functions.when(
                        functions.isnan("") | column.isNull(), functions.lit(1)
                    )
                )
                / functions.count(functions.lit(1))
            ).alias(f"{self.from_column}__completeness")
        ]


class GlobalProfiler(Profiler):
    def __init__(self):
        super().__init__(from_column="global")

    def profile(self) -> List[Column]:
        return [functions.sum(functions.lit(1)).alias("global__count")]
