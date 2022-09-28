from typing import List, Set

from sqlalchemy import Column as SqlAlchemyColumn
from sqlmodel import Field, SQLModel

from thoth.profiler import Metric
from thoth.util.custom_typing import pydantic_column_type


class Dataset(SQLModel, table=True):
    """Hold a dataset metadata.

    Attributes:
        uri: dataset URI.
        granularity: granularity key indicating the grain of the aggregation.
            E.g. 'DAY'.
        metrics: list of profiling metrics this dataset is being monitored with.

    """

    uri: str = Field(primary_key=True)
    ts_column: str
    columns: List[str] = Field(
        sa_column=SqlAlchemyColumn(pydantic_column_type(List[str]))
    )
    granularity: str
    metrics: List[Metric] = Field(
        sa_column=SqlAlchemyColumn(pydantic_column_type(List[Metric]))
    )

    def get_instances(self) -> Set[str]:
        """Get the set of all instances presented in the metrics."""
        return set(metric.instance for metric in self.metrics)
