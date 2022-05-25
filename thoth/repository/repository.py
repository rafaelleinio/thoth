import datetime
from abc import ABC, abstractmethod
from typing import Any, Dict, Final, List, NamedTuple, Optional, Union

import desert

from thoth.anomaly import AnomalyScoring
from thoth.profiler import ProfilingReport

PROFILING_REPORT_TABLE_NAME: Final[str] = "profiling_reports"
DATASET_COLUMN_NAME: Final[str] = "dataset"
TS_COLUMN_NAME: Final[str] = "ts"


class DataSourceClient(ABC):
    """Abstract data source client.

    Implementations for specific data sources must implement the supported queries, the
    abstract methods.

    """

    @abstractmethod
    def _upsert(
        self,
        records: List[Dict[str, Any]],
        dataset_name: str,
    ) -> None:
        """Needs to implement upsert query."""

    def upsert(
        self,
        records: Union[Dict[str, Any], List[Dict[str, Any]]],
        dataset_name: str,
    ) -> None:
        """Insert records or update on key match.

        Args:
            records: json record to upsert.
            dataset_name: name of the dataset to upsert the record.

        Returns:
            True if upsert was successful.

        """
        self._upsert(
            records=records if isinstance(records, list) else [records],
            dataset_name=dataset_name,
        )

    @abstractmethod
    def _get(
        self,
        table: str,
        dataset_name: str,
        max_ts: datetime.datetime,
        min_ts: datetime.datetime,
    ) -> List[Dict[str, Any]]:
        """Needs to implement get query."""

    def get(
        self,
        table: str,
        dataset_name: str,
        max_ts: datetime.datetime,
        min_ts: datetime.datetime,
    ) -> List[Dict[str, Any]]:
        return self._get(table, dataset_name, max_ts, min_ts)


class MemoryDataSourceClient(DataSourceClient):
    class _Keys(NamedTuple):
        dataset: str
        ts: str

    def __init__(self) -> None:
        self.profiling_reports: Dict[NamedTuple, Dict[str, Any]] = dict()

    def _upsert(
        self,
        records: List[Dict[str, Any]],
        dataset_name: str,
        **kwargs: Any,
    ) -> None:
        if dataset_name == PROFILING_REPORT_TABLE_NAME:
            self.profiling_reports.update(
                {
                    self._Keys(
                        record[DATASET_COLUMN_NAME], record[TS_COLUMN_NAME]
                    ): record
                    for record in records
                }
            )

    def _get(
        self,
        table: str,
        dataset_name: str,
        max_ts: datetime.datetime,
        min_ts: datetime.datetime,
    ) -> List[Dict[str, Any]]:
        table_collection = (
            self.profiling_reports if table == PROFILING_REPORT_TABLE_NAME else {}
        )
        return [
            value
            for keys, value in table_collection.items()
            if (
                datetime.datetime.isoformat(min_ts)
                <= getattr(keys, TS_COLUMN_NAME)
                <= datetime.datetime.isoformat(max_ts)
            )
            and getattr(keys, DATASET_COLUMN_NAME) == dataset_name
        ]


class MetricsRepository:
    """Base repository adapter.

    It relies on a DataSourceClient to connect to data sources.

    Attributes:
        client: connect to specific data sources and have implemented queries.

    """

    def __init__(self, client: DataSourceClient = MemoryDataSourceClient()):
        self.client = client

    def add_profiling_report(self, reports: List[ProfilingReport]) -> None:
        """Add profiling reports to the repository.

        Args:
            reports: profiling report metrics.

        Returns:
            True if successful.

        """
        report_records_json = desert.schema(ProfilingReport).dump(reports, many=True)
        self.client.upsert(
            records=report_records_json,
            dataset_name=PROFILING_REPORT_TABLE_NAME,
        )

    def get_profiling_history(
        self,
        dataset_name: str,
        max_ts: datetime.datetime = datetime.datetime.min,
        min_ts: datetime.datetime = datetime.datetime.max,
    ) -> List[ProfilingReport]:
        report_records_json = self.client.get(
            table=PROFILING_REPORT_TABLE_NAME,
            dataset_name=dataset_name,
            max_ts=max_ts,
            min_ts=min_ts,
        )
        reports: List[ProfilingReport] = desert.schema(ProfilingReport).load(
            report_records_json, many=True
        )
        return reports

    def add_anomaly_scoring(self, anomaly_scoring: AnomalyScoring) -> None:
        pass

    def get_anomaly_scoring(
        self, dataset_name: str, ts: Optional[datetime.datetime] = None
    ) -> AnomalyScoring:
        pass
