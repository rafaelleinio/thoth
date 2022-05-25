from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Set, Union

import desert
from sqlalchemy import MetaData, create_engine
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.engine.base import Engine
from sqlalchemy.schema import Table

from thoth.profiler import ProfilingReport


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
        key_columns: List[str],
        **kwargs: Any,
    ) -> bool:
        """Needs to implement upsert query."""

    def upsert(
        self,
        records: Union[Dict[str, Any], List[Dict[str, Any]]],
        dataset_name: str,
        key_columns: List[str],
        **kwargs: Any,
    ) -> bool:
        """Insert records or update on key match.

        Args:
            records: json record to upsert.
            dataset_name: name of the dataset to upsert the record.
            key_columns: key columns to match for updates.
            **kwargs: extra arguments supported by specific data sources.

        Returns:
            True if upsert was successful.

        """
        return self._upsert(
            records=records if isinstance(records, list) else [records],
            dataset_name=dataset_name,
            key_columns=key_columns,
            **kwargs,
        )


def _with_engine(func: Callable[..., Any]) -> Callable[..., Any]:
    def query_wrapper(*args: Any, **kwargs: Any) -> Any:
        self = args[0]
        connection_str = (
            f"postgresql+pg8000://{self.username}:{self.password}@"
            f"{self.host}:{self.port}/{self.database}"
        )
        engine: Engine = create_engine(url=connection_str)
        if not kwargs.get("engine"):
            kwargs["engine"] = engine
        output = func(*args, **kwargs)
        engine.dispose()
        return output

    return query_wrapper


def _get_table(table_name: str, schema: str, engine: Engine) -> Table:
    metadata = MetaData(bind=engine, schema=schema)
    return Table(table_name, metadata, autoload=True)


class PostgreSQLClient(DataSourceClient):
    """Connects and implements queries for PostgreSQL database.

    Args:
        username: username credential to log in database.
        password: password credential do log in database
        host: host of the database service.
        port: port number to connect.
        database: database name.
        schema: schema name.

    """

    def __init__(
        self,
        username: str,
        password: str,
        host: str,
        port: str,
        database: str,
        schema: str,
    ):
        self.username = username
        self.password = password
        self.host = host
        self.port = port
        self.database = database
        self.schema = schema

    @_with_engine
    def _upsert(
        self,
        records: List[Dict[str, Any]],
        dataset_name: str,
        key_columns: List[str],
        engine: Engine,
        **kwargs: Any,
    ) -> bool:
        table = _get_table(table_name=dataset_name, schema=self.schema, engine=engine)
        columns = [c.name for c in table.c]
        features = [c for c in columns if c not in key_columns]
        stmt = insert(table).values(records)
        stmt = stmt.on_conflict_do_update(
            constraint=table.primary_key,
            set_={f: getattr(stmt.excluded, f) for f in features},
        )
        engine.execute(stmt)
        return True

    @_with_engine
    def select_table(self, table_name: str, engine: Engine) -> Table:
        """..."""
        return _get_table(table_name=table_name, schema=self.schema, engine=engine)

    @_with_engine
    def query(self, query: str, engine) -> Any:
        """..."""
        return engine.execute(query)


class MetricsRepository:
    """Base repository adapter.

    It relies on a DataSourceClient to connect to data sources.

    Attributes:
        client: connect to specific data sources and have implemented queries.
        dataset_name: dataset identification.
        write_mode: what strategy to use when writing the records.

    """

    def __init__(self, client: Optional[DataSourceClient] = None):
        self.client = client or PostgreSQLClient(
            username="postgres",
            password="postgres",
            host="rleinio-test.cyl3qqwcwdae.us-west-1.rds.amazonaws.com",
            port="5432",
            database="metrics_store",
            schema="public",
        )

    def add(self, reports: List[ProfilingReport], **kwargs: Any) -> bool:
        """Add profiling reports to the repository.

        Args:
            reports: profiling report metrics.
            kwargs: extra key arguments for db client.

        Returns:
            True if successful.

        """
        report_records_json = desert.schema(ProfilingReport).dump(reports, many=True)
        return self.client.upsert(
            records=report_records_json,
            dataset_name="metrics",
            key_columns=["dataset", "ts"],
            **kwargs,
        )

    def get(self, dataset: str) -> List[ProfilingReport]:
        """..."""
        # table: Table = self.client.select_table(table_name="metrics")
        # result = table.select().filter(table.dataset == dataset)
        result = self.client.query(
            f"select dataset, "
            f"replace(replace(replace(ts, '2021-7', '2021-07'), '2021-8', '2021-08'), "
            f"'2021-9', '2021-09') as ts, schema, metrics from metrics "
            f"where dataset = '{dataset}' order by ts"
        )
        return desert.schema(ProfilingReport, many=True).load(
            [
                {
                    "dataset": r.dataset,
                    "ts": r.ts,
                    "schema": r.schema,
                    "metrics": r.metrics,
                }
                for r in result
            ]
        )

    def get_datasets(self) -> Set[str]:
        """..."""
        result = self.client.query(
            "select distinct dataset from metrics order by dataset"
        )
        return {r.dataset for r in result}
