# import datetime
# from typing import Any, Callable, Dict, List
#
# from sqlalchemy import MetaData, create_engine
# from sqlalchemy.dialects.postgresql import insert
# from sqlalchemy.engine.base import Engine
# from sqlalchemy.schema import Table
#
# from thoth.repository import DataSourceClient
#
#
# def _with_engine(func: Callable[..., Any]) -> Callable[..., Any]:
#     def query_wrapper(*args: Any, **kwargs: Any) -> Any:
#         self = args[0]
#         connection_str = (
#             f"postgresql+pg8000://{self.username}:{self.password}@"
#             f"{self.host}:{self.port}/{self.database}"
#         )
#         engine: Engine = create_engine(url=connection_str)
#         if not kwargs.get("engine"):
#             kwargs["engine"] = engine
#         output = func(*args, **kwargs)
#         engine.dispose()
#         return output
#
#     return query_wrapper
#
#
# def _get_table(table_name: str, schema: str, engine: Engine) -> Table:
#     metadata = MetaData(bind=engine, schema=schema)
#     return Table(table_name, metadata, autoload=True)
#
#
# class PostgreSQLClient(DataSourceClient):
#     """Connects and implements queries for PostgreSQL database.
#
#     Args:
#         username: username credential to log in database.
#         password: password credential do log in database
#         host: host of the database service.
#         port: port number to connect.
#         database: database name.
#         schema: schema name.
#
#     """
#
#     def _get(
#         self,
#         table: str,
#         dataset_name: str,
#         max_ts: datetime.datetime,
#         min_ts: datetime.datetime,
#     ) -> List[Dict[str, Any]]:
#         pass
#
#     def __init__(
#         self,
#         username: str,
#         password: str,
#         host: str,
#         port: str,
#         database: str,
#         schema: str,
#     ):
#         self.username = username
#         self.password = password
#         self.host = host
#         self.port = port
#         self.database = database
#         self.schema = schema
#
#     @_with_engine
#     def _upsert(
#         self,
#         records: List[Dict[str, Any]],
#         dataset_name: str,
#         engine: Engine,
#     ) -> bool:
#         table = _get_table(table_name=dataset_name, schema=self.schema, engine=engine)
#         columns = [c.name for c in table.c]
#         features = [c for c in columns if c not in key_columns]
#         stmt = insert(table).values(records)
#         stmt = stmt.on_conflict_do_update(
#             constraint=table.primary_key,
#             set_={f: getattr(stmt.excluded, f) for f in features},
#         )
#         engine.execute(stmt)
#         return True
#
#     @_with_engine
#     def select_table(self, table_name: str, engine: Engine) -> Table:
#         """..."""
#         return _get_table(table_name=table_name, schema=self.schema, engine=engine)
