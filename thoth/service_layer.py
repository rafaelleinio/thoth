import datetime
import os
from typing import List, Optional, Tuple

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from sqlalchemy.future import Engine as _FutureEngine
from sqlmodel import Session, SQLModel, create_engine

from thoth import anomaly, profiler, quality, repository


def _build_connection_string() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///:memory:")


def build_engine() -> _FutureEngine:
    return create_engine(_build_connection_string())


def init_db(engine: Optional[_FutureEngine] = None) -> None:
    """Initialize the database with all models declared in domain."""
    SQLModel.metadata.create_all(engine or build_engine())


def _build_repo(
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> repository.AbstractRepository:
    if repo:
        return repo
    if session:
        return repository.SqlRepository(session=session)
    raise ValueError("Both repo and session cannot be None, one must be set.")


def profile(
    df: DataFrame,
    dataset: str,
    ts_column: str,
    profiling_builder: Optional[profiler.ProfilingBuilder] = None,
    granularity: Optional[str] = None,
    spark: Optional[SparkSession] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> List[profiler.ProfilingReport]:
    profiling_records = profiler.profile(
        df=df,
        dataset=dataset,
        ts_column=ts_column,
        profiling_builder=profiling_builder,
        granularity=granularity,
        spark=spark,
    )
    repo = _build_repo(repo=repo, session=session)
    repo.add_profiling(records=profiling_records)
    return profiling_records


def optimize(
    dataset: str,
    profiling: Optional[List[profiler.ProfilingReport]] = None,
    last_n: Optional[int] = None,
    start_proportion: Optional[float] = None,
    target_confidence: Optional[float] = None,
    model_factory: Optional[anomaly.optimization.BaseModelFactory] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> anomaly.optimization.AnomalyOptimization:
    repo = _build_repo(repo=repo, session=session)
    profiling = profiling or repo.select_profiling(dataset=dataset)
    optimization = anomaly.optimization.optimize(
        profiling_history=profiling[-last_n:] if last_n else profiling,
        start_proportion=start_proportion,
        confidence=target_confidence,
        model_factory=model_factory,
    )
    repo.add_optimization(optimization=optimization)
    return optimization


def score(
    dataset: str,
    ts: datetime.datetime,
    optimization: Optional[anomaly.optimization.AnomalyOptimization] = None,
    profiling_history: Optional[List[profiler.ProfilingReport]] = None,
    model_factory: Optional[anomaly.optimization.BaseModelFactory] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> anomaly.AnomalyScoring:
    repo = _build_repo(repo=repo, session=session)
    profiling_history = profiling_history or repo.select_profiling(
        dataset=dataset,
        end_ts=ts,
    )
    optimization = optimization or repo.get_optimization(dataset=dataset)
    if not profiling_history or not optimization:
        raise ValueError(
            "profiling and optimization can't be None. Values were not found in repo."
        )

    scoring = anomaly.score(
        profiling_history=profiling_history,
        optimization=optimization,
        model_factory=model_factory,
    )
    repo.add_scoring(scoring=scoring)
    return scoring


def assess_quality(
    dataset: str,
    ts: datetime.datetime,
    optimization: Optional[anomaly.optimization.AnomalyOptimization] = None,
    scoring: Optional[anomaly.AnomalyScoring] = None,
    notification_handlers: Optional[List[quality.NotificationHandler]] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> bool:
    repo = _build_repo(repo=repo, session=session)
    scoring = scoring or repo.get_scoring(
        id_=anomaly.AnomalyScoring.build_id(dataset=dataset, ts=ts)
    )
    optimization = optimization or repo.get_optimization(dataset=dataset)
    if not scoring or not optimization:
        raise ValueError(
            "scoring and optimization can't be None. Values were not found in repo."
        )

    return quality.assess_quality(
        anomaly_optimization=optimization,
        anomaly_scoring=scoring,
        notification_handlers=notification_handlers,
    )


def profile_optimize(
    df: DataFrame,
    dataset: str,
    ts_column: str,
    profiling_builder: Optional[profiler.ProfilingBuilder] = None,
    granularity: Optional[str] = None,
    start_proportion: Optional[float] = None,
    target_confidence: Optional[float] = None,
    model_factory: Optional[anomaly.optimization.BaseModelFactory] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
    spark: Optional[SparkSession] = None,
) -> Tuple[List[profiler.ProfilingReport], anomaly.optimization.AnomalyOptimization]:
    logger.info("Pipeline started 👤 📈 ...")
    profiling = profile(
        df=df,
        dataset=dataset,
        ts_column=ts_column,
        profiling_builder=profiling_builder,
        granularity=granularity,
        spark=spark,
        repo=repo,
        session=session,
    )
    optimization = optimize(
        dataset=dataset,
        profiling=profiling,
        start_proportion=start_proportion,
        target_confidence=target_confidence,
        model_factory=model_factory,
        repo=repo,
        session=session,
    )
    logger.info("Pipeline finished!")
    return profiling, optimization


def assess_new_ts(
    df: DataFrame,
    ts: datetime.datetime,
    dataset: str,
    ts_column: str,
    profiling_builder: Optional[profiler.ProfilingBuilder] = None,
    granularity: Optional[str] = None,
    model_factory: Optional[anomaly.optimization.BaseModelFactory] = None,
    notification_handlers: Optional[List[quality.NotificationHandler]] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
    spark: Optional[SparkSession] = None,
) -> bool:
    logger.info("Pipeline started 👤 💯 🔍️ ...")
    repo = _build_repo(repo=repo, session=session)
    profiling_history = repo.select_profiling(dataset=dataset, end_ts=ts)
    optimization = repo.get_optimization(dataset=dataset)
    if not profiling_history:
        raise ValueError("Optimization or profiling history not found in repository.")

    [new_profiling] = profile(
        df=df,
        dataset=dataset,
        ts_column=ts_column,
        profiling_builder=profiling_builder,
        granularity=granularity,
        spark=spark,
        repo=repo,
        session=session,
    )
    last_profiling_history = profiling_history.pop()
    profiling_history += (
        [new_profiling]
        if last_profiling_history.ts == new_profiling.ts
        else [last_profiling_history, new_profiling]
    )

    scoring = score(
        dataset=dataset,
        ts=ts,
        optimization=optimization,
        profiling_history=profiling_history,
        model_factory=model_factory,
        repo=repo,
        session=session,
    )

    success = assess_quality(
        dataset=dataset,
        ts=ts,
        optimization=optimization,
        scoring=scoring,
        notification_handlers=notification_handlers,
        repo=repo,
        session=session,
    )
    logger.info("Pipeline finished!")
    return success
