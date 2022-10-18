import datetime
import os
from typing import List, Optional, Tuple

from loguru import logger
from pyspark.sql import DataFrame, SparkSession
from sqlalchemy.future import Engine as _FutureEngine
from sqlmodel import Session, SQLModel, create_engine

from thoth import anomaly, profiler, quality, repository, util
from thoth.dataset import Dataset
from thoth.profiler import Granularity


class ThothServiceError(Exception):
    """General error for the service layer."""


def _build_connection_string() -> str:
    return os.environ.get("DATABASE_URL", "sqlite:///:memory:")


def build_engine() -> _FutureEngine:
    """Creates a SQL connection engine."""
    return create_engine(_build_connection_string())


def init_db(engine: Optional[_FutureEngine] = None, clear: bool = False) -> None:
    """Initialize the database with all models declared in domain."""
    engine = engine or build_engine()
    if clear:
        SQLModel.metadata.drop_all(engine)
    SQLModel.metadata.create_all(engine)


def _build_repo(
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> repository.AbstractRepository:
    if repo:
        return repo
    if session:
        return repository.SqlRepository(session=session)
    raise ValueError("Both repo and session cannot be None, one must be set.")


def add_dataset(
    dataset: Dataset,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> None:
    """Register a new dataset in the Metrics Repository.

    Args:
        dataset: dataset metadata.
        repo: repository to save the resulting profiling metrics.
        session: sql session to be used by the repository.

    """
    repo = _build_repo(repo=repo, session=session)
    repo.add_dataset(dataset=dataset)


def get_datasets(
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> List[Dataset]:
    """Get all datasets registered in the Metrics Repository.

    Args:
        repo: repository to save the resulting profiling metrics.
        session: sql session to be used by the repository.

    """
    repo = _build_repo(repo=repo, session=session)
    return repo.get_datasets()


def get_dataset(
    dataset_uri: str,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> Optional[Dataset]:
    """Get a specific dataset given the uri.

    Args:
        dataset_uri: dataset URI.
        repo: repository to save the resulting profiling metrics.
        session: sql session to be used by the repository.

    """
    repo = _build_repo(repo=repo, session=session)
    return repo.get_dataset(dataset_uri=dataset_uri)


def select_profiling(
    dataset_uri: str,
    start_ts: Optional[datetime.datetime] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> List[profiler.ProfilingReport]:
    """Get the profiling history for a specific dataset.

    Args:
        dataset_uri: dataset URI.
        start_ts: minimum timestamp to filter (closed interval).
        repo: repository to save the resulting profiling metrics.
        session: sql session to be used by the repository.

    """
    repo = _build_repo(repo=repo, session=session)
    return repo.select_profiling(dataset_uri=dataset_uri, start_ts=start_ts)


def get_optimization(
    dataset_uri: str,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> Optional[anomaly.AnomalyOptimization]:
    """Get the optimization for a specific dataset.

    Args:
        dataset_uri: dataset URI.
        repo: repository to save the resulting profiling metrics.
        session: sql session to be used by the repository.

    """
    repo = _build_repo(repo=repo, session=session)
    return repo.get_optimization(dataset_uri=dataset_uri)


def get_scoring(
    dataset_uri: str,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> List[anomaly.AnomalyScoring]:
    """Get the optimization for a specific dataset.

    Args:
        dataset_uri: dataset URI.
        repo: repository to save the resulting profiling metrics.
        session: sql session to be used by the repository.

    """
    repo = _build_repo(repo=repo, session=session)
    return repo.select_scoring(dataset_uri=dataset_uri)


def profile(
    df: DataFrame,
    dataset_uri: str,
    profiling_builder: Optional[profiler.ProfilingBuilder] = None,
    spark: Optional[SparkSession] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> List[profiler.ProfilingReport]:
    """Run a Spark profiling aggregation pipeline on a given input dataset.

    Args:
        df: dataset's data as a Spark Dataframe.
        dataset_uri: dataset URI.
        profiling_builder: config describing the profiling analyzers to run.
        spark: spark context.
        repo: repository to save the resulting profiling metrics.
        session: sql session to be used by the repository.

    Returns:
        Collection of resulting profiling aggregations for the given dataset.
            The length of the collection is based on the given ts_column and defined
            granularity.

    """
    repo = _build_repo(repo=repo, session=session)

    dataset = repo.get_dataset(dataset_uri=dataset_uri)
    if not dataset:
        raise ThothServiceError(
            f"No dataset was found for the giving uri={dataset_uri}"
        )
    ts_column = dataset.ts_column
    granularity = dataset.granularity

    profiling_records = profiler.profile(
        df=df,
        dataset_uri=dataset_uri,
        ts_column=ts_column,
        profiling_builder=profiling_builder,
        granularity=granularity,
        spark=spark,
    )
    repo.add_profiling(dataset_uri=dataset_uri, records=profiling_records)
    url = util.dashboard.build_dashboard_link(
        dataset_uri=dataset_uri, view=util.dashboard.PROFILING_VIEW
    )
    logger.info(f"🧐 You can now check the profiling metrics in the dashboard: {url}")
    return profiling_records


def profile_create(
    df: DataFrame,
    dataset_uri: str,
    ts_column: str,
    profiling_builder: Optional[profiler.ProfilingBuilder] = None,
    granularity: str = Granularity.DAY,
    spark: Optional[SparkSession] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> List[profiler.ProfilingReport]:
    """Run a profiling pipeline and the creation of the dataset.

    For help with the arguments check the 'profile' service docstring.

    """
    repo = _build_repo(repo=repo, session=session)
    profiling_records = profiler.profile(
        df=df,
        dataset_uri=dataset_uri,
        ts_column=ts_column,
        profiling_builder=profiling_builder,
        granularity=granularity,
        spark=spark,
    )
    add_dataset(
        dataset=Dataset(
            uri=dataset_uri,
            ts_column=ts_column,
            columns=[c for c in df.columns if c != ts_column],
            granularity=granularity,
            metrics=profiling_records[-1].get_metrics(),
        ),
        repo=repo,
    )
    repo.add_profiling(dataset_uri=dataset_uri, records=profiling_records)
    return profiling_records


def optimize(
    dataset_uri: str,
    profiling: Optional[List[profiler.ProfilingReport]] = None,
    last_n: Optional[int] = None,
    start_proportion: Optional[float] = None,
    target_confidence: Optional[float] = None,
    model_factory: Optional[anomaly.BaseModelFactory] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> anomaly.AnomalyOptimization:
    """Optimize the anomaly strategy for a given dataset using its profiling history.

    Args:
        dataset_uri: dataset URI.
        profiling: profiling history to be used in optimization.
            If not given, the service will try to fetch the dataset's profiling history
            from the metrics repository.
        last_n: limit the training profiling history series to the last_n points.
            Very old profiling patterns in the series may not help to explain what to
            expect from the next batches of data. In this way, this attribute can be
            helpful in prioritizing only the most recent patterns.
        start_proportion: start proportion for the cross validation algorithm.
            0.5 by default, meaning it validates sequentially half of the series points.
        target_confidence: main objective for the optimization.
            0.95 by default, meaning that it will try to automatically find a model and
            threshold for each metric on the dataset in which no more than 5% of the
            points have higher anomaly scores than the resulting threshold.
        model_factory: factory defining all the models to be evaluated.
        repo: repository to save the resulting optimization.
        session: sql session to be used by the repository.

    Returns:
        Return the optimization result for the target dataset.

    """
    repo = _build_repo(repo=repo, session=session)
    profiling = profiling or repo.select_profiling(dataset_uri=dataset_uri)
    optimization = anomaly.optimize(
        profiling_history=profiling[-last_n:] if last_n else profiling,
        start_proportion=start_proportion,
        confidence=target_confidence,
        model_factory=model_factory,
    )
    repo.add_optimization(optimization=optimization)
    url = util.dashboard.build_dashboard_link(
        dataset_uri=dataset_uri, view=util.dashboard.OPTIMIZATION_VIEW
    )
    logger.info(f"🧐 You can now check the dataset optimization in the dashboard: {url}")
    return optimization


def score(
    dataset_uri: str,
    ts: datetime.datetime,
    optimization: Optional[anomaly.AnomalyOptimization] = None,
    profiling_history: Optional[List[profiler.ProfilingReport]] = None,
    model_factory: Optional[anomaly.BaseModelFactory] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> anomaly.AnomalyScoring:
    """Calculate the anomaly score for a target dataset timestamp batch.

    Args:
        dataset_uri: dataset URI.
        ts: target profiling batch timestamp to be scored.
        optimization: anomaly optimization for the dataset.
            If not given, the service will try to fetch the dataset's anomaly
            optimization from the metrics repository.
        profiling_history:
            If not given, the service will try to fetch the dataset's profiling history
            from the metrics repository.
        model_factory: factory defining the same models as used in the optimization.
        repo: repository to save the scoring.
        session: sql session to be used by the repository.

    Returns:
        Scoring event for the target dataset timestamp batch.

    """
    repo = _build_repo(repo=repo, session=session)
    profiling_history = profiling_history or repo.select_profiling(
        dataset_uri=dataset_uri,
        end_ts=ts,
    )
    optimization = optimization or repo.get_optimization(dataset_uri=dataset_uri)
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
    dataset_uri: str,
    ts: datetime.datetime,
    optimization: Optional[anomaly.AnomalyOptimization] = None,
    scoring: Optional[anomaly.AnomalyScoring] = None,
    notification_handlers: Optional[List[quality.NotificationHandler]] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
) -> bool:
    """Perform the quality assessment for a target dataset scoring timestamp.

    Args:
        dataset_uri: dataset URI.
        ts: target scoring timestamp to be assessed.
        optimization: anomaly optimization for the dataset.
            If not given, the service will try to fetch the dataset's anomaly
            optimization from the metrics repository.
        scoring: scoring event for the target dataset timestamp batch.
            If not given, the service will try to fetch the dataset's scoring event
            from the metrics repository.
        notification_handlers: handlers to be triggered in case of anomaly detected.
        repo: repository fetch the optimization and scoring.
        session: sql session to be used by the repository.

    Returns:
        False if anomaly is detected for the target scoring.

    """
    repo = _build_repo(repo=repo, session=session)
    scoring = scoring or repo.get_scoring(
        id_=anomaly.AnomalyScoring._build_id(dataset_uri=dataset_uri, ts=ts)
    )
    optimization = optimization or repo.get_optimization(dataset_uri=dataset_uri)
    if not scoring or not optimization:
        raise ValueError(
            "scoring and optimization can't be None. Values were not found in repo."
        )

    return quality.assess_quality(
        anomaly_optimization=optimization,
        anomaly_scoring=scoring,
        notification_handlers=notification_handlers,
    )


def profile_create_optimize(
    df: DataFrame,
    dataset_uri: str,
    ts_column: str,
    granularity: str = Granularity.DAY,
    profiling_builder: Optional[profiler.ProfilingBuilder] = None,
    optimize_last_n: Optional[int] = None,
    optimize_start_proportion: Optional[float] = None,
    optimize_target_confidence: Optional[float] = None,
    model_factory: Optional[anomaly.BaseModelFactory] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
    spark: Optional[SparkSession] = None,
) -> Tuple[List[profiler.ProfilingReport], anomaly.AnomalyOptimization]:
    """Run a profiling pipeline, creation of the dataset and the optimization together.

    This service is very useful to onboard a brand-new dataset to the metrics
    repository.

    For help with the arguments check 'profile' and 'optimize' services docstrings.

    """
    logger.info("Pipeline started 👤 📈 ...")
    profiling = profile_create(
        df=df,
        dataset_uri=dataset_uri,
        ts_column=ts_column,
        profiling_builder=profiling_builder,
        granularity=granularity,
        spark=spark,
        repo=repo,
        session=session,
    )
    optimization = optimize(
        dataset_uri=dataset_uri,
        profiling=profiling,
        last_n=optimize_last_n,
        start_proportion=optimize_start_proportion,
        target_confidence=optimize_target_confidence,
        model_factory=model_factory,
        repo=repo,
        session=session,
    )
    logger.info("Pipeline finished!")
    return profiling, optimization


def assess_new_ts(
    df: DataFrame,
    ts: datetime.datetime,
    dataset_uri: str,
    profiling_builder: Optional[profiler.ProfilingBuilder] = None,
    model_factory: Optional[anomaly.BaseModelFactory] = None,
    notification_handlers: Optional[List[quality.NotificationHandler]] = None,
    repo: Optional[repository.AbstractRepository] = None,
    session: Optional[Session] = None,
    spark: Optional[SparkSession] = None,
) -> bool:
    """Run a profiling pipeline, scoring and assessment for a new data batch timestamp.

    For help with the arguments check 'profile' and 'score' services docstrings.

    """
    logger.info("Pipeline started 👤 💯 🔍️ ...")
    logger.info(ts)
    repo = _build_repo(repo=repo, session=session)
    profiling_history = repo.select_profiling(dataset_uri=dataset_uri, end_ts=ts)
    optimization = repo.get_optimization(dataset_uri=dataset_uri)
    if not profiling_history:
        raise ValueError("Optimization or profiling history not found in repository.")

    [new_profiling] = profile(
        df=df,
        dataset_uri=dataset_uri,
        profiling_builder=profiling_builder,
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
        dataset_uri=dataset_uri,
        ts=ts,
        optimization=optimization,
        profiling_history=profiling_history,
        model_factory=model_factory,
        repo=repo,
        session=session,
    )

    success = assess_quality(
        dataset_uri=dataset_uri,
        ts=ts,
        optimization=optimization,
        scoring=scoring,
        notification_handlers=notification_handlers,
        repo=repo,
        session=session,
    )
    logger.info("Pipeline finished!")
    return success
