import datetime

import pytest
from pydeequ.analyzers import Mean
from sqlmodel import Session

from thoth import repository, service_layer
from thoth.anomaly import AnomalyScoring, Score
from thoth.anomaly.models import SimpleModelFactory
from thoth.anomaly.optimization import AnomalyOptimization, MetricOptimization
from thoth.dataset import Dataset
from thoth.profiler import Metric, ProfilingBuilder, Granularity
from thoth.quality import LogHandler
from thoth.service_layer import ThothServiceError


def test__build_repo_error():
    # act and assert
    with pytest.raises(ValueError):
        service_layer._build_repo()


def test__build_repo_with_session(session):
    # act
    output = service_layer._build_repo(session=session)

    # assert
    isinstance(output, repository.SqlRepository)


def test_assess_quality():
    """Test assess_quality service fetching optimization and scoring from repo."""
    # arrange
    engine = service_layer.build_engine()
    ts = datetime.datetime(2022, 1, 1)
    dataset = "my-dataset"
    metric = Metric(entity="Column", instance="f1", name="Mean")
    optimization = AnomalyOptimization(
        dataset_uri=dataset,
        confidence=0.95,
        metric_optimizations=[
            MetricOptimization(
                metric=metric,
                best_model_name="SimpleModel",
                threshold=0.2,
                validation_results=[],
            )
        ],
    )
    scoring = AnomalyScoring(
        dataset_uri=dataset,
        ts=ts,
        scores=[Score(metric=metric, value=0.19, predicted=123)],
    )

    # act
    with Session(engine) as session:
        service_layer.init_db(engine=engine)
        repo = service_layer._build_repo(session=session)

        repo.add_optimization(optimization=optimization)
        repo.add_scoring(scoring=scoring)
        [output_scoring] = repo.select_scoring(dataset_uri="my-dataset")

        success = service_layer.assess_quality(
            dataset_uri=dataset,
            ts=ts,
            session=session,
        )

    # assert
    assert success is True
    assert output_scoring == scoring


def test_e2e_flow_with_anomaly(json_data, spark_session, caplog):
    # arrange
    caplog.set_level("INFO")

    series = [
        dict(ts=data_point.get("ts"), value=data_point.get("value") * 1.5)
        if data_point.get("ts") == "1981-12-31"
        else data_point
        for data_point in json_data[-15:]
    ]
    normal_history_df = spark_session.createDataFrame(
        [data_point for data_point in series if data_point.get("ts") < "1981-12-31"],
        schema="ts string, value float",
    )
    new_anomalous_point_df = spark_session.createDataFrame(
        [data_point for data_point in series if data_point.get("ts") == "1981-12-31"],
        schema="ts string, value float",
    )

    model_factory = SimpleModelFactory()

    engine = service_layer.build_engine()

    # act
    with Session(engine) as session:
        service_layer.init_db(engine=engine)
        _, _ = service_layer.profile_create_optimize(
            df=normal_history_df,
            dataset_uri="temperatures",
            ts_column="ts",
            profiling_builder=ProfilingBuilder(analyzers=[Mean(column="value")]),
            model_factory=model_factory,
            session=session,
            spark=spark_session,
        )
        result = service_layer.assess_new_ts(
            df=new_anomalous_point_df,
            ts=datetime.datetime(year=1981, month=12, day=31),
            dataset_uri="temperatures",
            profiling_builder=ProfilingBuilder(analyzers=[Mean(column="value")]),
            model_factory=model_factory,
            session=session,
            spark=spark_session,
            notification_handlers=[LogHandler()],
        )

    # assert
    assert "Anomaly detected for ts=1981-12-31" in caplog.text
    assert result is False


def test_score_exception():
    engine = service_layer.build_engine()
    with Session(engine) as session:
        service_layer.init_db(engine=engine)
        with pytest.raises(ValueError):
            service_layer.score(
                dataset_uri="my-dataset", ts=datetime.datetime.utcnow(), session=session
            )


def test_assess_quality_exception():
    engine = service_layer.build_engine()
    with Session(engine) as session:
        service_layer.init_db(engine=engine)
        with pytest.raises(ValueError):
            service_layer.assess_quality(
                dataset_uri="my-dataset", ts=datetime.datetime.utcnow(), session=session
            )


def test_assess_new_ts_exception():
    engine = service_layer.build_engine()
    with Session(engine) as session:
        service_layer.init_db(engine=engine)
        with pytest.raises(ValueError):
            service_layer.assess_new_ts(
                df=None,
                ts=datetime.datetime.utcnow(),
                dataset_uri="my-dataset",
                session=session,
            )


def test_profile_exception(session):
    # act and assert
    with pytest.raises(ThothServiceError):
        service_layer.profile(df=None, dataset_uri="not-found", session=session)


def test_get_datasets(session):
    # arrange
    dataset = Dataset(
        uri="my-dataset",
        ts_column="ts",
        columns=["f1"],
        granularity=Granularity.DAY,
        metrics=[Metric(entity="Column", instance="f1", name="Mean")],
    )
    service_layer.add_dataset(dataset=dataset, session=session)

    # act
    output = service_layer.get_datasets(session=session)

    # assert
    assert output == [dataset]
