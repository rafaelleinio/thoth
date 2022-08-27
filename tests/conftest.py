import datetime
import json
import pathlib
import uuid
from typing import Any, Dict, List

import pytest
from sqlmodel import Session, SQLModel, create_engine

from thoth.profiler import Granularity, Metric, ProfilingReport, ProfilingValue


@pytest.fixture
def json_data() -> List[Dict[str, Any]]:
    with open(f"{pathlib.Path(__file__).parent.resolve()}/data.json") as f:
        return json.load(f)


@pytest.fixture
def base_profiling_history(json_data) -> List[ProfilingReport]:
    return [
        ProfilingReport(
            uuid=str(uuid.uuid4()),
            dataset="my_dataset",
            ts=datetime.datetime.fromisoformat(record["ts"]),
            granularity=Granularity.DAY,
            profiling_values=[
                ProfilingValue(
                    Metric(entity="Column", instance="f1", name="Mean"),
                    value=record["value"],
                )
            ],
        )
        for record in json_data
    ]


@pytest.fixture
def in_memory_db():
    return create_engine("sqlite:///:memory:")


@pytest.fixture
def session(in_memory_db):
    SQLModel.metadata.create_all(in_memory_db)
    yield Session(in_memory_db)
