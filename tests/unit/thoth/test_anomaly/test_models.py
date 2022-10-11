import datetime
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import pytest

from thoth.anomaly.models import (
    AutoProphetModel,
    AutoSarimaModel,
    ForecastValueError,
    Model,
    SimpleModel,
)
from thoth.base import Point


@pytest.fixture
def simple_json_data() -> List[Dict[str, Any]]:
    return [
        {"ts": datetime.datetime(2022, 4, 20), "value": 1},
        {"ts": datetime.datetime(2022, 4, 20), "value": 2},
        {"ts": datetime.datetime(2022, 4, 20), "value": 3},
        {"ts": datetime.datetime(2022, 4, 20), "value": 4},
        {"ts": datetime.datetime(2022, 4, 20), "value": 5},
    ]


class TestModel:
    def test_forecast(self):
        # arrange
        class ProblematicModel(Model):
            def _train(self, points: List[Point]) -> None:
                pass

            def _reset(self) -> None:
                pass

            def _forecast(self, n: int = 1) -> List[float]:
                return [np.NaN]

        model = ProblematicModel()

        # act and assert
        with pytest.raises(ForecastValueError):
            _ = model.forecast()


class TestSimpleModel:
    def test__add_windows_and_errors(self, simple_json_data):
        # arrange
        input_pdf = pd.DataFrame(simple_json_data)
        target_pdf = pd.concat(
            [
                input_pdf,
                pd.DataFrame(
                    [
                        {
                            "window_2": None,
                            "window_2_error": None,
                            "window_3": None,
                            "window_3_error": None,
                        },
                        {
                            "window_2": None,
                            "window_2_error": None,
                            "window_3": None,
                            "window_3_error": None,
                        },
                        {
                            "window_2": 1.5,
                            "window_2_error": 0.5,
                            "window_3": None,
                            "window_3_error": None,
                        },
                        {
                            "window_2": 2.5,
                            "window_2_error": 0.375,
                            "window_3": 2.0,
                            "window_3_error": 0.5,
                        },
                        {
                            "window_2": 3.5,
                            "window_2_error": 0.3,
                            "window_3": 3.0,
                            "window_3_error": 0.4,
                        },
                    ]
                ),
            ],
            axis=1,
        )
        model = SimpleModel(windows=[2, 3])

        # act
        output_pdf = model._add_windows_and_errors(input_pdf=input_pdf)

        # assert
        pd.testing.assert_frame_equal(output_pdf, target_pdf)

    def test_score(self, simple_json_data):
        # arrange
        model = SimpleModel(windows=[1])

        # act
        predicted, score = model.score(
            points=[Point(**record) for record in simple_json_data]
        )

        # assert
        assert predicted == 4.0
        assert score == 0.2
        assert model._best_window == 1

    def test__check_series_length(self):
        # arrange
        input_points = [
            Point(ts=datetime.datetime(2022, 1, 1), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 2), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 3), value=15.0),
        ]
        model = SimpleModel(windows=[1, 3])

        # act
        _ = model._check_series_length(train_points=input_points)

        # assert
        assert model._skip_windows == [3]

    def test__check_series_length_exception(self):
        # arrange
        input_points = [
            Point(ts=datetime.datetime(2022, 1, 1), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 2), value=15.0),
            Point(ts=datetime.datetime(2022, 1, 3), value=15.0),
        ]
        model = SimpleModel(windows=[3])

        # act and assert
        with pytest.raises(ValueError):
            _ = model._check_series_length(train_points=input_points)

    def test__forecast_error(self):
        # arrange
        model = SimpleModel()

        # act and assert
        with pytest.raises(RuntimeError):
            model.forecast(n=1)


class TestAutoSarimaModel:
    def test_score(self, json_data):
        # arrange
        model = AutoSarimaModel()

        # act
        _, score = model.score(
            points=[
                Point(
                    ts=datetime.datetime.fromisoformat(record["ts"]),
                    value=record["value"],
                )
                for record in json_data
            ]
        )

        # assert
        assert score < 0.18  # less than 18% of absolute percentage error


class TestAutoProphetModel:
    def test_score(self, json_data):
        # arrange
        model = AutoProphetModel()

        # act
        _, score = model.score(
            points=[
                Point(
                    ts=datetime.datetime.fromisoformat(record["ts"]),
                    value=record["value"],
                )
                for record in json_data
            ]
        )

        # assert
        assert score < 0.23
