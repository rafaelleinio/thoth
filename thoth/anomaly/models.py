import dataclasses
import warnings
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Type

import numpy as np
import pandas as pd
from merlion.models.automl.autoprophet import AutoProphet, AutoProphetConfig
from merlion.models.automl.autosarima import AutoSarima, AutoSarimaConfig
from merlion.utils.time_series import TimeSeries as MerlionTimeSeries
from merlion.utils.time_series import UnivariateTimeSeries

from thoth.anomaly.base import Point
from thoth.anomaly.error_metrics import ape


class ForecastValueError(Exception):
    pass


class Model(ABC):
    __name__ = "Base"

    @abstractmethod
    def _train(self, points: List[Point]) -> None:
        """."""

    @abstractmethod
    def _forecast(self, n: int = 1) -> List[float]:
        """."""

    @abstractmethod
    def _reset(self) -> None:
        """."""

    def reset(self) -> None:
        return self._reset()

    def train(self, points: List[Point]) -> None:
        self._train(points)

    def forecast(self, n: int = 1) -> List[float]:
        forecasts = self._forecast(n)
        if not all(
            (isinstance(value, float) and not np.isnan(value)) for value in forecasts
        ):
            raise ForecastValueError(
                f"Unexpected non-float forecast value(s) returned from model, "
                f"forecasts: {forecasts}"
            )
        return forecasts

    def score(self, points: List[Point]) -> Tuple[float, float]:
        train_points = points[:-1]
        target_point = points[-1]
        self.train(train_points)
        [predicted_value] = self.forecast(n=1)
        return predicted_value, ape(
            true_value=target_point.value, predicted_value=predicted_value
        )


class SimpleModel(Model):
    __name__ = "Simple"

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or [3, 5, 7, 30]
        self.best_window: Optional[int] = None
        self.train_data_pdf: pd.DataFrame = None
        self.skip_windows: List[int] = []

    def _reset(self) -> None:
        self.best_window = None
        self.train_data_pdf = None
        self.skip_windows = []

    def _add_windows_and_errors(
        self, input_pdf: pd.DataFrame, windows: Optional[List[int]] = None
    ) -> pd.DataFrame:
        pdf = input_pdf.copy()
        for window in windows or self.windows:
            pdf[f"window_{window}"] = pdf["value"].shift(1).rolling(window).mean()
            pdf[f"window_{window}_error"] = pdf.apply(
                func=lambda r: ape(
                    true_value=r["value"], predicted_value=r[f"window_{window}"]
                ),
                axis=1,
            )
        return pdf

    def _check_series_length(self, train_points: List[Point]) -> None:
        train_length = len(train_points)
        sorted_windows = sorted(self.windows)
        shortest_window = sorted_windows[0]
        if train_length <= shortest_window:
            raise ValueError(
                f"Given train data is smaller (length={train_length}) than the "
                f"shortest window (range={shortest_window})."
            )
        self.skip_windows = [w for w in sorted_windows if w >= train_length]

    def _train(self, points: List[Point]) -> None:
        # self._check_series_length(points)
        self.train_data_pdf = pd.DataFrame(
            [dataclasses.asdict(point) for point in points]
        )

        transformed_pdf = self._add_windows_and_errors(input_pdf=self.train_data_pdf)
        error_columns = [
            column for column in transformed_pdf.columns if column.endswith("error")
        ]
        mean_errors = [transformed_pdf[column].mean() for column in error_columns]

        index_min = min(range(len(mean_errors)), key=mean_errors.__getitem__)
        self.best_window = self.windows[index_min]

    def _forecast(self, n: int = 1) -> List[float]:
        if n > 1:
            raise NotImplementedError(
                "This model only supports the forecast window of n=1"
            )
        if not self.best_window:
            raise RuntimeError("Model must be trained first.")

        train_df_with_empty_row_in_the_end_pdf = pd.concat(
            [
                self.train_data_pdf,
                pd.DataFrame(
                    [[np.NaN] * self.train_data_pdf.shape[1]],
                    columns=self.train_data_pdf.columns,
                ),
            ],
            ignore_index=True,
        )
        transformed_pdf = self._add_windows_and_errors(
            input_pdf=train_df_with_empty_row_in_the_end_pdf,
            windows=[self.best_window],
        )
        value: float = transformed_pdf[f"window_{self.best_window}"].iloc[-1]
        return [value]


def _create_train_data_for_merlion_models(points: List[Point]) -> MerlionTimeSeries:
    ts_as_seconds = [p.ts.timestamp() for p in points]
    return MerlionTimeSeries(
        univariates=[
            UnivariateTimeSeries(
                time_stamps=ts_as_seconds, values=[p.value for p in points]
            )
        ]
    )


def _parse_forecast_for_merlion_models(args: Any) -> List[float]:
    forecast_ts, _ = args
    forecast_values: List[float] = [value for _, [value] in forecast_ts]
    return forecast_values


class AutoSarimaModel(Model):
    __name__ = "AutoSarima"

    def __init__(self, auto_sarima_model: Optional[AutoSarima] = None):
        self.model = auto_sarima_model or AutoSarima(
            config=AutoSarimaConfig(
                auto_pqPQ=True,
                auto_d=True,
                auto_D=True,
                auto_seasonality=True,
                approximation=True,
            )
        )

    def _reset(self) -> None:
        self.model.reset()

    def _train(self, points: List[Point]) -> None:
        self.model.train(
            _create_train_data_for_merlion_models(points),
            train_config={"enforce_stationarity": True, "enforce_invertibility": True},
        )

    def _forecast(self, n: int = 1) -> List[float]:
        return _parse_forecast_for_merlion_models(self.model.forecast(n))


class AutoProphetModel(Model):
    __name__ = "AutoProphet"

    def __init__(self, auto_prophet_model: Optional[AutoProphet] = None):
        self.model: AutoProphet = auto_prophet_model or AutoProphet(
            config=AutoProphetConfig()
        )

    def _reset(self) -> None:
        self.model.reset()

    def _train(self, points: List[Point]) -> None:
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            self.model.train(_create_train_data_for_merlion_models(points))

    def _forecast(self, n: int = 1) -> List[float]:
        timestamps = self.model.model.resample_time_stamps(n)
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            return _parse_forecast_for_merlion_models(
                self.model.forecast(time_stamps=timestamps)
            )


class BaseModelFactory:
    def __init__(self, models: Dict[str, Type[Model]]):
        self.models = models

    def create_model(
        self, name: str, model_kwargs: Optional[Dict[str, Any]] = None
    ) -> Model:
        return self.models[name](**(model_kwargs or {}))

    def create_all_models(self) -> List[Model]:
        return [model_cls() for model_cls in self.models.values()]


class DefaultModelFactory(BaseModelFactory):
    def __init__(
        self,
        extra_models: Optional[Dict[str, Type[Model]]] = None,
    ):
        super().__init__(
            models={
                SimpleModel.__name__: SimpleModel,
                AutoSarimaModel.__name__: AutoSarimaModel,
                AutoProphetModel.__name__: AutoProphetModel,
                **(extra_models or {}),
            }
        )


class SimpleModelFactory(BaseModelFactory):
    def __init__(self) -> None:
        super().__init__(models={SimpleModel.__name__: SimpleModel})
