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

from thoth.anomaly.error_metrics import ape
from thoth.base import Point


class ForecastValueError(Exception):
    """Error about unexpected non-float values return from a model."""


class Model(ABC):
    """Base class defining expected interface for forecast models."""

    @abstractmethod
    def _train(self, points: List[Point]) -> None:
        """Child class must implement this operation."""

    @abstractmethod
    def _forecast(self, n: int = 1) -> List[float]:
        """Child class must implement this operation."""

    @abstractmethod
    def _reset(self) -> None:
        """Child class must implement this operation."""

    def reset(self) -> None:
        """Reset the state of the model to the original object."""
        return self._reset()

    def train(self, points: List[Point]) -> None:
        """Train the model with the given series."""
        self._train(points)

    def forecast(self, n: int = 1) -> List[float]:
        """Use the trained model to forecast the next n points."""
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
        """Use a given series to train the model and anomaly score the last point.

        Args:
            points: series to support the training and scoring.
                The last point of the series is used as base for the scoring as the
                true value (calculating the error from it), while the other points of
                the series are used to train the model.

        Returns:
            Tuple containing the predicted value and the scoring (error) respectively.

        """
        train_points = points[:-1]
        target_point = points[-1]
        self.train(train_points)
        [predicted_value] = self.forecast(n=1)
        return predicted_value, ape(
            true_value=target_point.value, predicted_value=predicted_value
        )


class SimpleModel(Model):
    """Model that predicts the next point of the series with an average agg window.

    Attributes:
        windows: what window lengths to try to fit to the series.
            The model tries all the defined windows length and chooses the one with
            the smallest mean error.

    """

    def __init__(self, windows: Optional[List[int]] = None):
        self.windows = windows or [3, 5, 7, 30]
        self._best_window: Optional[int] = None
        self._train_data_pdf: Optional[pd.DataFrame] = None
        self._skip_windows: List[int] = []

    def _reset(self) -> None:
        self._best_window = None
        self._train_data_pdf = None
        self.windows = self.windows + self._skip_windows
        self._skip_windows = []

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
        self._skip_windows = [w for w in sorted_windows if w >= train_length]
        self.windows = [w for w in self.windows if w not in self._skip_windows]

    def _train(self, points: List[Point]) -> None:
        self._check_series_length(points)
        self._train_data_pdf = pd.DataFrame(
            [dataclasses.asdict(point) for point in points]
        )

        transformed_pdf = self._add_windows_and_errors(input_pdf=self._train_data_pdf)
        error_columns = [
            column for column in transformed_pdf.columns if column.endswith("error")
        ]
        mean_errors = [transformed_pdf[column].mean() for column in error_columns]

        index_min = min(range(len(mean_errors)), key=mean_errors.__getitem__)
        self._best_window = self.windows[index_min]

    def _forecast(self, n: int = 1) -> List[float]:
        if n > 1:
            raise NotImplementedError(
                "This model only supports the forecast window of n=1"
            )
        if not self._best_window or self._train_data_pdf is None:
            raise RuntimeError("Model must be trained first.")

        train_df_with_empty_row_in_the_end_pdf = pd.concat(
            [
                self._train_data_pdf,
                pd.DataFrame(
                    [[np.NaN] * self._train_data_pdf.shape[1]],
                    columns=self._train_data_pdf.columns,
                ),
            ],
            ignore_index=True,
        )
        transformed_pdf = self._add_windows_and_errors(
            input_pdf=train_df_with_empty_row_in_the_end_pdf,
            windows=[self._best_window],
        )
        value: float = transformed_pdf[f"window_{self._best_window}"].iloc[-1]
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
    """AutoML version of the SARIMA forecast model.

    Merlion documentation:
    https://opensource.salesforce.com/Merlion/latest/examples/advanced/1_AutoSARIMA_forecasting_tutorial.html?highlight=autosarima

    """

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
    """AutoML version of the Prophet forecast model.

    Merlion documentation:
    https://opensource.salesforce.com/Merlion/latest/merlion.models.automl.html

    """

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
        with warnings.catch_warnings():
            warnings.simplefilter(action="ignore", category=FutureWarning)
            forecast = self.model.forecast(n)
            return _parse_forecast_for_merlion_models(forecast)


class BaseModelFactory:
    """Factory to create models based on a given config."""

    def __init__(self, models: Dict[str, Type[Model]]):
        self.models = models

    def create_model(self, name: str) -> Model:
        """Instantiate one of the defined models given its name."""
        return self.models[name]()

    def create_all_models(self) -> List[Model]:
        """Creates all the models defined in the factory."""
        return [model_cls() for model_cls in self.models.values()]


class DefaultModelFactory(BaseModelFactory):
    """Factory that creates all the implemented models in the module."""

    def __init__(self, extra_models: Optional[Dict[str, Type[Model]]] = None):
        super().__init__(
            models={
                SimpleModel.__name__: SimpleModel,
                # AutoSarimaModel.__name__: AutoSarimaModel,
                AutoProphetModel.__name__: AutoProphetModel,
                **(extra_models or {}),
            }
        )


class SimpleModelFactory(BaseModelFactory):
    """Factory creates only the simple model."""

    def __init__(self) -> None:
        super().__init__(models={SimpleModel.__name__: SimpleModel})
