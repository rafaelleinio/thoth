from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from neuralprophet import NeuralProphet
from prophet import Prophet

from thoth.anomaly.error_metrics import APEMetric, ErrorMetric


@dataclass
class TrainMetrics:
    train_time: int
    start_ts: datetime
    end_ts: datetime
    errors: Optional[np.ndarray] = None


class ForecastModel(ABC):
    def __init__(
        self, model_kwargs: Dict[str, Any], ts_column: str, feature_column: str
    ):
        self.model_kwargs = model_kwargs
        self.model = self._create_model()
        self.ts_column = ts_column
        self.feature_column = feature_column
        self.is_trained = False
        self.train_metrics: Optional[TrainMetrics] = None

    @abstractmethod
    def _create_model(self) -> Any:
        pass

    @abstractmethod
    def _train(self, train_pdf: pd.DataFrame):
        pass

    def _calculate_train_errors(
        self, train_pdf, error_metric: ErrorMetric
    ) -> np.ndarray:
        predicted = self.predict(periods=1, include_history=True)[self.feature_column]
        true = train_pdf[self.feature_column]
        # print(
        #     f"enter _calculate_train_errors, "
        #     f"max_predicted_ts = {self.predict(periods=1, include_history=True)[self.ts_column].max()}, "
        #     f"max_true_ts = {train_pdf[self.ts_column].max()}")
        return error_metric.calculate_np_array(
            true_values=true.to_numpy(), predicted_values=predicted.to_numpy()
        )

    def train(
        self, train_pdf: pd.DataFrame, error_metric: ErrorMetric = APEMetric()
    ) -> TrainMetrics:
        train_log_start_ts = datetime.now()
        self._train(train_pdf)
        train_log_end_ts = datetime.now()
        self.is_trained = True
        self.train_metrics = TrainMetrics(
            train_time=int(
                (train_log_end_ts - train_log_start_ts).total_seconds() * 1000
            ),
            start_ts=train_pdf[self.ts_column].min().to_pydatetime(),
            end_ts=train_pdf[self.ts_column].max().to_pydatetime(),
        )
        self.train_metrics.errors = self._calculate_train_errors(
            train_pdf, error_metric
        )
        return self.train_metrics

    @abstractmethod
    def _predict(self, ts_pdf: pd.DataFrame) -> pd.DateFrame:
        pass

    def _make_ts_dataframe(
        self, periods: int, time_granularity: timedelta, include_history: bool
    ) -> pd.DataFrame:
        start_ts = (
            self.train_metrics.start_ts
            if include_history
            else (self.train_metrics.end_ts + time_granularity)
        )
        periods = (
            (
                (self.train_metrics.end_ts - self.train_metrics.start_ts)
                / time_granularity
            )
            + periods
            if include_history
            else periods
        )
        dates = pd.date_range(start=start_ts, periods=periods)
        return pd.DataFrame({self.ts_column: dates})

    def predict(
        self,
        periods: int = 1,
        time_granularity: timedelta = timedelta(days=1),
        include_history=False,
    ) -> pd.DataFrame:
        if not self.is_trained:
            raise RuntimeError("Model needs to be trained before.")
        ts_pdf = self._make_ts_dataframe(
            periods=periods,
            time_granularity=time_granularity,
            include_history=include_history,
        )
        return self._predict(ts_pdf)


class ProphetForecastModel(ForecastModel):
    def __init__(
        self, model_kwargs: Dict[str, Any], ts_column: str, feature_column: str
    ):
        super().__init__(model_kwargs, ts_column, feature_column)

    def _create_model(self) -> Prophet:
        return Prophet(**self.model_kwargs)

    def _train(self, train_pdf: pd.DataFrame) -> None:
        model: Prophet = self.model
        model.fit(
            train_pdf[[self.ts_column, self.feature_column]].rename(
                columns={self.ts_column: "ds", self.feature_column: "y"}
            )
        )

    def _predict(self, ts_pdf: pd.DataFrame) -> pd.DataFrame:
        input_ts_pdf = ts_pdf[[self.ts_column]].rename(columns={self.ts_column: "ds"})
        predicted_pdf: pd.DataFrame = self.model.predict(input_ts_pdf)
        return predicted_pdf[["ds", "yhat"]].rename(
            columns={"ds": self.ts_column, "yhat": self.feature_column}
        )


class NeuralProphetModel(ForecastModel):
    def __init__(
        self, model_kwargs: Dict[str, Any], ts_column: str, feature_column: str
    ):
        super().__init__(model_kwargs, ts_column, feature_column)

    def _create_model(self) -> NeuralProphet:
        return NeuralProphet(**self.model_kwargs)

    def _train(self, train_pdf: pd.DataFrame):
        model: NeuralProphet = self.model
        model.fit(
            train_pdf[[self.ts_column, self.feature_column]].rename(
                columns={self.ts_column: "ds", self.feature_column: "y"}
            ),
            freq="D",
        )

    def _predict(self, ts_pdf: pd.DataFrame) -> pd.DateFrame:
        model: NeuralProphet = self.model
        input_ts_pdf = ts_pdf[[self.ts_column]].rename(columns={self.ts_column: "ds"})
        input_ts_pdf["y"] = np.nan
        input_ts_pdf = model.make_future_dataframe(
            input_ts_pdf, n_historic_predictions=len(input_ts_pdf["ds"])
        )[:-1]
        predicted_pdf: pd.DataFrame = model.predict(input_ts_pdf)
        return predicted_pdf[["ds", "yhat1"]].rename(
            columns={"ds": self.ts_column, "yhat1": self.feature_column}
        )


class ModelsConstructor:
    MODELS = {"prophet": ProphetForecastModel, "neural-prophet": NeuralProphetModel}

    def list_models(self) -> List[str]:
        return list(self.MODELS.keys())

    def create_model(
        self,
        name: str,
        model_kwargs: Dict[str, Any],
        ts_column: str,
        feature_column: str,
    ) -> ForecastModel:
        if name not in self.list_models():
            raise ValueError(
                f"Model not defined, list of available models: {self.list_models()}"
            )
        model_class = self.MODELS[name]
        return model_class(
            model_kwargs=model_kwargs,
            ts_column=ts_column,
            feature_column=feature_column,
        )
