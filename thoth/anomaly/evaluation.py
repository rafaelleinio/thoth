from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd

from thoth.anomaly.error_metrics import APEMetric, ErrorMetric
from thoth.anomaly.models import ForecastModel, ModelsConstructor


@dataclass
class ModelValidationSpec:
    name: str
    model_kwargs_list: List[Dict[str, Any]]

    def __repr__(self):
        return f"<{self.name} ModelValidationSpec>"

    def __eq__(self, other):
        if not isinstance(other, ModelValidationSpec):
            return False
        return other.name == self.name

    def __hash__(self):
        return hash(self.name)


@dataclass
class CrossValidationMetrics:
    name: str
    timestamps: np.ndarray
    predicted_values: np.ndarray
    true_values: np.ndarray
    errors: np.ndarray
    errors_avg: float
    errors_stddev: float
    errors_max: float
    errors_min: float
    model_evaluation_results: List[ModelEvaluationResult]

    def __lt__(self, other):
        if self.errors_avg is None:
            return False
        if other.errors_avg is None:
            return True
        return self.errors_avg < other.errors_avg


@dataclass
class ModelEvaluationResult:
    name: str
    model: ForecastModel
    timestamp: datetime
    model_kwargs: Dict[str, Any]
    error: float
    predicted_value: float
    true_value: float


@dataclass
class ModelValidationResult:
    model_kwargs: Dict[str, Any]
    error: float

    def __lt__(self, other):
        if self.error is None:
            return False
        if other.error is None:
            return True
        return self.error < other.error


class CrossValidation:
    def __init__(
        self,
        validation_specs: Set[ModelValidationSpec],
        ts_column: str,
        feature_column: str,
        start_proportion: float,
        slide: int = 1,
        time_granularity: timedelta = timedelta(days=1),
        error_measure: ErrorMetric = APEMetric(),
    ):
        self.validation_specs = validation_specs
        self.ts_column = ts_column
        self.feature_column = feature_column
        self.start_proportion = start_proportion
        self.slide = slide
        self.time_granularity = time_granularity
        self.error_measure = error_measure

    def _define_step_dates(self):
        pass

    def _create_model(self, name: str, model_kwargs: Dict[str, Any]) -> ForecastModel:
        return ModelsConstructor().create_model(
            name=name,
            model_kwargs=model_kwargs,
            ts_column=self.ts_column,
            feature_column=self.feature_column,
        )

    def _filter_pdf(
        self,
        data_pdf: pd.DataFrame,
        end_ts: datetime,
        start_ts: Optional[datetime] = None,
    ) -> pd.DataFrame:
        end_filter_series = data_pdf[self.ts_column] <= end_ts
        filter_series = (
            (data_pdf[self.ts_column] >= start_ts) & end_filter_series
            if start_ts
            else end_filter_series
        )
        return data_pdf[filter_series]

    def _calculate_error(self, predicted_value: float, true_value: float) -> float:
        return self.error_measure.calculate(
            true_value=true_value, predicted_value=predicted_value
        )

    def _validate_ts(
        self,
        data_pdf: pd.DataFrame,
        timestamp: datetime,
        validation_spec: ModelValidationSpec,
    ) -> List[ModelValidationResult]:
        validation_train_ts = timestamp - (self.time_granularity * 2)
        # print(f"Validating for ts = {validation_train_ts}")
        validation_train_pdf = self._filter_pdf(
            data_pdf=data_pdf, end_ts=validation_train_ts
        )
        # print(f"validation train max_ts = {validation_train_pdf['ts'].max().to_pydatetime()}, min_ts = {validation_train_pdf['ts'].min().to_pydatetime()}")

        validation_results = []
        for model_kwargs in validation_spec.model_kwargs_list:
            # train
            validation_model = self._create_model(validation_spec.name, model_kwargs)
            validation_model.train(validation_train_pdf)

            # validation
            predicted_value = validation_model.predict()[self.feature_column].iloc[0]
            validation_ts = validation_train_ts + self.time_granularity
            true_value = self._filter_pdf(
                data_pdf, start_ts=validation_ts, end_ts=validation_ts
            )[self.feature_column].iloc[0]
            error = self._calculate_error(
                predicted_value=predicted_value, true_value=true_value
            )

            validation_results.append(
                ModelValidationResult(error=error, model_kwargs=model_kwargs)
            )

        return validation_results

    def _evaluate_ts(
        self,
        data_pdf: pd.DataFrame,
        timestamp: datetime,
        validation_spec: ModelValidationSpec,
    ) -> ModelEvaluationResult:
        # print(f"evaluation start for ts = {timestamp}")
        # model selection
        validation_results = self._validate_ts(data_pdf, timestamp, validation_spec)
        best_validation_result = min(validation_results)
        # print(f"best validation result = {best_validation_result}, others = {validation_results}")
        evaluation_model = self._create_model(
            name=validation_spec.name, model_kwargs=best_validation_result.model_kwargs
        )

        # model evaluation
        evaluation_train_ts = timestamp - self.time_granularity
        # print(f"evaluation train ts = {evaluation_train_ts}")
        evaluation_train_pdf = self._filter_pdf(
            data_pdf=data_pdf, end_ts=evaluation_train_ts
        )
        evaluation_model.train(evaluation_train_pdf)

        predicted_value: float = evaluation_model.predict()[self.feature_column].iloc[0]
        true_value: float = self._filter_pdf(
            data_pdf, start_ts=timestamp, end_ts=timestamp
        )[self.feature_column].iloc[0]
        error = self._calculate_error(
            predicted_value=predicted_value, true_value=true_value
        )
        # print(f"evaluation predict={predicted_value}, true={true_value}, error={error}")
        # print("-----------------END-TS---------------------\n")
        return ModelEvaluationResult(
            name=validation_spec.name,
            timestamp=timestamp,
            model_kwargs=evaluation_model.model_kwargs,
            error=error,
            predicted_value=predicted_value,
            true_value=true_value,
            model=evaluation_model,
        )

    def _get_ts_range(self, data_pdf: pd.DataFrame) -> list[datetime]:
        max_ts = data_pdf["ts"].max().to_pydatetime()
        min_ts = data_pdf["ts"].min().to_pydatetime()
        n = int((max_ts - min_ts) / self.time_granularity) + 1
        start = int(self.start_proportion * n)
        return [
            min_ts + (i * self.time_granularity) for i in range(start, n, self.slide)
        ]

    def run(self, data_pdf: pd.DataFrame) -> list[CrossValidationMetrics]:
        # print("run start...")
        sorted_data_pdf = data_pdf.sort_values(by=[self.ts_column])
        cross_validated_models = []
        for validation_spec in self.validation_specs:
            # print(f"Cross Validation starting: {validation_spec}...")
            ts_range = self._get_ts_range(sorted_data_pdf)
            model_evaluation_results = [
                self._evaluate_ts(
                    data_pdf=sorted_data_pdf,
                    timestamp=ts,
                    validation_spec=validation_spec,
                )
                for ts in ts_range
            ]
            errors = [result.error for result in model_evaluation_results]
            predicted_values = [
                result.predicted_value for result in model_evaluation_results
            ]
            true_values = [result.true_value for result in model_evaluation_results]
            cross_validated_models.append(
                CrossValidationMetrics(
                    name=validation_spec.name,
                    timestamps=np.array(ts_range),
                    predicted_values=np.array(predicted_values),
                    true_values=np.array(true_values),
                    errors=np.array(errors),
                    errors_avg=float(np.mean(errors)),
                    errors_stddev=float(np.std(errors)),
                    errors_max=np.max(errors),
                    errors_min=np.min(errors),
                    model_evaluation_results=model_evaluation_results,
                )
            )
            # print("-----------------END-CROSS-VALIDATION-SPEC---------------------\n-------------------------\n\n")
        return cross_validated_models
