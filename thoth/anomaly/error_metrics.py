from abc import ABC, abstractmethod

import numpy as np


class ErrorMetric(ABC):
    @abstractmethod
    def calculate(self, true_value: float, predicted_value: float) -> float:
        """."""

    def calculate_np_array(
        self, true_values: np.ndarray, predicted_values: np.ndarray
    ) -> np.ndarray:
        vec_calculate = np.vectorize(self.calculate)
        return vec_calculate(true_values, predicted_values)


class APEMetric(ErrorMetric):
    def __init__(self, safe_denominator=True):
        self.safe_denominator = safe_denominator

    def calculate(self, true_value: float, predicted_value: float) -> float:
        return min(np.abs(true_value - predicted_value) / true_value, 1.0)
