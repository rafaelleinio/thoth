import numpy as np


def ape(true_value: float, predicted_value: float) -> float:
    return float(min(np.abs(true_value - predicted_value) / true_value, 1.0))
