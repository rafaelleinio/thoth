import numpy as np


def ape(true_value: float, predicted_value: float) -> float:
    if true_value < 10**-4:
        raise ValueError(
            f"Trying to calculate APE for a true_value too close to zero which is "
            f"undefined and thus not supported. true_value={true_value} and "
            f"predicted_value={predicted_value}"
        )
    result = float(min(np.abs(true_value - predicted_value) / true_value, 1.0))
    return result
