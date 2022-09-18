import pytest

from thoth.anomaly.error_metrics import ape


def test_ape_error():
    with pytest.raises(ValueError):
        ape(true_value=0, predicted_value=0.001)
