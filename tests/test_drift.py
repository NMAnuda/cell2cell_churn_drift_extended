import pandas as pd
import numpy as np

from src.drift.drift_detector import detect_drift
from src.config import NUMERIC_FEATURES

def test_detect_drift_positive():
    np.random.seed(42)
    feature = NUMERIC_FEATURES[0]

    baseline = pd.DataFrame({feature: np.random.randn(2000)})
    current  = pd.DataFrame({feature: np.random.randn(2000) + 1.0})  # clear mean shift

    drifts, overall = detect_drift(baseline, current)

    assert feature in drifts
    assert drifts[feature]["drift_detected"] is True
    assert overall is True


def test_detect_drift_negative():
    np.random.seed(42)
    feature = NUMERIC_FEATURES[0]

    baseline = pd.DataFrame({feature: np.random.randn(2000)})
    current  = pd.DataFrame({feature: np.random.randn(2000)})  # same distribution

    drifts, overall = detect_drift(baseline, current)

    assert feature in drifts
    assert drifts[feature]["drift_detected"] is False
    assert overall is False
