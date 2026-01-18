import pytest
from src.drift.drift_detector import detect_drift
import pandas as pd
import numpy as np
from src.config import NUMERIC_FEATURES  # FIXED: Import real features

def test_drift():
    # FIXED: Use real feature name from config (e.g., 'MonthlyRevenue')
    feature = NUMERIC_FEATURES[0]  # First numeric feature
    # Dummy data with shift
    baseline = pd.DataFrame({feature: np.random.randn(1000)})
    current = pd.DataFrame({feature: np.random.randn(1000) + 1})  # Shifted for drift
    drifts, has_drift = detect_drift(baseline, current)
    assert has_drift == True  # Should detect shift
    print("Drift test passed!")