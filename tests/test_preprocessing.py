import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data.preprocessing import load_and_preprocess
from src.config import RAW_DATA

def test_preprocessing():
    df, scaler, le = load_and_preprocess()
    assert 'Churn' in df.columns
    assert df.shape[0] > 0
    print("Preprocessing test passed!")