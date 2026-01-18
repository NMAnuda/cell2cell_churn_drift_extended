import pytest

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.model.train import train_model

def test_train():
    results = train_model('data/batches/batch_0.csv', 'test')
    assert results['f1'] > 0.3
    print("Model train test passed!")