from fastapi.testclient import TestClient

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.api.app import app

client = TestClient(app)

def test_predict():
    response = client.post("/predict", json={"features": {"MonthlyRevenue": 50.0}})
    assert response.status_code == 200
    print("API test passed!")