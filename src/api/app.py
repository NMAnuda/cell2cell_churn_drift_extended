from fastapi import FastAPI, HTTPException
from src.api.schemas import PredictRequest
from src.model.inference import predict_churn
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import os
from src.config import  CATEGORICAL_FEATURES, NUMERIC_FEATURES

app = FastAPI(title="Churn API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
def predict(request: PredictRequest):
    try:
        print("sssssssssssssssssssss")
        # FIXED: Check model exists
        model_path = 'models/retrained_model.pkl'
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found — run pipeline first")
        print("sssdssssssssssssssssss")
        # FIXED: Ensure all features (pad missing with 0)
        features = request.features
        full_features = {f: features.get(f, 0.0) for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES}  # From config
        df = pd.DataFrame([full_features])
        print("ssssssasssssssssssssss")
        probs, preds = predict_churn(model_path, df)
        
        return {"churn_prob": probs[0], "prediction": preds[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.get("/health")
def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)