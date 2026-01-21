# src/api/app_full.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
import numpy as np
import json
import mlflow
from datetime import datetime
from src.model.inference import predict_churn
from src.model.evaluate import evaluate_model
from src.drift.drift_detector import detect_drift
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET, MLFLOW_TRACKING_URI

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

app = FastAPI(title="Churn MLOps API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------
# Pydantic schema for predict
# -----------------------------
class PredictRequest(BaseModel):
    features: dict

# -----------------------------
# Health Check
# -----------------------------
@app.get("/health")
def health():
    return {"status": "healthy"}

# -----------------------------
# Predict endpoint
# -----------------------------
@app.post("/predict")
def predict(request: PredictRequest):
    try:
        model_path = 'models/retrained_model.pkl'
        if not os.path.exists(model_path):
            raise HTTPException(status_code=404, detail="Model not found — run pipeline first")

        features = request.features
        full_features = {f: features.get(f, 0.0) for f in NUMERIC_FEATURES + CATEGORICAL_FEATURES}
        df = pd.DataFrame([full_features])
        probs, preds = predict_churn(model_path, df)
        return {"churn_prob": probs[0], "prediction": preds[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

# -----------------------------
# Latest metrics for dashboard
# -----------------------------
@app.get("/latest_metrics")
def latest_metrics():
    try:
        client = mlflow.tracking.MlflowClient()
        experiment = client.get_experiment_by_name("Churn Drift Pipeline")
        if not experiment:
            raise HTTPException(status_code=404, detail="Experiment not found — run pipeline first")

        runs = client.search_runs(experiment.experiment_id, order_by=["start_time DESC"], max_results=1)
        if not runs:
            raise HTTPException(status_code=404, detail="No MLflow runs found — run pipeline first")

        run = runs[0]
        metrics = run.data.metrics
        params = run.data.params

        response = {
            "baseline_f1": metrics.get("baseline_f1", 0.0),
            "retrained_f1": metrics.get("retrained_f1", 0.0),
            "delta_f1": metrics.get("delta_f1", 0.0),
            "baseline_auc": metrics.get("baseline_auc", 0.0),
            "retrained_auc": metrics.get("retrained_auc", 0.0),
            "drift_psi_avg": metrics.get("drift_psi_avg", 0.0),
            "drift_detected": metrics.get("drift_detected", False), 
            "retrained": params.get("retrained", "False")
        }
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"MLflow fetch error: {str(e)}")

# -----------------------------
# PR curves endpoint
# -----------------------------
@app.get("/pr_curves")
def pr_curves():
    try:
        baseline_img = "models/baseline_pr_curve.png"
        retrained_img = "models/retrained_pr_curve.png"
        for f in [baseline_img, retrained_img]:
            if not os.path.exists(f):
                raise HTTPException(status_code=404, detail=f"PR curve {f} missing — run pipeline first")

        return {
            "baseline": baseline_img,
            "retrained": retrained_img
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
def summarize_drift(drifts):
    psis = [float(v["psi"]) for v in drifts.values()]
    drifted = [bool(v["drift_detected"]) for v in drifts.values()]

    return {
        "drift_psi": float(np.mean(psis)) if psis else 0.0,
        "drift_detected": bool(any(drifted))
    }


# -----------------------------
# Drift check endpoint
# -----------------------------
@app.get("/drift")
def get_drift():
    baseline_df = pd.read_csv("data/batches/batch_0.csv")
    current_df = pd.read_csv("data/batches/batch_2.csv")
    print("dffffff")
    drifts, overall_drift = detect_drift(baseline_df, current_df)
    print("dffffffdd")
    summary = summarize_drift(drifts)

    return {
    "drifts": drifts,
    "drift_psi": summary["drift_psi"],
    "drift_detected": summary["drift_detected"]
}


# -----------------------------
# List available models
# -----------------------------
@app.get("/models")
def list_models():
    try:
        models_dir = "models"
        files = [f for f in os.listdir(models_dir) if f.endswith(".pkl")]
        model_info = []

        for m in files:
            if "baseline" in m.lower():
                meta_file = os.path.join(models_dir, "baseline_metadata.json")
                model_type = "baseline"
            elif "retrain" in m.lower():
                meta_file = os.path.join(models_dir, "retrained_metadata.json")
                model_type = "retrained"
            else:
                meta_file = None
                model_type = "unknown"

            if meta_file and os.path.exists(meta_file):
                with open(meta_file, "r") as f:
                    meta = json.load(f)
            else:
                meta = {}

            model_info.append({
                "name": m,
                "type": model_type,
                "f1": meta.get("f1"),
                "auc": meta.get("auc"),
                "threshold": meta.get("threshold"),
                "trained_on": meta.get("trained_on"),
                "timestamp": meta.get("timestamp")
            })

        return {"models": model_info}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -----------------------------
# Trigger pipeline run
# -----------------------------


@app.post("/run_pipeline")
def run_pipeline_endpoint():
    try:
        import sys
        import os
        sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        from scripts.run_pipeline import run_pipeline

        run_pipeline()
        return {"status": "Pipeline executed"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Pipeline error: {str(e)}")

# -----------------------------
# Trigger retrain manually
# -----------------------------
@app.post("/trigger_retrain")
def trigger_retrain_endpoint():
    try:
        import numpy as np
        import pandas as pd
        from src.pipeline.retrain_trigger import should_retrain
        from src.drift.drift_detector import detect_drift
        
        baseline_df = pd.read_csv("data/batches/batch_0.csv")
        current_df = pd.read_csv("data/batches/batch_2.csv")

        drifts, overall_drift = detect_drift(baseline_df, current_df)
        avg_psi = float(np.mean([d["psi"] for d in drifts.values()]))  # 👈 convert

        triggered = bool(should_retrain(avg_psi))  # 👈 convert

        return {
            "retrain_triggered": triggered,
            "avg_psi": avg_psi
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retrain error: {str(e)}")

# -----------------------------
# Run locally
# -----------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
