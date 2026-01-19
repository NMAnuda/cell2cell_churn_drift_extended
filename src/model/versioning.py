import mlflow
import mlflow.xgboost
from src.config import MLFLOW_TRACKING_URI

def log_model_version(model, name, f1, auc, params):
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_name=name):
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("auc", auc)
        mlflow.log_params(params)
        mlflow.xgboost.log_model(model, "model")
        print(f" Model versioned: {name} (F1 {f1:.3f})")

if __name__ == "__main__":
    from src.model.train import train_model
    results = train_model('data/batches/batch_0.csv', 'test')
    log_model_version(results['model'], 'test-v1', results['f1'], results['auc'], results['metadata']['params'])