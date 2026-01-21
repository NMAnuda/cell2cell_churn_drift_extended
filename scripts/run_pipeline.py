"""
Master Orchestrator: Wraps MLflow, calls core pipeline, handles scheduling.
- Logs all parameters/metrics/models dynamically from inner returns.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import logging
import mlflow
from datetime import datetime
import schedule
import time
import traceback
import joblib  # For model load

from src.pipeline.orchestrator import run_core_pipeline  # Inner orchestrator
from src.config import MLFLOW_TRACKING_URI, PSI_THRESHOLD ,N_BATCHES

# MLflow Setup
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Churn Drift Pipeline")

# Logging Setup
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def run_pipeline():
    logger.info("Pipeline started...")

    try:
        with mlflow.start_run(run_name=f"churn_pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}") as run:
            logger.info(f"MLflow Run ID: {run.info.run_id}")

            # Log parameters in outer run (dynamic where possible)
            mlflow.log_param("n_batches", N_BATCHES)
            mlflow.log_param("psi_threshold", PSI_THRESHOLD)
            mlflow.log_param("target", "Churn")
            logger.info("Parameters logged")

            # Call inner orchestrator for core steps
            f1_base, f1_re, delta_f1, has_drift, drift_psi_avg, threshold_base, threshold_re, auc_base, auc_re = run_core_pipeline()

   
            mlflow.log_metric("baseline_f1", f1_base)
            mlflow.log_metric("baseline_auc", auc_base)
            mlflow.log_metric("drift_detected", int(has_drift))
            mlflow.log_metric("drift_psi_avg", drift_psi_avg)
            mlflow.log_metric("retrained_f1", f1_re)
            mlflow.log_metric("retrained_auc", auc_re)
            mlflow.log_metric("delta_f1", delta_f1)
            mlflow.log_metric("final_f1", f1_re)
            mlflow.log_metric("final_auc", auc_re)
            mlflow.log_metric("improvement", delta_f1)
            logger.info("All metrics logged")


            mlflow.log_param("baseline_threshold", threshold_base)
            mlflow.log_param("drift_threshold", PSI_THRESHOLD)
            mlflow.log_param("retrained_threshold", threshold_re)
            mlflow.log_param("retrained", True if has_drift else False)
            logger.info("All parameters logged")

            from mlflow.xgboost import log_model
            baseline_model = joblib.load('models/baseline_model_improved.pkl')
            retrained_model = joblib.load('models/retrained_model.pkl')
            log_model(baseline_model, "baseline_model")
            log_model(retrained_model, "retrained_model")
            logger.info("Models logged to MLflow")

            # Summary print
            logger.info("=" * 60)
            logger.info("PIPELINE SUMMARY")
            logger.info("=" * 60)
            logger.info(f"Baseline F1: {f1_base:.3f}, AUC: {auc_base:.3f}, Threshold: {threshold_base:.3f}")
            logger.info(f"Retrained F1: {f1_re:.3f}, AUC: {auc_re:.3f}, Threshold: {threshold_re:.3f}")
            logger.info(f"Improvement: {delta_f1:.3f}")
            logger.info(f"Drift Detected: {has_drift} (PSI: {drift_psi_avg:.3f})")
            logger.info(f"MLflow Run ID: {run.info.run_id}")
            logger.info("=" * 60)

        logger.info("Pipeline finished successfully. MLflow run closed.")
        print("\nAll metrics logged to MLflow!")
        print(f"View results: mlflow ui --port 5000")
        print(f"Navigate to: http://localhost:5000")

    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        traceback.print_exc()
        raise

# Scheduler
def job():
    try:
        run_pipeline()
    except Exception as e:
        logger.error(f"Scheduled job failed: {e}")

if __name__ == "__main__":
    print("Running pipeline immediately...")
    job()

    print("Scheduler started (daily at 02:00)")
    print("Press Ctrl+C to stop")

    schedule.every().day.at("02:00").do(job)

    while True:
        schedule.run_pending()
        time.sleep(60)