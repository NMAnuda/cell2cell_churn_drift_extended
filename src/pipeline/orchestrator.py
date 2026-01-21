

import logging
import pandas as pd
from datetime import datetime
import traceback
import numpy as np
from src.data.preprocessing import load_and_preprocess
from src.data.batch_generator import generate_batches
from src.model.train import train_model
from src.drift.drift_detector import detect_drift
from src.config import USE_AWS, BATCHES_PATH, NUMERIC_FEATURES, CATEGORICAL_FEATURES, TARGET

# Conditional AWS imports
if USE_AWS:
    from src.aws.s3_upload import upload_to_s3
    from src.aws.trigger_retrain import trigger_retrain
    from src.aws.deploy_model import deploy_model

logger = logging.getLogger(__name__)

def run_core_pipeline():
    logger.info("Inner orchestrator started...")

    try:
        # 1. Preprocess & Batches
        logger.info("Step 1: Preprocessing...")
        df, scaler, le = load_and_preprocess()
        batches = generate_batches(df, n_batches=5)
        for i, batch in enumerate(batches):
            batch.to_csv(f"{BATCHES_PATH}/batch_{i}.csv", index=False)

        logger.info(f"Batches generated: {len(batches)}")

        # AWS: Upload batches if on
        if USE_AWS:
            for i in range(5):
                upload_to_s3(f"{BATCHES_PATH}/batch_{i}.csv")

        # 2. Baseline Train
        logger.info("Step 2: Training baseline...")
        baseline_results = train_model(f"{BATCHES_PATH}/batch_0.csv", "baseline")
        f1_base = baseline_results["f1"]
        auc_base = baseline_results["auc"]  # FIXED: Capture AUC
        threshold_base = baseline_results["threshold"]  # FIXED: Capture threshold
        best_model = baseline_results["model"]

        logger.info(f"Baseline: F1={f1_base:.3f}, AUC={auc_base:.3f}")

        # AWS: Upload baseline if on
        if USE_AWS:
            upload_to_s3('models/baseline_model_improved.pkl')

        # 3. Drift Detection
        logger.info("Step 3: Detecting drift...")
        baseline_df = pd.read_csv(f"{BATCHES_PATH}/batch_0.csv")
        current_df = pd.read_csv(f"{BATCHES_PATH}/batch_2.csv")

        drifted_df = current_df.copy()
        drifted_df["CustomerCareCalls"] += np.abs(drifted_df["CustomerCareCalls"]) * 0.15

        drifts, has_drift = detect_drift(baseline_df, drifted_df)
        drift_psi_avg = np.mean([d["psi"] for d in drifts.values()])

        logger.info(f"Drift detected: {has_drift}, PSI avg: {drift_psi_avg:.3f}")

        # AWS: Trigger if on
        if USE_AWS and has_drift:
            trigger_retrain(drift_psi_avg)

        # 4. Retrain if Drift
        f1_re = f1_base
        auc_re = auc_base
        model_re = best_model
        delta_f1 = 0
        threshold_re = threshold_base

        if has_drift:
            logger.info("Step 4: Retraining...")
            recent_data = pd.concat([pd.read_csv(f"{BATCHES_PATH}/batch_{i}.csv") for i in range(1, 5)], ignore_index=True)
            recent_path = f"{BATCHES_PATH}/recent_concat.csv"
            recent_data.to_csv(recent_path, index=False)
            retrain_results = train_model(recent_path, "retrained")
            f1_re = retrain_results["f1"]
            auc_re = retrain_results["auc"]  
            threshold_re = retrain_results["threshold"]  
            model_re = retrain_results["model"]
            delta_f1 = f1_re - f1_base

            logger.info(f"Retrained: F1={f1_re:.3f}, AUC={auc_re:.3f}")

            # AWS: Deploy if on
            if USE_AWS:
                deploy_model('models/retrained_model.pkl')

        logger.info("Inner orchestrator finished...")


        return f1_base, f1_re, delta_f1, has_drift, drift_psi_avg, threshold_base, threshold_re, auc_base, auc_re

    except Exception as e:
        logger.error(f"Inner orchestrator failed: {e}")
        traceback.print_exc()
        raise
