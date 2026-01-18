from src.drift.drift_detector import detect_drift
from src.drift.alerts import send_drift_alert
from src.config import PSI_THRESHOLD
import numpy as np
import pandas as pd

def run_drift_monitor(baseline_path, current_path):
    """Run drift check and alert if needed."""
    baseline_df = pd.read_csv(baseline_path)
    current_df = pd.read_csv(current_path)
    drifts, has_drift = detect_drift(baseline_df, current_df)
    psi_avg = np.mean([d['psi'] for d in drifts.values()])
    if has_drift and psi_avg > PSI_THRESHOLD:
        send_drift_alert('PSI Drift', psi_avg)
    print(f"Drift monitor: {has_drift} (avg PSI {psi_avg:.3f})")

if __name__ == "__main__":
    run_drift_monitor('data/batches/batch_0.csv', 'data/batches/batch_2.csv')