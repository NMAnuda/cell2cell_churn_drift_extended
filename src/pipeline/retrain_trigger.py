from src.config import PSI_THRESHOLD

def should_retrain(drift_psi_avg):
    """Simple logic: Retrain if PSI > threshold."""
    return drift_psi_avg > PSI_THRESHOLD

if __name__ == "__main__":
    print(should_retrain(1.398))  # True