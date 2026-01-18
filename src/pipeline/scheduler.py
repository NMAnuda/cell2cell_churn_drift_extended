import schedule
import time
from src.pipeline.orchestrator import run_core_pipeline

def scheduled_retrain():
    """Scheduled job."""
    run_core_pipeline()

# Example
schedule.every().day.at("02:00").do(scheduled_retrain)
while True:
    schedule.run_pending()
    time.sleep(60)