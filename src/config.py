import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Data Paths
DATA_PATH = os.path.join(BASE_DIR, "data")
RAW_DATA = os.path.join(DATA_PATH, "raw/cell2cellholdout.csv") 
PROCESSED_DATA = os.path.join(DATA_PATH, "processed/churn_processed.csv")
BATCHES_PATH = os.path.join(DATA_PATH, "batches")

# Drift Thresholds
PSI_THRESHOLD = 0.25
KS_THRESHOLD = 0.1

# Model Hyperparams
MODEL_PARAMS = {
    'n_estimators': 150,
    'max_depth': 5,
    'learning_rate': 0.01,
}

# Features (your existing)
NUMERIC_FEATURES = [
    'MonthlyRevenue', 'MonthlyMinutes', 'TotalRecurringCharge',
    'DirectorAssistedCalls', 'OverageMinutes', 'RoamingCalls',
    'PercChangeMinutes', 'PercChangeRevenues',
    'CustomerCareCalls', 'MonthsInService', 'HandsetPrice', 'DroppedCalls'
]
CATEGORICAL_FEATURES = ['IncomeGroup', 'OwnsMotorcycle', 'TruckOwner', 'Homeownership']
TARGET = 'Churn'

# AWS (your existing)
S3_BUCKET = 'your-churn-drift-bucket'
S3_RAW_PREFIX = 'raw/'
S3_LOGS_PREFIX = 'inference-logs/'
REGION = 'us-east-1'

# AWS Toggle
USE_AWS = False  # False for local, True for real AWS + .env
N_BATCHES = 5 

# MLflow - FIXED: Use environment variable or absolute path
MLFLOW_DIR = os.path.join(BASE_DIR, "mlflow_data")
os.makedirs(MLFLOW_DIR, exist_ok=True)

# Use env var if in Docker, else local path
MLFLOW_TRACKING_URI = os.getenv(
    'MLFLOW_TRACKING_URI', 
    f"sqlite:///{os.path.join(MLFLOW_DIR, 'mlflow.db')}"
)