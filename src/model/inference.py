import joblib
import pandas as pd
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES

def predict_churn(model_path, new_data_df, threshold=0.5):
    """Predict churn on new data."""
    model = joblib.load(model_path)
    X_new = new_data_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    probs = model.predict_proba(X_new)[:, 1]
    preds = (probs >= threshold).astype(int)
    return probs.tolist(), preds.tolist()

if __name__ == "__main__":
    # Example
    new_df = pd.DataFrame({'MonthlyRevenue': [50], 'CustomerCareCalls': [3]})
    probs, preds = predict_churn('models/baseline_model_improved.pkl', new_df)
    print(f"Probs: {probs}, Preds: {preds}")