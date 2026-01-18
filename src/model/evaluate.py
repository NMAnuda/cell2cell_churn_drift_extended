from sklearn.metrics import f1_score, roc_auc_score, classification_report, precision_recall_curve
import matplotlib.pyplot as plt
import joblib
import pandas as pd
from src.config import NUMERIC_FEATURES, CATEGORICAL_FEATURES
from xgboost import XGBClassifier  

def evaluate_model(model_path, test_path):
    """Evaluate loaded model on test data."""
  
    model = joblib.load(model_path)
    if hasattr(model, 'use_label_encoder'):
        model.set_params(use_label_encoder=False)  
    
    test_df = pd.read_csv(test_path)
    X_test = test_df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_test = test_df['Churn']

    probs = model.predict_proba(X_test)[:, 1]
    y_pred = (probs >= 0.5).astype(int)  # Default threshold

    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, probs)

    print(classification_report(y_test, y_pred))
    print(f"F1: {f1:.3f}, AUC: {auc:.3f}")

    # PR Curve
    precision, recall, _ = precision_recall_curve(y_test, probs)
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("PR Curve")
    plt.savefig("eval_pr_curve.png")
    plt.close()

    return f1, auc

if __name__ == "__main__":
    evaluate_model('models/retrained_model.pkl', 'data/batches/batch_2.csv')