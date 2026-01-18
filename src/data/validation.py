import pandas as pd
from great_expectations.dataset import PandasDataset  # FIXED: Correct import path
from src.config import NUMERIC_FEATURES  # FIXED: Import NUMERIC_FEATURES

def validate_schema(df):
    """Basic data quality checks."""
    assert 'Churn' in df.columns, "Missing target column"
    assert df['Churn'].mean() > 0.1, "Churn rate too low (<10%)"
    assert not df[NUMERIC_FEATURES].isnull().any().any(), "Missing numeric values"
    assert df[NUMERIC_FEATURES].nunique().min() > 1, "Constant numeric features"
    print("✅ Data schema validated")
    return df

# Advanced (optional)
def run_ge_validation(df):
    """Great Expectations for advanced checks."""
    ge_df = PandasDataset(df)
    ge_df.expect_column_values_to_not_be_null('Churn')
    ge_df.expect_column_mean_to_be_between('MonthlyRevenue', -3, 3)  # Scaled data
    results = ge_df.validate()
    print(f"GE Validation: {len(results.successes)} passed")
    return results

if __name__ == "__main__":
    # Test with dummy data
    dummy_df = pd.DataFrame({
        'Churn': [0, 1, 0, 1],
        'MonthlyRevenue': [-1.0, 2.0, 0.5, -0.5]
    })
    validated_df = validate_schema(dummy_df)
    print("Test passed!")