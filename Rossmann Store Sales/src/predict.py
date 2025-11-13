import joblib
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def test_model():
    df = pd.read_parquet("../data/processed/test_features.parquet")

    features = [f for f in df.columns if f != 'Sales']
    X_val = df[features]

    lgbm = joblib.load("../models/lgbm_model.joblib")

    y_pred = lgbm.predict(X_val)

    valid_predictions = pd.DataFrame({
        "Id": X_val.index,
        "Sales_Predicted": y_pred
    })

    valid_predictions.to_csv("../data/predictions/submission.csv", index=False)

    print("\nModel was tested succesfully")