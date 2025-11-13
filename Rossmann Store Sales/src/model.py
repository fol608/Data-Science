import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

def train_lgbm():
    df = pd.read_parquet("../data/processed/train_features.parquet")

    val_df = df[(df["Year"] == 2015) & (df["Month"] >= 2)]
    train_df = df[(df["Year"] != 2015) | (df["Month"] == 1) & (df["Year"] == 2015)]

    features = [f for f in df.columns if f != 'Sales']
    X_train = train_df[features]
    y_train = train_df['Sales']
    X_val = val_df[features]
    y_val = val_df['Sales']

    lgbm = lgb.LGBMRegressor(
        boosting_type='gbdt',     
        objective='regression',   
        metric='rmse',           
        n_estimators=2000,       
        learning_rate=0.05,       
        num_leaves=31,           
        max_depth=-1,             
        subsample=0.8,            
        colsample_bytree=0.8,     
        min_child_samples=20,     
        reg_alpha=0.1,            
        reg_lambda=0.1,           
        random_state=42,
        n_jobs=-1,
        device='gpu'
    )

    lgbm.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        eval_metric='rmse',
    )

    print("📊 Evaluation metrics:")

    y_pred = lgbm.predict(X_val)
    print('MAE:', mean_absolute_error(y_val, y_pred))
    print('RMSE:', mean_squared_error(y_val, y_pred))
    print('R2:', r2_score(y_val, y_pred))

    valid_predictions = pd.DataFrame({
        "Id": X_val.index,
        "Sales_Predicted": y_pred
    })
    valid_predictions.to_csv("../data/predictions/valid_predictions.csv", index=False)

    print("\n⚙️  Model saved succesfully")

    joblib.dump(lgbm, '../models/lgbm_model.joblib')