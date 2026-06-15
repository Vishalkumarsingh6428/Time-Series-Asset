"""
models.py — TheilSen, XGBoost training and evaluation.
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import TheilSenRegressor
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor
import joblib


def train_theilsen(X_train, y_train):
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("reg", TheilSenRegressor(random_state=42, max_iter=500))
    ])
    model.fit(X_train, y_train)
    return model


def train_xgboost_log_return(X_train, y_log_return_train):
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X_train)
    model = XGBRegressor(
        n_estimators=500,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        random_state=42,
    )
    model.fit(X_sc, y_log_return_train, verbose=False)
    return model, scaler


def evaluate_pipeline(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    r2   = float(r2_score(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    print(f"  R²   : {r2:.4f}")
    print(f"  RMSE : {rmse:.4f}")
    return {"r2": r2, "rmse": rmse, "y_pred": y_pred}


def save_model(model, path: str):
    joblib.dump(model, path)
    print(f"Saved → {path}")


def load_model(path: str):
    return joblib.load(path)