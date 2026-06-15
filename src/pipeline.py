"""
pipeline.py — End-to-end training pipeline
"""

import json, sys, os
import numpy as np
import pandas as pd
from sklearn.metrics import r2_score, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, os.path.dirname(__file__))

from preprocess import preprocess
from models import train_theilsen, train_xgboost_log_return, evaluate_pipeline, save_model
from visualize import plot_actual_vs_predicted, plot_residuals, plot_feature_importance

TARGET    = "Close"
DATA_PATH = "../data/raw/stocks/AAPL.csv"
TEST_SIZE = 0.2


def run():
    # 1. Preprocess
    print("=" * 40)
    print("STEP 1 — Preprocessing")
    print("=" * 40)
    df = preprocess(DATA_PATH, target_col=TARGET)
    df.to_csv("../data/processed/asset_prices_processed.csv")
    print(f"Processed shape: {df.shape}")

    feature_cols = [c for c in df.columns if c != TARGET]

    df["log_return"] = np.log(df[TARGET] / df[TARGET].shift(1))
    df["next_log_return"] = df["log_return"].shift(-1)
    df.dropna(inplace=True)

    X = df[feature_cols].values
    y_price = df[TARGET].values
    y_log_return = df["next_log_return"].values

    split = int(len(X) * (1 - TEST_SIZE))
    X_train, X_test   = X[:split], X[split:]
    yp_train, yp_test = y_price[:split], y_price[split:]
    ylr_train         = y_log_return[:split]

    # 2. TheilSen
    print("\n" + "=" * 40)
    print("STEP 2 — TheilSen Regressor")
    print("=" * 40)
    ts_model   = train_theilsen(X_train, yp_train)
    ts_metrics = evaluate_pipeline(ts_model, X_test, yp_test)
    save_model(ts_model, "../models/theilsen.pkl")
    plot_actual_vs_predicted(yp_test, ts_metrics["y_pred"],
                             "TheilSen: Actual vs Predicted",
                             save_path="../results/plots/theilsen_pred.png")
    plot_residuals(yp_test, ts_metrics["y_pred"],
                   save_path="../results/plots/theilsen_residuals.png")

    # 3. ARIMA — rolling one-step-ahead forecast
    print("\n" + "=" * 40)
    print("STEP 3 — ARIMA (Rolling Forecast)")
    print("=" * 40)
    history = list(yp_train)
    arima_preds = []

    for i in range(len(yp_test)):
        model = ARIMA(history, order=(5, 1, 0))
        model_fit = model.fit()
        pred = model_fit.forecast(steps=1)[0]
        arima_preds.append(pred)
        history.append(yp_test[i])
        if i % 100 == 0:
            print(f"  Progress: {i}/{len(yp_test)}")

    arima_preds = np.array(arima_preds)
    arima_r2   = float(r2_score(yp_test, arima_preds))
    arima_rmse = float(np.sqrt(mean_squared_error(yp_test, arima_preds)))
    print(f"  R²   : {arima_r2:.4f}")
    print(f"  RMSE : {arima_rmse:.4f}")

    save_model(ARIMA(yp_train, order=(5, 1, 0)).fit(), "../models/arima.pkl")
    plot_actual_vs_predicted(yp_test, arima_preds,
                             "ARIMA: Actual vs Predicted",
                             save_path="../results/plots/arima_pred.png")
    plot_residuals(yp_test, arima_preds,
                   save_path="../results/plots/arima_residuals.png")

    # 4. XGBoost (log return → reconstruct prices)
    print("\n" + "=" * 40)
    print("STEP 4 — XGBoost Regressor")
    print("=" * 40)
    xgb_model, xgb_scaler = train_xgboost_log_return(X_train, ylr_train)

    pred_log_returns = xgb_model.predict(xgb_scaler.transform(X_test))
    pred_prices = [yp_test[0]]
    for lr in pred_log_returns:
        pred_prices.append(pred_prices[-1] * np.exp(lr))
    pred_prices = np.array(pred_prices[:-1])

    xgb_r2   = float(r2_score(yp_test, pred_prices))
    xgb_rmse = float(np.sqrt(mean_squared_error(yp_test, pred_prices)))
    print(f"  R²   : {xgb_r2:.4f}")
    print(f"  RMSE : {xgb_rmse:.4f}")

    save_model((xgb_model, xgb_scaler), "../models/xgboost.pkl")
    plot_actual_vs_predicted(yp_test, pred_prices,
                             "XGBoost: Actual vs Predicted",
                             save_path="../results/plots/xgboost_pred.png")
    plot_residuals(yp_test, pred_prices,
                   save_path="../results/plots/xgboost_residuals.png")
    plot_feature_importance(xgb_model, feature_cols,
                            save_path="../results/plots/xgboost_feature_importance.png")

    # 5. Save metrics
    print("\n" + "=" * 40)
    print("STEP 5 — Saving Metrics")
    print("=" * 40)
    metrics = {
        "theilsen": {"r2": ts_metrics["r2"], "rmse": ts_metrics["rmse"]},
        "arima":    {"r2": arima_r2, "rmse": arima_rmse},
        "xgboost":  {"r2": xgb_r2,  "rmse": xgb_rmse},
    }
    with open("../results/metrics/metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print("\nAll done!")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    run()