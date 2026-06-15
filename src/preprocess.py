"""
preprocess.py — Data loading, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np


def load_data(filepath: str) -> pd.DataFrame:
    df = pd.read_csv(filepath, parse_dates=["Date"], index_col="Date")
    df.sort_index(inplace=True)
    return df


def remove_outliers(df: pd.DataFrame, col: str, z_thresh: float = 3.0) -> pd.DataFrame:
    z = (df[col] - df[col].mean()) / df[col].std()
    return df[np.abs(z) < z_thresh].copy()


def add_lag_features(df: pd.DataFrame, col: str, lags: list) -> pd.DataFrame:
    for lag in lags:
        df[f"{col}_lag_{lag}"] = df[col].shift(lag)
    return df


def add_rolling_features(df: pd.DataFrame, col: str, windows: list) -> pd.DataFrame:
    for w in windows:
        df[f"{col}_roll_mean_{w}"] = df[col].rolling(w).mean()
        df[f"{col}_roll_std_{w}"]  = df[col].rolling(w).std()
    return df


def preprocess(filepath: str, target_col: str = "Close") -> pd.DataFrame:
    df = load_data(filepath)

    # Drop leaky column — Adj Close is nearly identical to Close
    drop_cols = [c for c in ["Adj Close", "Adj_Close"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)

    df = remove_outliers(df, target_col)
    df = add_lag_features(df, target_col, lags=[1, 3, 7])
    df = add_rolling_features(df, target_col, windows=[7, 14, 30])

    # Stationary / ratio features — help XGBoost generalize across price levels
    df["pct_change_1"]      = df[target_col].pct_change(1)
    df["pct_change_5"]      = df[target_col].pct_change(5)
    df["high_low_ratio"]    = df["High"] / df["Low"]
    df["close_open_ratio"]  = df[target_col] / df["Open"]

    df.dropna(inplace=True)
    return df


if __name__ == "__main__":
    df = preprocess("../data/raw/stocks/AAPL.csv", target_col="Close")
    df.to_csv("../data/processed/asset_prices_processed.csv")
    print(f"Processed shape: {df.shape}")
    print(df.columns.tolist())