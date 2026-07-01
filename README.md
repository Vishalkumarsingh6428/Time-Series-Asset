# Time Series Asset Price Forecasting

A machine learning pipeline for forecasting next-day AAPL closing prices, comparing a robust linear model (TheilSen), a classical statistical model (ARIMA), and a gradient-boosted tree model (XGBoost) against a naive persistence baseline.

## Project Motivation

Daily equity prices are close to a random walk: the best naive guess for tomorrow's close is often just today's close. This project is built around that fact rather than around it — every model here is benchmarked against a **naive persistence baseline** (predict `Close_t = Close_{t-1}`), because raw R² on price *level* is a well-known misleading metric for near-random-walk series. A model that changes nothing from day to day can still score R² > 0.99 on price level simply because daily moves are small relative to the price range over 14 years. The metric that actually reveals whether a model has learned something is **whether it beats the naive baseline**, and ideally, its accuracy in *return* space rather than price-level space.

## Dataset

| Property | Details |
|---|---|
| Stock | AAPL |
| Raw rows | 9,909 |
| Rows after preprocessing | 9,879 |
| Period | 2010–2024 |
| Raw features | Open, High, Low, Close, Volume |
| Engineered features | Lag features (1, 3, 7 days), rolling mean/std (7, 14, 30 days), pct-change (1, 5 days), high/low ratio, close/open ratio |

**Data integrity notes:**
- 7 likely bad ticks were detected in `Close` via a rolling (not expanding) return-based z-score filter, and repaired through time-based interpolation rather than row deletion — this avoids creating gaps in the calendar index that would silently corrupt lag/rolling features.
- `Adj Close` is dropped, since it is derived almost entirely from `Close` and would otherwise leak the target.

## Methodology: Preventing Look-Ahead Leakage

Time-series forecasting is unusually easy to get wrong via subtle information leakage — a feature that technically wasn't dropped, but was computed such that it "knows" values from the same day (or later) as the target. This pipeline enforces strict causality:

- **All rolling statistics** (`rolling mean`, `rolling std`) are computed on a series shifted by one day, so the window for day `D` never includes day `D`'s own value.
- **All ratio and percent-change features** (`pct_change_1`, `pct_change_5`, `high_low_ratio`, `close_open_ratio`) are shifted by one day, so they describe the *previous* day's dynamics, not the current day's.
- **Same-day Open/High/Low/Volume are shifted by one day.** In practice, a day's High/Low/Volume only become fully known at the same time as its Close — using them concurrently to predict Close is a leak, not a legitimate feature.
- **Outlier detection uses a rolling window over returns**, not a global (whole-series) or expanding-from-inception mean/std over price level. Price is non-stationary (AAPL rose from ~$6 to ~$250 over the sample period), so any filter based on distance from a fixed or slowly-adapting mean will misclassify normal price growth as anomalous and silently delete valid rows.

Every feature at row `D` is therefore computable using only information available through `D-1`, and the target at row `D` (`Close_D`) is the one-step-ahead quantity actually being forecast.

## Results

| Model | R² (price level) | RMSE (price level) |
|---|---|---|
| Naive persistence baseline | 0.99745 | 2.913 |
| TheilSen | 0.99743 | 2.925 |
| ARIMA(5,1,0) | 0.99743 | 2.926 |
| XGBoost (log-return) | -3.684 | 124.889 |

**Key finding: no model beats the naive persistence baseline.** TheilSen and ARIMA land within a rounding error of simply predicting "tomorrow = today," which — given how leakage-free the pipeline now is — is the expected, honest result for a liquid large-cap equity's daily closing price. Their high R² is an artifact of the metric (price level barely moves day-to-day relative to its 14-year range), not evidence of predictive skill.

XGBoost is trained to predict next-day *log return* rather than price directly, and its price forecast is reconstructed by chaining predicted returns forward from a seed price. This reconstruction is sequential over ~1,976 test-set steps, so small per-step return errors compound multiplicatively into large price-level drift — which is why its price-level R² looks catastrophic even though its underlying return predictions may be more informative than the price-level metric suggests. A fairer comparison of all four models would be conducted directly in return space (see **Future Work**).

## Models Used

- **TheilSen Regressor** — A robust linear regression method resistant to outliers, using lagged and rolling statistical features of price.
- **ARIMA(5,1,0)** — A classical autoregressive model, refit at every step on an expanding history for genuine rolling one-step-ahead forecasts.
- **XGBoost** — Gradient-boosted decision trees trained to predict next-day log return (rather than raw price), with predictions reconstructed into price levels.

## Project Structure

```
ts_project/
├── data/
│   ├── raw/                # Original unprocessed data
│   └── processed/          # Cleaned & feature-engineered data
├── docs/                   # Additional documentation
├── models/                 # Saved .pkl model files
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── results/
│   ├── plots/              # Prediction & residual charts
│   └── metrics/            # metrics.json
├── src/
│   ├── preprocess.py       # Data loading & feature engineering
│   ├── models.py           # TheilSen & XGBoost training
│   ├── visualize.py        # Plotting utilities
│   └── pipeline.py         # End-to-end run script
├── .gitignore
├── requirements.txt
└── README.md
```

## Setup

```bash
pip install -r requirements.txt
```

## Usage

```bash
cd src
python pipeline.py
```

This runs the full pipeline: preprocessing, TheilSen training, rolling ARIMA forecasting, XGBoost training, naive-baseline evaluation, and metrics/plot generation. Note that the rolling ARIMA step refits the model at every test-set step and can take several minutes on the full dataset.

## Future Work

- **Evaluate all models in return space**, not just price level, to get a metric that isn't dominated by the "prices barely move day-to-day" effect. A model can legitimately fail to beat the naive baseline on price-level R² while still having learned real directional signal in returns.
- **Add a directional accuracy metric** (percentage of days where the predicted direction of movement matches the actual direction) — often more meaningful for trading-relevant evaluation than R²/RMSE on price.
- **Extend features beyond price-derived signals** — e.g., volatility indices, sector/index co-movement, macroeconomic indicators — since price-derived-only features on a near-random-walk asset are unlikely to yield much further improvement.
- **Walk-forward / expanding-window cross-validation** rather than a single train/test split, to check stability of results across different market regimes (e.g., 2010–2015 vs. 2020–2024).

## Author

Vishal Kumar Singh
IIT Madras
