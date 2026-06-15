# Model Notes

## ARIMA
- Order (p,d,q) selected via AIC/BIC grid search
- Used as statistical baseline for univariate price series
- Reference: `statsmodels.tsa.arima.model.ARIMA`

## TheilSen
- Robust to outliers; estimates median slope across all pairs
- Suitable when price data contains noise spikes
- Hyperparameters: `max_iter=500`, `random_state=42`

## XGBoost
- Input: lag features (1,3,7 days) + rolling mean/std (7,14,30 days)
- `n_estimators=500`, `learning_rate=0.05`, `max_depth=6`
- Best model: **R² = 0.9979**, **RMSE = 6.0203**

## Feature Engineering
| Feature | Description |
|---------|-------------|
| price_lag_1 | Price 1 day ago |
| price_lag_3 | Price 3 days ago |
| price_lag_7 | Price 7 days ago |
| price_roll_mean_7 | 7-day rolling average |
| price_roll_std_7  | 7-day rolling std dev |
| price_roll_mean_14 | 14-day rolling average |
| price_roll_mean_30 | 30-day rolling average |
