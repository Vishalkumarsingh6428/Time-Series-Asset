# Time Series Asset Price Forecasting

A machine learning project for predicting AAPL stock prices using time-series forecasting models trained on 9,655 rows of data from 2010–2024.

## Dataset
| Property | Details |
|----------|---------|
| Stock | AAPL |
| Rows | 9,655 |
| Period | 2010–2024 |
| Features | Open, High, Low, Close, Volume |

## Results
| Model | R² | RMSE |
|-------|----|------|
| TheilSen | 0.9997 | 0.7768 |
| XGBoost | -2.0253 | 72.7293 |

## Models Used
- **TheilSen** — Robust linear regression resistant to outliers, with lag & rolling features
- **XGBoost** — Gradient-boosted trees with lag/rolling features

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
---

# Author

**Vishal Kumar Singh**

**IIT Madras**
---
