# nifty-stock-screener-ml

A machine learning pipeline that predicts which Nifty 50 stocks are likely
to outperform the index over the next 30 trading days, with an interactive
risk analytics dashboard.

## Business question

**Which Nifty 50 stocks beat the index over the next 30 days — and what drives it?**

This is framed as a binary classification problem: label = 1 if the stock's
30-day forward return exceeds the Nifty 50's return over the same window.

## Key findings

- XGBoost achieved **67% accuracy** on holdout data (vs 50% random baseline)
- **P/E ratio, 30-day momentum, and RSI** were the strongest predictors
- IT and FMCG sectors showed higher predictability than cyclicals
- Simulated long-only portfolio (top 5 signals monthly): **Sharpe 1.34**, max drawdown **-8.2%**

## Features engineered

| Feature | Type | Finance rationale |
|---|---|---|
| RSI (14d) | Technical | Momentum oscillator |
| MACD | Technical | Trend + momentum |
| 30/60d return | Technical | Price momentum |
| Distance from SMA-20 | Technical | Mean reversion signal |
| 52-week position | Technical | Breakout / support level |
| Volatility (20d ann.) | Risk | Downside risk proxy |
| Volume spike | Technical | Institutional activity |
| Bollinger band width | Technical | Volatility squeeze |
| P/E ratio | Fundamental | Valuation |
| Beta | Fundamental | Systematic risk |

## Risk metrics (portfolio simulation)

| Metric | Value |
|---|---|
| Sharpe ratio | 1.34 |
| Sortino ratio | 1.61 |
| Max drawdown | -8.2% |
| VaR (95%) | -2.1% |
| CVaR (95%) | -3.0% |
| Win rate | 54.3% |

## Tech stack

`Python` · `pandas` · `yfinance` · `XGBoost` · `scikit-learn` · `Streamlit` · `matplotlib`

## Project structure

```
nifty-stock-screener-ml/
├── src/
│   ├── data_loader.py      # yfinance download + fundamentals
│   ├── features.py         # Technical + fundamental feature engineering
│   ├── model.py            # XGBoost + RF training with TimeSeriesSplit
│   └── risk_metrics.py     # Sharpe, VaR, drawdown, CVaR
├── dashboard/
│   └── app.py              # Streamlit interactive dashboard
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_eng.ipynb
│   └── 03_model.ipynb
└── requirements.txt
```

## How to run

```bash
# Install dependencies
pip install -r requirements.txt

# Download data
python src/data_loader.py

# Launch dashboard
streamlit run dashboard/app.py
```

## Author

**Dheeraj Bandam** — MBA (Business Analytics), IIM Amritsar  
[LinkedIn](https://www.linkedin.com/in/dheeraj-bandam-73386b24a/) · [Email](mailto:dheerajbandam@gmail.com)
