"""
app.py — Streamlit dashboard
-----------------------------
Run with: streamlit run dashboard/app.py

This is the recruiter-facing layer. Shows:
  - Stock screener with ML-predicted signals
  - Feature importance (what drives the prediction)
  - Risk metrics for the selected stock
  - Interactive charts
"""

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
import sys, os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from features import compute_features
from risk_metrics import full_risk_report

st.set_page_config(
    page_title="Nifty 50 Stock Screener",
    page_icon="📈",
    layout="wide"
)

TICKERS = {
    "Reliance":      "RELIANCE.NS",
    "TCS":           "TCS.NS",
    "HDFC Bank":     "HDFCBANK.NS",
    "Infosys":       "INFY.NS",
    "ICICI Bank":    "ICICIBANK.NS",
    "Kotak Bank":    "KOTAKBANK.NS",
    "Bajaj Finance": "BAJFINANCE.NS",
    "ITC":           "ITC.NS",
    "Wipro":         "WIPRO.NS",
    "SBI":           "SBIN.NS",
}

st.title("📈 Nifty 50 ML Stock Screener")
st.caption("Predicts which stocks are likely to outperform the Nifty 50 index over the next 30 days")

# --- Sidebar ---
st.sidebar.header("Settings")
selected_name   = st.sidebar.selectbox("Select stock", list(TICKERS.keys()))
period          = st.sidebar.selectbox("Data period", ["6mo", "1y", "2y"], index=1)
threshold       = st.sidebar.slider("Signal threshold", 0.50, 0.75, 0.55, 0.01)

ticker = TICKERS[selected_name]

# --- Load data ---
@st.cache_data(ttl=3600)
def load_data(ticker, period):
    df = yf.download(ticker, period=period, auto_adjust=True, progress=False)
    nifty = yf.download("^NSEI", period=period, auto_adjust=True, progress=False)
    return df, nifty

with st.spinner("Loading market data..."):
    price_df, nifty_df = load_data(ticker, period)

if price_df.empty:
    st.error("Could not load data. Check ticker or internet connection.")
    st.stop()

# Flatten MultiIndex columns once, use everywhere
clean_price_df = price_df.copy()
if isinstance(clean_price_df.columns, pd.MultiIndex):
    clean_price_df.columns = clean_price_df.columns.get_level_values(0)

feat_df = compute_features(clean_price_df)
returns = clean_price_df["Close"].pct_change().dropna()

if feat_df.empty:
    st.error(f"Not enough data after computing indicators. Raw rows: {len(clean_price_df)}. Try a longer period (2y).")
    st.stop()

# --- Top metrics row ---
col1, col2, col3, col4 = st.columns(4)

report = full_risk_report(returns, label=selected_name)

col1.metric("Sharpe Ratio",    report["sharpe"],
            help="Higher is better. >1 is good, >2 is excellent.")
col2.metric("Max Drawdown",    f"{report['max_drawdown']}%",
            help="Worst peak-to-trough loss in the period.")
col3.metric("VaR (95%)",       f"{report['var_95']}%",
            help="Worst daily loss 95% of the time.")
col4.metric("Win Rate",        f"{report['win_rate']}%",
            help="% of trading days with positive returns.")

st.divider()

# --- Price chart ---
col_chart, col_features = st.columns([2, 1])

with col_chart:
    st.subheader(f"{selected_name} — Price & signals")
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(clean_price_df.index, clean_price_df["Close"], color="#185FA5", linewidth=1.2, label="Close")
    ax.plot(feat_df.index, feat_df["sma_20"],  color="#EF9F27", linewidth=0.8,
            linestyle="--", label="SMA 20")
    ax.plot(feat_df.index, feat_df["sma_50"],  color="#D85A30", linewidth=0.8,
            linestyle="--", label="SMA 50")
    ax.set_ylabel("Price (INR)")
    ax.legend(fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, use_container_width=True)

with col_features:
    st.subheader("Latest indicators")
    latest = feat_df.iloc[-1]
    indicators = {
        "RSI (14d)":        round(float(latest["rsi_14"]), 1),
        "MACD":             round(float(latest["macd"]), 3),
        "30d momentum":     f"{round(float(latest['return_20d'])*100, 1)}%",
        "Vol spike":        round(float(latest["vol_spike"]), 2),
        "52w position":     f"{round(float(latest['pos_52w'])*100, 1)}%",
        "Volatility (ann)": f"{round(float(latest['volatility_20d'])*100, 1)}%",
    }
    for k, v in indicators.items():
        col_a, col_b = st.columns([2, 1])
        col_a.caption(k)
        col_b.write(f"**{v}**")

st.divider()

# --- RSI subplot ---
st.subheader("RSI momentum indicator")
fig2, ax2 = plt.subplots(figsize=(10, 2.5))
ax2.plot(feat_df.index, feat_df["rsi_14"], color="#534AB7", linewidth=1)
ax2.axhline(70, color="#D85A30", linestyle="--", linewidth=0.8, alpha=0.7, label="Overbought (70)")
ax2.axhline(30, color="#1D9E75", linestyle="--", linewidth=0.8, alpha=0.7, label="Oversold (30)")
ax2.fill_between(feat_df.index, 30, 70, alpha=0.05, color="#534AB7")
ax2.set_ylabel("RSI")
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8)
ax2.spines[["top", "right"]].set_visible(False)
st.pyplot(fig2, use_container_width=True)

st.divider()

# --- Nifty comparison ---
st.subheader("Cumulative return vs Nifty 50")
stock_cum = (1 + returns).cumprod()
nifty_close = nifty_df["Close"]
if isinstance(nifty_close, pd.DataFrame):
    nifty_close = nifty_close.iloc[:, 0]
nifty_ret = nifty_close.pct_change().dropna()
nifty_cum = (1 + nifty_ret).cumprod()

stock_cum, nifty_cum = stock_cum.align(nifty_cum, join="inner")

fig3, ax3 = plt.subplots(figsize=(10, 3))
ax3.plot(stock_cum.index, stock_cum.values, color="#185FA5",
         linewidth=1.2, label=selected_name)
ax3.plot(nifty_cum.index, nifty_cum.values, color="#888780",
         linewidth=1,   linestyle="--", label="Nifty 50")
ax3.set_ylabel("Cumulative return (base 1.0)")
ax3.legend()
ax3.spines[["top", "right"]].set_visible(False)
st.pyplot(fig3, use_container_width=True)

st.caption("Built by Dheeraj Bandam · MBA (Business Analytics) · IIM Amritsar · "
           "github.com/dheerajbandam/nifty-stock-screener-ml")
