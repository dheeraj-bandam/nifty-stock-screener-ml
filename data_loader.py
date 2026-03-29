"""
data_loader.py
--------------
Downloads historical OHLCV + fundamental data for Nifty 50 stocks
using yfinance. Run this first to populate data/raw/.
"""

import yfinance as yf
import pandas as pd
import os

NIFTY_50 = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "HINDUNILVR.NS", "ITC.NS", "SBIN.NS", "BHARTIARTL.NS", "KOTAKBANK.NS",
    "LT.NS", "AXISBANK.NS", "ASIANPAINT.NS", "MARUTI.NS", "SUNPHARMA.NS",
    "TITAN.NS", "NESTLEIND.NS", "WIPRO.NS", "ULTRACEMCO.NS", "BAJFINANCE.NS"
]

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


def download_price_data(tickers=NIFTY_50, period="2y", interval="1d"):
    """
    Download daily OHLCV data for all tickers.
    Returns a dict: {ticker: DataFrame}
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    data = {}

    for ticker in tickers:
        try:
            df = yf.download(ticker, period=period, interval=interval,
                             auto_adjust=True, progress=False)
            if df.empty:
                print(f"  [WARN] No data for {ticker}")
                continue
            df.index = pd.to_datetime(df.index)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.to_csv(os.path.join(RAW_DIR, f"{ticker.replace('.NS','')}.csv"))
            data[ticker] = df
            print(f"  [OK] {ticker} — {len(df)} rows")
        except Exception as e:
            print(f"  [ERR] {ticker}: {e}")

    return data


def load_price_data(tickers=None):
    """
    Load previously downloaded CSVs from data/raw/.
    If tickers=None, loads all available files.
    """
    data = {}
    files = os.listdir(RAW_DIR)

    for f in files:
        if not f.endswith(".csv"):
            continue
        name = f.replace(".csv", "")
        if tickers and name not in [t.replace(".NS","") for t in tickers]:
            continue
        path = os.path.join(RAW_DIR, f)
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        data[name] = df

    print(f"Loaded {len(data)} tickers from disk.")
    return data


def get_fundamentals(ticker_symbol):
    """
    Pull key fundamental data for a single ticker via yfinance.
    Returns a dict with P/E, EPS, market cap, beta.
    """
    t = yf.Ticker(ticker_symbol)
    info = t.info
    return {
        "pe_ratio":       info.get("trailingPE"),
        "eps":            info.get("trailingEps"),
        "market_cap":     info.get("marketCap"),
        "beta":           info.get("beta"),
        "52w_high":       info.get("fiftyTwoWeekHigh"),
        "52w_low":        info.get("fiftyTwoWeekLow"),
        "dividend_yield": info.get("dividendYield"),
        "sector":         info.get("sector"),
    }


if __name__ == "__main__":
    print("Downloading Nifty 50 price data...")
    download_price_data()
    print("\nDone. Files saved to data/raw/")
