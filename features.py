"""
features.py
-----------
Computes finance-specific technical + fundamental features
for each stock. These are what signal domain knowledge to recruiters.

Features computed:
  Technical  : RSI, momentum (5d/20d/60d), volatility, volume spike
  Price-based: 52-week position, distance from moving averages
  Fundamental: P/E ratio, beta (merged from fundamentals dict)
  Target     : Binary — did stock beat Nifty 50 over next 30 days?
"""

import pandas as pd
import numpy as np


def compute_rsi(series, window=14):
    """Relative Strength Index — momentum oscillator (0-100)."""
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_features(price_df, fundamentals=None):
    """
    Takes a single stock's OHLCV DataFrame and optional fundamentals dict.
    Returns a DataFrame with one row per date, all features computed.
    """
    df = price_df.copy()

    # Fix: newer yfinance returns MultiIndex columns like ("Close", "RELIANCE.NS")
    # Flatten to single level so df["Close"] works normally
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    close = df["Close"]
    volume = df["Volume"]

    # --- Momentum features ---
    df["return_5d"]  = close.pct_change(5)
    df["return_20d"] = close.pct_change(20)
    df["return_60d"] = close.pct_change(60)

    # --- RSI ---
    df["rsi_14"] = compute_rsi(close, window=14)

    # --- Moving averages ---
    df["sma_20"]  = close.rolling(20).mean()
    df["sma_50"]  = close.rolling(50).mean()
    df["ema_12"]  = close.ewm(span=12).mean()
    df["ema_26"]  = close.ewm(span=26).mean()
    df["macd"]    = df["ema_12"] - df["ema_26"]

    # Distance from 20-day SMA (mean reversion signal)
    df["dist_sma20"] = (close - df["sma_20"]) / df["sma_20"]

    # --- Volatility ---
    df["volatility_20d"] = close.pct_change().rolling(20).std() * np.sqrt(252)

    # --- Volume spike (unusual activity = institutional interest) ---
    df["vol_spike"] = volume / volume.rolling(20).mean()

    # --- 52-week position (0 = at low, 1 = at high) ---
    # Use expanding() so this works even with < 252 days of data
    high_52w = close.expanding(min_periods=20).max()
    low_52w  = close.expanding(min_periods=20).min()
    df["pos_52w"] = (close - low_52w) / (high_52w - low_52w + 1e-8)

    # --- Bollinger Band width (squeeze = breakout incoming) ---
    bb_std = close.rolling(20).std()
    df["bb_width"] = (2 * bb_std) / df["sma_20"]

    # --- Fundamental features (static, merged across all dates) ---
    if fundamentals:
        df["pe_ratio"] = fundamentals.get("pe_ratio")
        df["beta"]     = fundamentals.get("beta")
        df["div_yield"]= fundamentals.get("dividend_yield")
    else:
        df["pe_ratio"] = np.nan
        df["beta"]     = np.nan
        df["div_yield"]= np.nan

    # Only require the core indicators to be non-null (not fundamentals)
    core_cols = ["return_20d", "rsi_14", "sma_50", "volatility_20d",
                 "vol_spike", "pos_52w", "bb_width"]
    return df.dropna(subset=core_cols)


def create_target(price_df, nifty_df, forward_days=30):
    """
    Binary classification target:
      1 = stock return over next `forward_days` > Nifty 50 return
      0 = underperformed

    This frames it as a business question: "will this stock beat the index?"
    """
    stock_fwd = price_df["Close"].pct_change(forward_days).shift(-forward_days)
    nifty_fwd = nifty_df["Close"].pct_change(forward_days).shift(-forward_days)

    # Align on common dates
    stock_fwd, nifty_fwd = stock_fwd.align(nifty_fwd, join="inner")
    target = (stock_fwd > nifty_fwd).astype(int)
    target.name = "beat_nifty"
    return target


def build_feature_matrix(all_price_data, nifty_df, fundamentals_dict=None):
    """
    Loops through all stocks, computes features + target, stacks into
    one master DataFrame for model training.

    Parameters:
        all_price_data  : dict {ticker: DataFrame}
        nifty_df        : DataFrame of Nifty 50 index (^NSEI)
        fundamentals_dict: dict {ticker: {pe_ratio, beta, ...}}

    Returns:
        X : feature matrix (pd.DataFrame)
        y : target series (pd.Series)
        meta : DataFrame with ticker + date for each row
    """
    frames = []
    feature_cols = [
        "return_5d", "return_20d", "return_60d",
        "rsi_14", "macd", "dist_sma20",
        "volatility_20d", "vol_spike", "pos_52w",
        "bb_width", "pe_ratio", "beta", "div_yield"
    ]

    for ticker, price_df in all_price_data.items():
        try:
            fund = fundamentals_dict.get(ticker) if fundamentals_dict else None
            feat_df = compute_features(price_df, fundamentals=fund)
            target  = create_target(price_df, nifty_df)

            # Align features and target
            feat_df, target = feat_df.align(target, join="inner", axis=0)
            feat_df = feat_df[feature_cols].copy()
            feat_df["beat_nifty"] = target
            feat_df["ticker"] = ticker
            frames.append(feat_df)

        except Exception as e:
            print(f"  [WARN] Skipping {ticker}: {e}")

    master = pd.concat(frames).dropna()
    meta = master[["ticker"]].copy()
    y = master["beat_nifty"]
    X = master[feature_cols]

    print(f"Feature matrix: {X.shape[0]} rows x {X.shape[1]} features")
    print(f"Target distribution:\n{y.value_counts(normalize=True).round(3)}")
    return X, y, meta


if __name__ == "__main__":
    # Quick test with dummy data
    dates = pd.date_range("2022-01-01", periods=300, freq="B")
    np.random.seed(42)
    price = pd.DataFrame({
        "Close":  100 * (1 + np.random.randn(300) * 0.01).cumprod(),
        "Volume": np.random.randint(1_000_000, 5_000_000, 300)
    }, index=dates)

    feat = compute_features(price)
    print(feat.tail())
    print(f"\nComputed {feat.shape[1]} features for {len(feat)} trading days")
