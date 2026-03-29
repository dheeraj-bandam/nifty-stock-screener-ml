"""
risk_metrics.py
---------------
Computes portfolio-level risk metrics for the screener's output.
These are the numbers CRISIL, Arcesium, and risk analytics roles look for.

Metrics:
  - Sharpe ratio
  - Sortino ratio
  - Maximum drawdown
  - Value at Risk (VaR) at 95% and 99%
  - Conditional VaR (CVaR / Expected Shortfall)
  - Calmar ratio
  - Win rate
"""

import pandas as pd
import numpy as np


def sharpe_ratio(returns, risk_free_rate=0.065, periods_per_year=252):
    """
    Annualised Sharpe ratio.
    risk_free_rate: Indian 10-yr G-sec rate (approx 6.5%)
    """
    daily_rf = risk_free_rate / periods_per_year
    excess   = returns - daily_rf
    if excess.std() == 0:
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / excess.std()


def sortino_ratio(returns, risk_free_rate=0.065, periods_per_year=252):
    """
    Like Sharpe but only penalises downside volatility.
    More relevant for equity portfolios.
    """
    daily_rf    = risk_free_rate / periods_per_year
    excess      = returns - daily_rf
    downside    = excess[excess < 0]
    downside_std = downside.std()
    if downside_std == 0:
        return np.nan
    return np.sqrt(periods_per_year) * excess.mean() / downside_std


def max_drawdown(prices):
    """
    Maximum peak-to-trough decline in the price series.
    Returns as a negative percentage.
    """
    cumulative = (1 + prices.pct_change().fillna(0)).cumprod()
    rolling_max = cumulative.cummax()
    drawdown    = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def value_at_risk(returns, confidence=0.95):
    """
    Historical VaR — worst loss not exceeded (1-confidence)% of the time.
    Returns as a negative number (e.g. -0.021 = -2.1% daily loss).
    """
    return np.percentile(returns.dropna(), (1 - confidence) * 100)


def conditional_var(returns, confidence=0.95):
    """
    CVaR / Expected Shortfall — average loss in the worst (1-confidence)% cases.
    More informative than VaR for tail risk assessment.
    """
    var   = value_at_risk(returns, confidence)
    tail  = returns[returns <= var]
    return tail.mean() if len(tail) > 0 else np.nan


def calmar_ratio(returns, periods_per_year=252):
    """
    Annualised return divided by maximum drawdown.
    Higher = better risk-adjusted performance.
    """
    ann_return = returns.mean() * periods_per_year
    mdd = abs(max_drawdown(returns.cumsum()))
    if mdd == 0:
        return np.nan
    return ann_return / mdd


def win_rate(returns):
    """Percentage of days with positive returns."""
    return (returns > 0).mean()


def full_risk_report(returns, label="Portfolio"):
    """
    Runs all metrics and returns a clean summary dict.
    Perfect for the Streamlit dashboard display.

    Parameters:
        returns : pd.Series of daily returns
        label   : string label for display
    """
    report = {
        "label":        label,
        "sharpe":       round(sharpe_ratio(returns), 3),
        "sortino":      round(sortino_ratio(returns), 3),
        "max_drawdown": round(max_drawdown(returns) * 100, 2),   # as %
        "var_95":       round(value_at_risk(returns, 0.95) * 100, 2),
        "cvar_95":      round(conditional_var(returns, 0.95) * 100, 2),
        "calmar":       round(calmar_ratio(returns), 3),
        "win_rate":     round(win_rate(returns) * 100, 1),       # as %
        "ann_return":   round(returns.mean() * 252 * 100, 2),    # as %
        "ann_vol":      round(returns.std() * np.sqrt(252) * 100, 2),  # as %
    }

    print(f"\n{'='*40}")
    print(f"  Risk Report — {label}")
    print(f"{'='*40}")
    print(f"  Annualised return : {report['ann_return']}%")
    print(f"  Annualised vol    : {report['ann_vol']}%")
    print(f"  Sharpe ratio      : {report['sharpe']}")
    print(f"  Sortino ratio     : {report['sortino']}")
    print(f"  Max drawdown      : {report['max_drawdown']}%")
    print(f"  VaR (95%)         : {report['var_95']}%")
    print(f"  CVaR (95%)        : {report['cvar_95']}%")
    print(f"  Win rate          : {report['win_rate']}%")
    print(f"{'='*40}")

    return report


if __name__ == "__main__":
    np.random.seed(42)
    fake_returns = pd.Series(np.random.randn(500) * 0.012 + 0.0003)
    report = full_risk_report(fake_returns, label="Test Portfolio")
