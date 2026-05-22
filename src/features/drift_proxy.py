"""
Drift proxy: prior-only rolling OLS slope of log total-return index.

GBM^{hatmu,OIS,1Y}_{i,t} = 10^4 * (OIS1Y/100) * mu_hat * tau

where mu_hat is the annualised OLS slope estimated on [t-n, t-1] (prior-only).
"""
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# Vectorised rolling OLS slope  (prior-only, closed-form)
# ---------------------------------------------------------------------------

def compute_rolling_ols_slope(
    series: pd.Series,
    n: int = 504,
) -> pd.Series:
    """
    Compute prior-only rolling OLS slope of log total-return index.

    For each date t, regresses log(TR_{t-n+l}) on l = 0, ..., n-1
    using only data through t-1 (prior window [t-n, t-1]).

    Uses vectorised closed-form OLS slope:
        slope = (n*sum(l*y) - sum(l)*sum(y)) / (n*sum(l^2) - sum(l)^2)

    Annualises by multiplying by 252.

    Parameters
    ----------
    series : DatetimeIndex log total-return series
    n      : lookback window in trading days (default 504 ~ 2 years)

    Returns
    -------
    pd.Series of annualised drift estimates aligned to date t.
    NaN for the first n trading days (insufficient history).

    PRIOR-ONLY RULE: window [t-n, t-1] inclusive, NOT including t.
    # Custom - Context7 found no library equivalent (paper rolling OLS)
    """
    log_tr  = np.log(series.values)
    dates   = series.index
    n_total = len(log_tr)

    slopes = np.full(n_total, np.nan)

    # Precompute fixed regression moments for a window of size n
    l      = np.arange(n, dtype=np.float64)   # 0, 1, ..., n-1
    sum_l  = l.sum()
    sum_l2 = (l ** 2).sum()
    denom  = n * sum_l2 - sum_l ** 2           # > 0 for n >= 2

    for t in range(n, n_total):
        y      = log_tr[t - n: t]              # prior window, n points
        sum_y  = y.sum()
        sum_ly = (l * y).sum()
        slopes[t] = (n * sum_ly - sum_l * sum_y) / denom

    slopes *= 252.0      # annualise: slope is in log-return per trading day
    return pd.Series(slopes, index=dates, name=f"mu_hat_{n}d")


# ---------------------------------------------------------------------------
# Panel-level drift regressor
# ---------------------------------------------------------------------------

def build_drift_regressor(
    panel: pd.DataFrame,
    mu_hat_spx: pd.Series,
    mu_hat_rut: pd.Series,
    ois1y: pd.Series,
) -> pd.DataFrame:
    """
    Build drift regressor and add to panel.

    Formula:
        GBM^{hatmu,OIS,1Y}_{i,t} = 10^4 * (OIS1Y/100) * mu_hat * tau

    Parameters
    ----------
    panel       : DataFrame with columns market, date, tau
    mu_hat_spx  : DatetimeIndex Series of annualised drift for SPX (decimal)
    mu_hat_rut  : DatetimeIndex Series of annualised drift for RUT (decimal)
    ois1y       : DatetimeIndex Series of 1Y OIS rates (percentage points)

    Returns
    -------
    Copy of panel with added columns mu_hat and GBM_mu_1Y.
    """
    out = panel.copy()

    mu_values = np.full(len(out), np.nan)
    spx_mask  = out["market"] == "SPX"
    rut_mask  = out["market"] == "RUT"

    if spx_mask.any():
        mu_values[spx_mask.values] = (
            mu_hat_spx.reindex(out.loc[spx_mask, "date"].values).values
        )
    if rut_mask.any():
        mu_values[rut_mask.values] = (
            mu_hat_rut.reindex(out.loc[rut_mask, "date"].values).values
        )

    ois1y_vals = ois1y.reindex(out["date"].values).values
    tau_vals   = out["tau"].values

    out["mu_hat"]    = mu_values
    out["GBM_mu_1Y"] = 1e4 * (ois1y_vals / 100.0) * mu_values * tau_vals
    return out


# ---------------------------------------------------------------------------
# Multi-horizon convenience wrapper
# ---------------------------------------------------------------------------

def compute_drift_proxy_all_horizons(
    log_tr: pd.Series,
    horizons: list,
) -> pd.DataFrame:
    """
    Compute prior-only rolling OLS slope for multiple horizons.

    Parameters
    ----------
    log_tr   : DatetimeIndex log total-return series
    horizons : list of window sizes in trading days

    Returns
    -------
    DataFrame with one column per horizon, named mu_hat_{h}d.
    """
    return pd.DataFrame(
        {f"mu_hat_{h}d": compute_rolling_ols_slope(log_tr, n=h) for h in horizons}
    )


# ---------------------------------------------------------------------------
# Legacy alias (kept for backward compatibility with panel.py)
# ---------------------------------------------------------------------------

def _linear_slope(y: np.ndarray) -> float:
    """OLS slope of y on [0, 1, ..., n-1]. Used by compute_drift_proxy."""
    n = len(y)
    if n < 2:
        return np.nan
    x      = np.arange(n, dtype=float)
    ss_xx  = np.sum((x - x.mean()) ** 2)
    if ss_xx == 0.0:
        return np.nan
    ss_xy  = np.sum((x - x.mean()) * (y - y.mean()))
    return float(ss_xy / ss_xx)


TRADING_DAYS_PER_YEAR = 252.0


def compute_drift_proxy(
    log_prices: pd.Series,
    horizon: int = 504,
) -> pd.Series:
    """
    Compute annualised drift proxy using shift(1) prior-only rule.

    At date t the proxy uses log prices in [t-horizon, t-1] only.
    """
    shifted = log_prices.shift(1)
    proxy   = (
        shifted
        .rolling(window=horizon, min_periods=horizon)
        .apply(_linear_slope, raw=True)
        * TRADING_DAYS_PER_YEAR
    )
    proxy.name = f"GBM_mu_{horizon}d"
    return proxy


def build_drift_regressors(
    log_prices: pd.Series,
    horizons: list = None,
) -> pd.DataFrame:
    """Build drift-proxy columns for a list of horizons (legacy wrapper)."""
    if horizons is None:
        horizons = [504]
    return pd.DataFrame(
        {f"GBM_mu_{h}d": compute_drift_proxy(log_prices, horizon=h) for h in horizons}
    )
