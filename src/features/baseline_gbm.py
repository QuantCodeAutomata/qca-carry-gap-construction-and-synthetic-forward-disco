"""
GBM diffusion regressors for carry-gap regression.

Paper formula
-------------
GBM^{sigma,OIS,r}_{i,t} = 10^4 * (OIS_r/100) * (2/3) * (Vol/100) * sqrt(2*tau/pi)

Public API
----------
compute_gbm_sigma_ois(vol, ois_rate, tau)           scalar paper formula
compute_gbm_sigma_sq(log_ret, window, min_periods)  rolling annualised variance
build_gbm_regressors(log_ret, ois_1y, ois_10y, tau) per-date DataFrame
generate_synthetic_vol_ois(start, end, market, seed) full synthetic panel
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scalar paper formula
# ---------------------------------------------------------------------------

def compute_gbm_sigma_ois(
    vol: float,
    ois_rate: float,
    tau: float,
) -> float:
    """
    GBM^{sigma,OIS,r} = 10^4 * (OIS_r/100) * (2/3) * (Vol/100) * sqrt(2*tau/pi)

    Parameters
    ----------
    vol      : vol in percentage points (e.g. 20.0 for 20%)
    ois_rate : OIS rate in percentage points (e.g. 5.0 for 5%)
    tau      : time to maturity in years

    Returns
    -------
    GBM regressor value in basis points.
    """
    return (1e4 * (ois_rate / 100.0) * (2.0 / 3.0)
            * (vol / 100.0) * np.sqrt(2.0 * tau / np.pi))


# ---------------------------------------------------------------------------
# Rolling annualised variance
# ---------------------------------------------------------------------------

def compute_gbm_sigma_sq(
    log_ret: pd.Series,
    window: int = 252,
    min_periods: int = None,
) -> pd.Series:
    """
    Rolling annualised variance of log returns.

    Annualisation: variance_daily * 252.

    Parameters
    ----------
    log_ret     : daily log-return Series with DatetimeIndex
    window      : rolling window in trading days (default 252)
    min_periods : minimum observations required (default = window)

    Returns
    -------
    pd.Series of annualised variance, same index as log_ret.
    """
    mp = window if min_periods is None else min_periods
    return log_ret.rolling(window=window, min_periods=mp).var() * 252.0


# ---------------------------------------------------------------------------
# DataFrame-level builder (uses rolling vol + fixed-rate defaults)
# ---------------------------------------------------------------------------

def build_gbm_regressors(
    log_ret_series: pd.Series,
    ois_1y: float = 2.0,
    ois_10y: float = 3.5,
    tau: float = 1.0,
) -> pd.DataFrame:
    """
    Compute GBM_1Y and GBM_10Y regressors from a log-return Series.

    Uses ``compute_gbm_sigma_sq`` for the rolling vol and applies the paper
    formula with the supplied constant OIS rates and a single maturity tau.

    Parameters
    ----------
    log_ret_series : daily log-return Series with DatetimeIndex
    ois_1y         : 1-year OIS rate in percentage points (default 2.0)
    ois_10y        : 10-year OIS rate in percentage points (default 3.5)
    tau            : maturity in years used in the formula (default 1.0)

    Returns
    -------
    DataFrame with columns GBM_1Y and GBM_10Y, same index as log_ret_series.
    Values are in basis points.  First 251 rows are NaN (warm-up).
    """
    sigma_sq  = compute_gbm_sigma_sq(log_ret_series, window=252)
    sigma_pct = np.sqrt(sigma_sq) * 100.0          # annualised vol in pct

    factor = 1e4 * (2.0 / 3.0) * (sigma_pct / 100.0) * np.sqrt(2.0 * tau / np.pi)
    gbm_1y  = (ois_1y  / 100.0) * factor
    gbm_10y = (ois_10y / 100.0) * factor

    return pd.DataFrame(
        {'GBM_1Y': gbm_1y, 'GBM_10Y': gbm_10y},
        index=log_ret_series.index,
    )


# ---------------------------------------------------------------------------
# Synthetic data generator — full per-market panel
# ---------------------------------------------------------------------------

def _piecewise_linear(
    dates: pd.DatetimeIndex,
    key_dates: list,
    key_values: list,
) -> np.ndarray:
    ts     = dates.asi8.astype(np.float64)
    key_ts = pd.DatetimeIndex(key_dates).asi8.astype(np.float64)
    return np.interp(ts, key_ts, key_values)


def generate_synthetic_vol_ois(
    start_date: str,
    end_date: str,
    market: str = 'SPX',
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate a per-market synthetic panel of vol, OIS, and GBM regressors.

    Parameters
    ----------
    start_date  : e.g. '2016-01-01'
    end_date    : e.g. '2025-10-31'
    market      : 'SPX' (VIX-based) or 'RUT' (RVX-based)
    random_seed : RNG seed

    Returns
    -------
    DataFrame indexed by business-day DatetimeIndex with columns:
        log_return : daily log return of a synthetic GBM price path
        GBM_1Y    : paper-formula regressor using 1Y OIS (basis points)
        GBM_10Y   : paper-formula regressor using 10Y OIS (basis points)
        OIS_1Y    : 1-year OIS rate (percentage points)
        OIS_10Y   : 10-year OIS rate (percentage points)

    Notes
    -----
    VIX / RVX follow a mean-reverting CIR-like process with stress spikes in
    2018 Q4, 2020 Q1, and 2022 Q1.  OIS rates follow a piecewise-linear
    approximation of the Fed funds path 2016-2025.
    GBM columns use tau=1 year (a single-maturity proxy used in panel assembly).
    """
    rng   = np.random.RandomState(random_seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n     = len(dates)

    # ── VIX (SPX) via mean-reverting CIR ─────────────────────────────────
    kappa, theta, sigma_v = 4.0, 18.0, 7.0
    dt  = 1.0 / 252.0
    vol_arr = np.empty(n)
    vol_arr[0] = 15.0
    for i in range(1, n):
        v  = max(vol_arr[i - 1], 1.0)
        dv = kappa * (theta - v) * dt + sigma_v * np.sqrt(v * dt) * rng.randn()
        vol_arr[i] = max(v + dv, 5.0)

    date_arr = dates.to_numpy()
    for spike_str, spike_level, decay_days in [
        ("2018-10-01", 38.0, 20),
        ("2020-03-16", 82.0, 15),
        ("2022-01-24", 38.0, 20),
    ]:
        idx = int(np.searchsorted(date_arr, np.datetime64(spike_str)))
        if idx < n:
            n_decay = min(decay_days * 3, n - idx)
            pre  = vol_arr[idx - 1] if idx > 0 else theta
            ramp = np.linspace(pre, spike_level, min(5, n_decay))
            dec  = np.linspace(spike_level, pre, n_decay - len(ramp))
            arr  = np.concatenate([ramp, dec])
            vol_arr[idx: idx + len(arr)] = arr
    vol_arr = np.clip(vol_arr, 5.0, 90.0)

    # RUT uses RVX ≈ VIX × 1.25 + noise
    if market.upper() == 'RUT':
        vol_arr = vol_arr * 1.25 + rng.randn(n) * 1.5
        vol_arr = np.clip(vol_arr, 6.0, 110.0)

    # ── OIS 1Y piecewise-linear Fed path ─────────────────────────────────
    ois1y_key_dates = [
        "2016-01-01", "2016-12-01", "2017-06-01", "2017-12-01",
        "2018-06-01", "2018-12-01", "2019-04-01", "2019-10-01",
        "2020-01-01", "2020-04-01", "2021-01-01", "2022-01-01",
        "2022-05-01", "2022-10-01", "2023-01-01", "2023-07-01",
        "2024-01-01", "2024-07-01", "2025-01-01", "2025-10-31",
    ]
    ois1y_key_vals = [
        0.50, 0.65, 1.05, 1.30,
        1.90, 2.30, 2.45, 1.80,
        1.55, 0.10, 0.10, 0.15,
        1.00, 3.20, 4.30, 5.30,
        5.30, 5.10, 4.60, 4.00,
    ]
    ois1y_arr = _piecewise_linear(dates, ois1y_key_dates, ois1y_key_vals)
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.98 * ou[i - 1] + rng.randn() * 0.03
    ois1y_arr = np.clip(ois1y_arr + ou, 0.01, 7.0) / 100.0  # convert to decimal

    tp_key_vals = [1.2, 1.0, 0.8, 0.6, 0.5, 0.9, 1.1, 1.3, 1.5]
    term_premium = _piecewise_linear(dates, [
        "2016-01-01", "2018-01-01", "2019-01-01",
        "2020-01-01", "2020-07-01", "2022-01-01",
        "2023-01-01", "2024-01-01", "2025-10-31",
    ], tp_key_vals)
    # ois1y_arr already decimal; add term premium in pct-pts / 100
    ois10y_arr = np.clip(ois1y_arr + term_premium / 100.0, 0.005, 0.08)

    # ── Synthetic log returns (GBM) ───────────────────────────────────────
    annual_vol = 0.18 if market.upper() == 'SPX' else 0.22
    log_ret = ((0.08 - 0.5 * annual_vol ** 2) * dt
               + annual_vol * np.sqrt(dt) * rng.randn(n))

    # ── GBM regressors (tau=1 year proxy for panel assembly) ─────────────
    # ois1y_arr and ois10y_arr are in decimal; vol_arr in pct pts
    tau = 1.0
    factor = (1e4 * (2.0 / 3.0) * (vol_arr / 100.0)
              * np.sqrt(2.0 * tau / np.pi))
    gbm_1y  = ois1y_arr  * factor
    gbm_10y = ois10y_arr * factor

    return pd.DataFrame({
        'log_return': log_ret,
        'GBM_1Y':    gbm_1y,
        'GBM_10Y':   gbm_10y,
        'OIS_1Y':    ois1y_arr,
        'OIS_10Y':   ois10y_arr,
    }, index=dates)
