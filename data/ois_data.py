import os
import numpy as np
import pandas as pd

# ── Tenor definitions ────────────────────────────────────────────────────────
TENORS: list = [1 / 12, 3 / 12, 6 / 12, 1, 2, 5, 10]
TENOR_LABELS: list = ['1m', '3m', '6m', '1y', '2y', '5y', '10y']

# Spread of each tenor relative to the 3 M policy-rate proxy (pct points).
# Positive during normal (upward-sloping) curve; compressed / negative when
# the policy rate is very high (inverted-curve regime).
_BASE_SPREADS = {
    '1m':  -0.05,
    '3m':   0.00,
    '6m':   0.10,
    '1y':   0.20,
    '2y':   0.40,
    '5y':   0.80,
    '10y':  1.00,
}

# OIS noise parameters (Ornstein-Uhlenbeck around deterministic path)
_OU_KAPPA = 0.05        # daily mean-reversion speed
_OU_SIGMA = 0.015       # daily diffusion (pct points)
_CACHE_FILE = os.path.join('data', 'cache', 'ois_rates.parquet')
_RNG_SEED = 42


# ── Internal helpers ─────────────────────────────────────────────────────────

def _build_policy_rate_path(dates: pd.DatetimeIndex) -> np.ndarray:
    """
    Deterministic 3 M policy-rate proxy interpolated from historical key points.
    Values in percentage points.
    """
    key_dates = pd.to_datetime([
        '2014-01-01', '2015-06-01', '2015-12-16',
        '2016-06-01', '2016-12-14',
        '2017-06-14', '2017-12-13',
        '2018-09-26', '2018-12-19',
        '2019-08-01', '2019-10-31',
        '2020-03-03', '2020-03-16',
        '2021-06-01', '2021-12-31',
        '2022-03-16', '2022-05-04',
        '2022-06-15', '2022-07-27',
        '2022-09-21', '2022-11-02',
        '2022-12-14',
        '2023-02-01', '2023-03-22',
        '2023-05-03', '2023-07-26',
        '2024-09-18', '2024-11-07',
        '2024-12-18', '2025-12-31',
    ])
    key_rates = np.array([
        0.09, 0.12, 0.24,
        0.37, 0.54,
        0.91, 1.16,
        2.00, 2.27,
        2.13, 1.55,
        1.08, 0.09,
        0.08, 0.08,
        0.33, 0.77,
        1.58, 2.33,
        3.08, 3.83,
        4.33,
        4.58, 4.83,
        5.08, 5.33,
        5.08, 4.83,
        4.58, 4.00,
    ])

    # Convert to float timestamps for linear interpolation
    key_ts = key_dates.astype(np.int64).astype(float)
    query_ts = dates.astype(np.int64).astype(float)
    policy = np.interp(query_ts, key_ts, key_rates)
    return policy


def _add_ou_noise(policy: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Add mean-reverting (OU) noise around the deterministic policy-rate path."""
    n = len(policy)
    residual = np.zeros(n)
    innovations = rng.normal(0.0, _OU_SIGMA, size=n)
    for t in range(1, n):
        residual[t] = (1.0 - _OU_KAPPA) * residual[t - 1] + innovations[t]
    return policy + residual


def _curve_spreads(policy: np.ndarray) -> dict:
    """
    Compute per-tenor spreads that compress when the policy rate is high
    (inversion regime) and expand when it is low (normal regime).
    `inversion` in [0, 1]: 0 = normal curve, 1 = fully inverted.
    """
    inversion = np.clip(policy / 5.5, 0.0, 1.0)  # saturates at 5.5 % policy rate
    spreads = {}
    for label, base_spread in _BASE_SPREADS.items():
        # Inversion compresses positive spreads and pushes long end below short end
        if base_spread >= 0:
            spreads[label] = base_spread * (1.0 - 0.80 * inversion)
        else:
            spreads[label] = base_spread  # 1 M always a little below 3 M
    return spreads


# ── Public API ───────────────────────────────────────────────────────────────

def load_ois_rates(
    start_date: str,
    end_date: str,
    tenors: list = None,
) -> pd.DataFrame:
    """
    Load daily OIS rate term structure.

    Returns a DataFrame with a DatetimeIndex and one column per tenor.
    Column names are TENOR_LABELS = ['1m','3m','6m','1y','2y','5y','10y'].
    Rates are in percentage points (e.g. 5.0 = 5 %).

    Generates simulated realistic OIS rates reproducing Fed-cycle dynamics
    (ZIRP 2014-2021, hiking cycle 2022-2023, easing 2024-onward).
    Results are cached to data/cache/ois_rates.parquet.
    """
    if tenors is None:
        tenors = TENORS

    os.makedirs(os.path.dirname(_CACHE_FILE), exist_ok=True)

    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)

    if os.path.exists(_CACHE_FILE):
        cached = pd.read_parquet(_CACHE_FILE)
        cached.index = pd.to_datetime(cached.index)
        if cached.index.min() <= start_ts and cached.index.max() >= end_ts:
            return cached.loc[start_ts:end_ts, TENOR_LABELS]

    # Generate the full simulation window to build the cached file once
    sim_start = '2014-01-01'
    sim_end = '2025-12-31'
    all_dates = pd.bdate_range(sim_start, sim_end)

    rng = np.random.default_rng(_RNG_SEED)
    policy = _build_policy_rate_path(all_dates)
    policy_noisy = _add_ou_noise(policy, rng)
    policy_noisy = np.clip(policy_noisy, 0.01, 20.0)

    spreads_by_tenor = _curve_spreads(policy_noisy)

    df = pd.DataFrame(index=all_dates)
    for label in TENOR_LABELS:
        raw = policy_noisy + spreads_by_tenor[label]
        # Small independent noise per tenor
        tenor_noise = rng.normal(0.0, _OU_SIGMA * 0.5, size=len(all_dates))
        df[label] = np.clip(raw + tenor_noise, 0.01, 20.0).round(4)

    df.index.name = 'date'
    df.to_parquet(_CACHE_FILE)

    return df.loc[start_ts:end_ts, TENOR_LABELS]


def get_ois1y_series(start_date: str, end_date: str) -> pd.Series:
    """
    Return the daily 1 Y OIS rate series (pct points) for the given date range.
    """
    return load_ois_rates(start_date, end_date)['1y']


def get_ois10y_series(start_date: str, end_date: str) -> pd.Series:
    """
    Return the daily 10 Y OIS rate series (pct points) for the given date range.
    """
    return load_ois_rates(start_date, end_date)['10y']
