"""
National Financial Conditions Index (NFCI) — synthetic stand-in.

The Chicago Fed NFCI is a weekly index of US financial conditions;
positive values indicate tighter-than-average conditions and negative
values indicate looser-than-average conditions.

DOCUMENTED CHOICE: NFCI is published weekly (Wednesdays).
Mapping rule: forward-fill from latest available value to daily.

Since NFCI is not available through the data API, this module generates
a synthetic daily NFCI series that reproduces key historical features:
  - Range approximately -1 to +2
  - Spikes during stress: 2018 Q4, 2020 March, 2022 Q1
  - Generally negative (loose conditions) post-2020
"""
import numpy as np
import pandas as pd


def _piecewise_linear(
    dates: pd.DatetimeIndex,
    key_dates: list,
    key_values: list,
) -> np.ndarray:
    ts     = dates.asi8.astype(np.float64)
    key_ts = pd.DatetimeIndex(key_dates).asi8.astype(np.float64)
    return np.interp(ts, key_ts, key_values)


def load_nfci(
    start_date: str,
    end_date: str,
    random_seed: int = 42,
) -> pd.Series:
    """
    Load Chicago Fed NFCI series aligned to trading dates.

    DOCUMENTED CHOICE: NFCI is published weekly (Wednesdays).
    Mapping rule: forward-fill from latest available value to daily.

    Generates realistic synthetic NFCI based on known historical pattern:
    - Range approximately -1 to +2
    - Spikes during stress: 2018 Q4, 2020 March, 2022 Q1
    - Generally negative (loose conditions) post-2020

    Parameters
    ----------
    start_date  : "YYYY-MM-DD" inclusive
    end_date    : "YYYY-MM-DD" inclusive
    random_seed : RNG seed (NFCI is market-neutral; seed controls OU noise)

    Returns
    -------
    pd.Series with DatetimeIndex (business days) and NFCI values, named "NFCI".
    """
    rng = np.random.RandomState(random_seed)

    # Step 1: generate weekly (Wednesday) NFCI values
    all_bdays = pd.bdate_range(start_date, end_date)

    # Wednesdays only
    wednesdays = all_bdays[all_bdays.dayofweek == 2]
    n_weeks = len(wednesdays)

    # Deterministic NFCI level path (piecewise linear, matches historical shape)
    level_key_dates = [
        "2016-01-01", "2017-01-01", "2017-07-01",
        "2018-01-01", "2018-10-01", "2019-01-01", "2019-07-01",
        "2020-01-01", "2020-02-21", "2020-04-01",
        "2020-07-01", "2021-01-01", "2021-07-01",
        "2022-01-01", "2022-03-01", "2022-09-01",
        "2023-01-01", "2023-07-01",
        "2024-01-01", "2024-07-01",
        "2025-01-01", "2025-10-31",
    ]
    level_key_vals = [
        -0.35, -0.35, -0.40,
        -0.38, +0.20, -0.25, -0.35,
        -0.30, -0.30, +1.50,
        +0.10, -0.55, -0.65,
        -0.50, +0.10, +0.35,
        +0.20, -0.10,
        -0.20, -0.30,
        -0.35, -0.45,
    ]

    if n_weeks == 0:
        # fallback: no Wednesdays in range, use all bdays
        nfci = pd.Series(
            _piecewise_linear(all_bdays, level_key_dates, level_key_vals),
            index=all_bdays,
            name="NFCI",
        )
        return nfci.clip(-2.0, 3.0)

    det_level = _piecewise_linear(wednesdays, level_key_dates, level_key_vals)

    # OU noise around deterministic path
    phi      = 0.92       # weekly autocorrelation
    sigma_ou = 0.06
    ou       = np.zeros(n_weeks)
    for i in range(1, n_weeks):
        ou[i] = phi * ou[i - 1] + rng.randn() * sigma_ou

    weekly_nfci = np.clip(det_level + ou, -2.0, 3.0)
    weekly_series = pd.Series(weekly_nfci, index=wednesdays, name="NFCI")

    # Step 2: forward-fill to business days (DOCUMENTED CHOICE)
    # Reindex to all business days, then forward-fill; back-fill initial NaNs.
    daily = weekly_series.reindex(all_bdays).ffill().bfill()

    daily.name = "NFCI"
    return daily
