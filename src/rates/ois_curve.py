import numpy as np
import pandas as pd
from scipy.interpolate import interp1d
from typing import Optional


# ── Day-count convention ──────────────────────────────────────────────────────

def compute_tau_act365(
    value_date: pd.Timestamp,
    maturity_date: pd.Timestamp,
) -> float:
    """
    Compute time to maturity in years using the ACT/365 (Fixed) day-count
    convention.

    Parameters
    ----------
    value_date    : settlement / pricing date
    maturity_date : expiry / maturity date

    Returns
    -------
    float : (maturity_date - value_date).days / 365.0
    """
    return (maturity_date - value_date).days / 365.0


# ── Bootstrapping ─────────────────────────────────────────────────────────────

def bootstrap_ois_discount_factors(
    tenors: np.ndarray,
    rates: np.ndarray,
) -> tuple:
    """
    Bootstrap an OIS discount curve from par zero rates.

    Assumes continuously-compounded rates for simplicity
    (suitable for OIS, which is quoted near continuously-compounded).

    D(T) = exp(-r(T) * T)  where r(T) is the zero rate at tenor T.

    Parameters
    ----------
    tenors : 1-D array of tenors in years (must be strictly positive)
    rates  : 1-D array of par rates in **percentage points** (e.g. 5.0 = 5 %)
             same length and order as *tenors*

    Returns
    -------
    (tenors, discount_factors) as numpy float64 arrays
    """
    # Custom – no library equivalent for this simple OIS bootstrap
    # (paper: G = a + bK, consistent with continuously-compounded OIS dfs)
    rates_decimal = np.asarray(rates, dtype=float) / 100.0
    tenors_arr = np.asarray(tenors, dtype=float)
    dfs = np.exp(-rates_decimal * tenors_arr)
    return tenors_arr, dfs


# ── Interpolation ─────────────────────────────────────────────────────────────

def get_ois_discount_factor(
    tau: float,
    tenors: np.ndarray,
    dfs: np.ndarray,
) -> float:
    """
    Interpolate the OIS discount factor at maturity *tau* using
    log-linear interpolation (standard market convention).

    Values outside the tenor range are extrapolated using the boundary
    slopes of the log-discount curve (i.e. ``fill_value='extrapolate'``
    with linear extrapolation in log space).

    Parameters
    ----------
    tau    : time to maturity in years (ACT/365)
    tenors : 1-D array of pillar tenors in years
    dfs    : 1-D array of discount factors at the pillar tenors

    Returns
    -------
    float : interpolated discount factor D(tau)
    """
    log_dfs = np.log(np.asarray(dfs, dtype=float))
    interp = interp1d(
        np.asarray(tenors, dtype=float),
        log_dfs,
        kind='linear',
        fill_value='extrapolate',
        bounds_error=False,
    )
    return float(np.exp(interp(tau)))


# ── Daily curve construction ──────────────────────────────────────────────────

def build_daily_ois_discount_functions(
    ois_rates_df: pd.DataFrame,
    tenor_values: list,
) -> dict:
    """
    Build per-date OIS discount factor interpolation functions.

    Parameters
    ----------
    ois_rates_df : DataFrame with a DatetimeIndex; each row contains OIS par
                   rates in percentage points for the given date.  Column
                   **order** must match *tenor_values*.
    tenor_values : list of tenor values in years corresponding to the columns
                   of *ois_rates_df* (e.g. [1/12, 3/12, 6/12, 1, 2, 5, 10])

    Returns
    -------
    dict mapping ``pd.Timestamp -> (tenors_array, dfs_array)``
    where both arrays are numpy float64 and ready to be passed to
    ``get_ois_discount_factor``.
    """
    tenors_arr = np.asarray(tenor_values, dtype=float)
    curves: dict = {}

    for date, row in ois_rates_df.iterrows():
        rates_arr = row.values.astype(float)
        _, dfs = bootstrap_ois_discount_factors(tenors_arr, rates_arr)
        curves[pd.Timestamp(date)] = (tenors_arr, dfs)

    return curves


# ── Point lookup ──────────────────────────────────────────────────────────────

def get_maturity_ois_df(
    date: pd.Timestamp,
    tau: float,
    ois_curves: dict,
) -> Optional[float]:
    """
    Return the OIS discount factor for a specific *date* and maturity *tau*.

    Parameters
    ----------
    date       : pricing date (must be a key in *ois_curves*)
    tau        : time to maturity in years (ACT/365)
    ois_curves : dict produced by ``build_daily_ois_discount_functions``

    Returns
    -------
    float discount factor, or None if *date* is not in *ois_curves*
    """
    key = pd.Timestamp(date)
    if key not in ois_curves:
        return None
    tenors, dfs = ois_curves[key]
    return get_ois_discount_factor(tau, tenors, dfs)
