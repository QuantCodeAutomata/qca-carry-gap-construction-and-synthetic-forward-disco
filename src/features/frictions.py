"""
Transaction-cost (friction) regressor.

BA_over_tau = median_atm_bid_ask_spread / tau

The median ATM bid-ask spread BA^{med}_{i,t} is computed as the median of the
single closest-to-forward option per (market, date, expiration), then medianed
across expirations for each (market, date).
"""
import numpy as np
import pandas as pd
from typing import Optional


# ---------------------------------------------------------------------------
# ATM identification
# ---------------------------------------------------------------------------

def identify_atm_options(
    option_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
    atm_bandwidth: float = 0.05,
) -> pd.DataFrame:
    """
    Identify ATM options within +/- atm_bandwidth of F_hat.

    Parameters
    ----------
    option_df      : DataFrame with columns market, date, expiration,
                     strike, option_type, bid, ask, mid
    fwd_df         : DataFrame with columns market, date, expiration, F_hat
    atm_bandwidth  : relative distance threshold |strike/F_hat - 1| <= threshold

    Returns
    -------
    option_df with an added boolean column "atm".

    DOCUMENTED CHOICE: ATM defined as within 5% of implied forward.
    """
    out = option_df.merge(
        fwd_df[["market", "date", "expiration", "F_hat"]],
        on=["market", "date", "expiration"],
        how="left",
    )
    out["atm"] = (np.abs(out["strike"] / out["F_hat"] - 1.0) <= atm_bandwidth)
    return out


# ---------------------------------------------------------------------------
# Median ATM bid-ask spread
# ---------------------------------------------------------------------------

def compute_ba_med(
    option_df: pd.DataFrame,
    fwd_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute market-date median ATM bid-ask spread BA^{med}_{i,t}.

    For each (market, date, expiration) the single option with the strike
    closest to F_hat is selected.  The bid-ask spread (ask - bid) of that
    option is taken.  BA^{med}_{i,t} is then the cross-maturity median.

    Parameters
    ----------
    option_df : DataFrame with columns market, date, expiration,
                strike, bid, ask  (and optionally "atm" flag)
    fwd_df    : DataFrame with columns market, date, expiration, F_hat

    Returns
    -------
    DataFrame with columns market, date, BA_med.
    """
    df = option_df.merge(
        fwd_df[["market", "date", "expiration", "F_hat"]],
        on=["market", "date", "expiration"],
        how="inner",
    ).copy()

    df["_dist"] = np.abs(df["strike"] / df["F_hat"] - 1.0)
    df["_ba"]   = df["ask"] - df["bid"]

    # Per (market, date, expiration): select row with minimum distance
    grp = df.groupby(["market", "date", "expiration"])
    closest_idx = grp["_dist"].idxmin()
    closest = df.loc[closest_idx, ["market", "date", "expiration", "_ba"]].copy()

    # Per (market, date): median BA across expirations
    ba_med = (
        closest
        .groupby(["market", "date"])["_ba"]
        .median()
        .reset_index()
        .rename(columns={"_ba": "BA_med"})
    )
    return ba_med


# ---------------------------------------------------------------------------
# Friction regressor
# ---------------------------------------------------------------------------

def build_friction_regressor(
    panel: pd.DataFrame,
    ba_med_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge BA_med onto panel and compute BA_med/tau regressor.

    Parameters
    ----------
    panel      : DataFrame with columns market, date, tau (at minimum)
    ba_med_df  : DataFrame with columns market, date, BA_med

    Returns
    -------
    Copy of panel with added columns BA_med and BA_over_tau.
    """
    out = panel.merge(
        ba_med_df[["market", "date", "BA_med"]],
        on=["market", "date"],
        how="left",
    )
    out["BA_over_tau"] = out["BA_med"] / out["tau"]
    return out


# ---------------------------------------------------------------------------
# Synthetic generator
# ---------------------------------------------------------------------------

def generate_synthetic_ba_med(
    start_date: str,
    end_date: str,
    market: str = 'SPX',
    random_seed: int = 42,
) -> pd.Series:
    """
    Generate a synthetic daily ATM median bid-ask spread for one market.

    Parameters
    ----------
    start_date  : e.g. '2016-01-01'
    end_date    : e.g. '2025-10-31'
    market      : 'SPX' (tighter spreads) or 'RUT' (wider spreads)
    random_seed : RNG seed for reproducibility

    Returns
    -------
    pd.Series named 'BA_med' with a business-day DatetimeIndex.
    Values are in index points (positive, mean ~1.5 for SPX, ~2.1 for RUT).

    BA_med follows a log-normal AR(1) mean-reverting process with stress-
    period widening in late 2018, March 2020, and early 2022.
    """
    rng   = np.random.RandomState(random_seed)
    dates = pd.bdate_range(start=start_date, end=end_date)
    n     = len(dates)

    base_spread = 1.5
    base = base_spread if market.upper() == 'SPX' else base_spread * 1.4

    # Log-spread AR(1) mean-reverting process
    phi      = 0.97
    log_base = np.log(base)
    sigma_l  = 0.06
    log_spread    = np.empty(n)
    log_spread[0] = log_base
    innov = rng.randn(n - 1)
    for i in range(1, n):
        log_spread[i] = (
            phi * log_spread[i - 1]
            + (1 - phi) * log_base
            + sigma_l * innov[i - 1]
        )

    # Stress-period BA widening
    date_arr = dates.to_numpy()
    for spike_str, spike_add, decay_days in [
        ("2018-10-01", 2.0, 30),
        ("2020-03-15", 5.0, 40),
        ("2022-01-15", 1.5, 30),
    ]:
        idx = int(np.searchsorted(date_arr, np.datetime64(spike_str)))
        if idx < n:
            n_decay = min(decay_days, n - idx)
            decay = np.exp(-np.arange(n_decay) / (decay_days / 3.0))
            log_spread[idx: idx + n_decay] += np.log(1.0 + spike_add / base) * decay

    ba_vals = np.clip(np.exp(log_spread), 0.05, base * 20.0)
    return pd.Series(ba_vals, index=dates, name='BA_med')
