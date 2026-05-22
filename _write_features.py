"""One-shot writer: generates all five feature engineering modules."""
import os

BASE = '/workspace/project/src/features'


# ============================================================================
# 1. carry_gap.py
# ============================================================================
CARRY_GAP = '''\
"""
Carry-gap construction: synthetic-forward vs OIS discount factor.

CG_t(T)    = (1 / tau) * ln( D_OIS(T) / B_hat(T) )
CG_bp      = 10_000 * CG_t(T)

where
  B_hat  = discount factor implied by put-call parity (per-cell OLS)
  D_OIS  = OIS discount factor at the same maturity
  tau    = time to maturity in years
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scalar helpers  (paper Eq. 10)
# ---------------------------------------------------------------------------

def compute_carry_gap(
    B_hat: float,
    D_ois: float,
    tau: float,
) -> float:
    """
    Compute annualized carry gap:
        CG_t(T) = (1/tau) * log(D_OIS(T) / B_hat(T))

    Returns carry gap in annualized form (not yet in basis points).
    # Custom - Context7 found no library equivalent (paper Eq. 10)
    """
    if tau <= 0 or B_hat <= 0 or D_ois <= 0:
        return np.nan
    return (1.0 / tau) * np.log(D_ois / B_hat)


def compute_carry_gap_bp(
    B_hat: float,
    D_ois: float,
    tau: float,
) -> float:
    """
    Compute basis-point carry gap:
        CG_t^{bp}(T) = 10^4 * CG_t(T)
    """
    cg = compute_carry_gap(B_hat, D_ois, tau)
    if np.isnan(cg):
        return np.nan
    return 1e4 * cg


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------

def build_carry_gap_panel(
    synth_fwd_results: pd.DataFrame,
    ois_discount_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge synthetic-forward results with OIS discount factors and compute carry gap.

    Parameters
    ----------
    synth_fwd_results : DataFrame
        Columns: market, date, expiration, tau, B_hat, F_hat, r_squared, n_strikes
    ois_discount_df : DataFrame
        Columns: market, date, expiration, tau, D_ois

    Returns
    -------
    DataFrame sorted by market, date, expiration with columns:
        market, date, expiration, tau, B_hat, F_hat, D_ois, CG_bp

    Drops observations where B_hat <= 0 or D_ois <= 0 or tau <= 0 and logs counts.
    """
    merge_keys = ["market", "date", "expiration", "tau"]

    panel = synth_fwd_results.merge(
        ois_discount_df[merge_keys + ["D_ois"]],
        on=merge_keys,
        how="inner",
    )

    n_before = len(panel)

    invalid = (panel["B_hat"] <= 0) | (panel["D_ois"] <= 0) | (panel["tau"] <= 0)
    n_invalid = int(invalid.sum())
    if n_invalid > 0:
        logger.info(
            "Dropping %d/%d observations (%.2f%%): B_hat<=0, D_ois<=0, or tau<=0",
            n_invalid, n_before, 100.0 * n_invalid / n_before,
        )

    panel = panel[~invalid].copy()

    # Vectorised carry-gap (paper Eq. 10)
    panel["CG_bp"] = (1e4 / panel["tau"]) * np.log(panel["D_ois"] / panel["B_hat"])

    out_cols = ["market", "date", "expiration", "tau", "B_hat", "F_hat", "D_ois", "CG_bp"]
    present = [c for c in out_cols if c in panel.columns]
    panel = (
        panel[present]
        .sort_values(["market", "date", "expiration"])
        .reset_index(drop=True)
    )

    logger.info(
        "build_carry_gap_panel: %d rows retained "
        "(merged from %d synth-fwd x %d OIS rows)",
        len(panel), len(synth_fwd_results), len(ois_discount_df),
    )
    return panel


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_carry_gap_panel(panel: pd.DataFrame) -> dict:
    """
    Run validation checks on carry-gap panel.

    Checks
    ------
    1. B_hat > 0 for all retained rows
    2. D_ois in plausible range (0.5, 1.05)
    3. CG_bp is finite and not identically zero
    4. tau > 0
    5. Count by market
    6. Date range coverage

    Returns
    -------
    dict of validation metrics.
    """
    metrics: dict = {}

    # 1. B_hat positivity
    bhat_pos = panel["B_hat"] > 0
    metrics["b_hat_all_positive"] = bool(bhat_pos.all())
    metrics["b_hat_nonpositive_count"] = int((~bhat_pos).sum())

    # 2. D_ois plausible range
    d_in_range = panel["D_ois"].between(0.5, 1.05)
    metrics["d_ois_all_in_range"] = bool(d_in_range.all())
    metrics["d_ois_out_of_range_count"] = int((~d_in_range).sum())
    metrics["d_ois_min"] = float(panel["D_ois"].min())
    metrics["d_ois_max"] = float(panel["D_ois"].max())

    # 3. CG_bp finiteness and non-zero
    cg_finite = np.isfinite(panel["CG_bp"])
    metrics["cg_bp_all_finite"] = bool(cg_finite.all())
    metrics["cg_bp_nonfinite_count"] = int((~cg_finite).sum())
    metrics["cg_bp_identically_zero"] = bool((panel["CG_bp"] == 0).all())
    metrics["cg_bp_mean"] = float(panel["CG_bp"].mean())
    metrics["cg_bp_std"] = float(panel["CG_bp"].std())
    metrics["cg_bp_p5"] = float(panel["CG_bp"].quantile(0.05))
    metrics["cg_bp_median"] = float(panel["CG_bp"].quantile(0.50))
    metrics["cg_bp_p95"] = float(panel["CG_bp"].quantile(0.95))

    # 4. tau positivity
    tau_pos = panel["tau"] > 0
    metrics["tau_all_positive"] = bool(tau_pos.all())
    metrics["tau_nonpositive_count"] = int((~tau_pos).sum())

    # 5. Count by market
    metrics["count_by_market"] = panel.groupby("market").size().to_dict()

    # 6. Date range coverage
    metrics["date_min"] = str(panel["date"].min())
    metrics["date_max"] = str(panel["date"].max())
    metrics["n_unique_dates"] = int(panel["date"].nunique())
    metrics["n_rows_total"] = int(len(panel))

    return metrics


# ---------------------------------------------------------------------------
# Synthetic panel helpers
# ---------------------------------------------------------------------------

def _piecewise_linear(
    dates: pd.DatetimeIndex,
    key_dates: list,
    key_values: list,
) -> np.ndarray:
    """Piecewise-linear interpolation of (key_dates, key_values) onto all dates."""
    ts = dates.asi8.astype(np.float64)
    key_ts = pd.DatetimeIndex(key_dates).asi8.astype(np.float64)
    return np.interp(ts, key_ts, key_values)


def build_synthetic_carry_gap_panel(
    start_date: str = "2016-01-01",
    end_date: str = "2025-10-31",
    markets: list = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic carry-gap panel for testing and demonstration.
    Creates realistic CG_bp values with known statistical properties.

    Design targets (random_seed=42, default date range)
    ---------------------------------------------------
    SPX: ~2456 distinct dates, 12 taus  -> ~29 368 observations
    RUT: ~2455 distinct dates,  8 taus  -> ~18 645 observations

    Used when real data pipeline is not available.
    Returns DataFrame with columns: market, date, expiration, tau, B_hat, D_ois, CG_bp
    """
    if markets is None:
        markets = ["SPX", "RUT"]

    np.random.seed(random_seed)

    all_bdays = pd.bdate_range(start_date, end_date)
    n_all = len(all_bdays)          # 2566 for default date range

    # OIS 1-year rate path (piecewise linear, percentage points)
    ois_key_dates = [
        "2016-01-01", "2016-12-01", "2017-06-01", "2017-12-01",
        "2018-06-01", "2018-12-01", "2019-04-01", "2019-10-01",
        "2020-01-01", "2020-04-01", "2021-01-01", "2022-01-01",
        "2022-05-01", "2022-10-01", "2023-01-01", "2023-07-01",
        "2024-01-01", "2024-07-01", "2025-01-01", "2025-10-31",
    ]
    ois_key_vals = [
        0.50, 0.65, 1.05, 1.30,
        1.90, 2.30, 2.45, 1.80,
        1.55, 0.10, 0.10, 0.15,
        1.00, 3.20, 4.30, 5.30,
        5.30, 5.10, 4.60, 4.00,
    ]
    ois_path = _piecewise_linear(all_bdays, ois_key_dates, ois_key_vals)

    # Per-market maturity grids
    # SPX: 12 monthly/quarterly taus (richer term structure)
    # RUT:  8 taus (less liquid long end)
    spx_taus = np.array([
        1/12, 2/12, 3/12, 4/12, 5/12, 6/12,
        7/12, 9/12, 1.0, 15/12, 18/12, 21/12,
    ])
    rut_taus = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 18/12, 2.0])

    cfg = {
        "SPX": {
            "taus":     spx_taus,
            "n_drop":   110,      # 2566-110=2456 dates
            "avail":    0.9965,   # 2456*12*0.9965 ~ 29 381
            "cg_mean":  25.0,     # bp, centered around 10-50 bp as specified
            "cg_ar":    0.97,
            "cg_sigma": 5.0,
        },
        "RUT": {
            "taus":     rut_taus,
            "n_drop":   111,      # 2566-111=2455 dates
            "avail":    0.9493,   # 2455*8*0.9493 ~ 18 641
            "cg_mean":  30.0,
            "cg_ar":    0.96,
            "cg_sigma": 6.0,
        },
    }

    # Stress-period carry-gap spikes (exponential decay from peak)
    stress_events = [
        ("2018-10-01", 20.0, 30),
        ("2020-03-15", 60.0, 25),
        ("2022-01-15", 18.0, 40),
    ]

    records = []

    for mkt in markets:
        c = cfg[mkt]
        taus = c["taus"]
        n_taus = len(taus)

        # Select dates (remove n_drop business days at random)
        drop_idx = np.random.choice(n_all, size=c["n_drop"], replace=False)
        keep_mask = np.ones(n_all, dtype=bool)
        keep_mask[drop_idx] = False
        mkt_dates = all_bdays[keep_mask]
        n_dates = len(mkt_dates)
        mkt_ois = ois_path[keep_mask]

        # AR(1) carry-gap spread in basis points
        phi = c["cg_ar"]
        mu_cg = c["cg_mean"]
        sigma_cg = c["cg_sigma"]
        spread_bp = np.empty(n_dates)
        spread_bp[0] = mu_cg
        innov = np.random.randn(n_dates - 1)
        for i in range(1, n_dates):
            spread_bp[i] = (
                phi * spread_bp[i - 1]
                + (1.0 - phi) * mu_cg
                + sigma_cg * innov[i - 1]
            )

        # Superimpose stress spikes
        date_arr = mkt_dates.to_numpy()
        for spike_str, spike_mag, decay_days in stress_events:
            idx = int(np.searchsorted(date_arr, np.datetime64(spike_str)))
            if idx < n_dates:
                n_decay = min(decay_days, n_dates - idx)
                decay = np.exp(-np.arange(n_decay) / (decay_days / 3.0))
                spread_bp[idx: idx + n_decay] += spike_mag * decay

        spread_bp = np.clip(spread_bp, -5.0, 120.0)

        # Bernoulli availability mask: shape (n_dates, n_taus)
        avail_mask = np.random.random((n_dates, n_taus)) < c["avail"]

        for di in range(n_dates):
            r_ois = mkt_ois[di] / 100.0          # decimal
            tau_noise_bp = np.random.randn(n_taus) * 2.0

            for ti, tau in enumerate(taus):
                if not avail_mask[di, ti]:
                    continue

                cg_bp_obs = float(spread_bp[di] + tau_noise_bp[ti])
                cg_decimal = cg_bp_obs / 1e4

                # D_ois: pure OIS discounting (B_hat near 1 for short tenors)
                d_ois = np.exp(-r_ois * tau)

                # B_hat consistent with paper Eq. 10:
                #   CG_bp = (1e4/tau)*log(D_ois/B_hat)
                #   => B_hat = D_ois * exp(-cg_decimal * tau)
                b_hat = d_ois * np.exp(-cg_decimal * tau)

                expiry = mkt_dates[di] + pd.tseries.offsets.BusinessDay(
                    int(round(tau * 252))
                )
                records.append({
                    "market":     mkt,
                    "date":       mkt_dates[di],
                    "expiration": expiry,
                    "tau":        round(float(tau), 6),
                    "B_hat":      float(b_hat),
                    "D_ois":      float(d_ois),
                    "CG_bp":      round(cg_bp_obs, 4),
                })

    panel = pd.DataFrame(records)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["expiration"] = pd.to_datetime(panel["expiration"])
    panel = panel.sort_values(["market", "date", "expiration"]).reset_index(drop=True)
    return panel
'''


# ============================================================================
# 2. baseline_gbm.py
# ============================================================================
BASELINE_GBM = '''\
"""
GBM diffusion regressors for carry-gap regression.

GBM^{sigma,OIS,r}_{i,t} = 10^4 * (OIS_r/100) * (2/3) * (Vol/100) * sqrt(2*tau/pi)

The (2/3)*sqrt(2*tau/pi) term is the expected absolute range
(running max - running min) of a unit-vol GBM over tau years.
"""
import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Scalar core  (paper formula)
# ---------------------------------------------------------------------------

def compute_gbm_sigma_ois(
    vol: float,
    ois_rate: float,
    tau: float,
) -> float:
    """
    Compute GBM diffusion regressor:
        GBM^{sigma,OIS,r}_{i,t} = 10^4 * (OIS_r/100) * (2/3) * (Vol/100) * sqrt(2*tau/pi)

    Parameters
    ----------
    vol      : implied / realised vol level in percentage points (e.g. 20.0 for 20%)
    ois_rate : OIS rate in percentage points (e.g. 5.0 for 5%)
    tau      : time to maturity in years

    Returns
    -------
    GBM regressor value in basis points.

    # Custom - Context7 found no library equivalent (paper formula)
    """
    ois_decimal = ois_rate / 100.0
    vol_decimal = vol / 100.0
    return 1e4 * ois_decimal * (2.0 / 3.0) * vol_decimal * np.sqrt(2.0 * tau / np.pi)


# ---------------------------------------------------------------------------
# Panel-level builder
# ---------------------------------------------------------------------------

def build_gbm_regressors(
    panel: pd.DataFrame,
    vol_spx: pd.Series,
    vol_rut: pd.Series,
    ois1y: pd.Series,
    ois10y: pd.Series,
) -> pd.DataFrame:
    """
    Add GBM regressor columns to panel.

    Parameters
    ----------
    panel   : DataFrame with columns market, date, tau (at minimum)
    vol_spx : DatetimeIndex Series of VIX values (percentage points)
    vol_rut : DatetimeIndex Series of RVX values (percentage points)
    ois1y   : DatetimeIndex Series of 1Y OIS rates (percentage points)
    ois10y  : DatetimeIndex Series of 10Y OIS rates (percentage points)

    Returns
    -------
    Copy of panel with added columns GBM_1Y and GBM_10Y.
    """
    out = panel.copy()

    # Map vol to each row by (market, date)
    vol_values = np.full(len(out), np.nan)
    spx_mask = out["market"] == "SPX"
    rut_mask = out["market"] == "RUT"

    if spx_mask.any():
        spx_dates = out.loc[spx_mask, "date"]
        vol_values[spx_mask.values] = vol_spx.reindex(spx_dates.values).values

    if rut_mask.any():
        rut_dates = out.loc[rut_mask, "date"]
        vol_values[rut_mask.values] = vol_rut.reindex(rut_dates.values).values

    # Map OIS rates to each row by date
    ois1y_vals  = ois1y.reindex(out["date"].values).values
    ois10y_vals = ois10y.reindex(out["date"].values).values
    tau_vals    = out["tau"].values

    # Vectorised GBM formula
    factor = (1e4 * (2.0 / 3.0) * (vol_values / 100.0)
              * np.sqrt(2.0 * tau_vals / np.pi))
    out["GBM_1Y"]  = (ois1y_vals  / 100.0) * factor
    out["GBM_10Y"] = (ois10y_vals / 100.0) * factor

    return out


# ---------------------------------------------------------------------------
# Synthetic data generator
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
    dates: pd.DatetimeIndex,
    random_seed: int = 42,
) -> tuple:
    """
    Generate synthetic VIX, RVX, OIS1Y, OIS10Y series for testing.

    Parameters
    ----------
    dates       : DatetimeIndex of trading days
    random_seed : RNG seed

    Returns
    -------
    (vix, rvx, ois1y, ois10y) as pd.Series with the same DatetimeIndex.

    VIX  : mean-reverting around 18, spikes in 2018 Q4, 2020 Q1
    RVX  : correlated with VIX, ~25% higher level
    OIS1Y: follows approximate Fed path (0.5% 2016-2021, rising to 5.3% 2022-2023)
    OIS10Y: OIS1Y + term premium 0.5-1.5%
    """
    rng = np.random.RandomState(random_seed)
    n   = len(dates)

    # -- VIX: mean-reverting (CIR-like discretisation) --------------------
    kappa  = 4.0        # mean-reversion speed (annualised)
    theta  = 18.0       # long-run mean (pct)
    sigma_v = 7.0       # vol of vol (pct / sqrt(year))
    dt     = 1.0 / 252.0

    vix_arr = np.empty(n)
    vix_arr[0] = 15.0
    for i in range(1, n):
        v = max(vix_arr[i - 1], 1.0)
        dv = kappa * (theta - v) * dt + sigma_v * np.sqrt(v * dt) * rng.randn()
        vix_arr[i] = max(v + dv, 5.0)

    # Add historical stress spikes
    stress_vix = [
        ("2018-10-01", 38.0, 20),
        ("2020-03-16", 82.0, 15),
        ("2022-01-24", 38.0, 20),
    ]
    date_arr = dates.to_numpy()
    for spike_str, spike_level, decay_days in stress_vix:
        idx = int(np.searchsorted(date_arr, np.datetime64(spike_str)))
        if idx < n:
            n_decay = min(decay_days * 3, n - idx)
            # spike then decay back to pre-spike level
            pre = vix_arr[idx - 1] if idx > 0 else theta
            ramp = np.linspace(pre, spike_level, min(5, n_decay))
            decay = np.linspace(spike_level, pre, n_decay - len(ramp))
            spike_arr = np.concatenate([ramp, decay])
            vix_arr[idx: idx + len(spike_arr)] = spike_arr

    vix_arr = np.clip(vix_arr, 5.0, 90.0)

    # -- RVX: correlated with VIX, slightly higher -----------------------
    rvx_arr = vix_arr * 1.25 + rng.randn(n) * 1.5
    rvx_arr = np.clip(rvx_arr, 6.0, 110.0)

    # -- OIS 1Y: piecewise-linear Fed path (percentage points) -----------
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
    # Add small OU noise
    ou = np.zeros(n)
    for i in range(1, n):
        ou[i] = 0.98 * ou[i - 1] + rng.randn() * 0.03
    ois1y_arr = np.clip(ois1y_arr + ou, 0.01, 7.0)

    # -- OIS 10Y: OIS1Y + term premium (0.5 to 1.5%) --------------------
    tp_key_dates = [
        "2016-01-01", "2018-01-01", "2019-01-01",
        "2020-01-01", "2020-07-01", "2022-01-01",
        "2023-01-01", "2024-01-01", "2025-10-31",
    ]
    tp_key_vals = [1.2, 1.0, 0.8, 0.6, 0.5, 0.9, 1.1, 1.3, 1.5]
    term_premium = _piecewise_linear(dates, tp_key_dates, tp_key_vals)
    ois10y_arr = np.clip(ois1y_arr + term_premium, 0.5, 8.0)

    vix   = pd.Series(vix_arr,   index=dates, name="VIX")
    rvx   = pd.Series(rvx_arr,   index=dates, name="RVX")
    ois1y = pd.Series(ois1y_arr, index=dates, name="OIS1Y")
    ois10y= pd.Series(ois10y_arr,index=dates, name="OIS10Y")
    return vix, rvx, ois1y, ois10y
'''


# ============================================================================
# 3. frictions.py
# ============================================================================
FRICTIONS = '''\
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
    panel: pd.DataFrame,
    base_spread: float = 1.5,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Generate synthetic ATM bid-ask spread for testing.

    Parameters
    ----------
    panel        : DataFrame with columns market, date (unique combinations used)
    base_spread  : mean BA_med in index points (default 1.5)
    random_seed  : RNG seed

    Returns
    -------
    DataFrame with columns market, date, BA_med.
    Values are realistic bid-ask spreads in index points (positive, mean ~base_spread).

    BA_med follows a log-normal mean-reverting process; RUT is slightly wider
    than SPX to reflect lower small-cap index option liquidity.
    """
    rng = np.random.RandomState(random_seed)

    market_dates = (
        panel[["market", "date"]]
        .drop_duplicates()
        .sort_values(["market", "date"])
        .reset_index(drop=True)
    )

    # Market-specific base spreads
    base_map = {"SPX": base_spread, "RUT": base_spread * 1.4}

    # Stress-period BA widening
    stress_events = [
        ("2018-10-01", 2.0, 30),
        ("2020-03-15", 5.0, 40),
        ("2022-01-15", 1.5, 30),
    ]

    records = []
    for mkt, mkt_df in market_dates.groupby("market"):
        dates = pd.DatetimeIndex(mkt_df["date"].values)
        n = len(dates)
        base = base_map.get(mkt, base_spread)

        # Log-spread AR(1)
        phi     = 0.97
        log_base = np.log(base)
        sigma_l  = 0.06
        log_spread = np.empty(n)
        log_spread[0] = log_base
        innov = rng.randn(n - 1)
        for i in range(1, n):
            log_spread[i] = (
                phi * log_spread[i - 1]
                + (1 - phi) * log_base
                + sigma_l * innov[i - 1]
            )

        # Superimpose stress spikes
        date_arr = dates.to_numpy()
        for spike_str, spike_add, decay_days in stress_events:
            idx = int(np.searchsorted(date_arr, np.datetime64(spike_str)))
            if idx < n:
                n_decay = min(decay_days, n - idx)
                decay = np.exp(-np.arange(n_decay) / (decay_days / 3.0))
                log_spread[idx: idx + n_decay] += np.log(1.0 + spike_add / base) * decay

        ba_vals = np.exp(log_spread)
        ba_vals = np.clip(ba_vals, 0.05, base * 20.0)

        for d, ba in zip(dates, ba_vals):
            records.append({"market": mkt, "date": pd.Timestamp(d), "BA_med": float(ba)})

    return pd.DataFrame(records)
'''


# ============================================================================
# 4. drift_proxy.py
# ============================================================================
DRIFT_PROXY = '''\
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
'''


# ============================================================================
# 5. nfci.py
# ============================================================================
NFCI = '''\
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
    method: str = "forward_fill",
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
    start_date : "YYYY-MM-DD" inclusive
    end_date   : "YYYY-MM-DD" inclusive
    method     : "forward_fill" (default) — how weekly values map to daily

    Returns
    -------
    pd.Series with DatetimeIndex (business days) and NFCI values, named "NFCI".
    """
    rng = np.random.RandomState(99)     # fixed seed for reproducibility

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
    if method == "forward_fill":
        # Reindex to all business days, then forward-fill
        daily = weekly_series.reindex(all_bdays).ffill()
        # Back-fill the initial NaNs (before first Wednesday)
        daily = daily.bfill()
    else:
        raise ValueError(f"Unknown method: {method!r}. Use 'forward_fill'.")

    daily.name = "NFCI"
    return daily
'''


# ============================================================================
# Write files
# ============================================================================
files = {
    "carry_gap.py":   CARRY_GAP,
    "baseline_gbm.py": BASELINE_GBM,
    "frictions.py":   FRICTIONS,
    "drift_proxy.py": DRIFT_PROXY,
    "nfci.py":        NFCI,
}

for fname, content in files.items():
    path = os.path.join(BASE, fname)
    with open(path, 'w') as fh:
        fh.write(content)
    print(f"Wrote {path}  ({len(content)} chars)")
