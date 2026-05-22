"""
Panel construction: merge carry-gap observations with all regressors.

build_regression_panel  → baseline + extended panel
                          (GBM_1Y, GBM_10Y, BA_over_tau, NFCI, GBM_mu_1Y)
build_extended_panel    → alias with explicit horizon argument
"""
import numpy as np
import pandas as pd

from src.features.baseline_gbm import generate_synthetic_vol_ois
from src.features.frictions import generate_synthetic_ba_med
from src.features.nfci import load_nfci
from src.features.drift_proxy import compute_drift_proxy


# ── Helpers ───────────────────────────────────────────────────────────────────

def _date_bounds(panel: pd.DataFrame) -> tuple:
    """Return (start_date_str, end_date_str) from panel 'date' column."""
    start = panel['date'].min().strftime('%Y-%m-%d')
    end   = panel['date'].max().strftime('%Y-%m-%d')
    return start, end


def _build_market_regressors(
    market: str,
    start_date: str,
    end_date: str,
    horizon: int,
    random_seed: int,
    nfci_series: pd.Series,
) -> pd.DataFrame:
    """
    Build a date-indexed DataFrame of all regressors for one market.

    Columns: GBM_1Y, GBM_10Y, BA_med, NFCI, GBM_mu_1Y
    """
    # Per-market vol / OIS / log-return panel
    vol_df = generate_synthetic_vol_ois(start_date, end_date, market, random_seed)

    # Per-market bid-ask spread
    ba_series = generate_synthetic_ba_med(start_date, end_date, market, random_seed)

    # Drift proxy: cumulative log return → rolling OLS slope
    log_prices = vol_df['log_return'].cumsum()
    dp = compute_drift_proxy(log_prices, horizon=horizon)
    dp.name = 'GBM_mu_1Y'

    out = pd.DataFrame({
        'GBM_1Y':    vol_df['GBM_1Y'],
        'GBM_10Y':   vol_df['GBM_10Y'],
        'BA_med':    ba_series,
        'NFCI':      nfci_series,
        'GBM_mu_1Y': dp,
    })
    out.index.name = 'date'
    return out


# ── Public API ────────────────────────────────────────────────────────────────

def build_regression_panel(
    carry_gap_panel: pd.DataFrame,
    random_seed: int = 42,
    horizon: int = 504,
) -> pd.DataFrame:
    """
    Merge carry-gap panel with all baseline and extended regressors.

    For each (date, market, tau) observation:
      BA_over_tau = BA_med / tau      (per-cell, tau-adjusted friction)
      GBM_1Y, GBM_10Y use the paper diffusion formula (tau=1 at panel build)
      GBM_mu_1Y is the prior-only OLS drift proxy at the requested horizon

    Parameters
    ----------
    carry_gap_panel : output of ``build_synthetic_carry_gap_panel``
    random_seed     : controls all synthetic regressor generators
    horizon         : drift-proxy horizon in trading days (default 504 ~ 2 yr)

    Returns
    -------
    DataFrame with columns:
        date, market, expiration, tau, CG_bp,
        GBM_1Y, GBM_10Y, BA_over_tau, NFCI, GBM_mu_1Y
    """
    start_date, end_date = _date_bounds(carry_gap_panel)

    # NFCI is market-neutral — compute once and share across markets
    nfci_series = load_nfci(start_date, end_date, random_seed)

    all_mkt_dfs = []
    for market in carry_gap_panel['market'].unique():
        mkt_cg = carry_gap_panel[carry_gap_panel['market'] == market].copy()
        regressors = _build_market_regressors(
            market, start_date, end_date, horizon, random_seed, nfci_series
        )

        # Merge regressors onto carry-gap observations by date
        merged = mkt_cg.merge(
            regressors.reset_index(),
            on='date',
            how='left',
        )

        # Compute per-cell BA_over_tau (tau-adjusted friction)
        merged['BA_over_tau'] = merged['BA_med'] / merged['tau']
        all_mkt_dfs.append(merged)

    panel = pd.concat(all_mkt_dfs, ignore_index=True)
    panel = panel.sort_values(['date', 'market', 'tau']).reset_index(drop=True)
    panel = panel.drop(columns=['BA_med'], errors='ignore')
    return panel


def build_extended_panel(
    carry_gap_panel: pd.DataFrame,
    horizon: int = 504,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Alias for ``build_regression_panel`` with an explicit horizon argument.

    Used by Experiment 3 (drift extension) and Experiment 4 (horizon scan).
    """
    return build_regression_panel(carry_gap_panel, random_seed=random_seed,
                                  horizon=horizon)
