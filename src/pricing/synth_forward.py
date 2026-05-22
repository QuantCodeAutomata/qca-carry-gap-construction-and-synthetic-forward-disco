import numpy as np
import pandas as pd
import statsmodels.api as sm
from typing import Tuple, Optional


# ── Quote preparation ─────────────────────────────────────────────────────────

def compute_mid_quotes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute mid quotes as (bid + ask) / 2 and add (or overwrite) a 'mid' column.

    Parameters
    ----------
    df : DataFrame with 'bid' and 'ask' columns

    Returns
    -------
    Copy of *df* with a 'mid' column
    """
    out = df.copy()
    out['mid'] = (out['bid'] + out['ask']) / 2.0
    return out


# ── Quote filters ─────────────────────────────────────────────────────────────

def apply_quote_filters(
    df: pd.DataFrame,
    min_mid: float = 0.01,
    max_spread_ratio: float = 2.0,
    min_strikes: int = 5,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply documented cleaning filters to an option quote DataFrame.

    Filters applied sequentially (each step operates on survivors):

    1. Remove rows with missing bid or ask.
    2. Remove crossed quotes (bid > ask).
    3. Remove non-positive mid prices (mid ≤ 0).
    4. Remove abnormally low prices (mid < *min_mid*).
    5. Remove excessive bid-ask spreads (spread / mid > *max_spread_ratio*).

    The *min_strikes* parameter is accepted for API consistency; cell-level
    filtering (dropping cells with too few strikes) is performed downstream
    in ``process_option_panel``.

    Parameters
    ----------
    df               : option quote DataFrame; must have 'bid', 'ask', 'mid'
    min_mid          : minimum acceptable mid price
    max_spread_ratio : maximum acceptable (ask - bid) / mid ratio
    min_strikes      : forwarded to the caller (not applied here)

    Returns
    -------
    (clean_df, exclusion_log_df)
    exclusion_log_df has the same columns as *df* plus an 'exclusion_reason'
    column.  An empty DataFrame is returned when there are no exclusions.
    """
    if 'mid' not in df.columns:
        df = compute_mid_quotes(df)

    work = df.copy()
    exclusion_parts = []

    def _exclude(mask: pd.Series, reason: str) -> pd.DataFrame:
        exc = work.loc[mask].copy()
        exc['exclusion_reason'] = reason
        return exc

    # Filter 1: missing bid or ask
    mask = work['bid'].isna() | work['ask'].isna()
    if mask.any():
        exclusion_parts.append(_exclude(mask, 'missing_quotes'))
        work = work.loc[~mask].copy()

    # Filter 2: crossed quotes
    mask = work['bid'] > work['ask']
    if mask.any():
        exclusion_parts.append(_exclude(mask, 'crossed_quotes'))
        work = work.loc[~mask].copy()

    # Filter 3: non-positive mid
    mask = work['mid'] <= 0.0
    if mask.any():
        exclusion_parts.append(_exclude(mask, 'nonpositive_mid'))
        work = work.loc[~mask].copy()

    # Filter 4: mid below minimum
    mask = work['mid'] < min_mid
    if mask.any():
        exclusion_parts.append(_exclude(mask, 'mid_below_minimum'))
        work = work.loc[~mask].copy()

    # Filter 5: excessive bid-ask spread
    spread = work['ask'] - work['bid']
    mask = (spread / work['mid']) > max_spread_ratio
    if mask.any():
        exclusion_parts.append(_exclude(mask, 'excessive_spread'))
        work = work.loc[~mask].copy()

    if exclusion_parts:
        exclusion_log = pd.concat(exclusion_parts, ignore_index=True)
    else:
        exclusion_log = pd.DataFrame(columns=list(df.columns) + ['exclusion_reason'])

    return work.reset_index(drop=True), exclusion_log


# ── Synthetic forward ─────────────────────────────────────────────────────────

def compute_synthetic_forward(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the synthetic forward price G(K, T) = C(K, T) − P(K, T) for each
    matched call-put pair.

    Parameters
    ----------
    df : DataFrame with 'call_mid' and 'put_mid' columns (one row per strike)

    Returns
    -------
    Copy of *df* with an added 'G' column
    """
    out = df.copy()
    out['G'] = out['call_mid'] - out['put_mid']
    return out


# ── Per-cell OLS estimation ───────────────────────────────────────────────────

def estimate_discount_factor_cell(
    strikes: np.ndarray,
    synth_fwd: np.ndarray,
    min_strikes: int = 5,
) -> Tuple[Optional[float], Optional[float], Optional[float], int]:
    """
    Estimate the OIS discount factor B̂ and synthetic forward price F̂ for a
    single (market, date, expiration) cell via cross-sectional OLS.

    Put-call parity implies:
        G(K, T) = C(K, T) − P(K, T) = B·F − B·K
    so the regression
        G = a + b·K
    identifies  B̂ = −b  and  F̂ = a / B̂.

    # Custom – Context7 found no library equivalent (paper Eq. G = a + bK)

    Parameters
    ----------
    strikes    : array of strike prices K_i
    synth_fwd  : array of synthetic forward values G_i = C_i − P_i
    min_strikes: minimum number of observations required

    Returns
    -------
    (B_hat, F_hat, r_squared, n_strikes)
    Returns (None, None, None, n) when the cell has too few observations or
    the discount-factor estimate is non-positive.
    """
    n = len(strikes)
    if n < min_strikes:
        return None, None, None, n

    X = sm.add_constant(np.asarray(strikes, dtype=float), has_constant='add')
    model = sm.OLS(np.asarray(synth_fwd, dtype=float), X)
    result = model.fit()

    intercept = result.params[0]   # a  = B · F
    slope = result.params[1]       # b  = −B

    B_hat = -slope
    if B_hat <= 0:
        return None, None, float(result.rsquared), n

    F_hat = intercept / B_hat
    return float(B_hat), float(F_hat), float(result.rsquared), n


# ── Full pipeline ─────────────────────────────────────────────────────────────

def process_option_panel(
    option_df: pd.DataFrame,
    min_strikes: int = 5,
    min_mid: float = 0.01,
    max_spread_ratio: float = 2.0,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Process a full option quote panel and estimate per-cell discount factors.

    Pipeline
    --------
    1. Compute mid quotes.
    2. Apply quote-level cleaning filters.
    3. Match calls and puts by (date, market, strike, expiration).
    4. Compute synthetic forward  G = C_mid − P_mid.
    5. Run cross-sectional OLS per (market, date, expiration) cell.
    6. Return results and exclusion log.

    Parameters
    ----------
    option_df        : raw option quote DataFrame with columns
                       [market, date, expiration, strike, option_type,
                        bid, ask, mid, tau]
    min_strikes      : minimum matched pairs required per cell for OLS
    min_mid          : minimum acceptable mid price
    max_spread_ratio : maximum (ask − bid) / mid ratio

    Returns
    -------
    (results_df, exclusion_log_df)

    results_df columns:
        market, date, expiration, tau, B_hat, F_hat, r_squared, n_strikes

    exclusion_log_df columns:
        all columns from option_df + exclusion_reason
    """
    # Step 1: ensure mid column is present
    df = compute_mid_quotes(option_df)

    # Step 2: quote-level filters
    clean, exclusion_log = apply_quote_filters(
        df,
        min_mid=min_mid,
        max_spread_ratio=max_spread_ratio,
        min_strikes=min_strikes,
    )

    if clean.empty:
        empty_results = pd.DataFrame(
            columns=['market', 'date', 'expiration', 'tau',
                     'B_hat', 'F_hat', 'r_squared', 'n_strikes']
        )
        return empty_results, exclusion_log

    # Step 3: split and match calls to puts
    merge_cols = ['date', 'market', 'strike', 'expiration']

    calls = clean[clean['option_type'] == 'call'][
        merge_cols + ['mid', 'bid', 'ask', 'tau']
    ].copy()
    puts = clean[clean['option_type'] == 'put'][
        merge_cols + ['mid', 'bid', 'ask']
    ].copy()

    matched = calls.merge(
        puts,
        on=merge_cols,
        how='inner',
        suffixes=('_call', '_put'),
    ).rename(columns={
        'mid_call': 'call_mid',
        'ask_call': 'call_ask',
        'bid_call': 'call_bid',
        'mid_put':  'put_mid',
        'ask_put':  'put_ask',
        'bid_put':  'put_bid',
    })

    if matched.empty:
        empty_results = pd.DataFrame(
            columns=['market', 'date', 'expiration', 'tau',
                     'B_hat', 'F_hat', 'r_squared', 'n_strikes']
        )
        return empty_results, exclusion_log

    # Step 4: synthetic forward
    matched = compute_synthetic_forward(matched)

    # Step 5: per-cell OLS
    results = []
    group_cols = ['market', 'date', 'expiration']

    for (market, date, expiration), cell in matched.groupby(group_cols):
        tau = float(cell['tau'].iloc[0])
        strikes = cell['strike'].values
        synth_fwd = cell['G'].values

        B_hat, F_hat, r2, n = estimate_discount_factor_cell(
            strikes, synth_fwd, min_strikes=min_strikes
        )

        if B_hat is not None:
            results.append({
                'market':     market,
                'date':       date,
                'expiration': expiration,
                'tau':        tau,
                'B_hat':      B_hat,
                'F_hat':      F_hat,
                'r_squared':  r2,
                'n_strikes':  n,
            })

    if results:
        results_df = pd.DataFrame(results)
    else:
        results_df = pd.DataFrame(
            columns=['market', 'date', 'expiration', 'tau',
                     'B_hat', 'F_hat', 'r_squared', 'n_strikes']
        )

    return results_df, exclusion_log
