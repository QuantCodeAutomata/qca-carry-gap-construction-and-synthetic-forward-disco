import os
from datetime import timedelta

import numpy as np
import pandas as pd
from scipy.stats import norm

from massive import RESTClient

from .ois_data import get_ois1y_series
from .total_return import load_total_return_index

# ── Option surface parameters ─────────────────────────────────────────────────

# Moneyness grid: K / S from 0.85 to 1.15 in steps of 0.025 (13 strikes)
MONEYNESS_GRID: np.ndarray = np.round(np.arange(0.85, 1.15 + 1e-9, 0.025), 4)

# Maturities expressed in calendar days (ACT/365 convention)
MATURITY_DAYS: list = [30, 60, 91, 122, 152, 182, 213, 273, 365]

# Vol term-structure slope: additive increment per year of maturity
TERM_SLOPE: float = 0.02        # adds 2 pp per year

# Put-skew factor: vol adjusted by SKEW * (1 - K/S) so OTM puts are richer
SKEW_FACTOR: float = 0.25

# Vol floor applied after all adjustments
MIN_VOL: float = 0.05

# Minimum option price (bid floor)
MIN_PRICE: float = 0.01

# Bid-ask spread at ATM; widens for OTM options
SPREAD_BASE: float = 0.005      # 0.5 % of mid at ATM
SPREAD_OTM_COEFF: float = 0.04  # rate at which spread widens with |log(m)|

# Noise scale added to bid/ask quotes
NOISE_STD: float = 0.001

# Random seed for reproducibility
RNG_SEED: int = 42

# Realized-vol window (business days)
RVOL_WINDOW: int = 21


# ── Black-Scholes helper ──────────────────────────────────────────────────────

def _bs_call_put(
    S: float,
    K: np.ndarray,
    tau: float,
    r: float,
    sigma: np.ndarray,
) -> tuple:
    """
    Vectorized Black-Scholes call and put prices.

    Parameters
    ----------
    S : scalar spot price
    K : array of strike prices
    tau : scalar time to expiry (years, ACT/365)
    r : scalar continuously-compounded risk-free rate (decimal)
    sigma : array of implied vols (decimal, one per strike)

    Returns
    -------
    (calls, puts) as numpy arrays of the same shape as K / sigma
    """
    tau = max(tau, 1e-10)
    sigma = np.maximum(sigma, 1e-6)
    sqt = np.sqrt(tau)

    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * tau) / (sigma * sqt)
    d2 = d1 - sigma * sqt

    df = np.exp(-r * tau)
    calls = S * norm.cdf(d1) - K * df * norm.cdf(d2)
    puts = K * df * norm.cdf(-d2) - S * norm.cdf(-d1)

    # European no-arbitrage lower bounds (discounted intrinsic value)
    calls = np.maximum(calls, np.maximum(S - K * df, 0.0))
    puts = np.maximum(puts, np.maximum(K * df - S, 0.0))
    return calls, puts


# ── Core synthetic panel builder ──────────────────────────────────────────────

def generate_synthetic_option_panel(
    spot_series: pd.Series,
    vol_series: pd.Series,
    market: str,
    rate_series: pd.Series,
) -> pd.DataFrame:
    """
    Generate a synthetic daily option quote panel from spot, vol, and rate inputs.

    Parameters
    ----------
    spot_series  : DatetimeIndex, daily spot prices
    vol_series   : DatetimeIndex, daily annualized realized vol (decimal, e.g. 0.20)
    market       : 'SPX' or 'RUT'
    rate_series  : DatetimeIndex, daily annualized risk-free rate (decimal)

    Returns
    -------
    DataFrame with columns:
        market, date, expiration, strike, option_type, bid, ask, mid, tau
    """
    rng = np.random.default_rng(RNG_SEED)

    tau_grid = np.array(MATURITY_DAYS, dtype=float) / 365.0
    n_strikes = len(MONEYNESS_GRID)
    ln_m = np.log(MONEYNESS_GRID)  # log(K/S) at each moneyness level

    # Pre-compute spread ratios (same for calls and puts at each moneyness)
    spread_ratios = np.clip(
        SPREAD_BASE + SPREAD_OTM_COEFF * np.abs(ln_m),
        SPREAD_BASE,
        0.25,
    )

    # Align indices
    common_dates = spot_series.index.intersection(vol_series.index).intersection(rate_series.index)

    frames = []
    for date in common_dates:
        S = spot_series[date]
        base_vol = vol_series[date]
        r = rate_series[date]

        if np.isnan(S) or np.isnan(base_vol) or np.isnan(r) or S <= 0:
            continue

        base_vol = max(float(base_vol), MIN_VOL)
        r = float(r)
        K_grid = MONEYNESS_GRID * S

        for i, (tau, mat_days) in enumerate(zip(tau_grid, MATURITY_DAYS)):
            expiration = date + timedelta(days=mat_days)

            # Vol surface: term structure + put-side skew
            skew_adj = SKEW_FACTOR * np.maximum(0.0, 1.0 - MONEYNESS_GRID) * base_vol
            vol_grid = np.maximum(base_vol + TERM_SLOPE * tau + skew_adj, MIN_VOL)

            calls, puts = _bs_call_put(S, K_grid, tau, r, vol_grid)

            # Independent bid/ask noise for calls and puts
            noise = rng.normal(0.0, NOISE_STD, (4, n_strikes))

            c_bid = np.maximum(
                calls * (1.0 - spread_ratios / 2.0) * (1.0 + noise[0]),
                MIN_PRICE,
            )
            c_ask = np.maximum(
                calls * (1.0 + spread_ratios / 2.0) * (1.0 + noise[1]),
                c_bid + MIN_PRICE,
            )
            p_bid = np.maximum(
                puts * (1.0 - spread_ratios / 2.0) * (1.0 + noise[2]),
                MIN_PRICE,
            )
            p_ask = np.maximum(
                puts * (1.0 + spread_ratios / 2.0) * (1.0 + noise[3]),
                p_bid + MIN_PRICE,
            )

            c_mid = (c_bid + c_ask) / 2.0
            p_mid = (p_bid + p_ask) / 2.0

            call_block = pd.DataFrame({
                'market':      market,
                'date':        date,
                'expiration':  pd.Timestamp(expiration),
                'strike':      np.round(K_grid, 4),
                'option_type': 'call',
                'bid':         np.round(c_bid, 4),
                'ask':         np.round(c_ask, 4),
                'mid':         np.round(c_mid, 4),
                'tau':         tau,
            })
            put_block = pd.DataFrame({
                'market':      market,
                'date':        date,
                'expiration':  pd.Timestamp(expiration),
                'strike':      np.round(K_grid, 4),
                'option_type': 'put',
                'bid':         np.round(p_bid, 4),
                'ask':         np.round(p_ask, 4),
                'mid':         np.round(p_mid, 4),
                'tau':         tau,
            })
            frames.append(call_block)
            frames.append(put_block)

    if not frames:
        return pd.DataFrame(
            columns=['market', 'date', 'expiration', 'strike',
                     'option_type', 'bid', 'ask', 'mid', 'tau']
        )

    return pd.concat(frames, ignore_index=True)


# ── Public loader ─────────────────────────────────────────────────────────────

def get_massive_client() -> RESTClient:
    """Return authenticated massive REST client."""
    api_key = os.getenv('MASSIVE_TOKEN', '')
    return RESTClient(api_key=api_key)


def load_option_quotes_daily(
    market: str,
    start_date: str,
    end_date: str,
    snapshot_time: str = '15:45',
    cache_dir: str = 'data/cache',
) -> pd.DataFrame:
    """
    Load daily EOD option quotes for *market* ('SPX' or 'RUT').

    Because minute-level NBBO option data is unavailable through the massive
    API, this function builds a synthetic but realistic option quote panel:

    1. Downloads daily underlying prices (SPY for SPX, IWM for RUT) via the
       massive REST API (with GBM fallback if the API is unavailable).
    2. Computes annualized realized volatility from a 21-business-day rolling
       window of log returns, multiplied by sqrt(252).
    3. Obtains a daily 1 Y OIS risk-free rate from ``data.ois_data``.
    4. Calls ``generate_synthetic_option_panel`` to produce Black-Scholes
       mid prices with a term-structure slope and OTM skew, plus realistic
       bid-ask spreads and small random noise.

    Parameters
    ----------
    market        : 'SPX' or 'RUT'
    start_date    : 'YYYY-MM-DD'
    end_date      : 'YYYY-MM-DD'
    snapshot_time : EOD snapshot label (stored but not used in synthesis)
    cache_dir     : directory for parquet cache files

    Returns
    -------
    DataFrame with columns:
        market, date, expiration, strike, option_type, bid, ask, mid, tau
    """
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(
        cache_dir,
        f'{market.upper()}_{start_date}_{end_date}_options.parquet',
    )

    if os.path.exists(cache_path):
        df = pd.read_parquet(cache_path)
        df['date'] = pd.to_datetime(df['date'])
        df['expiration'] = pd.to_datetime(df['expiration'])
        return df

    # ── 1. Underlying price series ────────────────────────────────────────────
    tr = load_total_return_index(market, start_date, end_date, cache_dir)

    # ── 2. Realized volatility (21-day rolling std of log returns) ────────────
    log_ret = np.log(tr / tr.shift(1))
    rvol = log_ret.rolling(window=RVOL_WINDOW, min_periods=RVOL_WINDOW).std() * np.sqrt(252)

    # Fill initial NaN values with the first valid estimate; floor at MIN_VOL
    rvol = rvol.ffill().fillna(MIN_VOL)
    rvol = rvol.clip(lower=MIN_VOL)

    # ── 3. Risk-free rate (1 Y OIS, converted to decimal) ────────────────────
    ois_pct = get_ois1y_series(start_date, end_date)
    rate = ois_pct / 100.0

    # Align all series to a common date index
    common = tr.index.intersection(rvol.index).intersection(rate.index)
    tr_aligned = tr.reindex(common)
    rvol_aligned = rvol.reindex(common)
    rate_aligned = rate.reindex(common)

    # ── 4. Generate synthetic option panel ───────────────────────────────────
    panel = generate_synthetic_option_panel(
        spot_series=tr_aligned,
        vol_series=rvol_aligned,
        market=market.upper(),
        rate_series=rate_aligned,
    )

    panel.to_parquet(cache_path, index=False)
    return panel
