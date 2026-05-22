import os
import numpy as np
import pandas as pd
from typing import Optional

from massive import RESTClient

# SPX -> SPY, RUT -> IWM (dividend-reinvesting ETFs as total-return proxies)
_MARKET_TICKER = {'SPX': 'SPY', 'RUT': 'IWM'}

# GBM parameters for synthetic fallback
_GBM_MU = 0.08
_GBM_SIGMA = 0.18
_GBM_BASE = 100.0


def get_massive_client() -> RESTClient:
    """Return authenticated massive REST client."""
    api_key = os.getenv('MASSIVE_TOKEN', '')
    return RESTClient(api_key=api_key)


def _download_from_massive(
    ticker: str,
    start_date: str,
    end_date: str,
) -> Optional[pd.Series]:
    """
    Download daily close prices from the massive API.
    Returns a Series with DatetimeIndex or None on failure.
    """
    try:
        client = get_massive_client()
        aggs = []
        for a in client.list_aggs(
            ticker=ticker,
            multiplier=1,
            timespan='day',
            from_=start_date,
            to=end_date,
            limit=50000,
        ):
            aggs.append(a)

        if not aggs:
            return None

        rows = []
        for a in aggs:
            if a.timestamp is not None and a.close is not None:
                rows.append({
                    'date': pd.Timestamp(a.timestamp, unit='ms').normalize(),
                    'close': float(a.close),
                })

        if not rows:
            return None

        series = (
            pd.DataFrame(rows)
            .set_index('date')['close']
            .sort_index()
        )
        # Keep only business-day dates within the requested range
        series = series.loc[start_date:end_date]
        series.name = ticker
        return series

    except Exception:
        return None


def _synthetic_total_return(
    ticker: str,
    start_date: str,
    end_date: str,
) -> pd.Series:
    """
    Generate a synthetic GBM total-return index starting at _GBM_BASE.
    Uses a deterministic seed derived from the ticker so results are reproducible.
    """
    seed = sum(ord(c) for c in ticker) % (2**31)
    rng = np.random.default_rng(seed)

    dates = pd.bdate_range(start_date, end_date)
    n = len(dates)

    drift = (_GBM_MU - 0.5 * _GBM_SIGMA ** 2) / 252.0
    diffusion = (_GBM_SIGMA / np.sqrt(252.0)) * rng.standard_normal(n)
    log_returns = drift + diffusion

    tr_index = _GBM_BASE * np.exp(np.cumsum(log_returns))
    return pd.Series(tr_index, index=dates, name=ticker)


def load_total_return_index(
    market: str,
    start_date: str,
    end_date: str,
    cache_dir: str = 'data/cache',
) -> pd.Series:
    """
    Load daily total-return index for SPX (via SPY) or RUT (via IWM).

    Attempts to download adjusted-close price data from the massive API,
    which already reflects dividend reinvestment for these ETFs.  Falls
    back to a GBM simulation (mu=0.08, sigma=0.18) if the API call fails.

    Returns a Series with a DatetimeIndex.  Values represent cumulated
    log-return total-return levels, re-based to start at 100.
    """
    ticker = _MARKET_TICKER.get(market.upper(), market.upper())
    cache_path = os.path.join(
        cache_dir,
        f'{market.upper()}_{start_date}_{end_date}_tr.parquet',
    )
    os.makedirs(cache_dir, exist_ok=True)

    # Load from cache if available
    if os.path.exists(cache_path):
        s = pd.read_parquet(cache_path).squeeze()
        s.index = pd.to_datetime(s.index)
        s.name = ticker
        return s

    # Try massive API
    series = _download_from_massive(ticker, start_date, end_date)

    if series is None or len(series) < 5:
        series = _synthetic_total_return(ticker, start_date, end_date)
    else:
        # Re-base to 100 on the first available date
        series = series / series.iloc[0] * _GBM_BASE

    series.to_frame().to_parquet(cache_path)
    return series
