"""
Shared pytest fixtures for carry-gap test suite.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import pytest

from src.features.carry_gap import build_synthetic_carry_gap_panel
from src.features.panel import build_regression_panel, build_extended_panel
from config import BASELINE_REGRESSORS, EXTENDED_REGRESSORS


# ── Date range for fast test panels ──────────────────────────────────────────

_TEST_START = '2019-01-02'
_TEST_END   = '2022-12-30'
_TEST_SEED  = 7


# ── Carry-gap panel (shared across multiple test modules) ─────────────────────

@pytest.fixture(scope='session')
def cg_panel():
    """Synthetic carry-gap panel for 2019-2022 (fast)."""
    return build_synthetic_carry_gap_panel(
        start_date=_TEST_START,
        end_date=_TEST_END,
        random_seed=_TEST_SEED,
    )


@pytest.fixture(scope='session')
def sample_panel(cg_panel):
    """
    Full regression panel (baseline + extended regressors) for 2019-2022.
    Scope='session' so the slow panel build runs only once.
    """
    return build_extended_panel(cg_panel, horizon=126, random_seed=_TEST_SEED)


# ── Option-pricing fixtures ───────────────────────────────────────────────────

@pytest.fixture
def call_put_pair_exact():
    """
    Exact put-call parity data: C - P = B*(F - K) with B=0.97, F=100.
    Five matched call-put pairs at strikes [92, 96, 100, 104, 108].
    """
    B_true = 0.97
    F = 100.0
    strikes = np.array([92.0, 96.0, 100.0, 104.0, 108.0])
    G_vals  = B_true * (F - strikes)     # G = C − P = B·F − B·K

    # Splits arbitrarily: calls bear positive PCP spread
    call_mid = np.maximum(G_vals / 2 + 5, 0.10)
    put_mid  = call_mid - G_vals

    calls = pd.DataFrame({'strike': strikes, 'mid': call_mid,
                           'bid': call_mid * 0.95, 'ask': call_mid * 1.05})
    puts  = pd.DataFrame({'strike': strikes, 'mid': put_mid,
                           'bid': put_mid * 0.95, 'ask': put_mid * 1.05})
    return calls, puts, strikes, G_vals, B_true, F


@pytest.fixture
def raw_quotes_df():
    """Small option quote DataFrame with some bad quotes for filter tests."""
    data = {
        'bid': [1.00, 0.005, 2.00, 1.00, -0.10],
        'ask': [1.10, 0.015, 2.20, 5.00,  0.50],
        'mid': [1.05, 0.010, 2.10, 3.00,  0.20],
    }
    return pd.DataFrame(data)


# ── Log-price series for drift-proxy tests ────────────────────────────────────

@pytest.fixture
def linear_log_prices():
    """
    Log-price series increasing at exactly 0.001 per day.
    Slope annualised = 0.001 × 252 = 0.252.
    """
    n = 600
    idx = pd.date_range('2020-01-02', periods=n, freq='B')
    slope = 0.001
    values = slope * np.arange(n)
    return pd.Series(values, index=idx, name='log_price'), slope


@pytest.fixture
def random_log_prices():
    """Random walk log-price series for general drift-proxy tests."""
    rng = np.random.RandomState(_TEST_SEED)
    n = 800
    idx = pd.date_range('2019-01-02', periods=n, freq='B')
    log_ret = 0.0003 + rng.randn(n) * 0.012
    log_price = pd.Series(np.cumsum(log_ret), index=idx, name='log_price')
    return log_price
