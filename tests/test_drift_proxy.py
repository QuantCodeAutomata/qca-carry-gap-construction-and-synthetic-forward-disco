"""
Tests for src/features/drift_proxy.py

Covers: _linear_slope, compute_drift_proxy — including the prior-only rule,
NaN structure, known-value recovery, and annualisation.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.drift_proxy import (
    _linear_slope,
    compute_drift_proxy,
    TRADING_DAYS_PER_YEAR,
)


# ── _linear_slope ─────────────────────────────────────────────────────────────

class TestLinearSlope:
    def test_constant_series(self):
        """All values equal → slope = 0."""
        y = np.full(10, 5.0)
        assert _linear_slope(y) == 0.0

    def test_perfect_linear(self):
        """y = a + b*x → slope = b."""
        n = 20
        b = 0.003
        y = 0.5 + b * np.arange(n)
        result = _linear_slope(y)
        assert abs(result - b) < 1e-10

    def test_too_short(self):
        assert np.isnan(_linear_slope(np.array([1.0])))

    def test_empty(self):
        assert np.isnan(_linear_slope(np.array([])))

    def test_two_points(self):
        y = np.array([0.0, 1.0])   # slope = 1
        assert abs(_linear_slope(y) - 1.0) < 1e-10


# ── compute_drift_proxy ───────────────────────────────────────────────────────

class TestComputeDriftProxy:
    def test_rolling_slope_linear_series(self, linear_log_prices):
        """
        Log-price = slope × t → drift proxy after warm-up should be
        slope × 252 (annualised).
        """
        lp, slope = linear_log_prices
        horizon = 20
        dp = compute_drift_proxy(lp, horizon=horizon)

        # First valid position is at index `horizon` (0-indexed)
        valid = dp.dropna()
        assert len(valid) > 0, 'No valid drift proxy values produced'

        expected_annual = slope * TRADING_DAYS_PER_YEAR
        # All valid values should equal the expected slope exactly (linear)
        np.testing.assert_allclose(
            valid.values,
            expected_annual,
            atol=1e-6,
            err_msg='Drift proxy does not recover linear slope',
        )

    def test_rolling_slope_nan_first_n_days(self, linear_log_prices):
        """
        First `horizon` values of the drift proxy must be NaN because the
        rolling window (shifted by 1) needs `horizon` prior observations.
        """
        lp, _ = linear_log_prices
        horizon = 30
        dp = compute_drift_proxy(lp, horizon=horizon)

        # Positions [0 .. horizon-1] should all be NaN
        assert dp.iloc[:horizon].isna().all(), (
            f'Expected first {horizon} values to be NaN')
        # Position `horizon` onward should be non-NaN (for a long enough series)
        assert dp.iloc[horizon:].notna().any(), (
            'Expected at least some non-NaN values after warm-up')

    def test_prior_only_rule(self):
        """
        Changing data at and after position T must not affect drift proxy
        values where the rolling window lies entirely before T.
        """
        n = 200
        idx = pd.date_range('2020-01-02', periods=n, freq='B')
        horizon = 20
        T = 80   # split point (0-indexed)

        # Original series
        rng = np.random.RandomState(0)
        lp_orig = pd.Series(
            np.cumsum(rng.randn(n) * 0.01), index=idx, name='lp'
        )

        # Modified series: identical up to T, completely different after T
        lp_mod = lp_orig.copy()
        lp_mod.iloc[T:] = 9999.0    # huge values that would dominate any slope

        dp_orig = compute_drift_proxy(lp_orig, horizon=horizon)
        dp_mod  = compute_drift_proxy(lp_mod,  horizon=horizon)

        # At position t, window uses lp[t-horizon : t-1].
        # For t <= T-1, the window is [t-horizon, t-1] where t-1 < T-1 < T.
        # → all window positions are < T → dp unchanged.
        safe_end = T - 1   # inclusive; window end = t-1 < T for t <= T-1
        pd.testing.assert_series_equal(
            dp_orig.iloc[:safe_end],
            dp_mod.iloc[:safe_end],
            check_names=False,
            rtol=1e-10,
        )

    def test_annualization_factor(self):
        """
        Daily slope of k per day should annualise to k × 252.
        """
        n = 600
        idx = pd.date_range('2019-01-02', periods=n, freq='B')
        k = 0.0005   # daily slope
        lp = pd.Series(k * np.arange(n), index=idx)
        horizon = 50
        dp = compute_drift_proxy(lp, horizon=horizon)

        expected = k * TRADING_DAYS_PER_YEAR
        np.testing.assert_allclose(
            dp.dropna().values,
            expected,
            atol=1e-8,
        )

    def test_output_length_matches_input(self, random_log_prices):
        dp = compute_drift_proxy(random_log_prices, horizon=126)
        assert len(dp) == len(random_log_prices)

    def test_index_preserved(self, random_log_prices):
        dp = compute_drift_proxy(random_log_prices, horizon=126)
        pd.testing.assert_index_equal(dp.index, random_log_prices.index)

    def test_different_horizons_different_values(self, random_log_prices):
        dp_short = compute_drift_proxy(random_log_prices, horizon=63)
        dp_long  = compute_drift_proxy(random_log_prices, horizon=252)
        # After sufficient warm-up, values should differ
        valid_both = dp_short.notna() & dp_long.notna()
        assert not dp_short[valid_both].equals(dp_long[valid_both])
