"""
Tests for src/pricing/synth_forward.py

Covers: compute_mid_quotes, apply_quote_filters, estimate_discount_factor_cell.
"""
import numpy as np
import pandas as pd
import pytest

from src.pricing.synth_forward import (
    compute_mid_quotes,
    apply_quote_filters,
    estimate_discount_factor_cell,
)


# ── compute_mid_quotes ────────────────────────────────────────────────────────

class TestComputeMidQuotes:
    def test_basic(self):
        df = pd.DataFrame({'bid': [1.0, 2.0, 3.0], 'ask': [2.0, 3.0, 4.0]})
        result = compute_mid_quotes(df)
        expected = pd.Series([1.5, 2.5, 3.5])
        pd.testing.assert_series_equal(result['mid'].reset_index(drop=True),
                                       expected, check_names=False)

    def test_does_not_modify_original(self):
        df = pd.DataFrame({'bid': [1.0], 'ask': [3.0]})
        _ = compute_mid_quotes(df)
        assert 'mid' not in df.columns

    def test_returns_copy(self):
        df = pd.DataFrame({'bid': [1.0], 'ask': [3.0]})
        result = compute_mid_quotes(df)
        assert result is not df

    def test_symmetric(self):
        """mid = (bid + ask) / 2 exactly."""
        df = pd.DataFrame({'bid': [0.0, 5.0, 10.0],
                           'ask': [0.0, 5.0, 10.0]})
        result = compute_mid_quotes(df)
        pd.testing.assert_series_equal(result['mid'], result['bid'],
                                       check_names=False)


# ── apply_quote_filters ───────────────────────────────────────────────────────

class TestApplyQuoteFilters:
    def _make_df(self, bid, ask):
        mid = [(b + a) / 2 for b, a in zip(bid, ask)]
        return pd.DataFrame({'bid': bid, 'ask': ask, 'mid': mid})

    def test_removes_mid_below_minimum(self):
        df = self._make_df(
            bid=[0.001, 1.0,  2.0],
            ask=[0.015, 1.10, 2.20],
        )
        clean, excluded = apply_quote_filters(df, min_mid=0.01)
        # Row 0: mid = 0.008 < 0.01 → excluded
        assert len(clean) == 2
        assert 'mid_below_minimum' in excluded['exclusion_reason'].values

    def test_removes_crossed_quotes(self):
        df = self._make_df(
            bid=[1.10, 1.0],
            ask=[1.00, 2.0],
        )
        # Row 0: bid > ask → crossed
        df.loc[0, 'mid'] = 1.05  # re-set to avoid nonpositive mid filter
        clean, excluded = apply_quote_filters(df)
        assert len(clean) == 1
        assert 'crossed_quotes' in excluded['exclusion_reason'].values

    def test_removes_nonpositive_mid(self):
        df = pd.DataFrame({'bid': [0.0, 1.0], 'ask': [0.0, 2.0],
                           'mid': [0.0, 1.5]})
        clean, excluded = apply_quote_filters(df, min_mid=0.0)
        assert len(clean) == 1
        assert 'nonpositive_mid' in excluded['exclusion_reason'].values

    def test_removes_missing_bid_ask(self):
        df = pd.DataFrame({
            'bid': [np.nan, 1.0],
            'ask': [1.0, 2.0],
            'mid': [0.5, 1.5],
        })
        clean, excluded = apply_quote_filters(df)
        assert len(clean) == 1
        assert 'missing_quotes' in excluded['exclusion_reason'].values

    def test_removes_excessive_spread(self):
        # spread / mid > max_spread_ratio=0.5
        # bid=1, ask=3 → mid=2, spread=2, ratio=2/2=1.0 > 0.5 → excluded
        df = self._make_df(bid=[1.0, 1.0], ask=[3.0, 1.2])
        clean, excluded = apply_quote_filters(df, min_mid=0.01,
                                              max_spread_ratio=0.5)
        assert len(clean) == 1
        assert 'excessive_spread' in excluded['exclusion_reason'].values

    def test_clean_df_no_exclusions(self):
        df = self._make_df(bid=[1.0, 2.0], ask=[1.05, 2.05])
        clean, excluded = apply_quote_filters(df, min_mid=0.01,
                                              max_spread_ratio=2.0)
        assert len(clean) == 2
        assert len(excluded) == 0

    def test_returns_reset_index(self):
        df = self._make_df(bid=[0.001, 1.0, 2.0], ask=[0.015, 1.1, 2.1])
        clean, _ = apply_quote_filters(df, min_mid=0.01)
        assert list(clean.index) == list(range(len(clean)))


# ── estimate_discount_factor_cell ─────────────────────────────────────────────

class TestEstimateDiscountFactorCell:
    def test_known_values(self, call_put_pair_exact):
        """OLS should recover B=0.97 exactly from perfect PCP data."""
        _, _, strikes, G_vals, B_true, _ = call_put_pair_exact
        B_hat, F_hat, r2, n = estimate_discount_factor_cell(
            strikes, G_vals, min_strikes=5
        )
        assert B_hat is not None
        assert abs(B_hat - B_true) < 1e-8
        assert abs(r2 - 1.0) < 1e-8
        assert n == 5

    def test_negative_b_hat_dropped(self):
        """Positive slope (negative B_hat) → returns (None, None, r2, n)."""
        strikes   = np.array([90.0, 95.0, 100.0, 105.0, 110.0])
        # Craft G values that increase with K (opposite to put-call parity)
        G_vals    = 0.5 * (strikes - 90.0)   # positive slope → b>0 → B<0
        B_hat, F_hat, r2, n = estimate_discount_factor_cell(strikes, G_vals)
        assert B_hat is None
        assert F_hat is None
        assert r2 is not None    # r_squared still computed
        assert n == 5

    def test_min_strikes_requirement(self):
        """Fewer observations than min_strikes → all Nones."""
        strikes = np.array([100.0, 105.0])
        G_vals  = np.array([2.0, -2.0])
        B_hat, F_hat, r2, n = estimate_discount_factor_cell(
            strikes, G_vals, min_strikes=5
        )
        assert B_hat is None
        assert F_hat is None
        assert r2 is None
        assert n == 2

    def test_f_hat_consistency(self, call_put_pair_exact):
        """F_hat should equal F used to generate data."""
        _, _, strikes, G_vals, B_true, F_true = call_put_pair_exact
        B_hat, F_hat, r2, _ = estimate_discount_factor_cell(strikes, G_vals)
        assert F_hat is not None
        assert abs(F_hat - F_true) < 1e-6

    def test_exact_5_strikes_accepted(self):
        """Exactly min_strikes observations should pass."""
        B = 0.95
        F = 100.0
        strikes = np.array([95.0, 97.5, 100.0, 102.5, 105.0])
        G_vals  = B * (F - strikes)
        B_hat, _, _, _ = estimate_discount_factor_cell(
            strikes, G_vals, min_strikes=5
        )
        assert B_hat is not None
        assert abs(B_hat - B) < 1e-8
