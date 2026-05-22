"""
Tests for src/features/baseline_gbm.py

Covers: compute_gbm_sigma_sq, build_gbm_regressors, generate_synthetic_vol_ois.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.baseline_gbm import (
    compute_gbm_sigma_sq,
    build_gbm_regressors,
    generate_synthetic_vol_ois,
)


# ── compute_gbm_sigma_sq ──────────────────────────────────────────────────────

class TestComputeGbmSigmaSq:
    def test_constant_series_zero_variance(self):
        """Constant log returns → variance = 0 → GBM_σ² = 0."""
        n = 300
        log_ret = pd.Series(np.full(n, 0.001),
                            index=pd.date_range('2020-01-02', periods=n, freq='B'))
        result = compute_gbm_sigma_sq(log_ret, window=252)
        assert result.dropna().abs().max() < 1e-10

    def test_known_variance(self):
        """
        Known daily σ from iid N(0, σ_daily):
        Expected annualised variance ≈ σ_daily² × 252.
        Test uses n=600 and 252-day window; Monte Carlo tolerance ≤ 3×σ_daily².
        """
        rng = np.random.default_rng(0)
        sigma_daily = 0.01
        n = 600
        log_ret = pd.Series(
            rng.normal(0, sigma_daily, n),
            index=pd.date_range('2020-01-02', periods=n, freq='B'),
        )
        result = compute_gbm_sigma_sq(log_ret, window=252)
        expected_mean = sigma_daily ** 2 * 252   # = 0.0252
        # Mean of rolling estimates should be close to expected (within 50%)
        assert abs(result.dropna().mean() - expected_mean) < expected_mean * 0.5

    def test_output_all_nonnegative(self):
        """Variance is always non-negative."""
        rng = np.random.default_rng(1)
        log_ret = pd.Series(rng.standard_normal(400),
                            index=pd.date_range('2020-01-02', periods=400, freq='B'))
        result = compute_gbm_sigma_sq(log_ret, window=252)
        assert (result.dropna() >= 0).all()

    def test_nan_before_window(self):
        """First (window-1) entries must be NaN when min_periods==window."""
        n = 300
        log_ret = pd.Series(np.random.randn(n),
                            index=pd.date_range('2020-01-02', periods=n, freq='B'))
        window = 60
        result = compute_gbm_sigma_sq(log_ret, window=window, min_periods=window)
        assert result.iloc[:window - 1].isna().all()
        assert result.iloc[window - 1:].notna().all()

    def test_min_periods_less_than_window(self):
        """min_periods=1 → first valid entry earlier."""
        log_ret = pd.Series(np.ones(50),
                            index=pd.date_range('2020-01-02', periods=50, freq='B'))
        result = compute_gbm_sigma_sq(log_ret, window=252, min_periods=10)
        # Some non-NaN values despite window > n
        assert result.notna().any()


# ── build_gbm_regressors ──────────────────────────────────────────────────────

class TestBuildGbmRegressors:
    @pytest.fixture
    def log_ret_series(self):
        rng = np.random.RandomState(42)
        n = 600
        return pd.Series(
            rng.randn(n) * 0.01,
            index=pd.date_range('2018-01-02', periods=n, freq='B'),
        )

    def test_returns_gbm_columns(self, log_ret_series):
        result = build_gbm_regressors(log_ret_series)
        assert 'GBM_1Y'  in result.columns
        assert 'GBM_10Y' in result.columns

    def test_gbm_1y_all_nonnegative(self, log_ret_series):
        result = build_gbm_regressors(log_ret_series)
        assert (result['GBM_1Y'].dropna() >= 0).all()

    def test_gbm_10y_all_nonnegative(self, log_ret_series):
        result = build_gbm_regressors(log_ret_series)
        assert (result['GBM_10Y'].dropna() >= 0).all()

    def test_no_nan_after_warmup(self, log_ret_series):
        """No NaN after the 252-day warm-up period."""
        result = build_gbm_regressors(log_ret_series)
        assert result.iloc[252:].notna().all().all()

    def test_index_alignment(self, log_ret_series):
        result = build_gbm_regressors(log_ret_series)
        assert (result.index == log_ret_series.index).all()


# ── generate_synthetic_vol_ois ────────────────────────────────────────────────

class TestGenerateSyntheticVolOis:
    @pytest.fixture(params=['SPX', 'RUT'])
    def vol_df(self, request):
        return generate_synthetic_vol_ois(
            '2020-01-02', '2022-12-30', request.param, random_seed=0
        )

    def test_columns_present(self, vol_df):
        for col in ['log_return', 'GBM_1Y', 'GBM_10Y', 'OIS_1Y']:
            assert col in vol_df.columns

    def test_ois_positive(self, vol_df):
        assert (vol_df['OIS_1Y'].dropna() > 0).all()

    def test_ois_bounded(self, vol_df):
        assert (vol_df['OIS_1Y'] <= 0.10).all()

    def test_gbm_1y_positive_after_warmup(self, vol_df):
        assert (vol_df['GBM_1Y'].dropna() > 0).all()

    def test_different_markets_differ(self):
        spx = generate_synthetic_vol_ois('2020-01-02', '2021-12-31', 'SPX',
                                         random_seed=0)
        rut = generate_synthetic_vol_ois('2020-01-02', '2021-12-31', 'RUT',
                                         random_seed=0)
        assert not spx['log_return'].equals(rut['log_return'])
