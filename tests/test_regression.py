"""
Tests for src/models/baseline_regression.py and extended_regression.py

Covers: prepare_regression_data, fit_baseline_hac, compute_insample_metrics,
        run_baseline_regression, format_coefficient_table,
        run_extended_regression.
"""
import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

from src.models.baseline_regression import (
    BASELINE_REGRESSORS,
    prepare_regression_data,
    fit_baseline_hac,
    compute_insample_metrics,
    run_baseline_regression,
    run_baseline_both_markets,
    format_coefficient_table,
)
from src.models.extended_regression import (
    EXTENDED_REGRESSORS,
    run_extended_regression,
    run_extended_both_markets,
    compare_models,
)


# ── prepare_regression_data ───────────────────────────────────────────────────

class TestPrepareRegressionData:
    def test_filters_to_market(self, sample_panel):
        X, y = prepare_regression_data(sample_panel, 'SPX', BASELINE_REGRESSORS)
        # All rows must belong to SPX (no RUT contamination)
        spx_rows = sample_panel[sample_panel['market'] == 'SPX']
        assert len(y) <= len(spx_rows)

    def test_drops_nan_rows(self, sample_panel):
        panel_w_nan = sample_panel.copy()
        panel_w_nan.loc[panel_w_nan['market'] == 'SPX', 'GBM_1Y'] = np.nan
        X, y = prepare_regression_data(panel_w_nan, 'SPX', BASELINE_REGRESSORS)
        assert len(X) == 0   # all rows have NaN in GBM_1Y

    def test_adds_constant_column(self, sample_panel):
        X, y = prepare_regression_data(sample_panel, 'SPX', BASELINE_REGRESSORS)
        assert 'const' in X.columns

    def test_x_y_same_length(self, sample_panel):
        X, y = prepare_regression_data(sample_panel, 'SPX', BASELINE_REGRESSORS)
        assert len(X) == len(y)

    def test_raises_on_missing_market_col(self):
        bad = pd.DataFrame({'CG_bp': [1.0], 'GBM_1Y': [0.1]})
        with pytest.raises(KeyError):
            prepare_regression_data(bad, 'SPX', ['GBM_1Y'])


# ── fit_baseline_hac ──────────────────────────────────────────────────────────

class TestFitBaselineHac:
    @pytest.fixture
    def simple_xy(self):
        rng = np.random.RandomState(0)
        n = 200
        X_raw = rng.randn(n, 3)
        y = 1.5 + 0.3 * X_raw[:, 0] - 0.2 * X_raw[:, 1] + rng.randn(n) * 0.5
        X = pd.DataFrame(X_raw, columns=['x1', 'x2', 'x3'])
        X = sm.add_constant(X, has_constant='add')
        y = pd.Series(y)
        return X, y

    def test_returns_result_object(self, simple_xy):
        X, y = simple_xy
        result = fit_baseline_hac(X, y, max_lags=5)
        assert hasattr(result, 'params')
        assert hasattr(result, 'pvalues')
        assert hasattr(result, 'tvalues')
        assert hasattr(result, 'bse')

    def test_coefficient_count(self, simple_xy):
        X, y = simple_xy
        result = fit_baseline_hac(X, y, max_lags=5)
        assert len(result.params) == X.shape[1]

    def test_hac_se_different_from_ols_se(self, simple_xy):
        X, y = simple_xy
        hac_result = fit_baseline_hac(X, y, max_lags=5)
        ols_result = sm.OLS(y, X).fit()
        # HAC SEs should generally differ from homoskedastic OLS SEs
        assert not np.allclose(hac_result.bse.values, ols_result.bse.values,
                               atol=1e-4)


# ── compute_insample_metrics ──────────────────────────────────────────────────

class TestComputeInsampleMetrics:
    def test_perfect_fit(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        metrics = compute_insample_metrics(y, y)
        assert abs(metrics['R2'] - 1.0) < 1e-10
        assert abs(metrics['RMSE'])    < 1e-10
        assert abs(metrics['MAE'])     < 1e-10
        assert abs(metrics['Corr'] - 1.0) < 1e-10

    def test_zero_fit(self):
        """Predict the mean everywhere → R2 = 0."""
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_hat = pd.Series([3.0, 3.0, 3.0, 3.0, 3.0])
        metrics = compute_insample_metrics(y, y_hat)
        assert abs(metrics['R2']) < 1e-10

    def test_r2_in_range(self, sample_panel):
        from src.models.baseline_regression import run_baseline_regression
        res = run_baseline_regression(sample_panel, 'SPX')
        r2 = res['metrics']['R2']
        assert -5.0 < r2 <= 1.0

    def test_adj_r2_computed_when_k_provided(self):
        y = pd.Series(np.random.randn(100))
        y_hat = y + 0.1 * np.random.randn(100)
        metrics = compute_insample_metrics(y, y_hat, k=4)
        assert 'adj_R2' in metrics

    def test_no_adj_r2_without_k(self):
        y = pd.Series(np.random.randn(100))
        y_hat = y + 0.1 * np.random.randn(100)
        metrics = compute_insample_metrics(y, y_hat)
        assert 'adj_R2' not in metrics

    def test_rmse_nonnegative(self):
        y = pd.Series([1.0, 2.0, 3.0])
        y_hat = pd.Series([1.5, 2.5, 3.5])
        assert compute_insample_metrics(y, y_hat)['RMSE'] >= 0


# ── run_baseline_regression ───────────────────────────────────────────────────

class TestRunBaselineRegression:
    def test_runs_without_error(self, sample_panel):
        res = run_baseline_regression(sample_panel, 'SPX')
        assert 'result' in res
        assert 'metrics' in res
        assert 'coefficients_df' in res

    def test_rut_runs_without_error(self, sample_panel):
        res = run_baseline_regression(sample_panel, 'RUT')
        assert res['result'] is not None

    def test_coefficient_names(self, sample_panel):
        res = run_baseline_regression(sample_panel, 'SPX')
        params = res['result'].params
        for reg in BASELINE_REGRESSORS:
            assert reg in params.index, f'{reg} missing from regression params'

    def test_both_markets_run(self, sample_panel):
        results = run_baseline_both_markets(sample_panel)
        assert 'SPX' in results
        assert 'RUT' in results


# ── format_coefficient_table ──────────────────────────────────────────────────

class TestFormatCoefficientTable:
    @pytest.fixture
    def fitted_result(self, sample_panel):
        return run_baseline_regression(sample_panel, 'SPX')['result']

    def test_columns_present(self, fitted_result):
        df = format_coefficient_table(fitted_result, 'TestModel')
        for col in ['coef', 'hac_se', 't_stat', 'p_value', 'significance']:
            assert col in df.columns

    def test_model_column(self, fitted_result):
        df = format_coefficient_table(fitted_result, 'MyModel')
        assert (df['model'] == 'MyModel').all()

    def test_significance_stars(self, fitted_result):
        df = format_coefficient_table(fitted_result, 'Test')
        # stars should be in the allowed set
        allowed = {'', '*', '**', '***'}
        assert set(df['significance'].unique()).issubset(allowed)

    def test_no_nan_in_core_columns(self, fitted_result):
        df = format_coefficient_table(fitted_result, 'Test')
        assert df[['coef', 'hac_se', 't_stat', 'p_value']].notna().all().all()


# ── run_extended_regression ───────────────────────────────────────────────────

class TestRunExtendedRegression:
    def test_has_drift_coef(self, sample_panel):
        """Extended model must include GBM_mu_1Y in the coefficient table."""
        # Drop rows where GBM_mu_1Y is NaN (drift warm-up)
        valid = sample_panel.dropna(subset=['GBM_mu_1Y'])
        res = run_extended_regression(valid, 'SPX')
        assert 'GBM_mu_1Y' in res['result'].params.index

    def test_more_regressors_than_baseline(self, sample_panel):
        valid = sample_panel.dropna(subset=['GBM_mu_1Y'])
        base = run_baseline_regression(valid, 'SPX')
        ext  = run_extended_regression(valid, 'SPX')
        assert len(ext['result'].params) > len(base['result'].params)

    def test_compare_models_has_required_cols(self, sample_panel):
        valid = sample_panel.dropna(subset=EXTENDED_REGRESSORS)
        base_all  = run_baseline_both_markets(valid)
        ext_all   = run_extended_both_markets(valid)
        comp = compare_models(base_all, ext_all, 'SPX')
        for col in ['metric', 'baseline', 'extended', 'delta']:
            assert col in comp.columns
