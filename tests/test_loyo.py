"""
Tests for src/evaluation/loyo.py

Covers: get_calendar_year_folds, pooled_oos_r2, run_loyo_fold,
        run_loyo_validation — including the no-look-ahead guarantee.
"""
import numpy as np
import pandas as pd
import pytest

from src.evaluation.loyo import (
    get_calendar_year_folds,
    pooled_oos_r2,
    run_loyo_fold,
    run_loyo_validation,
    run_loyo_both_markets,
    format_loyo_summary,
)
from config import BASELINE_REGRESSORS, EXTENDED_REGRESSORS


# ── get_calendar_year_folds ───────────────────────────────────────────────────

class TestGetCalendarYearFolds:
    def test_folds_cover_all_years(self, sample_panel):
        years_in_panel = set(sample_panel['date'].dt.year.unique().tolist())
        folds = get_calendar_year_folds(sample_panel)
        assert set(folds) == years_in_panel

    def test_folds_sorted(self, sample_panel):
        folds = get_calendar_year_folds(sample_panel)
        assert folds == sorted(folds)

    def test_single_year_panel(self):
        dates = pd.date_range('2021-01-04', '2021-12-31', freq='B')
        panel = pd.DataFrame({
            'date': dates,
            'market': 'SPX',
            'CG_bp': np.random.randn(len(dates)),
        })
        folds = get_calendar_year_folds(panel)
        assert folds == [2021]

    def test_returns_list(self, sample_panel):
        assert isinstance(get_calendar_year_folds(sample_panel), list)


# ── pooled_oos_r2 ─────────────────────────────────────────────────────────────

class TestPooledOosR2:
    def test_perfect_predictions(self):
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(pooled_oos_r2(y, y) - 1.0) < 1e-10

    def test_mean_prediction(self):
        """Predicting the grand mean gives R2=0."""
        y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_hat = pd.Series([3.0] * 5)
        assert abs(pooled_oos_r2(y, y_hat)) < 1e-10

    def test_known_value(self):
        """y=[1,2,3,4,5], y_hat=[1.1,2.1,3.1,4.1,5.1] → R2=0.995."""
        y     = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
        y_hat = pd.Series([1.1, 2.1, 3.1, 4.1, 5.1])
        # SS_res = 5 × 0.1² = 0.05;  SS_tot = (4+1+0+1+4) = 10
        expected = 1 - 0.05 / 10.0
        assert abs(pooled_oos_r2(y, y_hat) - expected) < 1e-10

    def test_can_be_negative(self):
        """Terrible predictions → negative R2."""
        y     = pd.Series([1.0, 2.0, 3.0])
        y_hat = pd.Series([10.0, 20.0, 30.0])
        assert pooled_oos_r2(y, y_hat) < 0


# ── run_loyo_fold ─────────────────────────────────────────────────────────────

class TestRunLoyoFold:
    def test_runs_without_error(self, sample_panel):
        years = get_calendar_year_folds(sample_panel)
        fold = run_loyo_fold(sample_panel, years[0], BASELINE_REGRESSORS,
                             market='SPX')
        assert fold is not None

    def test_returns_expected_keys(self, sample_panel):
        years = get_calendar_year_folds(sample_panel)
        fold = run_loyo_fold(sample_panel, years[0], BASELINE_REGRESSORS,
                             market='SPX')
        for key in ['holdout_year', 'result', 'y_pred', 'y_test', 'metrics']:
            assert key in fold

    def test_none_on_missing_year(self, sample_panel):
        """Holdout year not in panel → returns None."""
        fold = run_loyo_fold(sample_panel, 1900, BASELINE_REGRESSORS,
                             market='SPX')
        assert fold is None

    def test_pred_and_test_same_length(self, sample_panel):
        years = get_calendar_year_folds(sample_panel)
        fold = run_loyo_fold(sample_panel, years[0], BASELINE_REGRESSORS,
                             market='SPX')
        assert len(fold['y_pred']) == len(fold['y_test'])


# ── run_loyo_validation ───────────────────────────────────────────────────────

class TestRunLoyoValidation:
    def test_loyo_returns_valid_metrics(self, sample_panel):
        res = run_loyo_validation(sample_panel, BASELINE_REGRESSORS,
                                  market='SPX')
        assert 'pooled_R2'  in res
        assert 'mean_R2'    in res
        assert 'fold_metrics' in res

    def test_fold_count_matches_years(self, sample_panel):
        years = get_calendar_year_folds(sample_panel)
        res   = run_loyo_validation(sample_panel, BASELINE_REGRESSORS,
                                    market='SPX')
        # Number of valid folds ≤ number of calendar years
        assert len(res['fold_results']) <= len(years)

    def test_pooled_r2_is_scalar(self, sample_panel):
        res = run_loyo_validation(sample_panel, BASELINE_REGRESSORS,
                                  market='SPX')
        assert isinstance(res['pooled_R2'], float)
        assert np.isfinite(res['pooled_R2'])

    def test_no_lookahead_in_loyo(self, sample_panel):
        """
        Changing the held-out year's CG_bp values must not alter OOS
        predictions for that year.  The model is trained exclusively on
        the remaining years and therefore cannot see the modified targets.
        """
        years = get_calendar_year_folds(sample_panel)
        holdout_year = years[0]

        panel_modified = sample_panel.copy()
        mask = panel_modified['date'].dt.year == holdout_year
        # Replace CG_bp in the holdout year with extreme noise
        panel_modified.loc[mask, 'CG_bp'] = (
            1e6 * np.random.default_rng(99).standard_normal(mask.sum())
        )

        res_orig = run_loyo_validation(sample_panel, BASELINE_REGRESSORS,
                                       market='SPX')
        res_mod  = run_loyo_validation(panel_modified, BASELINE_REGRESSORS,
                                       market='SPX')

        fold_orig = next(
            f for f in res_orig['fold_results']
            if f['holdout_year'] == holdout_year
        )
        fold_mod = next(
            f for f in res_mod['fold_results']
            if f['holdout_year'] == holdout_year
        )

        # OOS predictions must be identical (model never saw holdout-year targets)
        pd.testing.assert_series_equal(
            fold_orig['y_pred'].reset_index(drop=True),
            fold_mod['y_pred'].reset_index(drop=True),
            check_names=False,
            rtol=1e-8,
            atol=1e-8,
        )

    def test_coefficient_signs_matrix_shape(self, sample_panel):
        res = run_loyo_validation(sample_panel, BASELINE_REGRESSORS,
                                  market='SPX')
        sign_mat = res['coefficient_signs']
        # Rows = years, columns include regressors + const
        assert sign_mat.shape[1] >= len(BASELINE_REGRESSORS)
        assert sign_mat.shape[0] == len(res['fold_results'])

    def test_n_positive_r2_bounded(self, sample_panel):
        res = run_loyo_validation(sample_panel, BASELINE_REGRESSORS,
                                  market='SPX')
        assert 0 <= res['n_positive_R2'] <= len(res['fold_results'])


# ── run_loyo_both_markets ─────────────────────────────────────────────────────

class TestRunLoyoBothMarkets:
    def test_both_markets_present(self, sample_panel):
        valid = sample_panel.dropna(subset=EXTENDED_REGRESSORS)
        res = run_loyo_both_markets(valid, BASELINE_REGRESSORS,
                                    EXTENDED_REGRESSORS)
        assert 'SPX' in res
        assert 'RUT' in res

    def test_both_models_present(self, sample_panel):
        valid = sample_panel.dropna(subset=EXTENDED_REGRESSORS)
        res = run_loyo_both_markets(valid, BASELINE_REGRESSORS,
                                    EXTENDED_REGRESSORS)
        assert 'baseline' in res['SPX']
        assert 'extended' in res['SPX']

    def test_format_loyo_summary(self, sample_panel):
        valid = sample_panel.dropna(subset=EXTENDED_REGRESSORS)
        res = run_loyo_both_markets(valid, BASELINE_REGRESSORS,
                                    EXTENDED_REGRESSORS)
        summary = format_loyo_summary(res)
        assert 'pooled_R2' in summary.columns
        assert len(summary) == 4   # 2 markets × 2 models
