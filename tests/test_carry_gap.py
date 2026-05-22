"""
Tests for src/features/carry_gap.py

Covers: compute_carry_gap, build_synthetic_carry_gap_panel,
        validate_carry_gap_panel.
"""
import numpy as np
import pandas as pd
import pytest

from src.features.carry_gap import (
    compute_carry_gap,
    build_synthetic_carry_gap_panel,
    validate_carry_gap_panel,
)


# ── compute_carry_gap ─────────────────────────────────────────────────────────

class TestComputeCarryGap:
    def test_basic(self):
        """When B_hat = B_OIS, CG_bp = 0."""
        cg = compute_carry_gap(B_hat=0.98, B_OIS=0.98, tau=0.5)
        assert abs(cg) < 1e-10

    def test_positive_gap(self):
        """B_hat > B_OIS implies implied DF cheaper → positive CG."""
        cg = compute_carry_gap(B_hat=0.99, B_OIS=0.98, tau=1.0)
        assert cg > 0

    def test_bp_scale(self):
        """Known analytic value: CG = (1/tau) * ln(B_hat/B_OIS) * 10000."""
        B_hat, B_OIS, tau = 0.99, 0.98, 1.0
        expected = np.log(B_hat / B_OIS) * 10_000.0
        assert abs(compute_carry_gap(B_hat, B_OIS, tau) - expected) < 1e-8

    def test_tau_scaling(self):
        """Doubling tau halves CG_bp for the same log(B_hat/B_OIS)."""
        cg1 = compute_carry_gap(0.99, 0.98, 0.5)
        cg2 = compute_carry_gap(0.99, 0.98, 1.0)
        assert abs(cg1 - 2 * cg2) < 1e-8

    def test_nan_on_zero_B_hat(self):
        cg = compute_carry_gap(B_hat=0.0, B_OIS=0.98, tau=1.0)
        assert np.isnan(cg)

    def test_nan_on_negative_B_hat(self):
        cg = compute_carry_gap(B_hat=-0.01, B_OIS=0.98, tau=1.0)
        assert np.isnan(cg)

    def test_nan_on_zero_B_OIS(self):
        cg = compute_carry_gap(B_hat=0.98, B_OIS=0.0, tau=1.0)
        assert np.isnan(cg)

    def test_nan_on_negative_tau(self):
        cg = compute_carry_gap(B_hat=0.98, B_OIS=0.97, tau=-0.1)
        assert np.isnan(cg)

    def test_nan_on_zero_tau(self):
        cg = compute_carry_gap(B_hat=0.98, B_OIS=0.97, tau=0.0)
        assert np.isnan(cg)

    def test_nan_on_inf_input(self):
        cg = compute_carry_gap(B_hat=np.inf, B_OIS=0.97, tau=1.0)
        assert np.isnan(cg)

    def test_large_positive_gap(self):
        """B_hat = B_OIS * exp(0.01) with tau=1 → CG_bp ≈ 100."""
        B_OIS = 0.95
        B_hat = B_OIS * np.exp(0.01)    # 100 bp annualised
        cg = compute_carry_gap(B_hat, B_OIS, tau=1.0)
        assert abs(cg - 100.0) < 1e-8


# ── build_synthetic_carry_gap_panel ───────────────────────────────────────────

class TestBuildSyntheticCarryGapPanel:
    def test_shape(self, cg_panel):
        """Panel must have the expected columns."""
        required = {'date', 'market', 'expiration', 'tau', 'B_hat', 'B_OIS', 'CG_bp'}
        assert required.issubset(set(cg_panel.columns))

    def test_markets(self, cg_panel):
        assert set(cg_panel['market'].unique()) == {'SPX', 'RUT'}

    def test_row_count(self, cg_panel):
        """Must have multiple dates and maturities per market."""
        spx = cg_panel[cg_panel['market'] == 'SPX']
        assert spx['date'].nunique() > 100
        assert spx['tau'].nunique() >= 2

    def test_dates_in_range(self, cg_panel):
        assert cg_panel['date'].min() >= pd.Timestamp('2018-01-01')
        assert cg_panel['date'].max() <= pd.Timestamp('2023-12-31')

    def test_all_B_hat_positive(self, cg_panel):
        assert (cg_panel['B_hat'] > 0).all()

    def test_all_B_OIS_positive(self, cg_panel):
        assert (cg_panel['B_OIS'] > 0).all()

    def test_no_nan_CG_bp(self, cg_panel):
        assert cg_panel['CG_bp'].notna().all()

    def test_reproducibility(self):
        """Same seed → identical panels."""
        p1 = build_synthetic_carry_gap_panel('2020-01-02', '2020-06-30',
                                              random_seed=99)
        p2 = build_synthetic_carry_gap_panel('2020-01-02', '2020-06-30',
                                              random_seed=99)
        pd.testing.assert_frame_equal(p1, p2)

    def test_different_seeds_differ(self):
        """Different seeds → different CG values."""
        p1 = build_synthetic_carry_gap_panel('2020-01-02', '2020-06-30',
                                              random_seed=1)
        p2 = build_synthetic_carry_gap_panel('2020-01-02', '2020-06-30',
                                              random_seed=2)
        assert not p1['CG_bp'].equals(p2['CG_bp'])


# ── validate_carry_gap_panel ──────────────────────────────────────────────────

class TestValidateCarryGapPanel:
    def test_pass_on_valid_panel(self, cg_panel):
        val = validate_carry_gap_panel(cg_panel)
        assert val['pass'] is True

    def test_detects_nan_CG(self, cg_panel):
        bad = cg_panel.copy()
        bad.loc[0, 'CG_bp'] = np.nan
        val = validate_carry_gap_panel(bad)
        assert val['frac_nan_CG'] > 0
        assert val['pass'] is False

    def test_detects_negative_B_hat(self, cg_panel):
        bad = cg_panel.copy()
        bad.loc[0, 'B_hat'] = -0.01
        val = validate_carry_gap_panel(bad)
        assert val['all_B_hat_positive'] is False
        assert val['pass'] is False

    def test_n_obs_count(self, cg_panel):
        val = validate_carry_gap_panel(cg_panel)
        assert val['n_obs'] == len(cg_panel)

    def test_n_markets(self, cg_panel):
        val = validate_carry_gap_panel(cg_panel)
        assert val['n_markets'] == 2
