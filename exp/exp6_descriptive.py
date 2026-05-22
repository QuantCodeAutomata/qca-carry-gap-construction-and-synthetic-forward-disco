"""
Experiment 6: Descriptive statistics and exploratory data analysis.

Produces summary tables and charts for all panel variables, including
carry gap, regressors, and their pairwise correlations.

Outputs
-------
results/exp6_descriptive_stats.csv   – full descriptive statistics table
results/exp6_correlation_matrix.png  – Pearson correlation heatmap
results/exp6_regressor_timeseries.png – time series of all regressors
results/exp6_acf_cg.png              – ACF/PACF of CG_bp (SPX, 6M maturity)
results/exp6_cg_vs_vol.png           – scatter of CG_bp vs GBM_1Y by market
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

try:
    from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
    _HAS_SM_PLOTS = True
except ImportError:
    _HAS_SM_PLOTS = False

from src.features.panel import build_extended_panel
from config import (
    RESULTS_DIR, RANDOM_SEED, MAIN_DRIFT_HORIZON,
    DATA_CACHE_DIR, BASELINE_REGRESSORS, EXTENDED_REGRESSORS,
)

_ALL_VARS = ['CG_bp'] + EXTENDED_REGRESSORS


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exp6(carry_gap_panel: pd.DataFrame = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load and build full panel ──────────────────────────────────────────
    if carry_gap_panel is None:
        print('Loading carry-gap panel from cache...')
        carry_gap_panel = pd.read_parquet(
            f'{DATA_CACHE_DIR}/carry_gap_panel.parquet'
        )

    print(f'Building extended panel (horizon={MAIN_DRIFT_HORIZON})...')
    panel = build_extended_panel(
        carry_gap_panel, horizon=MAIN_DRIFT_HORIZON, random_seed=RANDOM_SEED
    )

    # ── 2. Descriptive statistics ─────────────────────────────────────────────
    desc_rows = []
    for mkt in ['SPX', 'RUT']:
        sub = panel[panel['market'] == mkt][_ALL_VARS].dropna()
        for col in _ALL_VARS:
            s = sub[col]
            desc_rows.append({
                'market':   mkt,
                'variable': col,
                'count':    int(s.count()),
                'mean':     float(s.mean()),
                'std':      float(s.std()),
                'min':      float(s.min()),
                'p25':      float(s.quantile(0.25)),
                'p50':      float(s.median()),
                'p75':      float(s.quantile(0.75)),
                'max':      float(s.max()),
                'skew':     float(s.skew()),
                'kurt':     float(s.kurtosis()),
            })
    desc_df = pd.DataFrame(desc_rows)
    print('\nDescriptive statistics:')
    print(desc_df.to_string(index=False))
    desc_df.to_csv(f'{RESULTS_DIR}/exp6_descriptive_stats.csv', index=False)
    print(f'\nSaved: {RESULTS_DIR}/exp6_descriptive_stats.csv')

    # ── 3. Correlation matrix ─────────────────────────────────────────────────
    _plot_correlation_matrix(panel, RESULTS_DIR)

    # ── 4. Regressor time series ──────────────────────────────────────────────
    _plot_regressor_timeseries(panel, RESULTS_DIR)

    # ── 5. ACF / PACF of CG_bp ───────────────────────────────────────────────
    _plot_acf_cg(panel, RESULTS_DIR)

    # ── 6. CG_bp vs GBM_1Y scatter ───────────────────────────────────────────
    _plot_cg_vs_vol(panel, RESULTS_DIR)

    return panel, desc_df


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_correlation_matrix(panel: pd.DataFrame, results_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, mkt in zip(axes, ['SPX', 'RUT']):
        sub = panel[panel['market'] == mkt][_ALL_VARS].dropna()
        corr = sub.corr()
        im = ax.imshow(corr.values, cmap='RdBu', vmin=-1, vmax=1)
        ax.set_xticks(range(len(corr.columns)))
        ax.set_xticklabels(corr.columns, rotation=45, ha='right', fontsize=9)
        ax.set_yticks(range(len(corr.index)))
        ax.set_yticklabels(corr.index, fontsize=9)
        ax.set_title(f'{mkt}: Pearson Correlation')
        for i in range(len(corr)):
            for j in range(len(corr.columns)):
                ax.text(j, i, f'{corr.values[i, j]:.2f}',
                        ha='center', va='center', fontsize=7,
                        color='white' if abs(corr.values[i, j]) > 0.6 else 'black')
        plt.colorbar(im, ax=ax, shrink=0.8)
    plt.suptitle('Pairwise Pearson Correlation', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp6_correlation_matrix.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp6_correlation_matrix.png')


def _plot_regressor_timeseries(panel: pd.DataFrame, results_dir: str):
    vars_to_plot = EXTENDED_REGRESSORS
    n = len(vars_to_plot)
    fig, axes = plt.subplots(n, 1, figsize=(14, 2.5 * n), sharex=True)
    colors = {'SPX': 'steelblue', 'RUT': 'tomato'}
    for ax, var in zip(axes, vars_to_plot):
        for mkt in ['SPX', 'RUT']:
            ts = (panel[panel['market'] == mkt]
                  .groupby('date')[var].mean())
            ax.plot(ts.index, ts.values, linewidth=0.8,
                    color=colors[mkt], alpha=0.85, label=mkt)
        ax.set_ylabel(var, fontsize=9)
        ax.legend(fontsize=8, loc='upper right')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    axes[-1].set_xlabel('Date')
    plt.suptitle('Regressor Time Series', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp6_regressor_timeseries.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp6_regressor_timeseries.png')


def _plot_acf_cg(panel: pd.DataFrame, results_dir: str, tau_target: float = 0.4986):
    """ACF and PACF of CG_bp for SPX at ~6M maturity."""
    sub = panel[
        (panel['market'] == 'SPX') &
        (panel['tau'].round(4) == round(tau_target, 4))
    ]['CG_bp'].dropna()

    if len(sub) < 50:
        # Fall back to all taus
        sub = panel[panel['market'] == 'SPX']['CG_bp'].dropna()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if _HAS_SM_PLOTS and len(sub) > 20:
        plot_acf(sub, lags=min(60, len(sub) // 2 - 1), ax=axes[0], zero=False)
        plot_pacf(sub, lags=min(60, len(sub) // 2 - 1), ax=axes[1], zero=False)
    else:
        # Manual ACF
        n = len(sub)
        max_lag = min(60, n // 2)
        acf_vals = [sub.autocorr(lag=lag) for lag in range(1, max_lag + 1)]
        axes[0].stem(range(1, max_lag + 1), acf_vals, use_line_collection=True)
        axes[0].axhline(0, color='black', linewidth=0.8)
        ci = 1.96 / np.sqrt(n)
        axes[0].axhline(ci, color='blue', linestyle='--', linewidth=0.8)
        axes[0].axhline(-ci, color='blue', linestyle='--', linewidth=0.8)
        axes[1].set_visible(False)

    axes[0].set_title('ACF of CG_bp (SPX ~6M)')
    axes[1].set_title('PACF of CG_bp (SPX ~6M)')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp6_acf_cg.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp6_acf_cg.png')


def _plot_cg_vs_vol(panel: pd.DataFrame, results_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mkt in zip(axes, ['SPX', 'RUT']):
        sub = panel[panel['market'] == mkt][['CG_bp', 'GBM_1Y']].dropna()
        # Sample up to 5000 points for readability
        idx = np.random.default_rng(0).choice(len(sub),
                                               size=min(5000, len(sub)),
                                               replace=False)
        s = sub.iloc[idx]
        ax.scatter(s['GBM_1Y'], s['CG_bp'], s=2, alpha=0.3,
                   color='steelblue', rasterized=True)
        corr = float(sub.corr().loc['CG_bp', 'GBM_1Y'])
        ax.set_title(f'{mkt}: CG_bp vs GBM_1Y  (ρ={corr:.3f})', fontsize=11)
        ax.set_xlabel('GBM_1Y (annualised variance)')
        ax.set_ylabel('CG_bp')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp6_cg_vs_vol.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp6_cg_vs_vol.png')


if __name__ == '__main__':
    run_exp6()
