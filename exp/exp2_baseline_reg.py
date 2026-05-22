"""
Experiment 2: Baseline implementation-risk regression.

Regresses CG_bp on GBM_1Y, GBM_10Y, BA_over_tau, NFCI with
OLS + HAC(Newey-West, maxlags=21) standard errors.

Outputs
-------
results/exp2_SPX_baseline_coefs.csv  – SPX coefficient table
results/exp2_RUT_baseline_coefs.csv  – RUT coefficient table
results/exp2_baseline_metrics.csv    – in-sample R², RMSE, MAE, Corr
results/exp2_baseline_coefs.png      – coefficient comparison chart
results/exp2_actual_vs_fitted.png    – actual vs fitted scatter
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.features.panel import build_regression_panel
from src.models.baseline_regression import (
    run_baseline_both_markets,
    format_coefficient_table,
    BASELINE_REGRESSORS,
)
from config import RESULTS_DIR, RANDOM_SEED, HAC_MAX_LAGS, DATA_CACHE_DIR


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exp2(carry_gap_panel: pd.DataFrame = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load panel ─────────────────────────────────────────────────────────
    if carry_gap_panel is None:
        print('Loading carry-gap panel from cache...')
        carry_gap_panel = pd.read_parquet(
            f'{DATA_CACHE_DIR}/carry_gap_panel.parquet'
        )

    # ── 2. Build regression panel ─────────────────────────────────────────────
    print('Building regression panel with baseline regressors...')
    reg_panel = build_regression_panel(carry_gap_panel, random_seed=RANDOM_SEED)
    print(f'  Regression panel: {len(reg_panel):,} rows, '
          f'{reg_panel["date"].nunique()} dates')

    # ── 3. Baseline regressions ───────────────────────────────────────────────
    print('\nRunning baseline regressions...')
    results = run_baseline_both_markets(reg_panel, max_lags=HAC_MAX_LAGS)

    all_coefs = []
    metrics_rows = []
    for mkt in ['SPX', 'RUT']:
        res = results[mkt]
        print(f'\n=== {mkt} Baseline Regression ===')
        coef_table = res['coefficients_df'].copy()
        coef_table['market'] = mkt
        print(coef_table[['coef', 'hac_se', 't_stat', 'p_value',
                           'significance']].to_string())
        coef_table.to_csv(
            f'{RESULTS_DIR}/exp2_{mkt.lower()}_baseline_coefs.csv'
        )
        all_coefs.append(coef_table)
        print('  Metrics:', res['metrics'])
        row = {'market': mkt}
        row.update(res['metrics'])
        metrics_rows.append(row)

    metrics_df = pd.DataFrame(metrics_rows)
    metrics_df.to_csv(f'{RESULTS_DIR}/exp2_baseline_metrics.csv', index=False)
    print(f'\nSaved metrics to {RESULTS_DIR}/exp2_baseline_metrics.csv')

    # ── 4. Coefficient comparison chart ───────────────────────────────────────
    _plot_coef_comparison(all_coefs, RESULTS_DIR)

    # ── 5. Actual vs fitted scatter ───────────────────────────────────────────
    _plot_actual_vs_fitted(results, RESULTS_DIR)

    return results, reg_panel


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_coef_comparison(all_coefs: list, results_dir: str):
    """Bar chart comparing HAC-significant coefficients for SPX and RUT."""
    coefs = pd.concat(all_coefs)
    regs = BASELINE_REGRESSORS  # skip const
    fig, axes = plt.subplots(1, len(regs), figsize=(4 * len(regs), 5), sharey=False)
    colors = {'SPX': 'steelblue', 'RUT': 'tomato'}
    for ax, reg in zip(axes, regs):
        for i, mkt in enumerate(['SPX', 'RUT']):
            row = coefs[(coefs['market'] == mkt) & (coefs.index == reg)]
            if len(row) == 0:
                continue
            c = float(row['coef'].iloc[0])
            se = float(row['hac_se'].iloc[0])
            ax.bar([i], [c], yerr=[[se], [se]], capsize=5,
                   color=colors[mkt], alpha=0.8, label=mkt)
            ax.axhline(0, color='black', linewidth=0.6)
        ax.set_title(reg, fontsize=10)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['SPX', 'RUT'])
        ax.set_ylabel('Coefficient')
    # de-duplicate legend
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colors.values()]
    fig.legend(handles, list(colors.keys()), loc='upper right',
               bbox_to_anchor=(1.0, 1.0))
    plt.suptitle('Baseline Regression: HAC Coefficients', fontsize=13)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp2_baseline_coefs.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp2_baseline_coefs.png')


def _plot_actual_vs_fitted(results: dict, results_dir: str):
    """Scatter of actual vs fitted CG_bp for each market."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mkt in zip(axes, ['SPX', 'RUT']):
        res = results[mkt]
        y_actual = res['y'].values
        y_fitted = res['result'].fittedvalues.values
        ax.scatter(y_fitted, y_actual, s=1, alpha=0.25, color='steelblue',
                   rasterized=True)
        lo = min(y_actual.min(), y_fitted.min())
        hi = max(y_actual.max(), y_fitted.max())
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        r2 = res['metrics']['R2']
        ax.set_title(f'{mkt}: Actual vs Fitted  (R²={r2:.3f})', fontsize=11)
        ax.set_xlabel('Fitted CG_bp')
        ax.set_ylabel('Actual CG_bp')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp2_actual_vs_fitted.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp2_actual_vs_fitted.png')


if __name__ == '__main__':
    run_exp2()
