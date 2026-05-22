"""
Experiment 3: Extended model — adding the drift proxy GBM_mu_1Y.

Compares baseline (4 regressors) with extended model (5 regressors).
Documents the incremental explanatory power of the drift proxy.

Outputs
-------
results/exp3_SPX_extended_coefs.csv  – extended SPX coefficient table
results/exp3_RUT_extended_coefs.csv  – extended RUT coefficient table
results/exp3_model_comparison.csv    – R², adj-R², RMSE, MAE for both specs
results/exp3_drift_coef_tstat.png    – t-statistic of GBM_mu_1Y by market
results/exp3_model_comparison.png    – bar chart: baseline vs extended metrics
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from src.features.panel import build_extended_panel
from src.models.baseline_regression import (
    run_baseline_both_markets,
    BASELINE_REGRESSORS,
)
from src.models.extended_regression import (
    run_extended_both_markets,
    compare_models,
    EXTENDED_REGRESSORS,
)
from config import (
    RESULTS_DIR, RANDOM_SEED, HAC_MAX_LAGS,
    MAIN_DRIFT_HORIZON, DATA_CACHE_DIR,
)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exp3(carry_gap_panel: pd.DataFrame = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load carry-gap panel ───────────────────────────────────────────────
    if carry_gap_panel is None:
        print('Loading carry-gap panel from cache...')
        carry_gap_panel = pd.read_parquet(
            f'{DATA_CACHE_DIR}/carry_gap_panel.parquet'
        )

    # ── 2. Build extended panel ───────────────────────────────────────────────
    print(f'Building extended panel (drift horizon = {MAIN_DRIFT_HORIZON} days)...')
    ext_panel = build_extended_panel(
        carry_gap_panel,
        horizon=MAIN_DRIFT_HORIZON,
        random_seed=RANDOM_SEED,
    )
    print(f'  Panel: {len(ext_panel):,} rows, '
          f'{ext_panel["GBM_mu_1Y"].notna().sum():,} non-NaN drift values')

    # ── 3. Fit both models ────────────────────────────────────────────────────
    print('\nFitting baseline model...')
    base_results = run_baseline_both_markets(ext_panel, max_lags=HAC_MAX_LAGS)

    print('Fitting extended model...')
    ext_results = run_extended_both_markets(ext_panel, max_lags=HAC_MAX_LAGS)

    # ── 4. Print and save coefficient tables ──────────────────────────────────
    comparison_rows = []
    for mkt in ['SPX', 'RUT']:
        print(f'\n=== {mkt} Extended Regression ===')
        coef_table = ext_results[mkt]['coefficients_df'].copy()
        print(coef_table[['coef', 'hac_se', 't_stat', 'p_value',
                           'significance']].to_string())
        coef_table.to_csv(
            f'{RESULTS_DIR}/exp3_{mkt.lower()}_extended_coefs.csv'
        )

        comp = compare_models(base_results, ext_results, mkt)
        comp.insert(0, 'market', mkt)
        comparison_rows.append(comp)
        print(f'\n  Model comparison ({mkt}):')
        print(comp.to_string(index=False))

    comparison_df = pd.concat(comparison_rows, ignore_index=True)
    comparison_df.to_csv(f'{RESULTS_DIR}/exp3_model_comparison.csv', index=False)
    print(f'\nSaved: {RESULTS_DIR}/exp3_model_comparison.csv')

    # ── 5. Drift-coefficient t-stat plot ─────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 5))
    markets = ['SPX', 'RUT']
    colors  = ['steelblue', 'tomato']
    for i, (mkt, col) in enumerate(zip(markets, colors)):
        res  = ext_results[mkt]['result']
        if 'GBM_mu_1Y' not in res.tvalues.index:
            continue
        t    = float(res.tvalues['GBM_mu_1Y'])
        p    = float(res.pvalues['GBM_mu_1Y'])
        bar  = ax.bar(i, t, color=col, alpha=0.85, label=mkt)
        star = ('***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.10 else '')
        ax.text(i, t + np.sign(t) * 0.15, f'{t:.2f}{star}',
                ha='center', va='bottom' if t >= 0 else 'top', fontsize=10)
    ax.axhline(0,    color='black', linewidth=0.8)
    ax.axhline(1.96, color='grey',  linewidth=0.8, linestyle='--', label='±1.96')
    ax.axhline(-1.96, color='grey', linewidth=0.8, linestyle='--')
    ax.set_xticks([0, 1])
    ax.set_xticklabels(markets)
    ax.set_ylabel('HAC t-statistic')
    ax.set_title(f'Drift proxy GBM_mu_1Y  (horizon={MAIN_DRIFT_HORIZON} days)')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/exp3_drift_coef_tstat.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {RESULTS_DIR}/exp3_drift_coef_tstat.png')

    # ── 6. Metrics bar chart ──────────────────────────────────────────────────
    _plot_model_comparison(base_results, ext_results, RESULTS_DIR)

    return base_results, ext_results, comparison_df


# ── Helper ────────────────────────────────────────────────────────────────────

def _plot_model_comparison(base_results: dict, ext_results: dict, results_dir: str):
    metrics = ['R2', 'RMSE', 'MAE']
    n_metrics = len(metrics)
    fig, axes = plt.subplots(1, n_metrics, figsize=(4 * n_metrics, 5))

    for ax, metric in zip(axes, metrics):
        xs = np.arange(2)
        base_vals = [base_results[m]['metrics'].get(metric, np.nan)
                     for m in ['SPX', 'RUT']]
        ext_vals  = [ext_results[m]['metrics'].get(metric, np.nan)
                     for m in ['SPX', 'RUT']]
        width = 0.35
        ax.bar(xs - width / 2, base_vals, width, label='Baseline',
               color='steelblue', alpha=0.8)
        ax.bar(xs + width / 2, ext_vals,  width, label='Extended',
               color='darkorange', alpha=0.8)
        ax.set_xticks(xs)
        ax.set_xticklabels(['SPX', 'RUT'])
        ax.set_title(metric)
        ax.legend(fontsize=8)
    plt.suptitle('Baseline vs Extended Model', fontsize=12)
    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp3_model_comparison.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp3_model_comparison.png')


if __name__ == '__main__':
    run_exp3()
