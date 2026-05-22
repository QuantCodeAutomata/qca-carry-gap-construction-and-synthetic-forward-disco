"""
Experiment 5: Leave-One-Year-Out (LOYO) out-of-sample validation.

Validates both baseline and extended models via LOYO cross-validation.
For each calendar year the model is re-trained on all other years and
predictions are made for the held-out year.

Outputs
-------
results/exp5_loyo_summary.csv         – pooled R², mean R², RMSE, MAE
results/exp5_loyo_fold_metrics.csv    – per-year metrics for all combos
results/exp5_loyo_r2_by_year.png      – OOS R² bar chart by year
results/exp5_loyo_stability.png       – coefficient sign / significance heatmap
results/exp5_loyo_oos_scatter.png     – pooled OOS actual vs predicted
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
from src.evaluation.loyo import (
    run_loyo_both_markets,
    format_loyo_summary,
)
from src.evaluation.stability import (
    format_stability_table,
    coefficient_sign_matrix,
)
from config import (
    RESULTS_DIR, RANDOM_SEED, HAC_MAX_LAGS,
    MAIN_DRIFT_HORIZON, DATA_CACHE_DIR,
    BASELINE_REGRESSORS, EXTENDED_REGRESSORS,
)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exp5(carry_gap_panel: pd.DataFrame = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load carry-gap panel ───────────────────────────────────────────────
    if carry_gap_panel is None:
        print('Loading carry-gap panel from cache...')
        carry_gap_panel = pd.read_parquet(
            f'{DATA_CACHE_DIR}/carry_gap_panel.parquet'
        )

    # ── 2. Build extended panel ───────────────────────────────────────────────
    print(f'Building extended panel (horizon={MAIN_DRIFT_HORIZON})...')
    ext_panel = build_extended_panel(
        carry_gap_panel, horizon=MAIN_DRIFT_HORIZON, random_seed=RANDOM_SEED
    )

    # ── 3. LOYO validation ────────────────────────────────────────────────────
    print('Running LOYO validation (baseline + extended, SPX + RUT)...')
    loyo_results = run_loyo_both_markets(
        ext_panel,
        baseline_regressors=BASELINE_REGRESSORS,
        extended_regressors=EXTENDED_REGRESSORS,
    )

    # ── 4. Summary table ──────────────────────────────────────────────────────
    summary = format_loyo_summary(loyo_results)
    print('\nLOYO Summary:')
    print(summary.to_string())
    summary.to_csv(f'{RESULTS_DIR}/exp5_loyo_summary.csv')
    print(f'\nSaved: {RESULTS_DIR}/exp5_loyo_summary.csv')

    # Per-year metrics
    fold_rows = []
    for mkt, models in loyo_results.items():
        for model_name, res in models.items():
            if not res:
                continue
            fm = res['fold_metrics'].copy()
            fm['market'] = mkt
            fm['model'] = model_name
            fold_rows.append(fm)
    if fold_rows:
        fold_df = pd.concat(fold_rows)
        fold_df.to_csv(f'{RESULTS_DIR}/exp5_loyo_fold_metrics.csv')
        print(f'Saved: {RESULTS_DIR}/exp5_loyo_fold_metrics.csv')

    # ── 5. OOS R² bar chart ───────────────────────────────────────────────────
    _plot_loyo_r2_by_year(loyo_results, RESULTS_DIR)

    # ── 6. Stability heatmap ──────────────────────────────────────────────────
    _plot_stability_heatmap(loyo_results, RESULTS_DIR)

    # ── 7. OOS scatter ────────────────────────────────────────────────────────
    _plot_oos_scatter(loyo_results, RESULTS_DIR)

    return loyo_results, summary


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _plot_loyo_r2_by_year(loyo_results: dict, results_dir: str):
    n_mkts = 2
    models = ['baseline', 'extended']
    fig, axes = plt.subplots(n_mkts, 1, figsize=(12, 7), sharex=False)
    if n_mkts == 1:
        axes = [axes]
    colors = {'baseline': 'steelblue', 'extended': 'darkorange'}

    for ax, mkt in zip(axes, ['SPX', 'RUT']):
        for model_name in models:
            res = loyo_results.get(mkt, {}).get(model_name, {})
            if not res or 'fold_metrics' not in res:
                continue
            fm = res['fold_metrics']
            x = fm.index.astype(float)
            ax.plot(x, fm['R2'], marker='o', markersize=4,
                    linewidth=1.2, label=model_name,
                    color=colors[model_name])
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.set_title(f'{mkt}: LOYO Out-of-Sample R² by Year', fontsize=11)
        ax.set_ylabel('OOS R²')
        ax.set_xlabel('Held-out year')
        ax.legend(fontsize=9)

    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp5_loyo_r2_by_year.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp5_loyo_r2_by_year.png')


def _plot_stability_heatmap(loyo_results: dict, results_dir: str):
    """Sign stability heatmap for the extended model."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mkt in zip(axes, ['SPX', 'RUT']):
        res = loyo_results.get(mkt, {}).get('extended', {})
        if not res or 'coefficient_signs' not in res:
            ax.set_visible(False)
            continue
        sign_mat = res['coefficient_signs'].drop(columns=['const'], errors='ignore')
        if sign_mat.empty:
            ax.set_visible(False)
            continue
        im = ax.imshow(sign_mat.values, aspect='auto', cmap='RdBu',
                       vmin=-1, vmax=1)
        ax.set_yticks(range(len(sign_mat)))
        ax.set_yticklabels(sign_mat.index.astype(str), fontsize=8)
        ax.set_xticks(range(len(sign_mat.columns)))
        ax.set_xticklabels(sign_mat.columns, rotation=45, ha='right', fontsize=8)
        ax.set_title(f'{mkt}: Extended Model Coefficient Signs')
        plt.colorbar(im, ax=ax, label='Sign (+1 / −1)')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp5_loyo_stability.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp5_loyo_stability.png')


def _plot_oos_scatter(loyo_results: dict, results_dir: str):
    """Pooled OOS actual vs predicted scatter."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, mkt in zip(axes, ['SPX', 'RUT']):
        res = loyo_results.get(mkt, {}).get('extended', {})
        if not res or 'pooled_predictions' not in res:
            ax.set_visible(False)
            continue
        y_act  = res['pooled_actuals']
        y_pred = res['pooled_predictions']
        ax.scatter(y_pred, y_act, s=1, alpha=0.2, color='steelblue',
                   rasterized=True)
        lo = min(float(y_act.min()), float(y_pred.min()))
        hi = max(float(y_act.max()), float(y_pred.max()))
        ax.plot([lo, hi], [lo, hi], 'k--', linewidth=1)
        r2 = res.get('pooled_R2', np.nan)
        ax.set_title(f'{mkt}: Pooled OOS  (R²={r2:.3f})', fontsize=11)
        ax.set_xlabel('Predicted CG_bp')
        ax.set_ylabel('Actual CG_bp')
    plt.tight_layout()
    plt.savefig(f'{results_dir}/exp5_loyo_oos_scatter.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {results_dir}/exp5_loyo_oos_scatter.png')


if __name__ == '__main__':
    run_exp5()
