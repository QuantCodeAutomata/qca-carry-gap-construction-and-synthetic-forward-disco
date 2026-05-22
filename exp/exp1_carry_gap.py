"""
Experiment 1: Carry-gap construction and panel diagnostics.

Builds the core dependent-variable panel (CG_bp) and writes it to the
data/cache directory so downstream experiments can load it directly.

Outputs
-------
data/cache/carry_gap_panel.parquet   – full panel
results/exp1_panel_stats.csv         – per-market summary statistics
results/exp1_cg_distribution.png     – CG_bp histogram by market
results/exp1_cg_timeseries.png       – daily median CG_bp time series
results/exp1_cg_by_maturity.png      – CG_bp by maturity bucket
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

from src.features.carry_gap import build_synthetic_carry_gap_panel, validate_carry_gap_panel
from config import START_DATE, END_DATE, RESULTS_DIR, RANDOM_SEED, DATA_CACHE_DIR


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exp1():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(DATA_CACHE_DIR, exist_ok=True)

    # ── 1. Build panel ────────────────────────────────────────────────────────
    print('Building synthetic carry-gap panel...')
    panel = build_synthetic_carry_gap_panel(
        start_date=START_DATE,
        end_date=END_DATE,
        random_seed=RANDOM_SEED,
    )
    panel.to_parquet(f'{DATA_CACHE_DIR}/carry_gap_panel.parquet', index=False)
    print(f'  Saved {len(panel):,} rows to {DATA_CACHE_DIR}/carry_gap_panel.parquet')

    # ── 2. Validate ───────────────────────────────────────────────────────────
    val = validate_carry_gap_panel(panel)
    print('\nValidation:')
    for k, v in val.items():
        print(f'  {k}: {v}')

    # ── 3. Summary statistics ─────────────────────────────────────────────────
    stats = panel.groupby('market').agg(
        n_obs      = ('CG_bp', 'count'),
        n_dates    = ('date',  'nunique'),
        mean_CG_bp = ('CG_bp', 'mean'),
        std_CG_bp  = ('CG_bp', 'std'),
        min_CG_bp  = ('CG_bp', 'min'),
        max_CG_bp  = ('CG_bp', 'max'),
        p25_CG_bp  = ('CG_bp', lambda x: x.quantile(0.25)),
        p75_CG_bp  = ('CG_bp', lambda x: x.quantile(0.75)),
    ).reset_index()
    print('\nPanel statistics:')
    print(stats.to_string(index=False))
    stats.to_csv(f'{RESULTS_DIR}/exp1_panel_stats.csv', index=False)

    # ── 4. Distribution plot ──────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mkt in zip(axes, ['SPX', 'RUT']):
        sub = panel[panel['market'] == mkt]['CG_bp']
        ax.hist(sub, bins=80, color='steelblue', edgecolor='white',
                linewidth=0.3, alpha=0.85)
        ax.axvline(sub.mean(), color='tomato', linewidth=1.5,
                   linestyle='--', label=f'Mean = {sub.mean():.1f} bp')
        ax.axvline(0, color='black', linewidth=0.8, linestyle=':')
        ax.set_title(f'{mkt}: Carry Gap Distribution', fontsize=12)
        ax.set_xlabel('CG (basis points)')
        ax.set_ylabel('Count')
        ax.legend(fontsize=9)
    plt.suptitle('Carry-Gap Distribution by Market', fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/exp1_cg_distribution.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'\nSaved: {RESULTS_DIR}/exp1_cg_distribution.png')

    # ── 5. Time-series plot ───────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    colors = {'SPX': 'steelblue', 'RUT': 'tomato'}
    for ax, mkt in zip(axes, ['SPX', 'RUT']):
        ts = (panel[panel['market'] == mkt]
              .groupby('date')['CG_bp']
              .agg(['median', lambda x: x.quantile(0.25), lambda x: x.quantile(0.75)])
              .rename(columns={'median': 'med', '<lambda_0>': 'q25', '<lambda_1>': 'q75'}))
        ax.fill_between(ts.index, ts['q25'], ts['q75'],
                        alpha=0.25, color=colors[mkt])
        ax.plot(ts.index, ts['med'], linewidth=0.9, color=colors[mkt], label=mkt)
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.set_ylabel('CG (bp)')
        ax.set_title(f'{mkt}: Daily Median Carry Gap (IQR shaded)', fontsize=11)
        ax.legend(fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/exp1_cg_timeseries.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {RESULTS_DIR}/exp1_cg_timeseries.png')

    # ── 6. CG by maturity bucket ──────────────────────────────────────────────
    tau_labels = {
        0.0822: '1M',  0.1644: '2M',  0.2493: '3M',
        0.3342: '4M',  0.4164: '5M',  0.4986: '6M',
        0.5836: '7M',  0.7479: '9M',  1.0000: '12M',
    }
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, mkt in zip(axes, ['SPX', 'RUT']):
        sub = panel[panel['market'] == mkt]
        tau_vals = sorted(sub['tau'].unique())
        means = [sub[sub['tau'].round(4) == round(t, 4)]['CG_bp'].mean()
                 for t in tau_vals]
        stds  = [sub[sub['tau'].round(4) == round(t, 4)]['CG_bp'].std()
                 for t in tau_vals]
        xlabels = [tau_labels.get(round(t, 4), f'{t:.2f}y') for t in tau_vals]
        x = np.arange(len(tau_vals))
        ax.bar(x, means, yerr=stds, capsize=4, color='steelblue', alpha=0.8,
               error_kw={'linewidth': 1.0})
        ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
        ax.set_xticks(x)
        ax.set_xticklabels(xlabels, fontsize=9)
        ax.set_title(f'{mkt}: Mean CG_bp by Maturity')
        ax.set_xlabel('Maturity')
        ax.set_ylabel('CG (bp)')
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/exp1_cg_by_maturity.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {RESULTS_DIR}/exp1_cg_by_maturity.png')

    return panel, val, stats


if __name__ == '__main__':
    run_exp1()
