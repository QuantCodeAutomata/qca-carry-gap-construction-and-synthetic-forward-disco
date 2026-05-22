"""
Experiment 4: Drift-proxy horizon scan.

For every horizon N in HORIZON_GRID, builds GBM_mu_{N} and runs the
extended regression.  Records the t-statistic, p-value, and in-sample R²
of the drift coefficient as a function of N.

Outputs
-------
results/exp4_horizon_scan_SPX.csv   – per-horizon stats for SPX
results/exp4_horizon_scan_RUT.csv   – per-horizon stats for RUT
results/exp4_horizon_scan.png       – t-stat profile curve (both markets)
results/exp4_r2_scan.png            – R² profile curve (both markets)
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
from src.models.extended_regression import run_extended_both_markets
from config import (
    RESULTS_DIR, RANDOM_SEED, HAC_MAX_LAGS,
    MAIN_DRIFT_HORIZON, HORIZON_GRID, DATA_CACHE_DIR,
)


# ── Core scan ─────────────────────────────────────────────────────────────────

def scan_horizons(
    carry_gap_panel: pd.DataFrame,
    horizon_grid: list,
    random_seed: int,
    max_lags: int,
) -> dict:
    """
    Run extended regression for every horizon in horizon_grid.

    Returns dict keyed by market, values are DataFrames with columns:
        horizon, t_stat, p_value, coef, R2
    """
    scan_rows = {mkt: [] for mkt in ['SPX', 'RUT']}
    n = len(horizon_grid)

    for i, h in enumerate(horizon_grid):
        if (i + 1) % 10 == 0 or (i + 1) == n:
            print(f'  Horizon {h:4d} ({i + 1}/{n})...', end='\r')

        # Build panel for this horizon
        ext_panel = build_extended_panel(
            carry_gap_panel, horizon=h, random_seed=random_seed
        )

        # Skip if drift column is all-NaN
        if ext_panel['GBM_mu_1Y'].notna().sum() < 30:
            continue

        results = run_extended_both_markets(ext_panel, max_lags=max_lags)

        for mkt in ['SPX', 'RUT']:
            res = results[mkt]['result']
            if 'GBM_mu_1Y' not in res.tvalues.index:
                continue
            scan_rows[mkt].append({
                'horizon': h,
                't_stat':  float(res.tvalues['GBM_mu_1Y']),
                'p_value': float(res.pvalues['GBM_mu_1Y']),
                'coef':    float(res.params['GBM_mu_1Y']),
                'R2':      float(results[mkt]['metrics']['R2']),
            })

    print()
    return {
        mkt: pd.DataFrame(rows).sort_values('horizon').reset_index(drop=True)
        for mkt, rows in scan_rows.items()
        if rows
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def run_exp4(carry_gap_panel: pd.DataFrame = None):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    # ── 1. Load carry-gap panel ───────────────────────────────────────────────
    if carry_gap_panel is None:
        print('Loading carry-gap panel from cache...')
        carry_gap_panel = pd.read_parquet(
            f'{DATA_CACHE_DIR}/carry_gap_panel.parquet'
        )

    # ── 2. Scan ───────────────────────────────────────────────────────────────
    print(f'Scanning {len(HORIZON_GRID)} horizons...')
    scan_results = scan_horizons(
        carry_gap_panel, HORIZON_GRID, RANDOM_SEED, HAC_MAX_LAGS
    )

    # ── 3. Save CSVs ─────────────────────────────────────────────────────────
    for mkt, df in scan_results.items():
        path = f'{RESULTS_DIR}/exp4_horizon_scan_{mkt}.csv'
        df.to_csv(path, index=False)
        print(f'Saved: {path}')
        # Print the horizon with max |t_stat|
        best = df.loc[df['t_stat'].abs().idxmax()]
        print(f'  {mkt}: best horizon = {int(best.horizon)}, '
              f't = {best.t_stat:.2f}, p = {best.p_value:.4f}')

    # ── 4. t-stat profile plot ────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    colors = {'SPX': 'steelblue', 'RUT': 'tomato'}
    for mkt, df in scan_results.items():
        ax.plot(df['horizon'], df['t_stat'], linewidth=1.5,
                color=colors[mkt], label=mkt)
    ax.axhline(1.96,  color='grey', linewidth=0.8, linestyle='--', label='±1.96')
    ax.axhline(-1.96, color='grey', linewidth=0.8, linestyle='--')
    ax.axvline(MAIN_DRIFT_HORIZON, color='black', linewidth=1.0,
               linestyle=':', label=f'Main horizon ({MAIN_DRIFT_HORIZON}d)')
    ax.set_xlabel('Drift horizon N (trading days)')
    ax.set_ylabel('HAC t-statistic of GBM_mu_N')
    ax.set_title('Horizon Scan: Drift Coefficient t-Statistic')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/exp4_horizon_scan.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {RESULTS_DIR}/exp4_horizon_scan.png')

    # ── 5. R² profile plot ────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    for mkt, df in scan_results.items():
        ax.plot(df['horizon'], df['R2'], linewidth=1.5,
                color=colors[mkt], label=mkt)
    ax.axvline(MAIN_DRIFT_HORIZON, color='black', linewidth=1.0,
               linestyle=':', label=f'Main horizon ({MAIN_DRIFT_HORIZON}d)')
    ax.set_xlabel('Drift horizon N (trading days)')
    ax.set_ylabel('In-sample R²')
    ax.set_title('Horizon Scan: In-Sample R²')
    ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(f'{RESULTS_DIR}/exp4_r2_scan.png', dpi=150,
                bbox_inches='tight')
    plt.close()
    print(f'Saved: {RESULTS_DIR}/exp4_r2_scan.png')

    return scan_results


if __name__ == '__main__':
    run_exp4()
