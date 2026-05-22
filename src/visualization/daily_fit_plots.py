"""
Daily fit plots: actual vs fitted carry gap for baseline and extended models.

Generates time-series overlays of realised and model-fitted CG_bp, residual
time series, and rolling-window in-sample R² to visualise structural stability.

Usage
-----
    python src/visualization/daily_fit_plots.py
Or call ``plot_daily_fits(baseline_results, extended_results, panel, save_dir)``.
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from config import RESULTS_DIR, RANDOM_SEED, MAIN_DRIFT_HORIZON, DATA_CACHE_DIR


# ── Rolling R² helper ─────────────────────────────────────────────────────────

def _rolling_r2(y_actual: pd.Series,
                y_fitted: pd.Series,
                window: int = 252) -> pd.Series:
    """Compute rolling in-sample R² over a given window."""
    resid = (y_actual - y_fitted) ** 2
    dev   = (y_actual - y_actual.mean()) ** 2
    ss_res = resid.rolling(window=window, min_periods=window // 2).sum()
    ss_tot = dev.rolling(window=window, min_periods=window // 2).sum()
    r2 = 1 - ss_res / ss_tot.replace(0, np.nan)
    return r2.clip(-1, 1)


# ── Main plotting function ────────────────────────────────────────────────────

def plot_daily_fits(
    baseline_results: dict,
    extended_results: dict,
    panel: pd.DataFrame,
    save_dir: str = None,
):
    """
    Generate daily actual-vs-fitted time series for both models and both markets.

    Parameters
    ----------
    baseline_results : output of ``run_baseline_both_markets``
    extended_results : output of ``run_extended_both_markets``
    panel            : regression panel (must contain 'date' and 'market')
    save_dir         : directory for output figures
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    for mkt in ['SPX', 'RUT']:
        _plot_market_fit(
            mkt=mkt,
            base_res=baseline_results[mkt],
            ext_res=extended_results[mkt],
            panel=panel,
            save_dir=save_dir,
        )


def _plot_market_fit(mkt, base_res, ext_res, panel, save_dir):
    """Three-panel plot: actual vs fitted, residuals, rolling R²."""

    # ── Retrieve date-indexed actuals and fitted values ───────────────────────
    # The regression panel is sorted by (date, tau) per market.
    # We average across tau per date for a clean daily time series.
    mkt_panel = panel[panel['market'] == mkt].copy()
    mkt_panel['date'] = pd.to_datetime(mkt_panel['date'])

    # Baseline
    base_y  = pd.Series(base_res['y'].values, index=base_res['y'].index)
    base_yf = pd.Series(base_res['result'].fittedvalues.values,
                        index=base_res['y'].index)

    # Extended
    ext_y   = pd.Series(ext_res['y'].values, index=ext_res['y'].index)
    ext_yf  = pd.Series(ext_res['result'].fittedvalues.values,
                        index=ext_res['y'].index)

    # Align date: use the integer positional index for x-axis
    # Map back to dates via merge with panel
    def _add_dates(y_series, the_panel):
        """Add date column from panel row positions."""
        sub = the_panel[['CG_bp']].dropna().reset_index(drop=True)
        dates = the_panel.dropna(subset=['CG_bp'])['date'].values
        out = pd.DataFrame({'y': y_series.values,
                            'date': dates[:len(y_series)]})
        return out.set_index('date')['y']

    base_df  = mkt_panel[mkt_panel['CG_bp'].notna()]
    date_idx = base_df['date'].values

    # Guard: index alignment
    n = min(len(date_idx), len(base_y))
    dates = pd.to_datetime(date_idx[:n])
    base_actual  = pd.Series(base_y.values[:n],  index=dates)
    base_fitted  = pd.Series(base_yf.values[:n], index=dates)

    n2 = min(len(date_idx), len(ext_y))
    dates2 = pd.to_datetime(date_idx[:n2])
    ext_actual   = pd.Series(ext_y.values[:n2],  index=dates2)
    ext_fitted   = pd.Series(ext_yf.values[:n2], index=dates2)

    # ── Aggregate to daily mean per date ──────────────────────────────────────
    def _daily(s: pd.Series) -> pd.Series:
        return s.groupby(s.index).mean()

    b_act = _daily(base_actual)
    b_fit = _daily(base_fitted)
    e_act = _daily(ext_actual)
    e_fit = _daily(ext_fitted)

    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True)

    # ── Panel 1: Actual vs fitted ─────────────────────────────────────────────
    ax = axes[0]
    ax.plot(b_act.index, b_act.values, linewidth=0.7, color='black',
            alpha=0.9, label='Actual')
    ax.plot(b_fit.index, b_fit.values, linewidth=1.0, color='steelblue',
            alpha=0.85, label='Baseline fitted')
    ax.plot(e_fit.index, e_fit.values, linewidth=1.0, color='darkorange',
            alpha=0.85, linestyle='--', label='Extended fitted')
    ax.set_ylabel('CG_bp')
    ax.set_title(f'{mkt}: Daily Mean CG_bp — Actual vs Fitted', fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 2: Residuals ────────────────────────────────────────────────────
    ax = axes[1]
    b_resid = b_act - b_fit
    e_resid = e_act - e_fit
    ax.plot(b_resid.index, b_resid.values, linewidth=0.7, color='steelblue',
            alpha=0.75, label='Baseline residual')
    ax.plot(e_resid.index, e_resid.values, linewidth=0.7, color='darkorange',
            alpha=0.75, linestyle='--', label='Extended residual')
    ax.axhline(0, color='black', linewidth=0.6)
    ax.set_ylabel('Residual (bp)')
    ax.set_title(f'{mkt}: Regression Residuals', fontsize=11)
    ax.legend(fontsize=9)

    # ── Panel 3: Rolling R² ───────────────────────────────────────────────────
    ax = axes[2]
    roll_r2_base = _rolling_r2(b_act, b_fit)
    roll_r2_ext  = _rolling_r2(e_act, e_fit)
    ax.plot(roll_r2_base.index, roll_r2_base.values, linewidth=1.0,
            color='steelblue', label='Baseline rolling R²')
    ax.plot(roll_r2_ext.index, roll_r2_ext.values, linewidth=1.0,
            color='darkorange', linestyle='--', label='Extended rolling R²')
    ax.axhline(0, color='black', linewidth=0.6, linestyle=':')
    ax.set_ylabel('Rolling R² (252d window)')
    ax.set_title(f'{mkt}: Rolling In-Sample R²', fontsize=11)
    ax.legend(fontsize=9)
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.suptitle(f'{mkt}: Daily Fit Diagnostics', fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(save_dir, f'daily_fit_{mkt.lower()}.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')


# ── Standalone runner ─────────────────────────────────────────────────────────

def main():
    from src.features.panel import build_extended_panel
    from src.models.baseline_regression import run_baseline_both_markets
    from src.models.extended_regression import run_extended_both_markets

    panel_path = f'{DATA_CACHE_DIR}/carry_gap_panel.parquet'
    if not os.path.exists(panel_path):
        from src.features.carry_gap import build_synthetic_carry_gap_panel
        from config import START_DATE, END_DATE
        cg = build_synthetic_carry_gap_panel(START_DATE, END_DATE, RANDOM_SEED)
    else:
        cg = pd.read_parquet(panel_path)

    panel = build_extended_panel(cg, horizon=MAIN_DRIFT_HORIZON,
                                 random_seed=RANDOM_SEED)

    base_res = run_baseline_both_markets(panel)
    ext_res  = run_extended_both_markets(panel)

    plot_daily_fits(base_res, ext_res, panel)


if __name__ == '__main__':
    main()
