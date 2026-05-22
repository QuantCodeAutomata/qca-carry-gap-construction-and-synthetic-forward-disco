"""
Figure 1.1: Cumulative carry gap vs cumulative log total return.

Plots the time series of the 5–7 month maturity bucket's daily median carry
gap (CG_bp) alongside the cumulative log total return of the underlying
index.  The visual juxtaposition motivates why drift enters the regression.

Usage
-----
    python src/visualization/figure_1_1.py
Or import and call ``plot_figure_1_1(panel, tr_spx, tr_rut, save_dir)``.
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

from config import (
    START_DATE, END_DATE, RESULTS_DIR, RANDOM_SEED,
    FIG_TAU_MIN, FIG_TAU_MAX, DATA_CACHE_DIR,
)


def _load_or_build_panel():
    cache = f'{DATA_CACHE_DIR}/carry_gap_panel.parquet'
    if os.path.exists(cache):
        return pd.read_parquet(cache)
    from src.features.carry_gap import build_synthetic_carry_gap_panel
    return build_synthetic_carry_gap_panel(
        start_date=START_DATE, end_date=END_DATE, random_seed=RANDOM_SEED
    )


def _synthetic_tr(market: str, dates: pd.DatetimeIndex) -> pd.Series:
    """Synthetic total-return index rebased to 1 at the first date."""
    seed = 0 if market == 'SPX' else 7
    rng = np.random.RandomState(seed)
    n = len(dates)
    mu = 0.07 / 252
    sigma = (0.18 if market == 'SPX' else 0.22) / np.sqrt(252)
    log_ret = mu + sigma * rng.randn(n)
    cum_log = np.cumsum(log_ret)
    return pd.Series(cum_log, index=dates, name=f'{market}_log_tr')


def plot_figure_1_1(
    panel: pd.DataFrame = None,
    save_dir: str = None,
) -> str:
    """
    Generate Figure 1.1 and save to disk.

    Parameters
    ----------
    panel    : carry-gap panel (loaded from cache if None)
    save_dir : directory to save the figure (defaults to RESULTS_DIR)

    Returns
    -------
    Path to the saved figure.
    """
    if save_dir is None:
        save_dir = RESULTS_DIR
    os.makedirs(save_dir, exist_ok=True)

    if panel is None:
        panel = _load_or_build_panel()

    panel['date'] = pd.to_datetime(panel['date'])

    # ── Maturity filter: 5–7 months ───────────────────────────────────────────
    mask_tau = (panel['tau'] >= FIG_TAU_MIN) & (panel['tau'] <= FIG_TAU_MAX)
    panel_56 = panel[mask_tau].copy()

    if panel_56.empty:
        raise ValueError(
            f'No observations in tau range [{FIG_TAU_MIN:.3f}, {FIG_TAU_MAX:.3f}]. '
            f'Available taus: {sorted(panel["tau"].unique())}')

    # ── Daily median CG per market ────────────────────────────────────────────
    cg_spx = (panel_56[panel_56['market'] == 'SPX']
              .groupby('date')['CG_bp'].median())
    cg_rut = (panel_56[panel_56['market'] == 'RUT']
              .groupby('date')['CG_bp'].median())

    dates = cg_spx.index
    tr_spx = _synthetic_tr('SPX', dates)
    tr_rut = _synthetic_tr('RUT', dates)

    # ── Cumulative carry gap ──────────────────────────────────────────────────
    cum_cg_spx = cg_spx.cumsum() / 10_000  # convert bp to decimal
    cum_cg_rut = cg_rut.cumsum() / 10_000

    # ── Figure ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    for ax, mkt, cum_cg, cum_tr, c1, c2 in [
        (axes[0], 'SPX', cum_cg_spx, tr_spx, 'steelblue', 'darkorange'),
        (axes[1], 'RUT', cum_cg_rut, tr_rut, 'tomato',    'darkorange'),
    ]:
        ax2 = ax.twinx()

        ax.plot(cum_cg.index, cum_cg.values,
                linewidth=1.2, color=c1, label='Cumulative CG (left)')
        ax2.plot(cum_tr.index, cum_tr.values,
                 linewidth=1.0, color=c2, alpha=0.85,
                 linestyle='--', label='Cum. log TR (right)')

        ax.axhline(0, color='black', linewidth=0.5, linestyle=':')
        ax.set_ylabel('Cumulative carry gap (decimal)', color=c1, fontsize=10)
        ax2.set_ylabel('Cumulative log total return', color=c2, fontsize=10)
        ax.set_title(f'{mkt}: Cumulative CG vs Cum. Log TR  '
                     f'(5–7 month maturity bucket)', fontsize=11)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=9,
                  loc='upper left')
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))

    plt.suptitle('Figure 1.1: Carry Gap vs Total Return (5–7 month ATM)',
                 fontsize=13, y=1.01)
    plt.tight_layout()

    out_path = os.path.join(save_dir, 'figure_1_1.png')
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'Saved: {out_path}')
    return out_path


if __name__ == '__main__':
    plot_figure_1_1()
