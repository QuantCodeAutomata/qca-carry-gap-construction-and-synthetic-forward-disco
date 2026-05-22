"""
Carry-gap construction: synthetic-forward vs OIS discount factor.

CG_bp(T) = (10_000 / tau) * ln( B_hat(T) / B_OIS(T) )

where
  B_hat  = discount factor implied by put-call parity (per-cell OLS)
  B_OIS  = OIS discount factor at the same maturity (also written D_OIS)
  tau    = time to maturity in years

A positive carry gap means the option-implied discount factor exceeds
the OIS discount factor (B_hat > B_OIS), i.e. the market is willing
to pay an above-OIS price for optionality.
"""
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Scalar helpers  (paper Eq. 10)
# ---------------------------------------------------------------------------

def compute_carry_gap(
    B_hat: float,
    B_OIS: float,
    tau: float,
) -> float:
    """
    Compute basis-point carry gap:
        CG_bp(T) = (10_000 / tau) * log(B_hat(T) / B_OIS(T))

    Parameters
    ----------
    B_hat : option-implied discount factor from put-call parity
    B_OIS : OIS discount factor at the same maturity (also called D_OIS)
    tau   : time to maturity in years

    Returns
    -------
    Carry gap in basis points (annualised by the 1/tau factor).
    Returns nan when any guard condition fails.

    # Custom - Context7 found no library equivalent (paper Eq. 10)
    """
    if tau <= 0 or B_hat <= 0 or B_OIS <= 0:
        return np.nan
    if not (np.isfinite(B_hat) and np.isfinite(B_OIS) and np.isfinite(tau)):
        return np.nan
    return (1e4 / tau) * np.log(B_hat / B_OIS)


def compute_carry_gap_bp(
    B_hat: float,
    B_OIS: float,
    tau: float,
) -> float:
    """
    Alias for ``compute_carry_gap``.  Retained for backward compatibility.
    """
    return compute_carry_gap(B_hat, B_OIS, tau)


# ---------------------------------------------------------------------------
# Panel builder
# ---------------------------------------------------------------------------

def build_carry_gap_panel(
    synth_fwd_results: pd.DataFrame,
    ois_discount_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge synthetic-forward results with OIS discount factors and compute carry gap.

    Parameters
    ----------
    synth_fwd_results : DataFrame
        Columns: market, date, expiration, tau, B_hat, F_hat, r_squared, n_strikes
    ois_discount_df : DataFrame
        Columns: market, date, expiration, tau, and one of B_OIS or D_ois

    Returns
    -------
    DataFrame sorted by market, date, expiration with columns:
        market, date, expiration, tau, B_hat, F_hat, B_OIS, CG_bp

    Drops observations where B_hat <= 0 or B_OIS <= 0 or tau <= 0 and logs counts.
    """
    merge_keys = ["market", "date", "expiration", "tau"]

    # Accept either column name for the OIS discount factor
    ois_col = "B_OIS" if "B_OIS" in ois_discount_df.columns else "D_ois"
    panel = synth_fwd_results.merge(
        ois_discount_df[merge_keys + [ois_col]],
        on=merge_keys,
        how="inner",
    )
    if ois_col == "D_ois":
        panel = panel.rename(columns={"D_ois": "B_OIS"})

    n_before = len(panel)

    invalid = (panel["B_hat"] <= 0) | (panel["B_OIS"] <= 0) | (panel["tau"] <= 0)
    n_invalid = int(invalid.sum())
    if n_invalid > 0:
        logger.info(
            "Dropping %d/%d observations (%.2f%%): B_hat<=0, B_OIS<=0, or tau<=0",
            n_invalid, n_before, 100.0 * n_invalid / n_before,
        )

    panel = panel[~invalid].copy()

    # Vectorised carry-gap: CG_bp = (10000/tau) * log(B_hat / B_OIS)
    panel["CG_bp"] = (1e4 / panel["tau"]) * np.log(panel["B_hat"] / panel["B_OIS"])

    out_cols = ["market", "date", "expiration", "tau", "B_hat", "F_hat", "B_OIS", "CG_bp"]
    present = [c for c in out_cols if c in panel.columns]
    panel = (
        panel[present]
        .sort_values(["market", "date", "expiration"])
        .reset_index(drop=True)
    )

    logger.info(
        "build_carry_gap_panel: %d rows retained "
        "(merged from %d synth-fwd x %d OIS rows)",
        len(panel), len(synth_fwd_results), len(ois_discount_df),
    )
    return panel


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_carry_gap_panel(panel: pd.DataFrame) -> dict:
    """
    Run validation checks on carry-gap panel.

    Checks
    ------
    1. B_hat > 0 for all retained rows
    2. B_OIS in plausible range (0.5, 1.05)
    3. CG_bp is finite (no NaN) and not identically zero
    4. tau > 0
    5. Count by market
    6. Date range coverage

    Returns
    -------
    dict of validation metrics.  Required keys consumed by tests:
        pass                  – True iff all critical checks pass
        all_B_hat_positive    – bool
        frac_nan_CG           – float, fraction of CG_bp that is NaN
        n_obs                 – int, total row count
        n_markets             – int, number of distinct markets
        b_hat_all_positive    – alias for all_B_hat_positive
        cg_bp_all_finite      – bool, True iff no non-finite CG_bp
        count_by_market       – dict {market: count}
        date_min / date_max   – str
        n_unique_dates        – int
    """
    metrics: dict = {}

    # ── 1. B_hat positivity ───────────────────────────────────────────────
    bhat_pos = panel["B_hat"] > 0
    all_bhat_positive = bool(bhat_pos.all())
    metrics["all_B_hat_positive"] = all_bhat_positive
    metrics["b_hat_all_positive"] = all_bhat_positive   # legacy alias
    metrics["b_hat_nonpositive_count"] = int((~bhat_pos).sum())

    # ── 2. B_OIS / D_ois plausible range ─────────────────────────────────
    bois_col = "B_OIS" if "B_OIS" in panel.columns else "D_ois"
    if bois_col in panel.columns:
        b_in_range = panel[bois_col].between(0.5, 1.05)
        metrics["b_ois_all_in_range"] = bool(b_in_range.all())
        metrics["b_ois_min"] = float(panel[bois_col].min())
        metrics["b_ois_max"] = float(panel[bois_col].max())

    # ── 3. CG_bp NaN fraction and finiteness ─────────────────────────────
    nan_mask = panel["CG_bp"].isna()
    frac_nan = float(nan_mask.mean())
    metrics["frac_nan_CG"] = frac_nan
    cg_finite = np.isfinite(panel["CG_bp"].fillna(np.nan))
    metrics["cg_bp_all_finite"] = bool(cg_finite.all())
    metrics["cg_bp_identically_zero"] = bool((panel["CG_bp"].dropna() == 0).all())
    metrics["cg_bp_mean"] = float(panel["CG_bp"].mean())
    metrics["cg_bp_std"] = float(panel["CG_bp"].std())
    metrics["cg_bp_p5"] = float(panel["CG_bp"].quantile(0.05))
    metrics["cg_bp_median"] = float(panel["CG_bp"].quantile(0.50))
    metrics["cg_bp_p95"] = float(panel["CG_bp"].quantile(0.95))

    # ── 4. tau positivity ────────────────────────────────────────────────
    tau_pos = panel["tau"] > 0
    metrics["tau_all_positive"] = bool(tau_pos.all())
    metrics["tau_nonpositive_count"] = int((~tau_pos).sum())

    # ── 5. Count by market ───────────────────────────────────────────────
    metrics["count_by_market"] = panel.groupby("market").size().to_dict()
    metrics["n_markets"] = int(panel["market"].nunique())

    # ── 6. Date range coverage ───────────────────────────────────────────
    metrics["date_min"] = str(panel["date"].min())
    metrics["date_max"] = str(panel["date"].max())
    metrics["n_unique_dates"] = int(panel["date"].nunique())
    metrics["n_obs"] = int(len(panel))
    metrics["n_rows_total"] = int(len(panel))  # legacy alias

    # ── Overall pass ─────────────────────────────────────────────────────
    metrics["pass"] = bool(
        all_bhat_positive
        and frac_nan == 0.0
        and metrics["cg_bp_all_finite"]
        and metrics["tau_all_positive"]
    )

    return metrics


# ---------------------------------------------------------------------------
# Synthetic panel helpers
# ---------------------------------------------------------------------------

def _piecewise_linear(
    dates: pd.DatetimeIndex,
    key_dates: list,
    key_values: list,
) -> np.ndarray:
    """Piecewise-linear interpolation of (key_dates, key_values) onto all dates."""
    ts = dates.asi8.astype(np.float64)
    key_ts = pd.DatetimeIndex(key_dates).asi8.astype(np.float64)
    return np.interp(ts, key_ts, key_values)


def build_synthetic_carry_gap_panel(
    start_date: str = "2016-01-01",
    end_date: str = "2025-10-31",
    markets: list = None,
    random_seed: int = 42,
) -> pd.DataFrame:
    """
    Build a synthetic carry-gap panel for testing and demonstration.
    Creates realistic CG_bp values with known statistical properties.

    Design targets (random_seed=42, default date range)
    ---------------------------------------------------
    SPX: ~2456 distinct dates, 12 taus  -> ~29 368 observations
    RUT: ~2455 distinct dates,  8 taus  -> ~18 645 observations

    Used when real data pipeline is not available.
    Returns DataFrame with columns: market, date, expiration, tau, B_hat, B_OIS, CG_bp
    """
    if markets is None:
        markets = ["SPX", "RUT"]

    np.random.seed(random_seed)

    all_bdays = pd.bdate_range(start_date, end_date)
    n_all = len(all_bdays)          # 2566 for default date range

    # OIS 1-year rate path (piecewise linear, percentage points)
    ois_key_dates = [
        "2016-01-01", "2016-12-01", "2017-06-01", "2017-12-01",
        "2018-06-01", "2018-12-01", "2019-04-01", "2019-10-01",
        "2020-01-01", "2020-04-01", "2021-01-01", "2022-01-01",
        "2022-05-01", "2022-10-01", "2023-01-01", "2023-07-01",
        "2024-01-01", "2024-07-01", "2025-01-01", "2025-10-31",
    ]
    ois_key_vals = [
        0.50, 0.65, 1.05, 1.30,
        1.90, 2.30, 2.45, 1.80,
        1.55, 0.10, 0.10, 0.15,
        1.00, 3.20, 4.30, 5.30,
        5.30, 5.10, 4.60, 4.00,
    ]
    ois_path = _piecewise_linear(all_bdays, ois_key_dates, ois_key_vals)

    # Per-market maturity grids
    # SPX: 12 monthly/quarterly taus (richer term structure)
    # RUT:  8 taus (less liquid long end)
    spx_taus = np.array([
        1/12, 2/12, 3/12, 4/12, 5/12, 6/12,
        7/12, 9/12, 1.0, 15/12, 18/12, 21/12,
    ])
    rut_taus = np.array([1/12, 2/12, 3/12, 6/12, 9/12, 1.0, 18/12, 2.0])

    cfg = {
        "SPX": {
            "taus":     spx_taus,
            "n_drop":   110,      # 2566-110=2456 dates
            "avail":    0.9965,   # 2456*12*0.9965 ~ 29 381
            "cg_mean":  25.0,     # bp, centered around 10-50 bp as specified
            "cg_ar":    0.97,
            "cg_sigma": 5.0,
        },
        "RUT": {
            "taus":     rut_taus,
            "n_drop":   111,      # 2566-111=2455 dates
            "avail":    0.9493,   # 2455*8*0.9493 ~ 18 641
            "cg_mean":  30.0,
            "cg_ar":    0.96,
            "cg_sigma": 6.0,
        },
    }

    # Stress-period carry-gap spikes (exponential decay from peak)
    stress_events = [
        ("2018-10-01", 20.0, 30),
        ("2020-03-15", 60.0, 25),
        ("2022-01-15", 18.0, 40),
    ]

    records = []

    for mkt in markets:
        c = cfg[mkt]
        taus = c["taus"]
        n_taus = len(taus)

        # Select dates (remove n_drop business days at random)
        drop_idx = np.random.choice(n_all, size=c["n_drop"], replace=False)
        keep_mask = np.ones(n_all, dtype=bool)
        keep_mask[drop_idx] = False
        mkt_dates = all_bdays[keep_mask]
        n_dates = len(mkt_dates)
        mkt_ois = ois_path[keep_mask]

        # AR(1) carry-gap spread in basis points
        phi = c["cg_ar"]
        mu_cg = c["cg_mean"]
        sigma_cg = c["cg_sigma"]
        spread_bp = np.empty(n_dates)
        spread_bp[0] = mu_cg
        innov = np.random.randn(n_dates - 1)
        for i in range(1, n_dates):
            spread_bp[i] = (
                phi * spread_bp[i - 1]
                + (1.0 - phi) * mu_cg
                + sigma_cg * innov[i - 1]
            )

        # Superimpose stress spikes
        date_arr = mkt_dates.to_numpy()
        for spike_str, spike_mag, decay_days in stress_events:
            idx = int(np.searchsorted(date_arr, np.datetime64(spike_str)))
            if idx < n_dates:
                n_decay = min(decay_days, n_dates - idx)
                decay = np.exp(-np.arange(n_decay) / (decay_days / 3.0))
                spread_bp[idx: idx + n_decay] += spike_mag * decay

        spread_bp = np.clip(spread_bp, -5.0, 120.0)

        # Bernoulli availability mask: shape (n_dates, n_taus)
        avail_mask = np.random.random((n_dates, n_taus)) < c["avail"]

        for di in range(n_dates):
            r_ois = mkt_ois[di] / 100.0          # decimal
            tau_noise_bp = np.random.randn(n_taus) * 2.0

            for ti, tau in enumerate(taus):
                if not avail_mask[di, ti]:
                    continue

                cg_bp_obs = float(spread_bp[di] + tau_noise_bp[ti])
                cg_decimal = cg_bp_obs / 1e4

                # B_OIS: pure OIS discounting
                b_ois = np.exp(-r_ois * tau)

                # B_hat consistent with: CG_bp = (1e4/tau)*log(B_hat/B_OIS)
                #   => B_hat = B_OIS * exp(cg_decimal * tau)
                b_hat = b_ois * np.exp(cg_decimal * tau)

                expiry = mkt_dates[di] + pd.tseries.offsets.BusinessDay(
                    int(round(tau * 252))
                )
                records.append({
                    "market":     mkt,
                    "date":       mkt_dates[di],
                    "expiration": expiry,
                    "tau":        round(float(tau), 6),
                    "B_hat":      float(b_hat),
                    "B_OIS":      float(b_ois),
                    "CG_bp":      round(cg_bp_obs, 4),
                })

    panel = pd.DataFrame(records)
    panel["date"] = pd.to_datetime(panel["date"])
    panel["expiration"] = pd.to_datetime(panel["expiration"])
    panel = panel.sort_values(["market", "date", "expiration"]).reset_index(drop=True)
    return panel
