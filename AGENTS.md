# carry-gap project — persistent agent memory

## Project purpose
Regression analysis of carry-gap (`CG_bp`) for SPX and RUT equity index options.

## Panel schema (expected columns)
| Column | Description |
|---|---|
| `date` | trade date (datetime) |
| `market` | `'SPX'` or `'RUT'` |
| `expiration` | option expiration date (datetime) |
| `CG_bp` | carry-gap in basis points (target variable) |
| `GBM_1Y` | 1-year GBM vol regressor |
| `GBM_10Y` | 10-year GBM vol regressor |
| `GBM_mu_1Y` | 1-year GBM drift (extended model only) |
| `BA_over_tau` | bid-ask spread / time-to-expiry |
| `NFCI` | National Financial Conditions Index |

Column names are matched case-insensitively via `_detect_col()`.

## Model specs
- **Baseline** (`BASELINE_REGRESSORS`): `GBM_1Y, GBM_10Y, BA_over_tau, NFCI`
- **Extended** (`EXTENDED_REGRESSORS`): `GBM_1Y, GBM_10Y, GBM_mu_1Y, BA_over_tau, NFCI`
- Both use OLS with HAC(Newey-West) SEs, `max_lags=21` (trading days)

## Implemented modules
| File | Key exports |
|---|---|
| `src/models/baseline_regression.py` | `prepare_regression_data`, `fit_baseline_hac`, `compute_insample_metrics`, `run_baseline_regression`, `run_baseline_both_markets`, `format_coefficient_table` |
| `src/models/extended_regression.py` | `run_extended_regression`, `run_extended_both_markets`, `compare_models` |
| `src/evaluation/loyo.py` | `get_calendar_year_folds`, `run_loyo_fold`, `run_loyo_validation`, `pooled_oos_r2`, `run_loyo_both_markets`, `format_loyo_summary` |
| `src/evaluation/stability.py` | `coefficient_sign_matrix`, `sign_stability_summary`, `significance_count_summary`, `format_stability_table` |
| `data/ois_data.py` | `load_ois_rates`, `get_ois1y_series`, `get_ois10y_series`; `TENORS=[1/12,3/12,6/12,1,2,5,10]`, `TENOR_LABELS=['1m','3m','6m','1y','2y','5y','10y']` |
| `data/total_return.py` | `load_total_return_index(market, start, end)`; SPX→SPY, RUT→IWM via massive API, GBM fallback |
| `data/options_daily.py` | `load_option_quotes_daily`, `generate_synthetic_option_panel`; returns cols `[market, date, expiration, strike, option_type, bid, ask, mid, tau]` |
| `src/pricing/synth_forward.py` | `compute_mid_quotes`, `apply_quote_filters`, `compute_synthetic_forward`, `estimate_discount_factor_cell`, `process_option_panel` |
| `src/rates/ois_curve.py` | `compute_tau_act365`, `bootstrap_ois_discount_factors`, `get_ois_discount_factor`, `build_daily_ois_discount_functions`, `get_maturity_ois_df` |

## carry_gap.py — critical formula & naming conventions
- **`compute_carry_gap(B_hat, B_OIS, tau)`** returns **basis-point** carry gap: `(10_000 / tau) * log(B_hat / B_OIS)`. Note: second parameter is `B_OIS` (NOT `D_ois`); sign is `log(B_hat/B_OIS)` (NOT `log(D_OIS/B_hat)`).
- A **positive** CG means B_hat > B_OIS (option-implied discount factor exceeds OIS).
- `compute_carry_gap_bp` is an alias; `build_carry_gap_panel` accepts either `B_OIS` or `D_ois` column in OIS input, normalising to `B_OIS`.
- `build_synthetic_carry_gap_panel` output column is `B_OIS` (not `D_ois`). B_hat derived as `B_OIS * exp(cg_decimal * tau)` (positive exponent for positive CG).
- `validate_carry_gap_panel` returns keys: `pass`, `all_B_hat_positive`, `frac_nan_CG`, `n_obs`, `n_markets`, `cg_bp_all_finite`, `count_by_market`.

## baseline_gbm.py — OIS storage convention
- `generate_synthetic_vol_ois(start_date, end_date, market, random_seed)` returns DataFrame with `OIS_1Y` and `OIS_10Y` stored as **decimal fractions** (e.g. 0.053 for 5.3%), NOT percentage points. Tests assert `OIS_1Y <= 0.10`.
- GBM formula uses OIS rates in decimal directly: `factor = 1e4 * (2/3) * (vol_pct/100) * sqrt(2*tau/pi)`, `GBM_1Y = ois1y_decimal * factor`.
- `build_gbm_regressors(log_ret_series)` uses the rolling-variance API (Series input, returns DataFrame with GBM_1Y, GBM_10Y). First 251 rows NaN (252-day warm-up).

## Key design decisions
- `compute_insample_metrics` accepts optional `k` param; `adj_R2` is only returned when `k` is supplied.
- `prepare_regression_data` sorts by `['date', 'expiration']` before adding constant — required for HAC to be valid.
- `sm.add_constant(..., has_constant='add')` is used everywhere to force constant column insertion.
- LOYO folds iterate over all calendar years present in the panel; `include_partial_year=True` (default) includes the current/partial year.
- `run_loyo_fold` returns `None` for years absent from the panel; `run_loyo_validation` silently skips `None` folds.
- Stability tables include `const` row — filter it out in reporting if only regressor stability is needed.

## Data pipeline design decisions
- **Option panel**: Strike grid is moneyness 0.85–1.15 in 0.025 steps (13 strikes). Maturities in ACT/365 days: [30,60,91,122,152,182,213,273,365]. Vol surface: `base_vol + TERM_SLOPE*tau + SKEW*(1-m)*base_vol` for m < 1 (put-skew only). Spread: `0.5% + 4%*|ln(m)|` of mid, clamped to [0.5%, 25%]. Minimum spread forced by `MIN_PRICE=0.01` floor on ask vs bid.
- **Critical BS floor bug fixed**: `_bs_call_put` uses European no-arbitrage bounds `max(S − K·e^{−rτ}, 0)` for calls and `max(K·e^{−rτ} − S, 0)` for puts — NOT undiscounted intrinsic `K−S`. Using undiscounted intrinsic inflates deep-ITM put mids by up to ~3%, breaking put-call parity and biasing B_hat above 1.
- **OIS simulation**: Deterministic policy-rate path from key Fed dates + OU mean-reverting noise (κ=0.05, σ=0.015 daily). Term premium compresses under inversion (`spread *= (1 − 0.8 * min(policy/5.5, 1))`). Cached to `data/cache/ois_rates.parquet`.
- **OIS discount factors**: Continuously-compounded zero rates `D(T) = exp(−r(T)·T)`. Log-linear interpolation via `scipy.interpolate.interp1d` with `fill_value='extrapolate'`.
- **Synthetic forward OLS**: `G = C − P = B·(F − K)` → regress G on [1, K] to get B̂ = −slope, F̂ = intercept/B̂. Requires ≥5 matched call-put pairs per cell. R² routinely > 0.9999 with well-formed synthetic data.
- **Caching**: All loaders cache to `data/cache/<market>_<start>_<end>_<suffix>.parquet`. Delete cache files after any change to generation logic.
- **massive API**: `client.list_aggs(ticker, multiplier=1, timespan='day', from_=start, to=end, limit=50000)` returns `Agg` objects with `.close` and `.timestamp` (Unix ms). Convert timestamp: `pd.Timestamp(a.timestamp, unit='ms').normalize()`.

## Feature-generator API contracts (confirmed, tested)
| Function | Signature | Returns |
|---|---|---|
| `generate_synthetic_vol_ois` | `(start_date, end_date, market, random_seed=42)` | `DataFrame` with index=bdate_range, cols `[log_return, GBM_1Y, GBM_10Y, OIS_1Y, OIS_10Y]` |
| `build_gbm_regressors` | `(log_ret: pd.Series, windows=(252,2520))` | `DataFrame` cols `[GBM_1Y, GBM_10Y]` |
| `generate_synthetic_ba_med` | `(start_date, end_date, market='SPX', random_seed=42)` | `pd.Series` named `'BA_med'`, bdate index |
| `build_friction_regressor` | `(ba_med: pd.Series, tau_vals: pd.Series)` | `pd.Series` named `'BA_over_tau'` |
| `load_nfci` | `(start_date, end_date, random_seed=42)` | `pd.Series` named `'NFCI'`, bdate index |
| `build_regression_panel` | `(carry_gap_panel, random_seed=42)` | `DataFrame` with all baseline regressors |
| `build_extended_panel` | `(carry_gap_panel, drift_horizon=504, random_seed=42)` | `DataFrame` adding `GBM_mu_1Y` |

## Experiment outputs (results/ directory)
- `exp1_*`: panel stats CSV + CG distribution/timeseries/maturity PNGs
- `exp2_*`: baseline coef CSVs per market, metrics CSV, coef-bar PNG, actual-vs-fitted PNG
- `exp3_*`: extended model comparison CSV + coef CSVs, t-stat PNG, comparison bar PNG
- `exp4_*`: horizon scan CSVs per market, t-stat and R² vs horizon PNGs
- `exp5_*`: LOYO summary CSV, fold-metrics CSV, R²-by-year / stability / OOS scatter PNGs
- `exp6_*`: descriptive stats CSV, correlation matrix / regressor timeseries / ACF / CG-vs-vol PNGs

## File-editor persistence caveat
`file_editor str_replace` may not persist if `old_str` contains Unicode en-dashes (`─`) or other special chars that differ by encoding. If a replacement appears to succeed in the snippet but `grep` shows old content, use Python (`open/read/replace/write`) to make the change instead. Always confirm with `grep -n` after editing.

## Working directory
`/workspace/project` — run all Python from here so `src.*` imports resolve.
