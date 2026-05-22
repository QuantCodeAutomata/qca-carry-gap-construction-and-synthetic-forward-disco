import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, List, Tuple, Optional
from src.models.baseline_regression import fit_baseline_hac, compute_insample_metrics


def _detect_col(columns: pd.Index, name: str) -> Optional[str]:
    """Return the actual column name matching `name` case-insensitively, or None."""
    for col in columns:
        if col.lower() == name.lower():
            return col
    return None


def _prepare_xy(
    df: pd.DataFrame,
    regressors: list,
    target: str,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X (with constant) and y from a pre-filtered DataFrame.
    Sorts by date then expiration, drops NaN rows for regressors and target.
    """
    date_col = _detect_col(df.columns, 'date')
    exp_col = _detect_col(df.columns, 'expiration')
    sort_cols = [c for c in [date_col, exp_col] if c is not None]

    working = df.copy()
    if sort_cols:
        working = working.sort_values(sort_cols)

    sub = working[regressors + [target]].dropna()
    X = sm.add_constant(sub[regressors], has_constant='add')
    y = sub[target].astype(float)
    return X, y


def get_calendar_year_folds(
    panel: pd.DataFrame,
    date_col: str = 'date',
) -> List[int]:
    """Return sorted list of unique calendar years in panel."""
    actual_col = _detect_col(panel.columns, date_col) or date_col
    return sorted(panel[actual_col].dt.year.unique().tolist())


def run_loyo_fold(
    panel: pd.DataFrame,
    holdout_year: int,
    regressors: list,
    target: str = 'CG_bp',
    market: str = 'SPX',
    max_lags: int = 21,
) -> Optional[Dict]:
    """
    Run one LOYO fold:
    1. Split panel into train (year != holdout_year) and test (year == holdout_year)
    2. Fit OLS with HAC on training data
    3. Predict on holdout year
    4. Compute holdout metrics: R2, RMSE, MAE, Corr, n_obs
    5. Return dict with all outputs including coefficient estimates
    """
    market_col = _detect_col(panel.columns, 'market')
    if market_col is None:
        raise KeyError("Panel must contain a 'market' column.")
    date_col = _detect_col(panel.columns, 'date')
    if date_col is None:
        raise KeyError("Panel must contain a 'date' column.")

    df = panel[panel[market_col] == market].copy()
    years = df[date_col].dt.year

    train_df = df[years != holdout_year]
    test_df = df[years == holdout_year]

    if len(train_df) == 0 or len(test_df) == 0:
        return None

    X_train, y_train = _prepare_xy(train_df, regressors, target)
    X_test, y_test = _prepare_xy(test_df, regressors, target)

    if len(X_train) == 0 or len(X_test) == 0:
        return None

    result = fit_baseline_hac(X_train, y_train, max_lags=max_lags)

    y_pred = pd.Series(result.predict(X_test), index=y_test.index, name=target)

    metrics = compute_insample_metrics(y_test, y_pred)
    metrics['n_obs'] = int(len(y_test))

    return {
        'holdout_year': holdout_year,
        'result': result,
        'y_pred': y_pred,
        'y_test': y_test,
        'metrics': metrics,
        'params': result.params,
        'coef_signs': np.sign(result.params),
        'coef_pvalues': result.pvalues,
    }


def run_loyo_validation(
    panel: pd.DataFrame,
    regressors: list,
    target: str = 'CG_bp',
    market: str = 'SPX',
    max_lags: int = 21,
    include_partial_year: bool = True,
) -> Dict:
    """
    Run full LOYO validation across all calendar years.

    Returns dict with:
    - fold_results: list of per-fold dicts
    - fold_metrics: DataFrame with year, R2, RMSE, MAE, Corr, n_obs
    - pooled_predictions: Series of all holdout predictions
    - pooled_actuals: Series of all holdout actuals
    - pooled_R2: R2 on concatenated predictions vs actuals
    - mean_R2: mean of yearly R2 values
    - median_R2: median of yearly R2 values
    - mean_RMSE: mean RMSE
    - mean_MAE: mean MAE
    - n_positive_R2: count of folds with positive R2
    - coefficient_signs: DataFrame with sign by fold and regressor
    - coefficient_significance: DataFrame with HAC significance by fold
    """
    years = get_calendar_year_folds(panel)

    if not include_partial_year:
        current_year = pd.Timestamp.now().year
        years = [y for y in years if y < current_year]

    fold_results = []
    for year in years:
        fold = run_loyo_fold(panel, year, regressors, target, market, max_lags)
        if fold is not None:
            fold_results.append(fold)

    if not fold_results:
        return {}

    # Per-fold metrics DataFrame
    metric_rows = []
    for fold in fold_results:
        row = {'year': fold['holdout_year']}
        row.update(fold['metrics'])
        metric_rows.append(row)
    fold_metrics = pd.DataFrame(metric_rows).set_index('year')

    # Pooled OOS predictions and actuals
    pooled_predictions = pd.concat([f['y_pred'] for f in fold_results])
    pooled_actuals = pd.concat([f['y_test'] for f in fold_results])

    # Coefficient stability DataFrames: rows=year, cols=regressors+const
    sign_rows: Dict[int, pd.Series] = {}
    sig_rows: Dict[int, pd.Series] = {}
    for fold in fold_results:
        year = fold['holdout_year']
        sign_rows[year] = np.sign(fold['params'])
        sig_rows[year] = (fold['coef_pvalues'] < 0.05).astype(int)

    coef_sign_df = pd.DataFrame(sign_rows).T
    coef_sign_df.index.name = 'year'

    coef_sig_df = pd.DataFrame(sig_rows).T
    coef_sig_df.index.name = 'year'

    return {
        'fold_results': fold_results,
        'fold_metrics': fold_metrics,
        'pooled_predictions': pooled_predictions,
        'pooled_actuals': pooled_actuals,
        'pooled_R2': pooled_oos_r2(pooled_actuals, pooled_predictions),
        'mean_R2': float(fold_metrics['R2'].mean()),
        'median_R2': float(fold_metrics['R2'].median()),
        'mean_RMSE': float(fold_metrics['RMSE'].mean()),
        'mean_MAE': float(fold_metrics['MAE'].mean()),
        'n_positive_R2': int((fold_metrics['R2'] > 0).sum()),
        'coefficient_signs': coef_sign_df,
        'coefficient_significance': coef_sig_df,
    }


def pooled_oos_r2(
    actuals: pd.Series,
    predictions: pd.Series,
) -> float:
    """
    Compute pooled OOS R2 from concatenated actuals and predictions.
    R2 = 1 - SS_res / SS_tot
    where SS_tot uses grand mean of actuals.
    """
    ss_res = np.sum((actuals - predictions) ** 2)
    ss_tot = np.sum((actuals - actuals.mean()) ** 2)
    return 1.0 - ss_res / ss_tot


def run_loyo_both_markets(
    panel: pd.DataFrame,
    baseline_regressors: list,
    extended_regressors: list,
) -> Dict:
    """
    Run LOYO validation for both markets and both model specifications.
    Returns nested dict: {market: {model: results}}
    """
    results: Dict = {}
    for market in ['SPX', 'RUT']:
        results[market] = {
            'baseline': run_loyo_validation(
                panel, baseline_regressors, market=market,
            ),
            'extended': run_loyo_validation(
                panel, extended_regressors, market=market,
            ),
        }
    return results


def format_loyo_summary(
    loyo_results: Dict,
) -> pd.DataFrame:
    """
    Format LOYO results into summary table.
    Rows: market x model combinations
    Columns: pooled_R2, mean_R2, median_R2, mean_RMSE, mean_MAE, n_positive_R2
    """
    summary_keys = ['pooled_R2', 'mean_R2', 'median_R2', 'mean_RMSE', 'mean_MAE', 'n_positive_R2']
    rows = []
    for market, models in loyo_results.items():
        for model_name, res in models.items():
            if not res:
                continue
            row = {'market': market, 'model': model_name}
            for k in summary_keys:
                row[k] = res.get(k, np.nan)
            rows.append(row)

    df = pd.DataFrame(rows, columns=['market', 'model'] + summary_keys)
    df = df.set_index(['market', 'model'])
    return df
