import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Tuple, Optional

# Using statsmodels OLS with HAC - Context7 confirmed

BASELINE_REGRESSORS = ['GBM_1Y', 'GBM_10Y', 'BA_over_tau', 'NFCI']


def _detect_col(columns: pd.Index, name: str) -> Optional[str]:
    """Return the actual column name matching `name` case-insensitively, or None."""
    for col in columns:
        if col.lower() == name.lower():
            return col
    return None


def prepare_regression_data(
    panel: pd.DataFrame,
    market: str,
    regressors: list,
    target: str = 'CG_bp',
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare X, y for regression from panel.
    Filters to market, drops rows with NaN in any regressor or target.
    Sorts by date, then expiration (for HAC).
    Returns (X_with_const, y).
    """
    market_col = _detect_col(panel.columns, 'market')
    if market_col is None:
        raise KeyError("Panel must contain a 'market' column.")
    df = panel[panel[market_col] == market].copy()

    date_col = _detect_col(df.columns, 'date')
    exp_col = _detect_col(df.columns, 'expiration')

    sort_cols = [c for c in [date_col, exp_col] if c is not None]
    if sort_cols:
        df = df.sort_values(sort_cols)

    df = df[regressors + [target]].dropna()

    X = sm.add_constant(df[regressors], has_constant='add')
    y = df[target].astype(float)
    return X, y


def fit_baseline_hac(
    X: pd.DataFrame,
    y: pd.Series,
    max_lags: int = 21,
) -> sm.regression.linear_model.RegressionResultsWrapper:
    """
    Fit OLS with HAC(Newey-West) standard errors.
    max_lags=21 trading days as specified in paper.

    Uses cov_type='HAC' with cov_kwds={'maxlags': max_lags}.
    Data must be sorted by date before calling.

    # Using statsmodels OLS with HAC - Context7 confirmed
    """
    model = sm.OLS(y, X)
    result = model.fit(cov_type='HAC', cov_kwds={'maxlags': max_lags})
    return result


def compute_insample_metrics(
    y_actual: pd.Series,
    y_fitted: pd.Series,
    k: Optional[int] = None,
) -> Dict[str, float]:
    """
    Compute in-sample metrics:
    - R2: coefficient of determination
    - adj_R2: adjusted R2 (computed when k, number of regressors excl. constant, is provided)
    - RMSE: root mean squared error in bp
    - MAE: mean absolute error in bp
    - Corr: Pearson correlation
    """
    ss_res = np.sum((y_actual - y_fitted) ** 2)
    ss_tot = np.sum((y_actual - y_actual.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    n = len(y_actual)
    rmse = np.sqrt(np.mean((y_actual - y_fitted) ** 2))
    mae = np.mean(np.abs(y_actual - y_fitted))
    corr = np.corrcoef(y_actual, y_fitted)[0, 1]
    metrics: Dict[str, float] = {'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Corr': corr}
    if k is not None and n > k + 1:
        metrics['adj_R2'] = 1 - (1 - r2) * (n - 1) / (n - k - 1)
    return metrics


def run_baseline_regression(
    panel: pd.DataFrame,
    market: str,
    max_lags: int = 21,
) -> Dict:
    """
    Run full baseline regression for one market.
    Returns dict with: result, metrics, coefficients_df, X, y
    """
    X, y = prepare_regression_data(panel, market, BASELINE_REGRESSORS)
    result = fit_baseline_hac(X, y, max_lags=max_lags)
    metrics = compute_insample_metrics(
        y, pd.Series(result.fittedvalues, index=y.index),
        k=len(BASELINE_REGRESSORS),
    )
    coef_df = format_coefficient_table(result, model_name='Baseline')
    return {
        'result': result,
        'metrics': metrics,
        'coefficients_df': coef_df,
        'X': X,
        'y': y,
    }


def run_baseline_both_markets(
    panel: pd.DataFrame,
    max_lags: int = 21,
) -> Dict:
    """
    Run baseline regression for SPX and RUT.
    Returns dict with market keys.
    """
    return {
        market: run_baseline_regression(panel, market, max_lags=max_lags)
        for market in ['SPX', 'RUT']
    }


def format_coefficient_table(
    result: sm.regression.linear_model.RegressionResultsWrapper,
    model_name: str = 'Baseline',
) -> pd.DataFrame:
    """
    Format coefficient table with HAC SE, t-stats, p-values, significance stars.
    Columns: coef, hac_se, t_stat, p_value, significance
    """
    params = result.params
    bse = result.bse
    tvalues = result.tvalues
    pvalues = result.pvalues

    def sig_stars(p):
        if p < 0.01:
            return '***'
        elif p < 0.05:
            return '**'
        elif p < 0.10:
            return '*'
        return ''

    df = pd.DataFrame({
        'coef': params,
        'hac_se': bse,
        't_stat': tvalues,
        'p_value': pvalues,
        'significance': [sig_stars(p) for p in pvalues],
    })
    df.index.name = 'variable'
    df['model'] = model_name
    return df
