import pandas as pd
import numpy as np
import statsmodels.api as sm
from typing import Dict, Tuple, Optional
from src.models.baseline_regression import (
    fit_baseline_hac,
    compute_insample_metrics,
    format_coefficient_table,
    prepare_regression_data,
)

EXTENDED_REGRESSORS = ['GBM_1Y', 'GBM_10Y', 'GBM_mu_1Y', 'BA_over_tau', 'NFCI']


def run_extended_regression(
    panel: pd.DataFrame,
    market: str,
    max_lags: int = 21,
) -> Dict:
    """
    Run extended regression (baseline + drift term) for one market.
    Extended spec: CG_bp = alpha + phi1*GBM_1Y + phi10*GBM_10Y + psi*GBM_mu_1Y
                           + beta*BA_over_tau + gamma*NFCI
    Returns dict with: result, metrics, coefficients_df, X, y
    """
    X, y = prepare_regression_data(panel, market, EXTENDED_REGRESSORS)
    result = fit_baseline_hac(X, y, max_lags=max_lags)
    metrics = compute_insample_metrics(
        y, pd.Series(result.fittedvalues, index=y.index),
        k=len(EXTENDED_REGRESSORS),
    )
    coef_df = format_coefficient_table(result, model_name='Extended')
    return {
        'result': result,
        'metrics': metrics,
        'coefficients_df': coef_df,
        'X': X,
        'y': y,
    }


def run_extended_both_markets(
    panel: pd.DataFrame,
    max_lags: int = 21,
) -> Dict:
    """
    Run extended regression for SPX and RUT.
    Returns dict with market keys.
    """
    return {
        market: run_extended_regression(panel, market, max_lags=max_lags)
        for market in ['SPX', 'RUT']
    }


def compare_models(
    baseline_results: Dict,
    extended_results: Dict,
    market: str,
) -> pd.DataFrame:
    """
    Side-by-side comparison of baseline vs extended model metrics.
    Columns: metric, baseline, extended, delta
    """
    base_metrics = baseline_results[market]['metrics']
    ext_metrics = extended_results[market]['metrics']

    all_keys = sorted(set(base_metrics) | set(ext_metrics))
    rows = []
    for key in all_keys:
        b_val = base_metrics.get(key, np.nan)
        e_val = ext_metrics.get(key, np.nan)
        if not (np.isnan(b_val) or np.isnan(e_val)):
            delta = e_val - b_val
        else:
            delta = np.nan
        rows.append({'metric': key, 'baseline': b_val, 'extended': e_val, 'delta': delta})

    return pd.DataFrame(rows, columns=['metric', 'baseline', 'extended', 'delta'])
