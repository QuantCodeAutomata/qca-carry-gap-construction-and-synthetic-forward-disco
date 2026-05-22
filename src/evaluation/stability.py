import pandas as pd
import numpy as np
from typing import Dict, List


def coefficient_sign_matrix(
    loyo_results: Dict,
    market: str,
    model: str,
) -> pd.DataFrame:
    """
    Build coefficient sign matrix: rows=folds, cols=regressors.
    Sign: +1, -1, or 0 (zero by convention if very small).
    """
    res = loyo_results[market][model]
    sign_df = res['coefficient_signs'].copy()
    # Replace near-zero entries (|coef| treated as 0 in sign step) with 0
    # np.sign already returns 0 for zero inputs; matrix is already sign-valued
    return sign_df


def sign_stability_summary(
    sign_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarize sign stability:
    - n_positive: count of folds with positive coefficient
    - n_negative: count of folds with negative coefficient
    - pct_stable: percentage of folds with most-common sign
    """
    rows = []
    for col in sign_matrix.columns:
        signs = sign_matrix[col].dropna()
        n_pos = int((signs > 0).sum())
        n_neg = int((signs < 0).sum())
        n_total = len(signs)
        pct_stable = max(n_pos, n_neg) / n_total * 100.0 if n_total > 0 else np.nan
        rows.append({
            'regressor': col,
            'n_positive': n_pos,
            'n_negative': n_neg,
            'pct_stable': pct_stable,
        })
    return pd.DataFrame(rows, columns=['regressor', 'n_positive', 'n_negative', 'pct_stable']).set_index('regressor')


def significance_count_summary(
    loyo_results: Dict,
    market: str,
    model: str,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Count folds where each coefficient is HAC-significant at level alpha.
    """
    res = loyo_results[market][model]
    fold_results = res['fold_results']

    sig_rows: Dict[int, pd.Series] = {}
    for fold in fold_results:
        year = fold['holdout_year']
        sig_rows[year] = (fold['coef_pvalues'] < alpha).astype(int)

    sig_df = pd.DataFrame(sig_rows).T
    sig_df.index.name = 'year'

    n_folds = len(sig_df)
    rows = []
    for col in sig_df.columns:
        n_sig = int(sig_df[col].sum())
        rows.append({
            'regressor': col,
            'n_significant': n_sig,
            'n_folds': n_folds,
            'pct_significant': n_sig / n_folds * 100.0 if n_folds > 0 else np.nan,
        })
    return pd.DataFrame(
        rows,
        columns=['regressor', 'n_significant', 'n_folds', 'pct_significant'],
    ).set_index('regressor')


def format_stability_table(
    loyo_results: Dict,
    markets: list = ['SPX', 'RUT'],
    model: str = 'extended',
) -> pd.DataFrame:
    """
    Format combined stability table for both markets.
    Includes sign stability and HAC significance counts per regressor.
    """
    dfs = []
    for market in markets:
        sign_mat = coefficient_sign_matrix(loyo_results, market, model)
        sign_summary = sign_stability_summary(sign_mat)
        sig_summary = significance_count_summary(loyo_results, market, model)

        combined = sign_summary.join(
            sig_summary[['n_significant', 'pct_significant']], how='left'
        )
        combined.insert(0, 'market', market)
        combined.insert(1, 'model', model)
        dfs.append(combined)

    return pd.concat(dfs)
