"""Global configuration for carry-gap experiments."""

START_DATE = '2016-01-01'
END_DATE = '2025-10-31'
MARKETS = ['SPX', 'RUT']
MAIN_DRIFT_HORIZON = 504          # trading days (~2 calendar years)
HAC_MAX_LAGS = 21
ATM_BANDWIDTH = 0.05
MIN_STRIKES = 5
MIN_MID = 0.01
MAX_SPREAD_RATIO = 2.0
RANDOM_SEED = 42
RESULTS_DIR = 'results'
DATA_CACHE_DIR = 'data/cache'

BASELINE_REGRESSORS = ['GBM_1Y', 'GBM_10Y', 'BA_over_tau', 'NFCI']
EXTENDED_REGRESSORS = ['GBM_1Y', 'GBM_10Y', 'GBM_mu_1Y', 'BA_over_tau', 'NFCI']

# Horizon scan grid: every 10 trading days from 126 to 756, plus key values
HORIZON_GRID = sorted(set(
    list(range(126, 757, 10)) +
    [481, 487, 500, 504, 514, 525, 542, 551, 572, 581]
))

# ATM maturity bucket for Figure 1.1 (5–7 months in years)
FIG_TAU_MIN = 5 / 12
FIG_TAU_MAX = 7 / 12
