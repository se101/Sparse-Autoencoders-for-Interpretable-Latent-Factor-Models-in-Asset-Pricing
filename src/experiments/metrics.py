from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AlphaSummary:
    mean_abs_alpha: float
    root_mean_square_alpha: float
    significant_alpha_count: int
    asset_count: int


def total_r2(actual: np.ndarray, fitted: np.ndarray) -> float:
    """GKX-style total R2: panel variation explained by contemporaneous factors."""

    actual, fitted = _aligned_arrays(actual, fitted)
    denominator = np.nansum(actual**2)
    if denominator == 0:
        return np.nan
    return 1.0 - np.nansum((actual - fitted) ** 2) / denominator


def predictive_r2(actual: np.ndarray, predicted: np.ndarray) -> float:
    """GKX-style predictive R2 for one-step-ahead conditional expected returns."""

    actual, predicted = _aligned_arrays(actual, predicted)
    denominator = np.nansum(actual**2)
    if denominator == 0:
        return np.nan
    return 1.0 - np.nansum((actual - predicted) ** 2) / denominator


def annualized_sharpe(returns: pd.Series | np.ndarray, periods_per_year: int = 12) -> float:
    values = pd.Series(returns).dropna().astype(float)
    if values.empty:
        return np.nan
    volatility = values.std(ddof=1)
    if volatility == 0:
        return np.nan
    return float(np.sqrt(periods_per_year) * values.mean() / volatility)


def prediction_decile_spread(
    frame: pd.DataFrame,
    date_col: str,
    return_col: str,
    prediction_col: str,
    quantiles: int = 10,
) -> pd.Series:
    """Return monthly high-minus-low spreads sorted on predicted returns."""

    spreads: list[tuple[pd.Timestamp, float]] = []
    for date, group in frame.dropna(subset=[return_col, prediction_col]).groupby(date_col):
        if group[prediction_col].nunique() < quantiles:
            continue
        buckets = pd.qcut(group[prediction_col], quantiles, labels=False, duplicates="drop")
        low = group.loc[buckets == buckets.min(), return_col].mean()
        high = group.loc[buckets == buckets.max(), return_col].mean()
        spreads.append((pd.Timestamp(date), float(high - low)))
    return pd.Series(dict(spreads)).sort_index()


def pricing_error_summary(
    frame: pd.DataFrame,
    asset_col: str,
    residual_col: str,
    annualization: int = 12,
    t_threshold: float = 3.0,
) -> AlphaSummary:
    """Summarize unconditional alphas from out-of-sample residuals."""

    alphas = []
    tstats = []
    for _, group in frame.dropna(subset=[residual_col]).groupby(asset_col):
        residuals = group[residual_col].astype(float)
        if residuals.empty:
            continue
        alpha = residuals.mean()
        se = residuals.std(ddof=1) / np.sqrt(len(residuals))
        alphas.append(alpha * annualization)
        tstats.append(np.nan if se == 0 else alpha / se)

    alpha_array = np.asarray(alphas, dtype=float)
    t_array = np.asarray(tstats, dtype=float)
    if alpha_array.size == 0:
        return AlphaSummary(np.nan, np.nan, 0, 0)

    return AlphaSummary(
        mean_abs_alpha=float(np.nanmean(np.abs(alpha_array))),
        root_mean_square_alpha=float(np.sqrt(np.nanmean(alpha_array**2))),
        significant_alpha_count=int(np.nansum(np.abs(t_array) > t_threshold)),
        asset_count=int(alpha_array.size),
    )


def factor_tangency_sharpe(factors: pd.DataFrame, target_monthly_vol: float = 0.01) -> float:
    """Ex post tangency Sharpe for a factor return panel."""

    numeric = factors.dropna().astype(float)
    if numeric.empty:
        return np.nan
    mean = numeric.mean().to_numpy()
    cov = numeric.cov().to_numpy()
    try:
        weights = np.linalg.solve(cov, mean)
    except np.linalg.LinAlgError:
        weights = np.linalg.pinv(cov) @ mean
    returns = numeric.to_numpy() @ weights
    vol = np.std(returns, ddof=1)
    if vol > 0:
        returns = returns * (target_monthly_vol / vol)
    return annualized_sharpe(returns)


def _aligned_arrays(actual: np.ndarray, fitted: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(actual, dtype=float)
    b = np.asarray(fitted, dtype=float)
    mask = np.isfinite(a) & np.isfinite(b)
    return a[mask], b[mask]

