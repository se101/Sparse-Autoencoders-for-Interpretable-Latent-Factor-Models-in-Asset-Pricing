from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from experiments.metrics import predictive_r2, total_r2


@dataclass(frozen=True)
class ModelScores:
    model: str
    factors: int
    total_r2: float
    predictive_r2: float


def fit_time_series_factor_model(
    returns: pd.DataFrame,
    factors: pd.DataFrame,
) -> pd.DataFrame:
    """Fit no-intercept time-series betas and return fitted values."""

    aligned_returns, aligned_factors = returns.align(factors, join="inner", axis=0)
    y = aligned_returns.to_numpy(dtype=float)
    x = aligned_factors.to_numpy(dtype=float)
    betas = np.linalg.pinv(x) @ y
    fitted = x @ betas
    return pd.DataFrame(fitted, index=aligned_returns.index, columns=aligned_returns.columns)


def fit_pca_factor_model(returns: pd.DataFrame, n_factors: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Static linear latent factor model estimated with SVD/PCA."""

    clean = returns.dropna(axis=1, how="any").dropna(axis=0, how="any")
    values = clean.to_numpy(dtype=float)
    _, _, vt = np.linalg.svd(values, full_matrices=False)
    loadings = vt[:n_factors, :].T
    factors = values @ loadings
    fitted = factors @ loadings.T
    factor_names = [f"pca_{i + 1}" for i in range(n_factors)]
    return (
        pd.DataFrame(fitted, index=clean.index, columns=clean.columns),
        pd.DataFrame(factors * singular_values[:n_factors], index=clean.index, columns=factor_names),
    )


def score_panel_model(
    model_name: str,
    n_factors: int,
    actual: pd.DataFrame,
    fitted: pd.DataFrame,
    expected: pd.DataFrame | None = None,
) -> ModelScores:
    actual_aligned, fitted_aligned = actual.align(fitted, join="inner", axis=0)
    actual_aligned, fitted_aligned = actual_aligned.align(fitted_aligned, join="inner", axis=1)
    expected_values = fitted_aligned if expected is None else expected.reindex_like(fitted_aligned)
    return ModelScores(
        model=model_name,
        factors=n_factors,
        total_r2=total_r2(actual_aligned.to_numpy(), fitted_aligned.to_numpy()),
        predictive_r2=predictive_r2(actual_aligned.to_numpy(), expected_values.to_numpy()),
    )


def build_managed_portfolios(
    panel: pd.DataFrame,
    date_col: str,
    return_col: str,
    characteristic_cols: tuple[str, ...],
) -> pd.DataFrame:
    """Build GKX characteristic-managed portfolios x_t from firm returns and characteristics."""

    records: list[dict[str, float | pd.Timestamp]] = []
    for date, group in panel.groupby(date_col):
        row: dict[str, float | pd.Timestamp] = {"date": pd.Timestamp(date)}
        returns = group[return_col].astype(float)
        row["market_ew"] = float(returns.mean())
        for col in characteristic_cols:
            row[col] = float((group[col].astype(float) * returns).mean())
        records.append(row)
    return pd.DataFrame.from_records(records).sort_values("date").set_index("date")


class ConditionalAutoencoderUnavailable(RuntimeError):
    pass


def train_conditional_autoencoder(
    panel: pd.DataFrame,
    managed_portfolios: pd.DataFrame,
    characteristic_cols: tuple[str, ...],
    return_col: str,
    n_factors: int,
    hidden_layers: tuple[int, ...],
    seed: int,
    epochs: int = 100,
    learning_rate: float = 1e-3,
    l1_penalty: float = 0.0,
) -> pd.DataFrame:
    """Train a GKX-style conditional autoencoder and return fitted row-level values.

    The factor side is linear in characteristic-managed portfolios. The beta side
    is `CA0` when `hidden_layers` is empty and `CA1..CA3` for 32/16/8 hidden
    beta layers, matching the paper's architecture targets.
    """

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise ConditionalAutoencoderUnavailable(
            "PyTorch is required for conditional autoencoder training. Install requirements.txt first."
        ) from exc

    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    month_index = {date: idx for idx, date in enumerate(managed_portfolios.index)}
    train = panel[panel["date"].isin(month_index)].copy()
    train["month_idx"] = train["date"].map(month_index)

    z = torch.tensor(train[list(characteristic_cols)].to_numpy(dtype=np.float32), device=device)
    y = torch.tensor(train[return_col].to_numpy(dtype=np.float32), device=device)
    month_ids = torch.tensor(train["month_idx"].to_numpy(dtype=np.int64), device=device)
    x = torch.tensor(managed_portfolios.to_numpy(dtype=np.float32), device=device)

    beta_net = _make_beta_network(z.shape[1], n_factors, hidden_layers, nn).to(device)
    factor_layer = nn.Linear(x.shape[1], n_factors, bias=True).to(device)
    optimizer = torch.optim.Adam([*beta_net.parameters(), *factor_layer.parameters()], lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        betas = beta_net(z)
        factors = factor_layer(x)[month_ids]
        fitted = (betas * factors).sum(dim=1)
        loss = torch.mean((y - fitted) ** 2)
        if l1_penalty:
            penalty = sum(param.abs().sum() for param in beta_net.parameters())
            loss = loss + l1_penalty * penalty
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        fitted_values = (beta_net(z) * factor_layer(x)[month_ids]).sum(dim=1).detach().cpu().numpy()

    out = train[["date", "permno"]].copy()
    out["fitted"] = fitted_values
    return out


def _make_beta_network(
    input_dim: int,
    output_dim: int,
    hidden_layers: tuple[int, ...],
    nn_module,
):
    layers = []
    current = input_dim
    for hidden in hidden_layers:
        layers.append(nn_module.Linear(current, hidden))
        layers.append(nn_module.BatchNorm1d(hidden))
        layers.append(nn_module.ReLU())
        current = hidden
    layers.append(nn_module.Linear(current, output_dim))
    return nn_module.Sequential(*layers)

