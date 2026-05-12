from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExplainedAlphaSummary:
    model: str
    explained_fraction: float
    explained_fraction_with_market: float | None
    test_assets: int


@dataclass(frozen=True)
class AutoencoderDiagnostics:
    reconstruction_mse: float
    encoder_weight_sparsity: float


def variance_normalize(frame: pd.DataFrame) -> pd.DataFrame:
    values = frame.astype(float)
    variance = values.var(ddof=1).replace(0, np.nan)
    return values / np.sqrt(variance)


def pca_latent_factors(frame: pd.DataFrame, n_factors: int, prefix: str) -> pd.DataFrame:
    clean = frame.dropna(axis=1, how="any").dropna(axis=0, how="any")
    values = clean.to_numpy(dtype=float)
    _, _, vt = np.linalg.svd(values, full_matrices=False)
    loadings = vt[:n_factors, :].T
    factors = values @ loadings
    columns = [f"{prefix}_{i + 1}" for i in range(n_factors)]
    return pd.DataFrame(factors, index=clean.index, columns=columns)


def recursive_latent_factors(frame: pd.DataFrame, n_factors: int, prefix: str) -> pd.DataFrame:
    residual = frame.dropna(axis=1, how="any").dropna(axis=0, how="any").copy()
    factors = []
    for i in range(n_factors):
        one_factor = pca_latent_factors(residual, 1, f"{prefix}_{i + 1}").iloc[:, 0]
        factors.append(one_factor)
        loading = np.linalg.pinv(one_factor.to_numpy()[:, None]) @ residual.to_numpy()
        residual = residual - np.outer(one_factor.to_numpy(), loading.ravel())
    return pd.concat(factors, axis=1)


def sparse_autoencoder_latent_factors(
    frame: pd.DataFrame,
    n_factors: int,
    hidden_layers: tuple[int, ...],
    activation: str,
    l1_penalty: float,
    epochs: int,
    learning_rate: float,
    seed: int,
    prefix: str,
) -> tuple[pd.DataFrame, AutoencoderDiagnostics]:
    """Train a reconstruction autoencoder and return encoder factors plus fit diagnostics."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for sparse autoencoder factor-zoo experiments.") from exc

    clean = frame.dropna(axis=1, how="any").dropna(axis=0, how="any")
    values = clean.to_numpy(dtype=np.float32)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(values, device=device)

    encoder = _make_mlp(values.shape[1], n_factors, hidden_layers, activation, nn).to(device)
    decoder = _make_mlp(n_factors, values.shape[1], tuple(reversed(hidden_layers)), activation, nn).to(device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=learning_rate)

    for _ in range(epochs):
        optimizer.zero_grad()
        latent = encoder(x)
        reconstructed = decoder(latent)
        loss = torch.mean((x - reconstructed) ** 2)
        if l1_penalty:
            loss = loss + l1_penalty * _l1_norm(encoder)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        latent_values = encoder(x).detach().cpu().numpy()
        reconstructed_values = decoder(encoder(x)).detach().cpu().numpy()

    columns = [f"{prefix}_{i + 1}" for i in range(n_factors)]
    diagnostics = AutoencoderDiagnostics(
        reconstruction_mse=float(np.mean((values - reconstructed_values) ** 2)),
        encoder_weight_sparsity=_weight_sparsity(encoder),
    )
    return pd.DataFrame(latent_values, index=clean.index, columns=columns), diagnostics


def unexplained_alpha_fraction(
    test_assets: pd.DataFrame,
    factors: pd.DataFrame,
    model_name: str,
    market: pd.Series | None = None,
    t_threshold: float = 1.96,
) -> ExplainedAlphaSummary:
    factor_aligned = factors.dropna()
    assets, factor_aligned = test_assets.align(factor_aligned, join="inner", axis=0)
    explained = _fraction_with_insignificant_alphas(assets, factor_aligned, t_threshold)

    explained_with_market = None
    if market is not None:
        market_frame = market.rename("market").to_frame()
        augmented = factor_aligned.join(market_frame, how="inner")
        assets_augmented = assets.reindex(augmented.index)
        explained_with_market = _fraction_with_insignificant_alphas(
            assets_augmented,
            augmented,
            t_threshold,
        )

    return ExplainedAlphaSummary(
        model=model_name,
        explained_fraction=explained,
        explained_fraction_with_market=explained_with_market,
        test_assets=int(assets.shape[1]),
    )


def _fraction_with_insignificant_alphas(
    assets: pd.DataFrame,
    factors: pd.DataFrame,
    t_threshold: float,
) -> float:
    x = np.column_stack([np.ones(len(factors)), factors.to_numpy(dtype=float)])
    significant = 0
    tested = 0
    for col in assets.columns:
        y = assets[col].reindex(factors.index).to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if mask.sum() <= x.shape[1] + 1:
            continue
        beta = np.linalg.pinv(x[mask]) @ y[mask]
        residual = y[mask] - x[mask] @ beta
        dof = max(mask.sum() - x.shape[1], 1)
        sigma2 = float(residual.T @ residual / dof)
        cov = sigma2 * np.linalg.pinv(x[mask].T @ x[mask])
        alpha_se = float(np.sqrt(max(cov[0, 0], 0.0)))
        alpha_t = np.inf if alpha_se == 0 else beta[0] / alpha_se
        significant += int(abs(alpha_t) > t_threshold)
        tested += 1
    if tested == 0:
        return np.nan
    return float(1.0 - significant / tested)


def _make_mlp(
    input_dim: int,
    output_dim: int,
    hidden_layers: tuple[int, ...],
    activation: str,
    nn_module,
):
    layers = []
    current = input_dim
    for hidden in hidden_layers:
        layers.append(nn_module.Linear(current, hidden))
        layers.append(_activation_module(activation, nn_module))
        current = hidden
    layers.append(nn_module.Linear(current, output_dim))
    return nn_module.Sequential(*layers)


def _activation_module(activation: str, nn_module):
    if activation == "relu":
        return nn_module.ReLU()
    if activation == "tanh":
        return nn_module.Tanh()
    if activation == "gelu":
        return nn_module.GELU()
    if activation == "elu":
        return nn_module.ELU()
    raise ValueError(f"Unsupported activation: {activation}")


def _l1_norm(module) -> object:
    return sum(param.abs().sum() for name, param in module.named_parameters() if "weight" in name)


def _weight_sparsity(module, threshold: float = 1e-6) -> float:
    weights = [
        param.detach().cpu().numpy().ravel()
        for name, param in module.named_parameters()
        if "weight" in name
    ]
    if not weights:
        return np.nan
    all_weights = np.concatenate(weights)
    return float(np.mean(np.abs(all_weights) <= threshold))

