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


@dataclass(frozen=True)
class MispricingAutoencoderDiagnostics:
    reconstruction_mse: float
    mean_return_mse: float
    loss: float


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


def pca_variance_diagnostics(frame: pd.DataFrame, max_factors: int) -> pd.DataFrame:
    """Return scree and reconstruction diagnostics for rank-k PCA approximations."""

    clean = frame.dropna(axis=1, how="any").dropna(axis=0, how="any")
    values = clean.to_numpy(dtype=float)
    _, singular_values, _ = np.linalg.svd(values, full_matrices=False)
    component_ss = singular_values**2
    total_ss = float(component_ss.sum())
    denominator = values.shape[0] * values.shape[1]

    rows = []
    max_k = min(max_factors, len(singular_values))
    cumulative_ss = 0.0
    for k in range(1, max_k + 1):
        cumulative_ss += float(component_ss[k - 1])
        remaining_ss = max(total_ss - cumulative_ss, 0.0)
        rows.append(
            {
                "latent_factors": k,
                "singular_value": float(singular_values[k - 1]),
                "explained_variance": float(component_ss[k - 1] / max(values.shape[0] - 1, 1)),
                "explained_variance_ratio": float(component_ss[k - 1] / total_ss) if total_ss else np.nan,
                "cumulative_explained_variance_ratio": float(cumulative_ss / total_ss)
                if total_ss
                else np.nan,
                "reconstruction_mse": float(remaining_ss / denominator) if denominator else np.nan,
            }
        )

    return pd.DataFrame(rows)


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


def mispricing_penalized_linear_autoencoder_latent_factors(
    frame: pd.DataFrame,
    n_factors: int,
    gamma: float,
    epochs: int,
    learning_rate: float,
    seed: int,
    prefix: str,
) -> tuple[pd.DataFrame, MispricingAutoencoderDiagnostics]:
    """Train a linear AE with a penalty for mismatched average returns."""

    try:
        import torch
        from torch import nn
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for mispricing-penalized autoencoder experiments.") from exc

    clean = frame.dropna(axis=1, how="any").dropna(axis=0, how="any")
    values = clean.to_numpy(dtype=np.float32)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(values, device=device)

    encoder = nn.Linear(values.shape[1], n_factors, bias=True).to(device)
    decoder = nn.Linear(n_factors, values.shape[1], bias=True).to(device)
    optimizer = torch.optim.Adam([*encoder.parameters(), *decoder.parameters()], lr=learning_rate)

    last_loss = torch.tensor(np.nan, device=device)
    for _ in range(epochs):
        optimizer.zero_grad()
        latent = encoder(x)
        reconstructed = decoder(latent)
        reconstruction_loss = torch.mean((x - reconstructed) ** 2)
        mean_return_loss = torch.mean((torch.mean(x, dim=0) - torch.mean(reconstructed, dim=0)) ** 2)
        loss = reconstruction_loss + (1.0 + gamma) * mean_return_loss
        loss.backward()
        optimizer.step()
        last_loss = loss.detach()

    with torch.no_grad():
        latent_values = encoder(x).detach().cpu().numpy()
        reconstructed_values = decoder(encoder(x)).detach().cpu().numpy()

    columns = [f"{prefix}_{i + 1}" for i in range(n_factors)]
    diagnostics = MispricingAutoencoderDiagnostics(
        reconstruction_mse=float(np.mean((values - reconstructed_values) ** 2)),
        mean_return_mse=float(np.mean((values.mean(axis=0) - reconstructed_values.mean(axis=0)) ** 2)),
        loss=float(last_loss.detach().cpu().item()),
    )
    return pd.DataFrame(latent_values, index=clean.index, columns=columns), diagnostics


def unexplained_alpha_fraction(
    test_assets: pd.DataFrame,
    factors: pd.DataFrame,
    model_name: str,
    market: pd.Series | None = None,
    t_threshold: float = 1.96,
    se_method: str = "ols",
    hac_lags: int = 0,
) -> ExplainedAlphaSummary:
    """Fraction of test assets with |t(alpha)| < ``t_threshold`` in a time-series regression.

    ``se_method`` selects the variance estimator for ``alpha``:
      - "ols" (default): homoskedastic OLS sigma^2 (X'X)^-1
      - "hac": Newey-West HAC with ``hac_lags`` Bartlett lags (set hac_lags=0 for White/HC0)
    """

    factor_aligned = factors.dropna()
    assets, factor_aligned = test_assets.align(factor_aligned, join="inner", axis=0)
    explained = _fraction_with_insignificant_alphas(
        assets, factor_aligned, t_threshold, se_method=se_method, hac_lags=hac_lags
    )

    explained_with_market: float | None = None
    if market is not None:
        augmented = factor_aligned.join(market.rename("market").to_frame(), how="inner").dropna()
        assets_augmented = assets.reindex(augmented.index)
        explained_with_market = _fraction_with_insignificant_alphas(
            assets_augmented,
            augmented,
            t_threshold,
            se_method=se_method,
            hac_lags=hac_lags,
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
    se_method: str = "ols",
    hac_lags: int = 0,
) -> float:
    x = np.column_stack([np.ones(len(factors)), factors.to_numpy(dtype=float)])
    significant = 0
    tested = 0
    for col in assets.columns:
        y = assets[col].reindex(factors.index).to_numpy(dtype=float)
        mask = np.isfinite(y) & np.isfinite(x).all(axis=1)
        if mask.sum() <= x.shape[1] + 1:
            continue
        x_m = x[mask]
        y_m = y[mask]
        xtx_inv = np.linalg.pinv(x_m.T @ x_m)
        beta = xtx_inv @ x_m.T @ y_m
        residual = y_m - x_m @ beta
        if se_method == "ols":
            dof = max(mask.sum() - x_m.shape[1], 1)
            sigma2 = float(residual.T @ residual / dof)
            cov = sigma2 * xtx_inv
        elif se_method == "hac":
            cov = _newey_west_cov(x_m, residual, xtx_inv, hac_lags)
        else:
            raise ValueError(f"Unknown se_method: {se_method}")
        alpha_se = float(np.sqrt(max(cov[0, 0], 0.0)))
        alpha_t = np.inf if alpha_se == 0 else beta[0] / alpha_se
        significant += int(abs(alpha_t) > t_threshold)
        tested += 1
    if tested == 0:
        return np.nan
    return float(1.0 - significant / tested)


def _newey_west_cov(
    x: np.ndarray,
    residual: np.ndarray,
    xtx_inv: np.ndarray,
    lags: int,
) -> np.ndarray:
    """Newey-West HAC sandwich covariance for OLS coefficients."""

    n, k = x.shape
    u = residual.reshape(-1, 1) * x
    s = u.T @ u
    for lag in range(1, max(lags, 0) + 1):
        weight = 1.0 - lag / (lags + 1.0)
        gamma = u[lag:].T @ u[:-lag]
        s = s + weight * (gamma + gamma.T)
    dof = max(n - k, 1)
    return xtx_inv @ s @ xtx_inv * (n / dof)


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

