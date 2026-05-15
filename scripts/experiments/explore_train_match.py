"""Sweep variants that could move the Train-row explained fraction toward the paper.

Paper target ("Chimeras", Linear 6): Train ~83/81, FF ~4/48, HXZ ~6/82.
This script holds K=6 fixed and varies:
  - latent model: PCA, recursive PCA, mispricing-penalized linear AE (tied weights)
  - test panel: raw vs variance-normalized factor zoo (Train), plus FF 75 and global-q low/high
  - standard error: OLS vs Newey-West HAC with several lag lengths
  - t-stat threshold: 1.96 / 2.5 / 3.0
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_loading import read_fgx_factors_csv, read_fgx_risk_free_rate  # noqa: E402
from experiments.config import BEN_FACTOR_ZOO  # noqa: E402
from experiments.factor_zoo import (  # noqa: E402
    mispricing_penalized_linear_autoencoder_latent_factors,
    pca_latent_factors,
    recursive_latent_factors,
    unexplained_alpha_fraction,
    variance_normalize,
)

OUTPUT_DIR = BEN_FACTOR_ZOO.output_dir / "pca_test_portfolios"
TEST_PORTFOLIO_DIR = ROOT / "data" / "test_portfolios"

PAPER_TARGETS: dict[str, tuple[float, float]] = {
    "train_raw_no_mkt": (0.83, 0.81),
    "train_raw_with_mkt": (0.83, 0.81),
    "train_normalized_no_mkt": (0.83, 0.81),
    "train_normalized_with_mkt": (0.83, 0.81),
    "fama_french_75": (0.04, 0.48),
    "global_q_low_high": (0.06, 0.82),
}


def _restrict(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


def _to_excess_returns(frame: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    return frame.sub(risk_free, axis=0)


def _read_wide_monthly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    df = df.assign(date=parsed_dates.dt.to_period("M").dt.to_timestamp("M"))
    value_cols = [col for col in df.columns if col != "date"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date").drop_duplicates("date").set_index("date")


def _read_french_25_value_weighted_monthly(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    marker_idx = next(
        i for i, line in enumerate(lines) if "Average Value Weighted Returns -- Monthly" in line
    )
    header_idx = marker_idx + 1
    end_idx = next(
        i
        for i in range(header_idx + 1, len(lines))
        if not lines[i].strip() or not lines[i].split(",", 1)[0].strip().isdigit()
    )
    text = "\n".join(lines[header_idx:end_idx])
    df = pd.read_csv(pd.io.common.StringIO(text))
    df = df.rename(columns={df.columns[0]: "date"})
    parsed_dates = pd.to_datetime(df["date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)
    df = df.assign(date=parsed_dates)
    value_cols = [col for col in df.columns if col != "date"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    df[value_cols] = df[value_cols].replace({-99.99: np.nan, -999.0: np.nan})
    return df.sort_values("date").drop_duplicates("date").set_index("date")


def _load_fama_french_75() -> pd.DataFrame:
    files = {
        "bm": TEST_PORTFOLIO_DIR / "ff" / "25_Portfolios_5x5.csv",
        "op": TEST_PORTFOLIO_DIR / "ff" / "25_Portfolios_ME_OP_5x5.csv",
        "inv": TEST_PORTFOLIO_DIR / "ff" / "25_Portfolios_ME_INV_5x5.csv",
    }
    frames = []
    for prefix, path in files.items():
        frame = _read_french_25_value_weighted_monthly(path)
        frame = frame.rename(columns={col: f"{prefix}__{col}" for col in frame.columns})
        frames.append(frame)
    return pd.concat(frames, axis=1).sort_index()


def _train_mispricing_ae_tied(
    normalized: pd.DataFrame,
    n_factors: int,
    gamma: float,
    epochs: int,
    learning_rate: float,
    seed: int,
    weight_decay: float = 0.0,
) -> pd.DataFrame:
    """Linear AE with tied weights (decoder = encoder.T) and a mispricing penalty."""

    import torch
    from torch import nn

    clean = normalized.dropna(axis=1, how="any").dropna(axis=0, how="any")
    values = clean.to_numpy(dtype=np.float32)
    torch.manual_seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x = torch.tensor(values, device=device)

    encoder = nn.Linear(values.shape[1], n_factors, bias=True).to(device)
    decoder_bias = nn.Parameter(torch.zeros(values.shape[1], device=device))
    optimizer = torch.optim.Adam(
        [*encoder.parameters(), decoder_bias],
        lr=learning_rate,
        weight_decay=weight_decay,
    )

    for _ in range(epochs):
        optimizer.zero_grad()
        latent = encoder(x)
        reconstructed = latent @ encoder.weight + decoder_bias
        reconstruction_loss = torch.mean((x - reconstructed) ** 2)
        mean_loss = torch.mean((torch.mean(x, dim=0) - torch.mean(reconstructed, dim=0)) ** 2)
        loss = reconstruction_loss + gamma * mean_loss
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        latent_values = encoder(x).detach().cpu().numpy()

    columns = [f"linear_ae_tied_{i + 1}" for i in range(n_factors)]
    return pd.DataFrame(latent_values, index=clean.index, columns=columns)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    factors = read_fgx_factors_csv(BEN_FACTOR_ZOO.factors_path).set_index("date")
    risk_free = read_fgx_risk_free_rate(BEN_FACTOR_ZOO.factors_path)
    factors = _restrict(factors, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)
    risk_free = _restrict(risk_free.to_frame(), BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)["RF"]
    market = factors["MktRf"].astype(float)
    factors_no_mkt = factors.drop(columns=["MktRf"])

    normalized_no_mkt = variance_normalize(factors_no_mkt)
    normalized_all = variance_normalize(factors)

    pca_latents = pca_latent_factors(normalized_no_mkt, 6, "linear_k6")
    recursive_latents = recursive_latent_factors(normalized_no_mkt, 6, "recursive_k6")
    ae_latents_paper, _ = mispricing_penalized_linear_autoencoder_latent_factors(
        normalized_no_mkt,
        6,
        BEN_FACTOR_ZOO.mispricing_gamma,
        BEN_FACTOR_ZOO.autoencoder_epochs,
        BEN_FACTOR_ZOO.autoencoder_learning_rate,
        BEN_FACTOR_ZOO.autoencoder_seed,
        "linear_ae_paper_k6",
    )
    ae_latents_tied_g10 = _train_mispricing_ae_tied(
        normalized_no_mkt,
        6,
        gamma=10.0,
        epochs=10000,
        learning_rate=5e-3,
        seed=BEN_FACTOR_ZOO.autoencoder_seed,
    )
    ae_latents_tied_g100 = _train_mispricing_ae_tied(
        normalized_no_mkt,
        6,
        gamma=100.0,
        epochs=10000,
        learning_rate=5e-3,
        seed=BEN_FACTOR_ZOO.autoencoder_seed,
    )

    models: dict[str, pd.DataFrame] = {
        "pca_k6": pca_latents,
        "recursive_pca_k6": recursive_latents,
        "mispricing_ae_k6_default": ae_latents_paper,
        "mispricing_ae_k6_tied_g10_10kep": ae_latents_tied_g10,
        "mispricing_ae_k6_tied_g100_10kep": ae_latents_tied_g100,
    }

    ff_75 = _load_fama_french_75()
    ff_75_excess = _to_excess_returns(ff_75, risk_free)
    ff_75_excess = _restrict(ff_75_excess, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)

    hxz_panel = _read_wide_monthly(
        TEST_PORTFOLIO_DIR / "global_q" / "global_q_1way_monthly_low_high.csv"
    )
    hxz_panel = _to_excess_returns(hxz_panel, risk_free)
    hxz_panel = _restrict(hxz_panel, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)

    test_panels: dict[str, pd.DataFrame] = {
        "train_raw_no_mkt": factors_no_mkt,
        "train_raw_with_mkt": factors,
        "train_normalized_no_mkt": normalized_no_mkt,
        "train_normalized_with_mkt": normalized_all,
        "fama_french_75": ff_75_excess,
        "global_q_low_high": hxz_panel,
    }

    se_specs: list[tuple[str, str, int]] = [
        ("ols", "ols", 0),
        ("hac_l6", "hac", 6),
        ("hac_l12", "hac", 12),
    ]
    thresholds: list[float] = [1.96, 2.5, 3.0]

    rows = []
    for model_name, latent in models.items():
        for panel_name, panel in test_panels.items():
            target_no, target_with = PAPER_TARGETS[panel_name]
            for se_label, se_method, lags in se_specs:
                for t_thr in thresholds:
                    result = unexplained_alpha_fraction(
                        panel,
                        latent,
                        model_name,
                        market=market,
                        t_threshold=t_thr,
                        se_method=se_method,
                        hac_lags=lags,
                    )
                    rows.append(
                        {
                            "model": model_name,
                            "test_panel": panel_name,
                            "se": se_label,
                            "t_threshold": t_thr,
                            "test_assets": result.test_assets,
                            "no_mkt_pct": round(result.explained_fraction * 100.0, 2),
                            "with_mkt_pct": (
                                round(result.explained_fraction_with_market * 100.0, 2)
                                if result.explained_fraction_with_market is not None
                                else np.nan
                            ),
                            "paper_no_mkt_pct": round(target_no * 100.0, 2),
                            "paper_with_mkt_pct": round(target_with * 100.0, 2),
                            "abs_gap_no_mkt": round(
                                abs(result.explained_fraction - target_no) * 100.0, 2
                            ),
                            "abs_gap_with_mkt": (
                                round(
                                    abs(result.explained_fraction_with_market - target_with)
                                    * 100.0,
                                    2,
                                )
                                if result.explained_fraction_with_market is not None
                                else np.nan
                            ),
                        }
                    )

    out = pd.DataFrame(rows).sort_values(
        ["model", "test_panel", "se", "t_threshold"], kind="stable"
    )
    out_path = OUTPUT_DIR / "train_match_exploration.csv"
    out.to_csv(out_path, index=False)

    closest = out.assign(
        combined_gap=out["abs_gap_no_mkt"].fillna(0) + out["abs_gap_with_mkt"].fillna(0)
    ).nsmallest(10, "combined_gap")
    closest_path = OUTPUT_DIR / "train_match_top10.csv"
    closest.to_csv(closest_path, index=False)

    train_rows = out["test_panel"].isin(["train_raw_no_mkt"])
    ff_rows = out["test_panel"] == "fama_french_75"
    hxz_rows = out["test_panel"] == "global_q_low_high"
    panels_per_config = out.loc[
        train_rows | ff_rows | hxz_rows,
        ["model", "se", "t_threshold", "test_panel", "abs_gap_no_mkt", "abs_gap_with_mkt"],
    ]
    panels_per_config = panels_per_config.assign(
        panel_gap=panels_per_config["abs_gap_no_mkt"].fillna(0)
        + panels_per_config["abs_gap_with_mkt"].fillna(0)
    )
    aggregated = (
        panels_per_config.groupby(["model", "se", "t_threshold"])["panel_gap"].sum().reset_index()
    )
    aggregated = aggregated.sort_values("panel_gap", kind="stable")
    aggregated.to_csv(OUTPUT_DIR / "train_match_joint_ranking.csv", index=False)

    print(f"Wrote {out_path}")
    print(f"Wrote {closest_path}")
    print(f"Wrote {OUTPUT_DIR / 'train_match_joint_ranking.csv'}")
    print()
    print("Best (model, se, t_threshold) joint over Train/FF/HXZ (lower sum-gap = better):")
    print(aggregated.head(10).to_string(index=False))
    return 0



if __name__ == "__main__":
    raise SystemExit(main())
