from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from data_loading import read_fgx_factors_csv, read_fgx_risk_free_rate  # noqa: E402
from experiments.config import BEN_FACTOR_ZOO, to_jsonable  # noqa: E402
from experiments.factor_zoo import (  # noqa: E402
    sparse_autoencoder_latent_factors,
    unexplained_alpha_fraction,
    variance_normalize,
)


TEST_PORTFOLIO_DIR = ROOT / "data" / "test_portfolios"
OUTPUT_DIR = BEN_FACTOR_ZOO.output_dir / "simple_ae_test_portfolios"


def main() -> int:
    parser = argparse.ArgumentParser(description="Train single-hidden-layer AE on FGX zoo; evaluate on Train/FF/HXZ panels.")
    parser.add_argument(
        "--latent-factors",
        type=int,
        default=None,
        help="Bottleneck size (defaults to BEN_FACTOR_ZOO.latent_factors)",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Training epochs (defaults to BEN_FACTOR_ZOO.autoencoder_epochs)",
    )
    args = parser.parse_args()

    n_latent = args.latent_factors if args.latent_factors is not None else BEN_FACTOR_ZOO.latent_factors
    epochs = args.epochs if args.epochs is not None else BEN_FACTOR_ZOO.autoencoder_epochs

    missing = [
        path
        for path in (
            BEN_FACTOR_ZOO.factors_path,
            TEST_PORTFOLIO_DIR / "global_q" / "global_q_1way_monthly_low_high.csv",
            TEST_PORTFOLIO_DIR / "ff" / "25_Portfolios_5x5.csv",
            TEST_PORTFOLIO_DIR / "ff" / "25_Portfolios_ME_OP_5x5.csv",
            TEST_PORTFOLIO_DIR / "ff" / "25_Portfolios_ME_INV_5x5.csv",
        )
        if not path.exists()
    ]
    if missing:
        print("Simple AE test-portfolio data are not ready.")
        for path in missing:
            print(f"- Missing {path}")
        return 2

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    factors = read_fgx_factors_csv(BEN_FACTOR_ZOO.factors_path).set_index("date")
    risk_free = read_fgx_risk_free_rate(BEN_FACTOR_ZOO.factors_path)
    factors = _restrict(factors, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)
    risk_free = _restrict(risk_free.to_frame(), BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)["RF"]
    if "MktRf" not in factors.columns:
        raise RuntimeError("MktRf column missing from FGX factors file; required for the 'with-market' variant.")
    market = factors["MktRf"].astype(float)
    normalized = variance_normalize(factors)

    ae_models: dict[str, pd.DataFrame] = {}
    diagnostics_by_model: dict[str, dict[str, float]] = {}
    for activation in BEN_FACTOR_ZOO.simple_ae_activation_grid:
        for h in BEN_FACTOR_ZOO.simple_ae_single_hidden_sizes:
            prefix = f"ae_{activation}_h{h}"
            latent, diag = sparse_autoencoder_latent_factors(
                normalized,
                n_latent,
                (h,),
                activation,
                0.0,
                epochs,
                BEN_FACTOR_ZOO.autoencoder_learning_rate,
                BEN_FACTOR_ZOO.autoencoder_seed,
                prefix,
            )
            model_name = f"simple_ae_{activation}_h{h}_k{n_latent}"
            ae_models[model_name] = latent
            diagnostics_by_model[model_name] = {
                "reconstruction_mse": diag.reconstruction_mse,
                "encoder_weight_sparsity": diag.encoder_weight_sparsity,
            }
            latent.to_csv(OUTPUT_DIR / f"{model_name}_latent_factors.csv")

    test_panels: dict[str, pd.DataFrame] = {
        name: _to_excess_returns(frame, risk_free) for name, frame in _load_test_datasets().items()
    }
    test_panels["train_fgx_zoo"] = factors.drop(columns=[c for c in ("MktRf",) if c in factors.columns])

    coverage_rows = [_coverage_summary(name, frame, normalized.index) for name, frame in test_panels.items()]
    pd.DataFrame(coverage_rows).to_csv(OUTPUT_DIR / "test_portfolio_coverage.csv", index=False)

    summary_rows: list[dict[str, object]] = []
    for dataset_name, test_assets in test_panels.items():
        test_assets = _restrict(test_assets, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)
        for model_name, latent in ae_models.items():
            overlap = test_assets.index.intersection(latent.index)
            diag = diagnostics_by_model[model_name]
            if len(overlap) <= n_latent + 1:
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "family": "simple_autoencoder",
                        "latent_factors": n_latent,
                        "model": model_name,
                        "explained_fraction": np.nan,
                        "explained_fraction_with_market": np.nan,
                        "test_assets": int(test_assets.shape[1]),
                        "months_used": int(len(overlap)),
                        "status": "skipped_insufficient_overlap",
                        **diag,
                    }
                )
                continue
            result = unexplained_alpha_fraction(test_assets, latent, model_name, market=market)
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "family": "simple_autoencoder",
                    "latent_factors": n_latent,
                    **result.__dict__,
                    "months_used": int(len(overlap)),
                    "status": "ok",
                    **diag,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "simple_ae_test_portfolios_summary.csv", index=False)

    manifest = {
        "config": to_jsonable(BEN_FACTOR_ZOO),
        "experiment": {
            "latent_factors": n_latent,
            "epochs": epochs,
            "l1_penalty": 0.0,
            "architecture": "single_linear_hidden_between_input_and_latent_each_side",
        },
        "factor_input_shape": list(factors.shape),
        "test_portfolios_are_excess_returns": True,
        "risk_free_source": str(BEN_FACTOR_ZOO.factors_path),
        "summary_path": str(OUTPUT_DIR / "simple_ae_test_portfolios_summary.csv"),
        "coverage_path": str(OUTPUT_DIR / "test_portfolio_coverage.csv"),
        "datasets": coverage_rows,
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote simple AE test-portfolio results to {OUTPUT_DIR}")
    return 0


def _load_test_datasets() -> dict[str, pd.DataFrame]:
    return {
        "global_q_low_high": _read_wide_monthly(
            TEST_PORTFOLIO_DIR / "global_q" / "global_q_1way_monthly_low_high.csv"
        ),
        "fama_french_75": _load_fama_french_75(),
    }


def _read_wide_monthly(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "date" not in df.columns and "Date" in df.columns:
        df = df.rename(columns={"Date": "date"})
    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    df = df.assign(date=parsed_dates.dt.to_period("M").dt.to_timestamp("M"))
    value_cols = [col for col in df.columns if col != "date"]
    df[value_cols] = df[value_cols].apply(pd.to_numeric, errors="coerce")
    return df.sort_values("date").drop_duplicates("date").set_index("date")


def _to_excess_returns(frame: pd.DataFrame, risk_free: pd.Series) -> pd.DataFrame:
    return frame.sub(risk_free, axis=0)


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


def _read_french_25_value_weighted_monthly(path: Path) -> pd.DataFrame:
    lines = path.read_text(encoding="utf-8-sig").splitlines()
    marker_idx = next(i for i, line in enumerate(lines) if "Average Value Weighted Returns -- Monthly" in line)
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


def _coverage_summary(name: str, frame: pd.DataFrame, factor_index: pd.Index) -> dict[str, object]:
    restricted = _restrict(frame, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)
    overlap = restricted.index.intersection(factor_index)
    return {
        "dataset": name,
        "date_min": None if frame.empty else frame.index.min(),
        "date_max": None if frame.empty else frame.index.max(),
        "assets": int(frame.shape[1]),
        "months_in_sample_window": int(len(restricted.index)),
        "overlap_months": int(len(overlap)),
        "missing_values_in_sample": int(restricted.isna().sum().sum()) if not restricted.empty else 0,
    }


def _restrict(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


if __name__ == "__main__":
    raise SystemExit(main())
