from __future__ import annotations

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
    mispricing_penalized_linear_autoencoder_latent_factors,
    pca_latent_factors,
    unexplained_alpha_fraction,
    variance_normalize,
)


TEST_PORTFOLIO_DIR = ROOT / "data" / "test_portfolios"
OUTPUT_DIR = BEN_FACTOR_ZOO.output_dir / "pca_test_portfolios"


def main() -> int:
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
        print("PCA test-portfolio data are not ready.")
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

    latent_models = {
        k: pca_latent_factors(normalized, k, f"linear_k{k}")
        for k in BEN_FACTOR_ZOO.latent_factor_grid
    }
    for k, latent in latent_models.items():
        latent.to_csv(OUTPUT_DIR / f"linear_pca_proxy_k{k}_latent_factors.csv")
    linear_ae_latent, linear_ae_diagnostics = mispricing_penalized_linear_autoencoder_latent_factors(
        normalized,
        BEN_FACTOR_ZOO.latent_factors,
        BEN_FACTOR_ZOO.mispricing_gamma,
        BEN_FACTOR_ZOO.autoencoder_epochs,
        BEN_FACTOR_ZOO.autoencoder_learning_rate,
        BEN_FACTOR_ZOO.autoencoder_seed,
        f"linear_ae_k{BEN_FACTOR_ZOO.latent_factors}",
    )
    linear_ae_model_name = f"linear_ae_mispricing_k{BEN_FACTOR_ZOO.latent_factors}"
    linear_ae_latent.to_csv(OUTPUT_DIR / f"{linear_ae_model_name}_latent_factors.csv")

    test_panels: dict[str, pd.DataFrame] = {
        name: _to_excess_returns(frame, risk_free) for name, frame in _load_test_datasets().items()
    }
    test_panels["train_fgx_zoo"] = factors.drop(columns=[c for c in ("MktRf",) if c in factors.columns])

    coverage_rows = [_coverage_summary(name, frame, normalized.index) for name, frame in test_panels.items()]
    coverage = pd.DataFrame(coverage_rows)
    coverage.to_csv(OUTPUT_DIR / "test_portfolio_coverage.csv", index=False)

    summary_rows = []
    for dataset_name, test_assets in test_panels.items():
        test_assets = _restrict(test_assets, BEN_FACTOR_ZOO.sample.start, BEN_FACTOR_ZOO.sample.end)
        for k, latent in latent_models.items():
            overlap = test_assets.index.intersection(latent.index)
            if len(overlap) <= k + 1:
                summary_rows.append(
                    {
                        "dataset": dataset_name,
                        "family": "linear_pca_proxy",
                        "latent_factors": k,
                        "model": f"linear_pca_proxy_k{k}",
                        "explained_fraction": np.nan,
                        "explained_fraction_with_market": np.nan,
                        "test_assets": int(test_assets.shape[1]),
                        "months_used": int(len(overlap)),
                        "status": "skipped_insufficient_overlap",
                    }
                )
                continue

            result = unexplained_alpha_fraction(
                test_assets, latent, f"linear_pca_proxy_k{k}", market=market
            )
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "family": "linear_pca_proxy",
                    "latent_factors": k,
                    **result.__dict__,
                    "months_used": int(len(overlap)),
                    "status": "ok",
                }
            )
        overlap = test_assets.index.intersection(linear_ae_latent.index)
        if len(overlap) <= BEN_FACTOR_ZOO.latent_factors + 1:
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "family": "linear_ae_mispricing",
                    "latent_factors": BEN_FACTOR_ZOO.latent_factors,
                    "model": linear_ae_model_name,
                    "explained_fraction": np.nan,
                    "explained_fraction_with_market": np.nan,
                    "test_assets": int(test_assets.shape[1]),
                    "months_used": int(len(overlap)),
                    "status": "skipped_insufficient_overlap",
                    **linear_ae_diagnostics.__dict__,
                }
            )
        else:
            result = unexplained_alpha_fraction(
                test_assets, linear_ae_latent, linear_ae_model_name, market=market
            )
            summary_rows.append(
                {
                    "dataset": dataset_name,
                    "family": "linear_ae_mispricing",
                    "latent_factors": BEN_FACTOR_ZOO.latent_factors,
                    **result.__dict__,
                    "months_used": int(len(overlap)),
                    "status": "ok",
                    **linear_ae_diagnostics.__dict__,
                }
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT_DIR / "pca_test_portfolios_summary.csv", index=False)
    comparison = _chimeras_linear6_comparison(summary)
    comparison.to_csv(OUTPUT_DIR / "chimeras_linear6_simulated_comparison.csv", index=False)

    manifest = {
        "config": to_jsonable(BEN_FACTOR_ZOO),
        "factor_input_shape": list(factors.shape),
        "test_portfolios_are_excess_returns": True,
        "risk_free_source": str(BEN_FACTOR_ZOO.factors_path),
        "linear_ae_mispricing_gamma": BEN_FACTOR_ZOO.mispricing_gamma,
        "datasets": coverage_rows,
        "summary_path": str(OUTPUT_DIR / "pca_test_portfolios_summary.csv"),
        "coverage_path": str(OUTPUT_DIR / "test_portfolio_coverage.csv"),
        "chimeras_linear6_comparison_path": str(OUTPUT_DIR / "chimeras_linear6_simulated_comparison.csv"),
    }
    (OUTPUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Wrote PCA test-portfolio results to {OUTPUT_DIR}")
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


def _chimeras_linear6_comparison(summary: pd.DataFrame) -> pd.DataFrame:
    """Compare our K=6 candidate "Linear 6" models with the quoted Chimeras values.

    Each Chimeras table cell is "X (Y)" where X is the explained fraction without a
    market factor and Y is the explained fraction once Mkt-Rf is appended to the
    latent factors. We emit one row per (dataset x candidate model).
    """

    paper_reference = {
        "train_fgx_zoo": {
            "paper_dataset": "Train",
            "paper_no_market_percent": 83,
            "paper_with_market_percent": 81,
        },
        "fama_french_75": {
            "paper_dataset": "FF",
            "paper_no_market_percent": 4,
            "paper_with_market_percent": 48,
        },
        "global_q_low_high": {
            "paper_dataset": "HXZ",
            "paper_no_market_percent": 6,
            "paper_with_market_percent": 82,
        },
    }
    candidates = {
        "pca_k6_baseline": (summary["family"] == "linear_pca_proxy") & (summary["latent_factors"] == 6),
        "mispricing_ae_k6": (summary["family"] == "linear_ae_mispricing")
        & (summary["latent_factors"] == 6),
    }

    rows = []
    for candidate_name, mask in candidates.items():
        observed = summary.loc[
            mask,
            [
                "dataset",
                "model",
                "explained_fraction",
                "explained_fraction_with_market",
                "test_assets",
                "months_used",
                "status",
            ],
        ].copy()
        for _, row in observed.iterrows():
            ref = paper_reference.get(row["dataset"])
            if ref is None:
                continue
            our_no = row["explained_fraction"]
            our_with = row["explained_fraction_with_market"]
            rows.append(
                {
                    "dataset": row["dataset"],
                    "paper_dataset": ref["paper_dataset"],
                    "candidate": candidate_name,
                    "our_model": row["model"],
                    "our_no_market_percent": (float(our_no) * 100.0) if pd.notna(our_no) else np.nan,
                    "our_with_market_percent": (float(our_with) * 100.0)
                    if pd.notna(our_with)
                    else np.nan,
                    "paper_no_market_percent": ref["paper_no_market_percent"],
                    "paper_with_market_percent": ref["paper_with_market_percent"],
                    "test_assets": row["test_assets"],
                    "months_used": row["months_used"],
                    "status": row["status"],
                }
            )

    rows.append(
        {
            "dataset": "mutual_funds",
            "paper_dataset": "MF",
            "candidate": "paper_only",
            "our_model": None,
            "our_no_market_percent": np.nan,
            "our_with_market_percent": np.nan,
            "paper_no_market_percent": 25,
            "paper_with_market_percent": 68,
            "test_assets": np.nan,
            "months_used": np.nan,
            "status": "paper_reference_only_not_run",
        }
    )
    return pd.DataFrame(rows)


def _restrict(df: pd.DataFrame, start: str, end: str) -> pd.DataFrame:
    return df.loc[(df.index >= pd.Timestamp(start)) & (df.index <= pd.Timestamp(end))]


if __name__ == "__main__":
    raise SystemExit(main())
