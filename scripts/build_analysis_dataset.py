from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
ANALYSIS_DIR = DATA_DIR / "analysis"

MAX_MISSING_SHARE = 0.20


INPUTS = {
    "openap_factors": DATA_DIR / "factors.csv",
    "openap_sorted_portfolios": DATA_DIR / "openassetpricing_sorted_portfolio_returns.csv",
    "ff3": DATA_DIR / "fama_french_3_factors.csv",
    "ff5": DATA_DIR / "fama_french_5_factors.csv",
}


def main() -> None:
    ANALYSIS_DIR.mkdir(exist_ok=True)

    tables = {name: _read_monthly(path) for name, path in INPUTS.items()}
    common_months = sorted(set.intersection(*(set(df["month"]) for df in tables.values())))
    if not common_months:
        raise ValueError("No overlapping months across all input files.")

    aligned = {
        name: _restrict_to_months(df, common_months)
        for name, df in tables.items()
    }

    broad_factors, factor_missing = _filter_by_missingness(
        aligned["openap_factors"],
        MAX_MISSING_SHARE,
    )
    broad_sorted, sorted_missing = _filter_by_missingness(
        aligned["openap_sorted_portfolios"],
        MAX_MISSING_SHARE,
    )

    balanced_factors = _drop_columns_with_any_missing(aligned["openap_factors"])
    balanced_sorted = _drop_columns_with_any_missing(aligned["openap_sorted_portfolios"])

    outputs = {
        "openap_factors_80pct_available": broad_factors,
        "openap_sorted_portfolios_80pct_available": broad_sorted,
        "openap_factors_balanced": balanced_factors,
        "openap_sorted_portfolios_balanced": balanced_sorted,
        "fama_french_3_factors_aligned": aligned["ff3"],
        "fama_french_5_factors_aligned": aligned["ff5"],
    }

    for name, df in outputs.items():
        _write_analysis_csv(df, ANALYSIS_DIR / f"{name}.csv")

    missingness = pd.concat(
        [
            factor_missing.assign(dataset="openap_factors"),
            sorted_missing.assign(dataset="openap_sorted_portfolios"),
        ],
        ignore_index=True,
    )
    missingness.to_csv(ANALYSIS_DIR / "missingness_report.csv", index=False)

    summary = _build_summary(aligned, outputs, common_months)
    summary_path = ANALYSIS_DIR / "data_prep_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Wrote analysis data to: {ANALYSIS_DIR}")
    print(json.dumps(summary, indent=2))


def _read_monthly(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing input file: {path}")

    df = pd.read_csv(path)
    if "date" not in df.columns:
        raise ValueError(f"Expected a date column in {path}")

    df["date"] = pd.to_datetime(df["date"])
    df["month"] = df["date"].dt.to_period("M")
    df = df.sort_values("date").drop_duplicates(subset=["month"], keep="last")
    return df


def _restrict_to_months(df: pd.DataFrame, months: list[pd.Period]) -> pd.DataFrame:
    out = df[df["month"].isin(months)].copy()
    out = out.sort_values("month").reset_index(drop=True)
    out["date"] = out["month"].dt.to_timestamp("M")
    return out.drop(columns=["month"])


def _filter_by_missingness(
    df: pd.DataFrame,
    max_missing_share: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    value_cols = [col for col in df.columns if col != "date"]
    missing = (
        df[value_cols]
        .isna()
        .mean()
        .rename("missing_share")
        .reset_index()
        .rename(columns={"index": "column"})
        .sort_values(["missing_share", "column"])
        .reset_index(drop=True)
    )
    keep_cols = missing.loc[missing["missing_share"] <= max_missing_share, "column"].tolist()
    return df[["date", *keep_cols]].copy(), missing


def _drop_columns_with_any_missing(df: pd.DataFrame) -> pd.DataFrame:
    value_cols = [col for col in df.columns if col != "date"]
    keep_cols = [col for col in value_cols if not df[col].isna().any()]
    return df[["date", *keep_cols]].copy()


def _write_analysis_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def _build_summary(
    aligned: dict[str, pd.DataFrame],
    outputs: dict[str, pd.DataFrame],
    common_months: list[pd.Period],
) -> dict:
    summary = {
        "common_sample_start": str(common_months[0].to_timestamp("M").date()),
        "common_sample_end": str(common_months[-1].to_timestamp("M").date()),
        "common_months": len(common_months),
        "max_missing_share_for_80pct_available_files": MAX_MISSING_SHARE,
        "inputs": {},
        "outputs": {},
    }

    for name, df in aligned.items():
        summary["inputs"][name] = _frame_summary(df)

    for name, df in outputs.items():
        summary["outputs"][name] = _frame_summary(df)

    return summary


def _frame_summary(df: pd.DataFrame) -> dict:
    value_cols = [col for col in df.columns if col != "date"]
    return {
        "shape": list(df.shape),
        "date_min": str(pd.to_datetime(df["date"]).min().date()),
        "date_max": str(pd.to_datetime(df["date"]).max().date()),
        "value_columns": len(value_cols),
        "missing_values": int(df[value_cols].isna().sum().sum()),
    }


if __name__ == "__main__":
    main()
