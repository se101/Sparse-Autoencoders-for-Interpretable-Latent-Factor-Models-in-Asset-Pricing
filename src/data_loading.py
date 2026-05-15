from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"


@dataclass
class ProjectData:
    factors: pd.DataFrame
    test_portfolios: pd.DataFrame


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {path}")


def _find_date_column(columns: Iterable[str]) -> str:
    lowered = {c.lower(): c for c in columns}
    for name in ("date", "month", "yyyymm", "timestamp"):
        if name in lowered:
            return lowered[name]
    raise ValueError("Could not find a date column. Expected one of: date, month, yyyymm, timestamp.")


def _normalize_table(df: pd.DataFrame, table_name: str) -> pd.DataFrame:
    df = df.copy()
    date_col = _find_date_column(df.columns)
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.rename(columns={date_col: "date"})
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)

    if df.shape[1] < 2:
        raise ValueError(f"{table_name} must contain at least one non-date column.")

    return df


def read_fgx_factors_csv(path: Path) -> pd.DataFrame:
    """Load Feng–Giglio–Xiu (2020) Journal of Finance ``factors.csv``.

    Drops ``RF`` and scales factor returns from decimals to **percentage points**
    so they are comparable to monthly return panels often quoted in percent
    (e.g. Ken French-style CSVs and many test-portfolio files).
    """

    df = _read_table(path)
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "RF" not in df.columns:
        raise ValueError(f"Expected FGX factors file with an `RF` column: {path}")
    parsed_dates = pd.to_datetime(df["date"].astype(str), errors="coerce")
    df = df.assign(date=parsed_dates.dt.to_period("M").dt.to_timestamp("M"))
    df = df.sort_values("date").drop_duplicates(subset=["date"]).reset_index(drop=True)
    df = df.drop(columns=["RF"], errors="ignore")
    value_cols = [c for c in df.columns if c != "date"]
    if len(value_cols) != 150:
        raise ValueError(f"Expected 150 FGX factor columns in {path}, found {len(value_cols)}.")
    df[value_cols] = df[value_cols].astype(float) * 100.0
    return df


def read_fgx_risk_free_rate(path: Path) -> pd.Series:
    """Load the FGX replication ``RF`` column as monthly percent returns."""

    df = _read_table(path)
    if "Date" in df.columns and "date" not in df.columns:
        df = df.rename(columns={"Date": "date"})
    if "RF" not in df.columns:
        raise ValueError(f"Expected FGX factors file with an `RF` column: {path}")
    parsed_dates = pd.to_datetime(df["date"].astype(str), errors="coerce")
    out = df.assign(
        date=parsed_dates.dt.to_period("M").dt.to_timestamp("M"),
        RF=pd.to_numeric(df["RF"], errors="coerce") * 100.0,
    )
    out = out.sort_values("date").drop_duplicates(subset=["date"]).set_index("date")
    return out["RF"].rename("RF")


def load_project_data(
    factors_path: Path | None = None,
    test_portfolios_path: Path | None = None,
) -> ProjectData:
    factors_path = factors_path or (DATA_DIR / "factors.csv")
    test_portfolios_path = test_portfolios_path or (
        DATA_DIR / "test_portfolios" / "global_q" / "global_q_1way_monthly_low_high.csv"
    )

    if not factors_path.exists():
        raise FileNotFoundError(f"Missing factors file: {factors_path}")
    if not test_portfolios_path.exists():
        raise FileNotFoundError(f"Missing test portfolios file: {test_portfolios_path}")

    factors = read_fgx_factors_csv(factors_path)
    test_portfolios = _normalize_table(_read_table(test_portfolios_path), "test_portfolios")

    return ProjectData(factors=factors, test_portfolios=test_portfolios)


def summarize_project_data(data: ProjectData) -> dict:
    overlap = pd.merge(
        data.factors[["date"]],
        data.test_portfolios[["date"]],
        on="date",
        how="inner",
    )

    return {
        "factors_shape": data.factors.shape,
        "test_portfolios_shape": data.test_portfolios.shape,
        "factors_date_min": data.factors["date"].min(),
        "factors_date_max": data.factors["date"].max(),
        "test_portfolios_date_min": data.test_portfolios["date"].min(),
        "test_portfolios_date_max": data.test_portfolios["date"].max(),
        "overlap_months": len(overlap),
        "factors_missing": int(data.factors.isna().sum().sum()),
        "test_portfolios_missing": int(data.test_portfolios.isna().sum().sum()),
    }
