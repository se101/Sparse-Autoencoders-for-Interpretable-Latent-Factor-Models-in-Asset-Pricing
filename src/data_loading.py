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
    benchmarks: pd.DataFrame


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


def load_project_data(
    factors_path: Path | None = None,
    benchmarks_path: Path | None = None,
) -> ProjectData:
    factors_path = factors_path or (DATA_DIR / "factors.csv")
    benchmarks_path = benchmarks_path or (DATA_DIR / "openassetpricing_sorted_portfolio_returns.csv")

    if not factors_path.exists():
        raise FileNotFoundError(f"Missing factors file: {factors_path}")
    if not benchmarks_path.exists():
        raise FileNotFoundError(f"Missing benchmark file: {benchmarks_path}")

    factors = _normalize_table(_read_table(factors_path), "factors")
    benchmarks = _normalize_table(_read_table(benchmarks_path), "benchmarks")

    return ProjectData(factors=factors, benchmarks=benchmarks)


def summarize_project_data(data: ProjectData) -> dict:
    overlap = pd.merge(
        data.factors[["date"]],
        data.benchmarks[["date"]],
        on="date",
        how="inner",
    )

    return {
        "factors_shape": data.factors.shape,
        "benchmarks_shape": data.benchmarks.shape,
        "factors_date_min": data.factors["date"].min(),
        "factors_date_max": data.factors["date"].max(),
        "benchmarks_date_min": data.benchmarks["date"].min(),
        "benchmarks_date_max": data.benchmarks["date"].max(),
        "overlap_months": len(overlap),
        "factors_missing": int(data.factors.isna().sum().sum()),
        "benchmarks_missing": int(data.benchmarks.isna().sum().sum()),
    }
