from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = ROOT / "data" / "gkx" / "monthly_characteristics.csv"
DEFAULT_RETURNS = ROOT / "data" / "gkx" / "monthly_stock_returns.csv"

IDENTIFIER_COLUMNS = {
    "gvkey",
    "permno",
    "permco",
    "datadate",
    "date",
    "year",
    "month",
    "sic",
    "siccd",
    "ffi49",
    "exchcd",
    "shrcd",
    "ticker",
    "cusip",
    "ncusip",
}
RETURN_COLUMNS = {
    "ret",
    "retx",
    "ret_excess",
    "ret_total",
    "rf",
    "dlret",
    "dlstcd",
}
RAW_SCALE_COLUMNS = {
    "me",
    "prc",
    "shrout",
    "vol",
    "volume",
}


def main() -> None:
    args = _parse_args()
    df = _read_table(args.input)
    df = _normalize_keys(df)

    characteristic_columns = _select_characteristics(df, args.prefer_rank_columns)
    if not characteristic_columns:
        raise ValueError(f"No characteristic columns found in {args.input}")

    out = df[["permno", "date", *characteristic_columns]].copy()
    out = out.drop_duplicates(subset=["permno", "date"], keep="last")

    if args.align_to_returns:
        out = _align_to_returns(out, args.returns_path)

    out = out.sort_values(["date", "permno"]).reset_index(drop=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.output, index=False)

    print(f"Wrote GKX monthly characteristics: {args.output}")
    print(f"Rows: {len(out):,}")
    print(f"Stocks: {out['permno'].nunique():,}")
    print(f"Date range: {out['date'].min()} to {out['date'].max()}")
    print(f"Characteristics: {len(characteristic_columns)}")
    print("First characteristics:")
    for col in characteristic_columns[:20]:
        print(f"- {col}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert EquityCharacteristics output into data/gkx/monthly_characteristics.csv.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="EquityCharacteristics output, e.g. chars_rank_imputed.feather/csv/pkl.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path, default {DEFAULT_OUTPUT}.",
    )
    parser.add_argument(
        "--returns-path",
        type=Path,
        default=DEFAULT_RETURNS,
        help=f"GKX returns CSV used for key alignment, default {DEFAULT_RETURNS}.",
    )
    parser.add_argument(
        "--no-align-to-returns",
        dest="align_to_returns",
        action="store_false",
        help="Do not inner-join to the available monthly_stock_returns permno/date keys.",
    )
    parser.add_argument(
        "--include-all-numeric",
        dest="prefer_rank_columns",
        action="store_false",
        help="Use all numeric non-metadata columns instead of preferring rank_* columns.",
    )
    parser.set_defaults(align_to_returns=True, prefer_rank_columns=True)
    return parser.parse_args()


def _read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix in {".feather", ".ftr"}:
        return pd.read_feather(path)
    if suffix in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    if suffix in {".pkl", ".pickle"}:
        return pd.read_pickle(path)
    raise ValueError(f"Unsupported input file type: {path}")


def _normalize_keys(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(col) for col in out.columns]

    lower_to_original = {col.lower(): col for col in out.columns}
    if "permno" not in lower_to_original:
        raise ValueError("EquityCharacteristics output must include a permno column.")
    if "date" not in lower_to_original:
        raise ValueError("EquityCharacteristics output must include a date column aligned to return month.")

    out = out.rename(
        columns={
            lower_to_original["permno"]: "permno",
            lower_to_original["date"]: "date",
        }
    )
    out["permno"] = pd.to_numeric(out["permno"], errors="coerce").astype("Int64")
    out["date"] = pd.to_datetime(out["date"]) + pd.offsets.MonthEnd(0)
    out = out.dropna(subset=["permno", "date"])
    out["permno"] = out["permno"].astype(int)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def _select_characteristics(df: pd.DataFrame, prefer_rank_columns: bool) -> list[str]:
    rank_columns = [
        col
        for col in df.columns
        if col.lower().startswith("rank_") and pd.api.types.is_numeric_dtype(df[col])
    ]
    if prefer_rank_columns and rank_columns:
        return sorted(rank_columns)

    characteristic_columns = []
    for col in df.columns:
        lowered = col.lower()
        if lowered in IDENTIFIER_COLUMNS or lowered in RETURN_COLUMNS or lowered in RAW_SCALE_COLUMNS:
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            characteristic_columns.append(col)
    return sorted(characteristic_columns)


def _align_to_returns(chars: pd.DataFrame, returns_path: Path) -> pd.DataFrame:
    if not returns_path.exists():
        raise FileNotFoundError(f"Missing returns file for alignment: {returns_path}")
    keys = pd.read_csv(returns_path, usecols=["permno", "date"])
    keys["permno"] = pd.to_numeric(keys["permno"], errors="coerce").astype("Int64")
    keys = keys.dropna(subset=["permno", "date"])
    keys["permno"] = keys["permno"].astype(int)
    keys["date"] = pd.to_datetime(keys["date"]).dt.strftime("%Y-%m-%d")
    return chars.merge(keys.drop_duplicates(), on=["permno", "date"], how="inner", validate="one_to_one")


if __name__ == "__main__":
    main()
