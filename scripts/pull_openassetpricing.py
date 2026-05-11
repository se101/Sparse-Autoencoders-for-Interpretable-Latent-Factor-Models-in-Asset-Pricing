from __future__ import annotations

from pathlib import Path

import openassetpricing as oap
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    openap = oap.OpenAP()

    signal_doc = openap.dl_signal_doc("pandas")
    signal_doc_path = DATA_DIR / "openassetpricing_signal_metadata.csv"
    signal_doc.to_csv(signal_doc_path, index=False)

    ports = openap.dl_port("op", "pandas")
    raw_path = RAW_DIR / "openassetpricing_predictor_ports_full.csv"
    ports.to_csv(raw_path, index=False)

    value_col = _find_return_column(ports)
    port_label = ports["port"].astype(str)
    is_long_short = port_label.str.lower().isin({"ls", "l-s", "long-short"})

    long_short = ports[is_long_short]
    if long_short.empty:
        available_ports = sorted(port_label.unique())
        raise ValueError(f"Could not find long-short portfolios. Available port values: {available_ports}")

    factors = (
        long_short.pivot_table(
            index="date",
            columns="signalname",
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
        .sort_values("date")
    )

    factors_path = DATA_DIR / "factors.csv"
    factors.to_csv(factors_path, index=False)

    benchmark_ports = ports[~is_long_short].assign(
        benchmark=lambda df: df["signalname"].astype(str) + "_p" + df["port"].astype(str)
    )
    benchmarks = (
        benchmark_ports.pivot_table(
            index="date",
            columns="benchmark",
            values=value_col,
            aggfunc="first",
        )
        .reset_index()
        .rename_axis(columns=None)
        .sort_values("date")
    )
    benchmarks_path = DATA_DIR / "openassetpricing_sorted_portfolio_returns.csv"
    benchmarks.to_csv(benchmarks_path, index=False)

    print(f"Wrote signal metadata: {signal_doc_path}")
    print(f"Wrote raw portfolio data: {raw_path}")
    print(f"Wrote long-short factor panel: {factors_path}")
    print(f"Wrote benchmark portfolio panel: {benchmarks_path}")
    print(f"Signal metadata shape: {signal_doc.shape}")
    print(f"Raw shape: {ports.shape}")
    print(f"Factor shape: {factors.shape}")
    print(f"Benchmark shape: {benchmarks.shape}")
    print(f"Date range: {factors['date'].min()} to {factors['date'].max()}")


def _find_return_column(df: pd.DataFrame) -> str:
    for candidate in ("ret", "return", "ret_vw", "ret_ew"):
        if candidate in df.columns:
            return candidate
    numeric_cols = [
        col
        for col in df.select_dtypes(include="number").columns
        if col.lower() not in {"permno", "yyyymm"}
    ]
    if len(numeric_cols) == 1:
        return numeric_cols[0]
    raise ValueError(f"Could not infer return column. Columns: {list(df.columns)}")


if __name__ == "__main__":
    main()
