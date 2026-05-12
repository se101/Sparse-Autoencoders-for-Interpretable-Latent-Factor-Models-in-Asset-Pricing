from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Download or rebuild Open Asset Pricing panels.")
    parser.add_argument(
        "--from-raw",
        action="store_true",
        help="Rebuild model-ready panels from data/raw/openassetpricing_predictor_ports_full.csv without downloading.",
    )
    args = parser.parse_args(argv)

    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    raw_path = RAW_DIR / "openassetpricing_predictor_ports_full.csv"
    signal_doc_path = DATA_DIR / "openassetpricing_signal_metadata.csv"

    if args.from_raw:
        if not raw_path.exists():
            raise FileNotFoundError(f"Missing raw OpenAP file: {raw_path}")
        ports = pd.read_csv(raw_path)
        signal_doc = pd.read_csv(signal_doc_path) if signal_doc_path.exists() else None
    else:
        import openassetpricing as oap

        openap = oap.OpenAP()
        signal_doc = openap.dl_signal_doc("pandas")
        signal_doc.to_csv(signal_doc_path, index=False)
        ports = openap.dl_port("op", "pandas")
        ports.to_csv(raw_path, index=False)

    factors, benchmarks = derive_portfolio_panels(ports)

    factors_path = DATA_DIR / "factors.csv"
    factors.to_csv(factors_path, index=False)

    benchmarks_path = DATA_DIR / "openassetpricing_sorted_portfolio_returns.csv"
    benchmarks.to_csv(benchmarks_path, index=False)

    if signal_doc is not None:
        print(f"Wrote signal metadata: {signal_doc_path}")
        print(f"Signal metadata shape: {signal_doc.shape}")
    else:
        print(f"Skipped signal metadata refresh; no local metadata file found at {signal_doc_path}")
    print(f"Wrote raw portfolio data: {raw_path}")
    print(f"Wrote long-short factor panel: {factors_path}")
    print(f"Wrote benchmark portfolio panel: {benchmarks_path}")
    print(f"Raw shape: {ports.shape}")
    print(f"Factor shape: {factors.shape}")
    print(f"Benchmark shape: {benchmarks.shape}")
    print(f"Date range: {factors['date'].min()} to {factors['date'].max()}")


def derive_portfolio_panels(ports: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
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
    return factors, benchmarks


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
