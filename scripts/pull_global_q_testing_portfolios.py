from __future__ import annotations

from io import BytesIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GLOBAL_Q_DIR = DATA_DIR / "global_q"
RAW_DIR = DATA_DIR / "raw" / "global_q"

BASE_URL = "http://global-q.org/uploads/1/2/2/6/122679606"
CATEGORIES = {
    "mom": "momentum",
    "vvg": "value_versus_growth",
    "inv": "investment",
    "prof": "profitability",
    "intan": "intangibles",
    "fric": "frictions",
}


def main() -> None:
    GLOBAL_Q_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    panels: list[pd.DataFrame] = []
    for prefix, category in CATEGORIES.items():
        url = f"{BASE_URL}/{prefix}_monthly_2024.zip"
        raw_zip_path = RAW_DIR / f"{prefix}_monthly_2024.zip"
        print(f"Downloading {url}")
        archive_bytes = _download(url)
        raw_zip_path.write_bytes(archive_bytes)
        panels.extend(_parse_archive(archive_bytes, category))

    long_panel = pd.concat(panels, ignore_index=True)
    long_panel = long_panel.sort_values(["date", "category", "anomaly", "portfolio"]).reset_index(drop=True)

    long_path = GLOBAL_Q_DIR / "global_q_1way_monthly_long.csv"
    all_path = GLOBAL_Q_DIR / "global_q_1way_monthly_all_portfolios.csv"
    low_high_path = GLOBAL_Q_DIR / "global_q_1way_monthly_low_high.csv"
    metadata_path = GLOBAL_Q_DIR / "global_q_1way_monthly_metadata.csv"

    long_panel.to_csv(long_path, index=False)
    _build_all_portfolios(long_panel).to_csv(all_path, index=False)
    _build_low_high_portfolios(long_panel).to_csv(low_high_path, index=False)
    _build_metadata(long_panel).to_csv(metadata_path, index=False)

    print(f"Wrote long panel: {long_path}")
    print(f"Wrote all portfolio panel: {all_path}")
    print(f"Wrote low/high panel: {low_high_path}")
    print(f"Wrote metadata: {metadata_path}")
    print(f"Anomalies: {long_panel['anomaly_id'].nunique()}")
    print(f"Rows: {len(long_panel)}")
    print(f"Date range: {long_panel['date'].min()} to {long_panel['date'].max()}")


def _download(url: str) -> bytes:
    with urlopen(url, timeout=60) as response:
        return response.read()


def _parse_archive(archive_bytes: bytes, category: str) -> list[pd.DataFrame]:
    frames: list[pd.DataFrame] = []
    with ZipFile(BytesIO(archive_bytes)) as archive:
        for name in archive.namelist():
            if not name.lower().endswith(".csv"):
                continue
            anomaly = _anomaly_from_filename(name)
            with archive.open(name) as csv_file:
                df = pd.read_csv(csv_file)
            frames.append(_normalize_anomaly_frame(df, category, anomaly, name))
    return frames


def _normalize_anomaly_frame(
    df: pd.DataFrame,
    category: str,
    anomaly: str,
    source_file: str,
) -> pd.DataFrame:
    rank_cols = [col for col in df.columns if col.lower().startswith("rank_")]
    if len(rank_cols) != 1:
        raise ValueError(f"Expected one rank column in {source_file}, found {rank_cols}")
    rank_col = rank_cols[0]

    required = {"year", "month", rank_col, "nstocks", "ret_vw"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"{source_file} is missing columns: {sorted(missing)}")

    out = df[["year", "month", rank_col, "nstocks", "ret_vw"]].copy()
    out["date"] = pd.to_datetime(
        {
            "year": out["year"].astype(int),
            "month": out["month"].astype(int),
            "day": 1,
        }
    ) + pd.offsets.MonthEnd(0)
    out = out.rename(columns={rank_col: "portfolio"})
    out["category"] = category
    out["anomaly"] = anomaly
    out["anomaly_id"] = category + "__" + anomaly
    out["source_file"] = source_file
    out["portfolio"] = out["portfolio"].astype(int)
    out["nstocks"] = pd.to_numeric(out["nstocks"], errors="coerce")
    out["ret_vw"] = pd.to_numeric(out["ret_vw"], errors="coerce")
    return out[["date", "category", "anomaly", "anomaly_id", "portfolio", "nstocks", "ret_vw", "source_file"]]


def _anomaly_from_filename(path: str) -> str:
    stem = Path(path).stem
    if not stem.startswith("portf_") or not stem.endswith("_monthly_2024"):
        raise ValueError(f"Unexpected Global-q filename: {path}")
    return stem.removeprefix("portf_").removesuffix("_monthly_2024")


def _build_all_portfolios(long_panel: pd.DataFrame) -> pd.DataFrame:
    panel = long_panel.copy()
    panel["column"] = (
        panel["category"]
        + "__"
        + panel["anomaly"]
        + "__p"
        + panel["portfolio"].astype(str).str.zfill(2)
    )
    wide = panel.pivot_table(index="date", columns="column", values="ret_vw", aggfunc="first")
    return wide.sort_index().reset_index().rename_axis(columns=None)


def _build_low_high_portfolios(long_panel: pd.DataFrame) -> pd.DataFrame:
    endpoint_rows = []
    for (_, anomaly_id, date), group in long_panel.groupby(["category", "anomaly_id", "date"]):
        low_rank = group["portfolio"].min()
        high_rank = group["portfolio"].max()
        endpoint_rows.append(group.loc[group["portfolio"] == low_rank].assign(endpoint="low"))
        endpoint_rows.append(group.loc[group["portfolio"] == high_rank].assign(endpoint="high"))

    endpoints = pd.concat(endpoint_rows, ignore_index=True)
    endpoints["column"] = endpoints["anomaly_id"] + "__" + endpoints["endpoint"]
    wide = endpoints.pivot_table(index="date", columns="column", values="ret_vw", aggfunc="first")
    return wide.sort_index().reset_index().rename_axis(columns=None)


def _build_metadata(long_panel: pd.DataFrame) -> pd.DataFrame:
    grouped = long_panel.groupby(["category", "anomaly", "anomaly_id"])
    return (
        grouped.agg(
            date_min=("date", "min"),
            date_max=("date", "max"),
            portfolio_count=("portfolio", "nunique"),
            observations=("ret_vw", "count"),
            source_file=("source_file", "first"),
        )
        .reset_index()
        .sort_values(["category", "anomaly"])
    )


if __name__ == "__main__":
    main()
