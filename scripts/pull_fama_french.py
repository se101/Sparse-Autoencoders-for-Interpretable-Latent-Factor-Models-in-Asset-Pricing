from __future__ import annotations

from io import BytesIO, StringIO
from pathlib import Path
from urllib.request import urlopen
from zipfile import ZipFile

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

DATASETS = {
    "ff3": {
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_Factors_CSV.zip",
        "raw_name": "ken_french_ff3_csv.zip",
        "out_name": "fama_french_3_factors.csv",
    },
    "ff5": {
        "url": "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/F-F_Research_Data_5_Factors_2x3_CSV.zip",
        "raw_name": "ken_french_ff5_csv.zip",
        "out_name": "fama_french_5_factors.csv",
    },
}


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    RAW_DIR.mkdir(exist_ok=True)

    for name, spec in DATASETS.items():
        raw_path = RAW_DIR / spec["raw_name"]
        out_path = DATA_DIR / spec["out_name"]

        archive = _download(spec["url"])
        raw_path.write_bytes(archive)

        df = _parse_monthly_csv_zip(archive)
        df.to_csv(out_path, index=False)

        print(f"Wrote raw {name.upper()} archive: {raw_path}")
        print(f"Wrote clean {name.upper()} monthly factors: {out_path}")
        print(f"{name.upper()} shape: {df.shape}")
        print(f"{name.upper()} date range: {df['date'].min()} to {df['date'].max()}")


def _download(url: str) -> bytes:
    with urlopen(url) as response:
        return response.read()


def _parse_monthly_csv_zip(archive: bytes) -> pd.DataFrame:
    with ZipFile(BytesIO(archive)) as zip_file:
        csv_names = [name for name in zip_file.namelist() if name.lower().endswith(".csv")]
        if len(csv_names) != 1:
            raise ValueError(f"Expected one CSV in archive, found: {csv_names}")
        text = zip_file.read(csv_names[0]).decode("utf-8")

    lines = text.splitlines()
    header_idx = next(i for i, line in enumerate(lines) if line.startswith(","))
    end_idx = next(
        i
        for i in range(header_idx + 1, len(lines))
        if not lines[i].strip() or not lines[i].split(",", 1)[0].strip().isdigit()
    )

    monthly_text = "\n".join(lines[header_idx:end_idx])
    df = pd.read_csv(StringIO(monthly_text))
    df = df.rename(columns={df.columns[0]: "date"})
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m") + pd.offsets.MonthEnd(0)

    factor_cols = [col for col in df.columns if col != "date"]
    df[factor_cols] = df[factor_cols].apply(pd.to_numeric, errors="coerce")
    return df


if __name__ == "__main__":
    main()
