from __future__ import annotations

import argparse
import os
from pathlib import Path

import pandas as pd
import wrds


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
GKX_DIR = DATA_DIR / "gkx"
DEFAULT_START = "1957-03-01"
DEFAULT_END = "2016-12-31"


def main() -> None:
    args = _parse_args()
    GKX_DIR.mkdir(parents=True, exist_ok=True)

    username = args.username or os.environ.get("WRDS_USERNAME")
    db = wrds.Connection(wrds_username=username) if username else wrds.Connection()
    try:
        monthly = _query_crsp_monthly(db, args.start, args.end, args.common_shares_only)
        delistings = _query_crsp_delistings(db, args.start, args.end)
    finally:
        db.close()

    panel = _prepare_returns(monthly, delistings)
    panel = _merge_risk_free_rate(panel, args.ff_factors_path)
    panel = _finalize_panel(panel)

    out_path = GKX_DIR / "monthly_stock_returns.csv"
    panel.to_csv(out_path, index=False)

    print(f"Wrote GKX CRSP monthly returns: {out_path}")
    print(f"Rows: {len(panel):,}")
    print(f"Stocks: {panel['permno'].nunique():,}")
    print(f"Date range: {panel['date'].min()} to {panel['date'].max()}")
    print(f"Missing ret_excess: {panel['ret_excess'].isna().sum():,}")
    print(f"Common shares only: {args.common_shares_only}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Pull CRSP monthly stock returns from WRDS for GKX-style experiments.",
    )
    parser.add_argument("--username", help="WRDS username. Defaults to WRDS_USERNAME or interactive prompt.")
    parser.add_argument("--start", default=DEFAULT_START, help=f"Start date, default {DEFAULT_START}.")
    parser.add_argument("--end", default=DEFAULT_END, help=f"End date, default {DEFAULT_END}.")
    parser.add_argument(
        "--ff-factors-path",
        type=Path,
        default=DATA_DIR / "fama_french_3_factors.csv",
        help="Local Fama-French monthly file containing RF in percent.",
    )
    parser.add_argument(
        "--include-non-common-shares",
        action="store_true",
        help="Include all share codes. By default, keep shrcd 10 and 11 common shares.",
    )
    args = parser.parse_args()
    args.common_shares_only = not args.include_non_common_shares
    return args


def _query_crsp_monthly(
    db: wrds.Connection,
    start: str,
    end: str,
    common_shares_only: bool,
) -> pd.DataFrame:
    share_filter = "and n.shrcd in (10, 11)" if common_shares_only else ""
    query = f"""
        select
            m.permno,
            m.date,
            m.ret,
            m.retx,
            m.prc,
            m.shrout,
            n.exchcd,
            n.shrcd,
            n.siccd
        from crsp.msf as m
        left join crsp.msenames as n
          on m.permno = n.permno
         and n.namedt <= m.date
         and m.date <= n.nameendt
        where m.date between '{start}' and '{end}'
          and n.exchcd in (1, 2, 3)
          {share_filter}
    """
    return db.raw_sql(query, date_cols=["date"])


def _query_crsp_delistings(db: wrds.Connection, start: str, end: str) -> pd.DataFrame:
    query = f"""
        select
            permno,
            dlstdt,
            dlret,
            dlstcd
        from crsp.msedelist
        where dlstdt between '{start}' and '{end}'
    """
    return db.raw_sql(query, date_cols=["dlstdt"])


def _prepare_returns(monthly: pd.DataFrame, delistings: pd.DataFrame) -> pd.DataFrame:
    monthly = monthly.copy()
    monthly["date"] = pd.to_datetime(monthly["date"]) + pd.offsets.MonthEnd(0)

    delistings = delistings.copy()
    delistings["date"] = pd.to_datetime(delistings["dlstdt"]) + pd.offsets.MonthEnd(0)
    delistings = delistings.sort_values(["permno", "dlstdt"]).drop_duplicates(
        subset=["permno", "date"],
        keep="last",
    )

    panel = monthly.merge(
        delistings[["permno", "date", "dlret", "dlstcd"]],
        on=["permno", "date"],
        how="left",
        validate="many_to_one",
    )

    for col in ("ret", "retx", "dlret", "prc", "shrout"):
        panel[col] = pd.to_numeric(panel[col], errors="coerce")

    ret = panel["ret"]
    dlret = panel["dlret"]
    has_return = ret.notna() | dlret.notna()
    panel["ret_total"] = (1.0 + ret.fillna(0.0)) * (1.0 + dlret.fillna(0.0)) - 1.0
    panel.loc[~has_return, "ret_total"] = pd.NA
    panel["me"] = panel["prc"].abs() * panel["shrout"]
    return panel


def _merge_risk_free_rate(panel: pd.DataFrame, ff_factors_path: Path) -> pd.DataFrame:
    if not ff_factors_path.exists():
        raise FileNotFoundError(f"Missing Fama-French file for RF: {ff_factors_path}")

    ff = pd.read_csv(ff_factors_path, usecols=["date", "RF"])
    ff["date"] = pd.to_datetime(ff["date"]) + pd.offsets.MonthEnd(0)
    ff["rf"] = pd.to_numeric(ff["RF"], errors="coerce") / 100.0

    out = panel.merge(ff[["date", "rf"]], on="date", how="left", validate="many_to_one")
    out["ret_excess"] = out["ret_total"] - out["rf"]
    return out


def _finalize_panel(panel: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "permno",
        "date",
        "ret",
        "retx",
        "dlret",
        "ret_total",
        "rf",
        "ret_excess",
        "prc",
        "shrout",
        "me",
        "exchcd",
        "shrcd",
        "siccd",
        "dlstcd",
    ]
    out = panel[columns].sort_values(["date", "permno"]).reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


if __name__ == "__main__":
    main()
