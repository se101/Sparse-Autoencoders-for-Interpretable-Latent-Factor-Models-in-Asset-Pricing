from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from experiments.config import GKXExactConfig


ID_COLUMNS = ("permno", "date")
RETURN_CANDIDATES = ("ret_excess", "excess_return", "ret", "return")


@dataclass(frozen=True)
class ValidationIssue:
    severity: str
    message: str


@dataclass(frozen=True)
class GKXDataContract:
    returns_path: Path
    characteristics_path: Path
    id_column: str
    date_column: str
    return_column: str
    characteristic_columns: tuple[str, ...]


class DataContractError(ValueError):
    """Raised when a paper-specific dataset is missing or malformed."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = issues
        message = "\n".join(f"- {issue.severity}: {issue.message}" for issue in issues)
        super().__init__(message)


def validate_gkx_exact_data(config: GKXExactConfig) -> GKXDataContract:
    """Validate the exact GKX input files and return their inferred schema.

    Expected files:
    - monthly stock returns: one row per `permno` and month, with excess returns
      in `ret_excess` or a compatible return column name.
    - monthly characteristics: one row per `permno` and month, with 94 lagged,
      point-in-time characteristics already aligned to month t returns.
    """

    issues: list[ValidationIssue] = []
    if not config.returns_path.exists():
        issues.append(
            ValidationIssue(
                "error",
                f"Missing GKX returns file: {config.returns_path}. Expected columns include permno, date, ret_excess.",
            )
        )
    if not config.characteristics_path.exists():
        issues.append(
            ValidationIssue(
                "error",
                f"Missing GKX characteristics file: {config.characteristics_path}. Expected columns include permno, date, and 94 lagged characteristics.",
            )
        )
    if issues:
        raise DataContractError(issues)

    returns = pd.read_csv(config.returns_path, nrows=5)
    chars = pd.read_csv(config.characteristics_path, nrows=5)

    _require_columns(returns, config.returns_path, ID_COLUMNS, issues)
    _require_columns(chars, config.characteristics_path, ID_COLUMNS, issues)

    return_column = _first_existing(returns.columns, RETURN_CANDIDATES)
    if return_column is None:
        issues.append(
            ValidationIssue(
                "error",
                f"{config.returns_path} needs one of these return columns: {', '.join(RETURN_CANDIDATES)}.",
            )
        )

    characteristic_columns = tuple(col for col in chars.columns if col not in ID_COLUMNS)
    if len(characteristic_columns) != config.required_characteristics:
        issues.append(
            ValidationIssue(
                "warning" if characteristic_columns else "error",
                f"Expected {config.required_characteristics} GKX characteristics, found {len(characteristic_columns)}.",
            )
        )

    if any(issue.severity == "error" for issue in issues):
        raise DataContractError(issues)

    return GKXDataContract(
        returns_path=config.returns_path,
        characteristics_path=config.characteristics_path,
        id_column="permno",
        date_column="date",
        return_column=return_column or RETURN_CANDIDATES[0],
        characteristic_columns=characteristic_columns,
    )


def load_gkx_exact_panel(contract: GKXDataContract) -> pd.DataFrame:
    returns = pd.read_csv(contract.returns_path)
    chars = pd.read_csv(contract.characteristics_path)

    returns[contract.date_column] = pd.to_datetime(returns[contract.date_column])
    chars[contract.date_column] = pd.to_datetime(chars[contract.date_column])

    panel = returns.merge(
        chars,
        on=[contract.id_column, contract.date_column],
        how="inner",
        validate="one_to_one",
    )
    panel = panel.sort_values([contract.date_column, contract.id_column]).reset_index(drop=True)
    return panel


def rank_normalize_characteristics(
    panel: pd.DataFrame,
    characteristic_columns: tuple[str, ...],
    date_column: str = "date",
) -> pd.DataFrame:
    out = panel.copy()
    for col in characteristic_columns:
        monthly_median = out.groupby(date_column)[col].transform("median")
        out[col] = out[col].fillna(monthly_median)
        ranks = out.groupby(date_column)[col].rank(method="average", pct=True)
        out[col] = 2.0 * ranks - 1.0
    return out


def _first_existing(columns: pd.Index, candidates: tuple[str, ...]) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in candidates:
        if candidate.lower() in lowered:
            return lowered[candidate.lower()]
    return None


def _require_columns(
    df: pd.DataFrame,
    path: Path,
    required: tuple[str, ...],
    issues: list[ValidationIssue],
) -> None:
    lowered = {col.lower() for col in df.columns}
    for col in required:
        if col not in lowered:
            issues.append(ValidationIssue("error", f"{path} is missing required column `{col}`."))

