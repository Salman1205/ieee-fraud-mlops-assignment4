from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass
class ValidationReport:
    row_count: int
    col_count: int
    missing_ratio: dict[str, float]
    has_target: bool
    duplicate_rows: int


def validate_dataframe(df: pd.DataFrame, target_col: str = "isFraud") -> ValidationReport:
    missing_ratio = (df.isna().mean()).to_dict()
    return ValidationReport(
        row_count=len(df),
        col_count=df.shape[1],
        missing_ratio=missing_ratio,
        has_target=target_col in df.columns,
        duplicate_rows=int(df.duplicated().sum()),
    )


def validate_schema(df: pd.DataFrame, required_cols: list[str]) -> bool:
    return all(c in df.columns for c in required_cols)
