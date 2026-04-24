from __future__ import annotations

import pandas as pd


def add_frequency_encoding(
    train_df: pd.DataFrame, test_df: pd.DataFrame, columns: list[str]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    te = test_df.copy()
    for col in columns:
        freq = tr[col].value_counts(normalize=True)
        tr[f"{col}_freq"] = tr[col].map(freq).fillna(0.0)
        te[f"{col}_freq"] = te[col].map(freq).fillna(0.0)
    return tr, te


def add_target_encoding(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    columns: list[str],
    target_col: str,
    smoothing: int = 20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    tr = train_df.copy()
    te = test_df.copy()
    prior = tr[target_col].mean()
    for col in columns:
        stats = tr.groupby(col)[target_col].agg(["mean", "count"])
        smooth = (stats["count"] * stats["mean"] + smoothing * prior) / (stats["count"] + smoothing)
        tr[f"{col}_te"] = tr[col].map(smooth).fillna(prior)
        te[f"{col}_te"] = te[col].map(smooth).fillna(prior)
    return tr, te
