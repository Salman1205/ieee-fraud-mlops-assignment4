from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_ingestion import load_ieee_data
from src.preprocessing import preprocess_dataframe
from src.utils import dump_json, ensure_dir, load_yaml


def compute_numeric_drift(train_df: pd.DataFrame, test_df: pd.DataFrame, cols: list[str]) -> dict[str, float]:
    drift = {}
    for c in cols:
        drift[c] = float(wasserstein_distance(train_df[c], test_df[c]))
    return drift


def run_drift(config_path: str) -> None:
    cfg = load_yaml(config_path)
    target = cfg["data"]["target_col"]
    raw = load_ieee_data(cfg["data"]["raw_dir"], max_rows=cfg["data"].get("max_rows"))
    df, artifacts = preprocess_dataframe(raw, target_col=target)
    df = df.drop(columns=artifacts.categorical_cols, errors="ignore")

    sort_col = cfg["data"]["validation_split_date_col"]
    if sort_col in df.columns:
        df = df.sort_values(sort_col).reset_index(drop=True)
        cut = int(len(df) * 0.7)
        train_df = df.iloc[:cut].copy()
        later_df = df.iloc[cut:].copy()
    else:
        train_df, later_df = train_test_split(df, test_size=0.3, random_state=42, stratify=df[target])

    # Simulate new fraud pattern by shifting select numeric features.
    numeric_cols = [c for c in train_df.select_dtypes(include=[np.number]).columns if c != target][:5]
    for c in numeric_cols:
        later_df[c] = later_df[c] * 1.15

    X_train, y_train = train_df.drop(columns=[target]), train_df[target]
    X_later, y_later = later_df.drop(columns=[target]), later_df[target]
    model = XGBClassifier(n_estimators=180, max_depth=6, learning_rate=0.05)
    model.fit(X_train, y_train)
    recall_later = float((((model.predict(X_later) == 1) & (y_later == 1)).sum()) / max((y_later == 1).sum(), 1))

    drift_scores = compute_numeric_drift(train_df, later_df, numeric_cols)
    global_drift = float(np.mean(list(drift_scores.values()))) if drift_scores else 0.0
    out = {
        "global_drift_score": global_drift,
        "feature_drift_scores": drift_scores,
        "later_period_recall": recall_later,
        "drift_threshold": cfg["drift"]["threshold"],
        "drift_exceeds_threshold": global_drift > cfg["drift"]["threshold"],
    }
    out_dir = ensure_dir(Path(cfg["data"]["artifacts_dir"]) / "drift")
    dump_json(out, out_dir / "drift_report.json")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_drift(args.config)
