from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

from src.data_ingestion import load_ieee_data
from src.feature_engineering import add_frequency_encoding, add_target_encoding
from src.preprocessing import preprocess_dataframe
from src.utils import ensure_dir, load_yaml


def run_explainability(config_path: str, model_path: str | None = None) -> None:
    cfg = load_yaml(config_path)
    artifacts_dir = Path(cfg["data"]["artifacts_dir"])
    model_file = Path(model_path) if model_path else artifacts_dir / "models" / "best_model.pkl"
    model_obj = joblib.load(model_file)
    model = model_obj["model"] if isinstance(model_obj, dict) and "model" in model_obj else model_obj
    selector = model_obj.get("selector") if isinstance(model_obj, dict) else None

    df = load_ieee_data(cfg["data"]["raw_dir"], max_rows=cfg["data"].get("max_rows"))
    df, prep = preprocess_dataframe(df, target_col=cfg["data"]["target_col"])
    target_col = cfg["data"]["target_col"]
    cat_cols = prep.categorical_cols[:10]
    if cat_cols:
        df, _ = add_frequency_encoding(df, df.copy(), cat_cols)
        df, _ = add_target_encoding(df, df.copy(), cat_cols, target_col=target_col)
    X = df.drop(columns=[target_col] + prep.categorical_cols, errors="ignore")
    X_sample = X.sample(n=min(1000, len(X)), random_state=42)
    if selector is not None:
        X_used = selector.transform(X_sample)
        feature_names = [f"f_{i}" for i in range(X_used.shape[1])]
        X_used = pd.DataFrame(X_used, columns=feature_names)
    else:
        X_used = X_sample

    # Align inference matrix shape with trained model expectations.
    expected_features = getattr(model, "n_features_in_", None)
    if expected_features is None and hasattr(model, "booster_"):
        expected_features = model.booster_.num_feature()
    if isinstance(expected_features, int):
        current = X_used.shape[1]
        if current < expected_features:
            for i in range(expected_features - current):
                X_used[f"pad_feature_{i}"] = 0.0
        elif current > expected_features:
            X_used = X_used.iloc[:, :expected_features]

    explain_dir = ensure_dir(artifacts_dir / "explainability")
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        if len(importances) != len(X_used.columns):
            feature_names = [f"feature_{i}" for i in range(len(importances))]
        else:
            feature_names = list(X_used.columns)
        fi = pd.DataFrame({"feature": feature_names, "importance": importances}).sort_values("importance", ascending=False)
        fi.to_csv(explain_dir / "feature_importance.csv", index=False)

    explainer = shap.TreeExplainer(model)
    sv = explainer.shap_values(X_used)
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, X_used, show=False)
    plt.tight_layout()
    plt.savefig(explain_dir / "shap_summary.png", dpi=150)
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model-path", default=None)
    args = parser.parse_args()
    run_explainability(args.config, model_path=args.model_path)
