from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from src.data_ingestion import load_ieee_data
from src.data_validation import validate_dataframe, validate_schema
from src.evaluate import evaluate_binary
from src.feature_engineering import add_frequency_encoding, add_target_encoding
from src.imbalance import apply_smote
from src.preprocessing import preprocess_dataframe
from src.utils import dump_json, ensure_dir, load_yaml


def _prepare_dataset(config: dict):
    df = load_ieee_data(config["data"]["raw_dir"], max_rows=config["data"].get("max_rows"))
    report = validate_dataframe(df, target_col=config["data"]["target_col"])
    dump_json(report.__dict__, Path(config["data"]["artifacts_dir"]) / "validation_report.json")

    assert validate_schema(df, [config["data"]["target_col"]]), "Target column missing"
    df, artifacts = preprocess_dataframe(df, target_col=config["data"]["target_col"])
    target = config["data"]["target_col"]

    cat_cols = artifacts.categorical_cols[:10]
    if cat_cols:
        df, _ = add_frequency_encoding(df, df.copy(), cat_cols)
        df, _ = add_target_encoding(df, df.copy(), cat_cols, target_col=target)

    df = df.drop(columns=artifacts.categorical_cols, errors="ignore")
    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(
        X, y, test_size=config["training"]["test_size"], random_state=config["training"]["random_state"], stratify=y
    )


def run_training(config_path: str) -> None:
    config = load_yaml(config_path)
    artifacts_dir = ensure_dir(config["data"]["artifacts_dir"])
    models_dir = ensure_dir(artifacts_dir / "models")
    metrics_dir = ensure_dir(artifacts_dir / "metrics")

    X_train, X_test, y_train, y_test = _prepare_dataset(config)
    threshold = config["training"]["threshold"]
    rs = config["training"]["random_state"]
    pos_w = config["training"]["class_weight_positive"]

    xgb = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, scale_pos_weight=pos_w
    )
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=63, class_weight={0: 1, 1: pos_w})

    rf_selector = RandomForestClassifier(n_estimators=120, random_state=rs, class_weight={0: 1, 1: pos_w})
    rf_selector.fit(X_train, y_train)
    selector = SelectFromModel(rf_selector, prefit=True, threshold="median")
    X_train_h = selector.transform(X_train)
    X_test_h = selector.transform(X_test)
    rf_hybrid = RandomForestClassifier(n_estimators=250, random_state=rs, class_weight={0: 1, 1: pos_w})

    models = {"xgboost": xgb, "lightgbm": lgbm}
    metrics = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:, 1]
        score = evaluate_binary(y_test, proba, threshold=threshold)
        metrics[name] = score
        joblib.dump(model, models_dir / f"{name}.pkl")

    rf_hybrid.fit(X_train_h, y_train)
    proba_h = rf_hybrid.predict_proba(X_test_h)[:, 1]
    metrics["hybrid_rf_fs"] = evaluate_binary(y_test, proba_h, threshold=threshold)
    joblib.dump({"selector": selector, "model": rf_hybrid}, models_dir / "hybrid_rf_fs.pkl")

    X_train_sm, y_train_sm = apply_smote(X_train, y_train, random_state=rs)
    xgb_smote = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.05)
    xgb_smote.fit(X_train_sm, y_train_sm)
    metrics["xgboost_smote"] = evaluate_binary(y_test, xgb_smote.predict_proba(X_test)[:, 1], threshold=threshold)
    joblib.dump(xgb_smote, models_dir / "xgboost_smote.pkl")

    dump_json(metrics, metrics_dir / "model_metrics.json")

    best_name = max(metrics, key=lambda k: (metrics[k]["recall"], metrics[k]["auc_roc"]))
    if best_name == "hybrid_rf_fs":
        joblib.dump({"selector": selector, "model": rf_hybrid}, models_dir / "best_model.pkl")
    elif best_name == "xgboost_smote":
        joblib.dump(xgb_smote, models_dir / "best_model.pkl")
    else:
        joblib.dump(models[best_name], models_dir / "best_model.pkl")

    pd.DataFrame(metrics).T.to_csv(metrics_dir / "model_metrics.csv", index=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_training(args.config)
