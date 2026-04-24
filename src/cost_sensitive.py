from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

from src.evaluate import evaluate_binary
from src.train_models import _prepare_dataset
from src.utils import dump_json, ensure_dir, load_yaml


def business_cost(y_true, y_prob, fn_cost: float, fp_cost: float, threshold: float = 0.5) -> dict[str, float]:
    y_pred = (np.array(y_prob) >= threshold).astype(int)
    fn = float(((y_true == 1) & (y_pred == 0)).sum())
    fp = float(((y_true == 0) & (y_pred == 1)).sum())
    return {"false_negative_cost": fn * fn_cost, "false_positive_cost": fp * fp_cost, "total_cost": fn * fn_cost + fp * fp_cost}


def run_cost_sensitive(config_path: str) -> None:
    cfg = load_yaml(config_path)
    X_train, X_test, y_train, y_test = _prepare_dataset(cfg)
    threshold = cfg["training"]["threshold"]
    fn_cost = cfg["training"]["fn_cost"]
    fp_cost = cfg["training"]["fp_cost"]

    baseline = XGBClassifier(n_estimators=220, max_depth=6, learning_rate=0.05)
    cost_model = XGBClassifier(
        n_estimators=220, max_depth=6, learning_rate=0.05, scale_pos_weight=cfg["training"]["class_weight_positive"]
    )

    baseline.fit(X_train, y_train)
    cost_model.fit(X_train, y_train)

    p_base = baseline.predict_proba(X_test)[:, 1]
    p_cost = cost_model.predict_proba(X_test)[:, 1]

    result = {
        "standard_training": {
            "metrics": evaluate_binary(y_test, p_base, threshold=threshold),
            "business_impact": business_cost(y_test, p_base, fn_cost, fp_cost, threshold=threshold),
        },
        "cost_sensitive_training": {
            "metrics": evaluate_binary(y_test, p_cost, threshold=threshold),
            "business_impact": business_cost(y_test, p_cost, fn_cost, fp_cost, threshold=threshold),
        },
    }

    out_dir = ensure_dir(Path(cfg["data"]["artifacts_dir"]) / "cost_sensitive")
    dump_json(result, out_dir / "comparison.json")
    pd.DataFrame(
        [
            {"mode": "standard", **result["standard_training"]["metrics"], **result["standard_training"]["business_impact"]},
            {"mode": "cost_sensitive", **result["cost_sensitive_training"]["metrics"], **result["cost_sensitive_training"]["business_impact"]},
        ]
    ).to_csv(out_dir / "comparison.csv", index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_cost_sensitive(args.config)
