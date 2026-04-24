from __future__ import annotations

import argparse
import datetime as dt
import json
from pathlib import Path

from src.utils import dump_json, ensure_dir, load_yaml


def should_retrain(
    recall: float,
    recall_threshold: float,
    drift_score: float,
    drift_threshold: float,
    last_retrain_iso: str | None,
    periodic_days: int,
) -> dict:
    now = dt.datetime.now(dt.UTC)
    periodic_due = False
    if last_retrain_iso:
        last = dt.datetime.fromisoformat(last_retrain_iso)
        periodic_due = (now - last).days >= periodic_days

    threshold_trigger = recall < recall_threshold or drift_score > drift_threshold
    trigger = threshold_trigger or periodic_due
    return {
        "trigger_retraining": trigger,
        "threshold_trigger": threshold_trigger,
        "periodic_trigger": periodic_due,
        "timestamp_utc": now.isoformat(),
    }


def run_retraining_decision(config_path: str) -> None:
    cfg = load_yaml(config_path)
    artifacts = Path(cfg["data"]["artifacts_dir"])

    metrics_path = artifacts / "metrics" / "model_metrics.json"
    drift_path = artifacts / "drift" / "drift_report.json"
    state_path = artifacts / "retraining" / "state.json"

    recall = 1.0
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            metrics = json.load(f)
        best = max(metrics, key=lambda k: (metrics[k]["recall"], metrics[k]["auc_roc"]))
        recall = float(metrics[best]["recall"])

    drift_score = 0.0
    if drift_path.exists():
        with open(drift_path, "r", encoding="utf-8") as f:
            drift = json.load(f)
        drift_score = float(drift.get("global_drift_score", 0.0))

    last_retrain = None
    if state_path.exists():
        with open(state_path, "r", encoding="utf-8") as f:
            state = json.load(f)
        last_retrain = state.get("last_retrain_utc")

    decision = should_retrain(
        recall=recall,
        recall_threshold=0.8,
        drift_score=drift_score,
        drift_threshold=cfg["drift"]["threshold"],
        last_retrain_iso=last_retrain,
        periodic_days=cfg["drift"]["periodic_days"],
    )

    out_dir = ensure_dir(artifacts / "retraining")
    dump_json(decision, out_dir / "decision.json")
    if decision["trigger_retraining"]:
        dump_json({"last_retrain_utc": decision["timestamp_utc"]}, state_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    run_retraining_decision(args.config)
