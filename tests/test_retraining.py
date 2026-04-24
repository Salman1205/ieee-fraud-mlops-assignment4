from src.retraining import should_retrain


def test_threshold_trigger_on_low_recall():
    res = should_retrain(
        recall=0.6,
        recall_threshold=0.8,
        drift_score=0.05,
        drift_threshold=0.15,
        last_retrain_iso=None,
        periodic_days=7,
    )
    assert res["trigger_retraining"] is True
    assert res["threshold_trigger"] is True
