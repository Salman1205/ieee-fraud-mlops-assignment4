import pandas as pd

from src.drift_simulation import compute_numeric_drift


def test_drift_returns_scores():
    a = pd.DataFrame({"x": [1.0, 2.0, 3.0]})
    b = pd.DataFrame({"x": [2.0, 3.0, 4.0]})
    scores = compute_numeric_drift(a, b, ["x"])
    assert "x" in scores
    assert scores["x"] > 0
