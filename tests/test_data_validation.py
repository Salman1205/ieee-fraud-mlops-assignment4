import pandas as pd

from src.data_validation import validate_dataframe, validate_schema


def test_validate_dataframe():
    df = pd.DataFrame({"a": [1, None], "isFraud": [0, 1]})
    r = validate_dataframe(df, target_col="isFraud")
    assert r.row_count == 2
    assert r.has_target


def test_validate_schema():
    df = pd.DataFrame({"a": [1], "isFraud": [0]})
    assert validate_schema(df, ["a", "isFraud"])
