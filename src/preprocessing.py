from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer


@dataclass
class PreprocessArtifacts:
    numeric_cols: list[str]
    categorical_cols: list[str]


def preprocess_dataframe(
    df: pd.DataFrame,
    target_col: str = "isFraud",
    numeric_strategy: str = "median",
    categorical_strategy: str = "most_frequent",
) -> tuple[pd.DataFrame, PreprocessArtifacts]:
    data = df.copy()
    features = data.drop(columns=[target_col], errors="ignore")
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in features.columns if c not in numeric_cols]

    if numeric_cols:
        num_imputer = SimpleImputer(strategy=numeric_strategy)
        features[numeric_cols] = num_imputer.fit_transform(features[numeric_cols])
    if categorical_cols:
        cat_imputer = SimpleImputer(strategy=categorical_strategy)
        features[categorical_cols] = cat_imputer.fit_transform(features[categorical_cols])

    if target_col in data.columns:
        features[target_col] = data[target_col].values

    return features, PreprocessArtifacts(numeric_cols=numeric_cols, categorical_cols=categorical_cols)
