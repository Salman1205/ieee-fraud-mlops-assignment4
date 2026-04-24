from __future__ import annotations

import pandas as pd
from imblearn.over_sampling import SMOTE


def apply_smote(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)
    return X_resampled, y_resampled
