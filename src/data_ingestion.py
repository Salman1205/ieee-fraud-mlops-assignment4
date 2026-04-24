from pathlib import Path

import pandas as pd


def load_ieee_data(raw_dir: str, max_rows: int | None = None) -> pd.DataFrame:
    base = Path(raw_dir)
    tx = pd.read_csv(base / "train_transaction.csv", nrows=max_rows)
    identity_path = base / "train_identity.csv"
    if identity_path.exists():
        identity = pd.read_csv(identity_path, nrows=max_rows)
        df = tx.merge(identity, on="TransactionID", how="left")
    else:
        df = tx
    return df
