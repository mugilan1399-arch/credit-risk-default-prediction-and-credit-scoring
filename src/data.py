"""Loading and splitting the Give Me Some Credit data."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.config import DATA_DIR, RANDOM_STATE, TARGET, VAL_SIZE


def load_raw(data_dir=DATA_DIR):
    """Return (train, test, sample_entry) exactly as the CSVs hold them."""
    data_dir = Path(data_dir)
    train = pd.read_csv(data_dir / "cs-training.csv", index_col=0)
    test = pd.read_csv(data_dir / "cs-test.csv", index_col=0)
    sample_entry = pd.read_csv(data_dir / "sampleEntry.csv", index_col=0)
    return train, test, sample_entry


def features_and_target(df):
    """Split a frame into X (features) and y (target)."""
    return df.drop(columns=TARGET), df[TARGET]


def train_val_split(X, y, val_size=VAL_SIZE, random_state=RANDOM_STATE):
    """Stratified hold-out split, so both sides keep the ~6.7% default rate."""
    return train_test_split(
        X, y, test_size=val_size, stratify=y, random_state=random_state
    )


def load_split(data_dir=DATA_DIR):
    """Convenience: raw CSVs in, (X_train, X_val, y_train, y_val) out."""
    train, _, _ = load_raw(data_dir)
    X, y = features_and_target(train)
    return train_val_split(X, y)
