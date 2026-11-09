"""Data loading and preprocessing helpers for the Hull Tactical Market Prediction competition."""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

# Kaggle's input folder; on Colab we download the data to ./data instead.
KAGGLE_DIR = Path("/kaggle/input/hull-tactical-market-prediction")
LOCAL_DIR = Path("data")

TARGET = "market_forward_excess_returns"
# Columns that are only known *after* the day is over -> never use them as features.
LEAK_COLS = ["forward_returns", "risk_free_rate", "market_forward_excess_returns"]
ID_COL = "date_id"

# Feature families as described on the competition data page.
FEATURE_GROUPS = {
    "D": "Dummy / binary",
    "E": "Macro economic",
    "I": "Interest rate",
    "M": "Market dynamics / technical",
    "P": "Price / valuation",
    "S": "Sentiment",
    "V": "Volatility",
}


def get_data_dir() -> Path:
    """Return the folder holding train.csv / test.csv (Kaggle or local)."""
    if KAGGLE_DIR.exists():
        return KAGGLE_DIR
    env = os.environ.get("HULL_DATA_DIR")
    return Path(env) if env else LOCAL_DIR


def load_train(data_dir: str | Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else get_data_dir()
    df = pd.read_csv(data_dir / "train.csv")
    return df.sort_values(ID_COL).reset_index(drop=True)


def load_test(data_dir: str | Path | None = None) -> pd.DataFrame:
    data_dir = Path(data_dir) if data_dir else get_data_dir()
    return pd.read_csv(data_dir / "test.csv").sort_values(ID_COL).reset_index(drop=True)


def feature_columns(df: pd.DataFrame) -> list[str]:
    """All anonymised feature columns (D*, E*, I*, M*, P*, S*, V*)."""
    cols = []
    for c in df.columns:
        prefix, rest = c[:1], c[1:]
        if prefix in FEATURE_GROUPS and rest.isdigit():
            cols.append(c)
    return cols


def group_of(col: str) -> str:
    return col[:1]


def missing_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Missing ratio per column plus the first date_id where the column becomes available."""
    feats = feature_columns(df)
    rows = []
    for c in feats:
        first_valid = df.loc[df[c].notna(), ID_COL]
        rows.append({
            "feature": c,
            "group": group_of(c),
            "missing_ratio": df[c].isna().mean(),
            "first_valid_date_id": int(first_valid.iloc[0]) if len(first_valid) else None,
        })
    return pd.DataFrame(rows).sort_values("missing_ratio", ascending=False).reset_index(drop=True)


def drop_early_rows(df: pd.DataFrame, max_missing_ratio: float = 0.5) -> pd.DataFrame:
    """Drop the oldest rows where most features are still missing."""
    feats = feature_columns(df)
    row_missing = df[feats].isna().mean(axis=1)
    ok = row_missing <= max_missing_ratio
    if not ok.any():
        return df.iloc[0:0]
    start = ok.idxmax()  # first row that satisfies the threshold
    return df.loc[start:].reset_index(drop=True)


def fill_missing(X: pd.DataFrame, medians: pd.Series | None = None) -> tuple[pd.DataFrame, pd.Series]:
    """Forward-fill in time, then fill what is left with (train) medians.

    Returns the filled frame and the medians so the same values can be reused on validation/test.
    """
    X = X.ffill()
    if medians is None:
        medians = X.median(numeric_only=True)
    return X.fillna(medians).fillna(0.0), medians


def add_lagged_returns(df: pd.DataFrame) -> pd.DataFrame:
    """Add yesterday's known returns to the training data.

    On the test set Kaggle provides `lagged_forward_returns` etc. (the previous day's realised values).
    The training set does not, so we build the same columns by shifting the targets one day.
    """
    df = df.copy()
    for col in LEAK_COLS:
        lag_col = f"lagged_{col}"
        if lag_col not in df.columns and col in df.columns:
            df[lag_col] = df[col].shift(1)
    return df


def time_split(df: pd.DataFrame, valid_frac: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological hold-out split (no shuffling - this is a time series)."""
    n_valid = int(len(df) * valid_frac)
    return df.iloc[:-n_valid].copy(), df.iloc[-n_valid:].copy()


def walk_forward_folds(n: int, n_folds: int = 5, valid_size: int | None = None, gap: int = 0):
    """Expanding-window folds: train on [0, t), validate on [t + gap, t + gap + valid_size).

    Yields (train_idx, valid_idx) numpy arrays.
    """
    valid_size = valid_size or n // (n_folds + 1)
    for k in range(n_folds):
        valid_end = n - (n_folds - 1 - k) * valid_size
        valid_start = valid_end - valid_size
        train_end = valid_start - gap
        if train_end <= 0:
            continue
        yield np.arange(0, train_end), np.arange(valid_start, valid_end)
