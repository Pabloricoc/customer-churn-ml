"""Data loading and dataset splitting utilities."""

from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from .config import (
    ID_COLUMN,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


def load_data(data_path: Path) -> pd.DataFrame:
    """Load the raw churn dataset and apply deterministic type fixes."""
    df = pd.read_csv(data_path)

    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce",
    )

    return df


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Separate predictive features from the churn target."""
    X = df.drop(
        columns=[
            ID_COLUMN,
            TARGET_COLUMN,
        ]
    )

    y = df[TARGET_COLUMN]

    return X, y


def create_train_test_split(
    X: pd.DataFrame,
    y: pd.Series,
):
    """Create the reproducible stratified train/test split."""
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )