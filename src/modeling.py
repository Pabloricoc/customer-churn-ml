"""Reusable preprocessing and final-model utilities."""

import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import (
    FINAL_MODEL_PARAMS,
    FINAL_THRESHOLD,
    NUMERIC_FEATURES,
)


def build_preprocessor(feature_names) -> ColumnTransformer:
    """Build the numerical and categorical preprocessing workflow."""
    categorical_features = [
        feature
        for feature in feature_names
        if feature not in NUMERIC_FEATURES
    ]

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "cat",
                categorical_pipeline,
                categorical_features,
            ),
        ]
    )


def build_final_model(feature_names) -> Pipeline:
    """Build the finalized preprocessing and Gradient Boosting pipeline."""
    preprocessor = build_preprocessor(
        feature_names
    )

    classifier = GradientBoostingClassifier(
        **FINAL_MODEL_PARAMS
    )

    return Pipeline(
        steps=[
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


def predict_with_threshold(
    model: Pipeline,
    X,
    threshold: float = FINAL_THRESHOLD,
):
    """Return churn labels and positive-class probabilities."""
    class_labels = list(
        model.named_steps["classifier"].classes_
    )

    positive_class_index = class_labels.index(
        "Yes"
    )

    probabilities = model.predict_proba(
        X
    )[:, positive_class_index]

    predictions = np.where(
        probabilities >= threshold,
        "Yes",
        "No",
    )

    return predictions, probabilities