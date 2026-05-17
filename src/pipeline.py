"""End-to-end sklearn Pipeline for COVID-19 fake news classification.

Combines TF-IDF features on cleaned text with hand-crafted linguistic
features through a ColumnTransformer, then plugs in any sklearn-compatible
classifier. Provides convenience builders for the project's baseline models.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

DEFAULT_TEXT_COL = "cleaned_text"
DEFAULT_LINGUISTIC_COLS: List[str] = [
    "word_count", "char_count", "avg_word_length", "unique_word_count",
    "unique_word_ratio", "exclamation_count", "question_count",
    "uppercase_ratio", "punctuation_count", "hashtag_count",
    "mention_count", "url_count",
]


class TextSelector(BaseEstimator, TransformerMixin):
    """Pick a single text column from a DataFrame and return a 1-D array.

    Needed because TfidfVectorizer expects a 1-D iterable of strings,
    not a DataFrame slice.
    """

    def __init__(self, column: str = DEFAULT_TEXT_COL):
        self.column = column

    def fit(self, X: pd.DataFrame, y: Optional[np.ndarray] = None):
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        return X[self.column].astype(str).values


def build_feature_union(
    text_col: str = DEFAULT_TEXT_COL,
    linguistic_cols: Optional[List[str]] = None,
    tfidf_kwargs: Optional[dict] = None,
) -> ColumnTransformer:
    """Build the ColumnTransformer joining TF-IDF and scaled linguistic features."""
    linguistic_cols = linguistic_cols or DEFAULT_LINGUISTIC_COLS
    tfidf_kwargs = tfidf_kwargs or {
        "max_features": 10_000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95,
        "sublinear_tf": True,
    }

    text_pipe = Pipeline([
        ("select", TextSelector(column=text_col)),
        ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
    ])

    return ColumnTransformer(
        transformers=[
            ("text", text_pipe, [text_col]),
            ("linguistic", StandardScaler(with_mean=False), linguistic_cols),
        ],
        sparse_threshold=0.3,
    )


def build_pipeline(
    classifier: Optional[BaseEstimator] = None,
    text_col: str = DEFAULT_TEXT_COL,
    linguistic_cols: Optional[List[str]] = None,
    tfidf_kwargs: Optional[dict] = None,
) -> Pipeline:
    """Build the full ingestion → features → classifier Pipeline."""
    if classifier is None:
        classifier = LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )

    return Pipeline([
        ("features", build_feature_union(text_col, linguistic_cols, tfidf_kwargs)),
        ("clf", classifier),
    ])


def make_lr_pipeline(**kwargs: Any) -> Pipeline:
    return build_pipeline(LogisticRegression(
        max_iter=1000, class_weight="balanced", random_state=42, **kwargs))


def make_svm_pipeline(**kwargs: Any) -> Pipeline:
    return build_pipeline(LinearSVC(
        max_iter=2000, class_weight="balanced", random_state=42, **kwargs))


def make_rf_pipeline(**kwargs: Any) -> Pipeline:
    defaults = {"n_estimators": 200, "max_depth": 20, "n_jobs": -1, "random_state": 42}
    defaults.update(kwargs)
    return build_pipeline(RandomForestClassifier(**defaults))


def save_pipeline(pipe: Pipeline, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(pipe, f)


def load_pipeline(path: str | Path) -> Pipeline:
    with open(path, "rb") as f:
        return pickle.load(f)
