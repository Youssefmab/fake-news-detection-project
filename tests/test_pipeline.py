"""Unit tests for the end-to-end sklearn Pipeline (src/pipeline.py).

Run with:
    pytest tests/test_pipeline.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.pipeline import (  # noqa: E402
    DEFAULT_LINGUISTIC_COLS,
    TextSelector,
    build_pipeline,
    load_pipeline,
    make_lr_pipeline,
    make_rf_pipeline,
    make_svm_pipeline,
    save_pipeline,
)


@pytest.fixture
def toy_df() -> pd.DataFrame:
    """Minimal DataFrame matching the project schema."""
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame({
        "cleaned_text": [
            "covid vaccine is safe and effective" if i % 2 else
            "shocking 5g causes coronavirus hoax fake conspiracy"
            for i in range(n)
        ],
    })
    for col in DEFAULT_LINGUISTIC_COLS:
        df[col] = rng.uniform(0, 10, size=n)
    df["label"] = np.array([i % 2 for i in range(n)])
    return df


def test_text_selector_returns_1d_array(toy_df):
    selector = TextSelector(column="cleaned_text")
    out = selector.fit_transform(toy_df)
    assert out.ndim == 1
    assert len(out) == len(toy_df)
    assert isinstance(out[0], str)


def test_build_pipeline_returns_pipeline(toy_df):
    pipe = build_pipeline()
    assert isinstance(pipe, Pipeline)
    assert "features" in pipe.named_steps
    assert "clf" in pipe.named_steps


def test_pipeline_fit_predict_lr(toy_df):
    pipe = make_lr_pipeline()
    X = toy_df.drop(columns=["label"])
    y = toy_df["label"].values
    pipe.fit(X, y)
    preds = pipe.predict(X)
    assert preds.shape == y.shape
    assert set(np.unique(preds)).issubset({0, 1})
    # Easy synthetic task should be near-perfect on training set
    assert (preds == y).mean() > 0.85


def test_pipeline_predict_proba(toy_df):
    pipe = make_lr_pipeline()
    X = toy_df.drop(columns=["label"])
    y = toy_df["label"].values
    pipe.fit(X, y)
    proba = pipe.predict_proba(X)
    assert proba.shape == (len(X), 2)
    assert np.allclose(proba.sum(axis=1), 1.0)


def test_svm_pipeline_runs(toy_df):
    pipe = make_svm_pipeline()
    X = toy_df.drop(columns=["label"])
    y = toy_df["label"].values
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.8


def test_rf_pipeline_runs(toy_df):
    pipe = make_rf_pipeline(n_estimators=20, max_depth=5)
    X = toy_df.drop(columns=["label"])
    y = toy_df["label"].values
    pipe.fit(X, y)
    assert pipe.score(X, y) > 0.8


def test_save_and_load_pipeline_roundtrip(toy_df, tmp_path):
    pipe = make_lr_pipeline()
    X = toy_df.drop(columns=["label"])
    y = toy_df["label"].values
    pipe.fit(X, y)

    path = tmp_path / "pipe.pkl"
    save_pipeline(pipe, path)
    assert path.exists()

    loaded = load_pipeline(path)
    np.testing.assert_array_equal(loaded.predict(X), pipe.predict(X))


def test_pipeline_handles_unseen_text(toy_df):
    pipe = make_lr_pipeline()
    X = toy_df.drop(columns=["label"])
    y = toy_df["label"].values
    pipe.fit(X, y)

    new_row = X.iloc[[0]].copy()
    new_row["cleaned_text"] = "completely new vocabulary unseen at training time"
    pred = pipe.predict(new_row)
    assert pred.shape == (1,)
