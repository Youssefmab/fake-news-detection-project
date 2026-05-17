"""
Unit tests for evaluation / metrics utilities.

Run with:
    pytest tests/test_metrics.py -v
"""

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# Functions under test
# ═══════════════════════════════════════════════════════════════════════════
def compute_metrics(y_true, y_pred) -> dict:
    """
    Compute common classification metrics.

    Returns a dict with accuracy, precision, recall, and f1.
    """
    from sklearn.metrics import (
        accuracy_score,
        precision_score,
        recall_score,
        f1_score,
    )

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }


def confusion_matrix_plot(y_true, y_pred, output_path: str) -> str:
    """
    Generate a confusion-matrix heatmap and save it as an image.

    Parameters
    ----------
    y_true : array-like
        Ground-truth labels.
    y_pred : array-like
        Predicted labels.
    output_path : str
        File path where the image will be saved (PNG).

    Returns
    -------
    str
        The path to the saved image.
    """
    from sklearn.metrics import confusion_matrix
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt
    import seaborn as sns

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Fake", "Real"],
        yticklabels=["Fake", "Real"],
        ax=ax,
    )
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(output_path, dpi=100)
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# Tests — compute_metrics
# ═══════════════════════════════════════════════════════════════════════════
class TestComputeMetrics:
    """Tests for the compute_metrics function."""

    def test_perfect_predictions(self):
        y_true = [0, 0, 1, 1, 1]
        y_pred = [0, 0, 1, 1, 1]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 1.0
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["f1"] == 1.0

    def test_all_wrong_predictions(self):
        y_true = [0, 0, 1, 1]
        y_pred = [1, 1, 0, 0]
        metrics = compute_metrics(y_true, y_pred)
        assert metrics["accuracy"] == 0.0

    def test_known_values(self):
        # 2 TP, 1 FP, 1 FN, 1 TN
        y_true = [1, 1, 0, 1, 0]
        y_pred = [1, 1, 1, 0, 0]
        metrics = compute_metrics(y_true, y_pred)
        # Accuracy: 3/5 = 0.6
        assert abs(metrics["accuracy"] - 0.6) < 1e-9
        # Precision (for label 1): TP/(TP+FP) = 2/3
        assert abs(metrics["precision"] - 2 / 3) < 1e-9
        # Recall (for label 1): TP/(TP+FN) = 2/3
        assert abs(metrics["recall"] - 2 / 3) < 1e-9

    def test_return_types(self):
        metrics = compute_metrics([0, 1], [0, 1])
        for key in ("accuracy", "precision", "recall", "f1"):
            assert isinstance(metrics[key], float), f"{key} should be float"

    def test_single_class(self):
        """All predictions are the same class — should not raise."""
        metrics = compute_metrics([0, 0, 0], [0, 0, 0])
        assert metrics["accuracy"] == 1.0

    def test_metric_ranges(self):
        """All metric values must be between 0 and 1."""
        rng = np.random.RandomState(42)
        y_true = rng.randint(0, 2, size=100).tolist()
        y_pred = rng.randint(0, 2, size=100).tolist()
        metrics = compute_metrics(y_true, y_pred)
        for key, value in metrics.items():
            assert 0.0 <= value <= 1.0, f"{key}={value} is out of [0, 1]"


# ═══════════════════════════════════════════════════════════════════════════
# Tests — confusion_matrix_plot
# ═══════════════════════════════════════════════════════════════════════════
class TestConfusionMatrixPlot:
    """Tests for the confusion_matrix_plot function."""

    def test_creates_file(self):
        y_true = [0, 0, 1, 1, 0, 1]
        y_pred = [0, 1, 1, 1, 0, 0]
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "cm.png")
            result_path = confusion_matrix_plot(y_true, y_pred, output)
            assert os.path.isfile(result_path)
            # The file should be a valid PNG (starts with the PNG magic bytes)
            with open(result_path, "rb") as f:
                header = f.read(4)
            assert header[:4] == b"\x89PNG"

    def test_returns_correct_path(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "matrix.png")
            result_path = confusion_matrix_plot([0, 1], [0, 1], output)
            assert result_path == output

    def test_file_not_empty(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            output = os.path.join(tmpdir, "cm_test.png")
            confusion_matrix_plot([0, 1, 0], [0, 0, 1], output)
            assert os.path.getsize(output) > 0
