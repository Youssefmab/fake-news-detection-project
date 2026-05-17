"""Evaluation metrics and visualization for fake news classification."""

from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    auc,
    classification_report as sk_classification_report,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)


class Evaluator:
    """Computes and visualizes classification metrics."""

    def compute_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """Compute a comprehensive set of classification metrics.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            y_prob: Predicted probabilities for the positive class (optional).

        Returns:
            Dictionary of metric names to their values.
        """
        metrics: Dict[str, float] = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision_macro": precision_score(y_true, y_pred, average="macro", zero_division=0),
            "precision_micro": precision_score(y_true, y_pred, average="micro", zero_division=0),
            "precision_weighted": precision_score(y_true, y_pred, average="weighted", zero_division=0),
            "recall_macro": recall_score(y_true, y_pred, average="macro", zero_division=0),
            "recall_micro": recall_score(y_true, y_pred, average="micro", zero_division=0),
            "recall_weighted": recall_score(y_true, y_pred, average="weighted", zero_division=0),
            "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
            "f1_micro": f1_score(y_true, y_pred, average="micro", zero_division=0),
            "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
            "mcc": matthews_corrcoef(y_true, y_pred),
        }

        if y_prob is not None:
            try:
                metrics["auc_roc"] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics["auc_roc"] = float("nan")

        return metrics

    def classification_report(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> str:
        """Generate a detailed classification report.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            labels: Optional list of label names.

        Returns:
            Formatted classification report string.
        """
        return sk_classification_report(
            y_true, y_pred, target_names=labels, zero_division=0
        )

    def confusion_matrix_plot(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot and optionally save a confusion matrix heatmap.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            labels: Optional list of label names for axes.
            save_path: If provided, save the figure to this path.
        """
        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels or ["Class 0", "Class 1"],
            yticklabels=labels or ["Class 0", "Class 1"],
            ax=ax,
        )
        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")
        ax.set_title("Confusion Matrix")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"Confusion matrix saved to {save_path}")

        plt.show()

    def roc_curve_plot(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot and optionally save the ROC curve.

        Args:
            y_true: Ground-truth binary labels.
            y_prob: Predicted probabilities for the positive class.
            save_path: If provided, save the figure to this path.
        """
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {roc_auc:.4f})")
        ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random baseline")
        ax.set_xlim([0.0, 1.0])
        ax.set_ylim([0.0, 1.05])
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("Receiver Operating Characteristic (ROC) Curve")
        ax.legend(loc="lower right")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"ROC curve saved to {save_path}")

        plt.show()
