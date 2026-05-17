"""Error analysis utilities for fake news classification."""

from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class ErrorAnalyzer:
    """Provides tools for analyzing misclassifications and model behaviour."""

    def analyze_errors(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """Identify and return misclassified examples.

        Args:
            texts: Original text inputs.
            y_true: Ground-truth labels.
            y_pred: Predicted labels.

        Returns:
            DataFrame containing only the misclassified samples with
            columns: text, true_label, predicted_label.
        """
        errors_mask = np.array(y_true) != np.array(y_pred)
        error_df = pd.DataFrame(
            {
                "text": np.array(texts)[errors_mask],
                "true_label": np.array(y_true)[errors_mask],
                "predicted_label": np.array(y_pred)[errors_mask],
            }
        )
        error_df = error_df.reset_index(drop=True)
        print(f"Total misclassified: {len(error_df)} / {len(texts)} "
              f"({len(error_df) / len(texts) * 100:.2f}%)")
        return error_df

    def error_distribution(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, int]]:
        """Compute and plot the error distribution by class.

        Args:
            y_true: Ground-truth labels.
            y_pred: Predicted labels.
            labels: Optional list of class name strings.

        Returns:
            Dictionary mapping each class to counts of correct and incorrect predictions.
        """
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        unique_classes = np.unique(y_true)
        distribution: Dict[str, Dict[str, int]] = {}

        for cls in unique_classes:
            cls_mask = y_true == cls
            correct = int(np.sum(y_pred[cls_mask] == cls))
            incorrect = int(np.sum(y_pred[cls_mask] != cls))
            class_name = labels[cls] if labels and cls < len(labels) else str(cls)
            distribution[class_name] = {"correct": correct, "incorrect": incorrect}

        # Visualization
        class_names = list(distribution.keys())
        correct_counts = [d["correct"] for d in distribution.values()]
        incorrect_counts = [d["incorrect"] for d in distribution.values()]

        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(len(class_names))
        width = 0.35
        ax.bar(x - width / 2, correct_counts, width, label="Correct", color="steelblue")
        ax.bar(x + width / 2, incorrect_counts, width, label="Incorrect", color="salmon")
        ax.set_xticks(x)
        ax.set_xticklabels(class_names)
        ax.set_ylabel("Count")
        ax.set_title("Error Distribution by Class")
        ax.legend()
        plt.tight_layout()
        plt.show()

        return distribution

    def confidence_analysis(
        self,
        y_prob: np.ndarray,
        y_true: np.ndarray,
        y_pred: Optional[np.ndarray] = None,
    ) -> Tuple[pd.DataFrame, None]:
        """Analyze model confidence on correct vs incorrect predictions.

        If ``y_pred`` is not supplied it is derived from ``y_prob`` using argmax.

        Args:
            y_prob: Predicted probabilities (shape: [n_samples] or [n_samples, n_classes]).
            y_true: Ground-truth labels.
            y_pred: Predicted labels (optional; inferred from y_prob if absent).

        Returns:
            Tuple of (summary DataFrame, None). The plot is displayed inline.
        """
        y_prob = np.array(y_prob)
        y_true = np.array(y_true)

        # If probabilities are 2-D, take the max class probability as confidence
        if y_prob.ndim == 2:
            confidence = np.max(y_prob, axis=1)
            if y_pred is None:
                y_pred = np.argmax(y_prob, axis=1)
        else:
            confidence = y_prob
            if y_pred is None:
                y_pred = (y_prob >= 0.5).astype(int)

        correct_mask = np.array(y_true) == np.array(y_pred)

        summary = pd.DataFrame(
            {
                "group": ["Correct", "Incorrect"],
                "mean_confidence": [
                    confidence[correct_mask].mean() if correct_mask.any() else 0.0,
                    confidence[~correct_mask].mean() if (~correct_mask).any() else 0.0,
                ],
                "median_confidence": [
                    np.median(confidence[correct_mask]) if correct_mask.any() else 0.0,
                    np.median(confidence[~correct_mask]) if (~correct_mask).any() else 0.0,
                ],
                "count": [int(correct_mask.sum()), int((~correct_mask).sum())],
            }
        )

        # Plot confidence distributions
        fig, ax = plt.subplots(figsize=(8, 5))
        if correct_mask.any():
            ax.hist(
                confidence[correct_mask],
                bins=30,
                alpha=0.6,
                label="Correct",
                color="steelblue",
            )
        if (~correct_mask).any():
            ax.hist(
                confidence[~correct_mask],
                bins=30,
                alpha=0.6,
                label="Incorrect",
                color="salmon",
            )
        ax.set_xlabel("Confidence")
        ax.set_ylabel("Frequency")
        ax.set_title("Confidence Distribution: Correct vs Incorrect Predictions")
        ax.legend()
        plt.tight_layout()
        plt.show()

        print(summary.to_string(index=False))
        return summary, None

    def length_analysis(
        self,
        texts: List[str],
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> pd.DataFrame:
        """Analyze whether text length correlates with prediction errors.

        Args:
            texts: Original text inputs.
            y_true: Ground-truth labels.
            y_pred: Predicted labels.

        Returns:
            DataFrame with length statistics for correct and incorrect predictions.
        """
        lengths = np.array([len(t.split()) for t in texts])
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
        correct_mask = y_true == y_pred

        summary = pd.DataFrame(
            {
                "group": ["Correct", "Incorrect"],
                "mean_length": [
                    lengths[correct_mask].mean() if correct_mask.any() else 0.0,
                    lengths[~correct_mask].mean() if (~correct_mask).any() else 0.0,
                ],
                "median_length": [
                    np.median(lengths[correct_mask]) if correct_mask.any() else 0.0,
                    np.median(lengths[~correct_mask]) if (~correct_mask).any() else 0.0,
                ],
                "std_length": [
                    lengths[correct_mask].std() if correct_mask.any() else 0.0,
                    lengths[~correct_mask].std() if (~correct_mask).any() else 0.0,
                ],
                "count": [int(correct_mask.sum()), int((~correct_mask).sum())],
            }
        )

        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Box plot
        data_to_plot = []
        group_labels = []
        if correct_mask.any():
            data_to_plot.append(lengths[correct_mask])
            group_labels.append("Correct")
        if (~correct_mask).any():
            data_to_plot.append(lengths[~correct_mask])
            group_labels.append("Incorrect")

        axes[0].boxplot(data_to_plot, labels=group_labels)
        axes[0].set_ylabel("Text Length (words)")
        axes[0].set_title("Text Length Distribution by Prediction Outcome")

        # Histogram
        if correct_mask.any():
            axes[1].hist(
                lengths[correct_mask],
                bins=30,
                alpha=0.6,
                label="Correct",
                color="steelblue",
            )
        if (~correct_mask).any():
            axes[1].hist(
                lengths[~correct_mask],
                bins=30,
                alpha=0.6,
                label="Incorrect",
                color="salmon",
            )
        axes[1].set_xlabel("Text Length (words)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title("Text Length Histogram")
        axes[1].legend()

        plt.tight_layout()
        plt.show()

        print(summary.to_string(index=False))
        return summary
