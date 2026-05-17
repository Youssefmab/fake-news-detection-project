"""SHAP and LIME-based model interpretability for fake news classifiers."""

from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
import torch.nn as nn
from lime.lime_text import LimeTextExplainer
from transformers import PreTrainedTokenizer


class ModelInterpreter:
    """Provides LIME and SHAP explanations for transformer-based classifiers."""

    def __init__(
        self,
        model: nn.Module,
        tokenizer: PreTrainedTokenizer,
        device: torch.device,
        class_names: Optional[List[str]] = None,
        max_length: int = 256,
    ):
        """Initialize the interpreter.

        Args:
            model: Trained PyTorch classifier (BERTClassifier / RoBERTaClassifier).
            tokenizer: HuggingFace tokenizer matching the model.
            device: Torch device.
            class_names: Human-readable class names.
            max_length: Maximum token length for encoding.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.class_names = class_names or ["Real", "Fake"]
        self.max_length = max_length

        self.model.eval()

    # ------------------------------------------------------------------
    # Internal prediction helper
    # ------------------------------------------------------------------

    def _predict_proba(self, texts: List[str]) -> np.ndarray:
        """Predict class probabilities for a list of texts.

        Args:
            texts: List of raw text strings.

        Returns:
            Numpy array of shape (n_samples, n_classes) with probabilities.
        """
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = encodings["input_ids"].to(self.device)
        attention_mask = encodings["attention_mask"].to(self.device)

        with torch.no_grad():
            logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
            probs = torch.softmax(logits, dim=1).cpu().numpy()

        return probs

    # ------------------------------------------------------------------
    # LIME
    # ------------------------------------------------------------------

    def lime_explain(
        self, text: str, num_features: int = 10, num_samples: int = 500
    ):
        """Generate a LIME explanation for a single text.

        Args:
            text: The input text to explain.
            num_features: Number of top features to include in the explanation.
            num_samples: Number of perturbed samples LIME generates.

        Returns:
            A ``lime.explanation.Explanation`` object.
        """
        explainer = LimeTextExplainer(class_names=self.class_names)
        explanation = explainer.explain_instance(
            text,
            self._predict_proba,
            num_features=num_features,
            num_samples=num_samples,
        )
        return explanation

    def plot_lime_explanation(
        self,
        explanation,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a LIME explanation as a horizontal bar chart.

        Args:
            explanation: A LIME Explanation object.
            save_path: If provided, save the figure to this path.
        """
        fig = explanation.as_pyplot_figure()
        fig.set_size_inches(10, 6)
        plt.title("LIME Explanation - Feature Importance")
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"LIME explanation saved to {save_path}")

        plt.show()

    # ------------------------------------------------------------------
    # SHAP
    # ------------------------------------------------------------------

    def shap_explain(
        self, texts: List[str], num_samples: int = 100
    ) -> shap.Explanation:
        """Generate SHAP explanations for a list of texts.

        Uses a partition-based masker suitable for text data.

        Args:
            texts: List of texts to explain.
            num_samples: Number of evaluations for the SHAP explainer.

        Returns:
            A ``shap.Explanation`` object containing SHAP values.
        """
        masker = shap.maskers.Text(self.tokenizer)
        explainer = shap.Explainer(
            self._predict_proba,
            masker,
            output_names=self.class_names,
        )
        shap_values = explainer(texts, max_evals=num_samples)
        return shap_values

    def plot_shap_summary(
        self,
        shap_values: shap.Explanation,
        save_path: Optional[str] = None,
    ) -> None:
        """Plot a SHAP summary bar chart.

        Args:
            shap_values: SHAP Explanation object.
            save_path: If provided, save the figure to this path.
        """
        plt.figure(figsize=(10, 6))
        shap.plots.bar(shap_values, show=False)
        plt.title("SHAP Feature Importance Summary")
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"SHAP summary saved to {save_path}")

        plt.show()
