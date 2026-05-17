"""Model definitions for COVID-19 Fake News Detection."""

from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from transformers import AutoModel


# ======================================================================
# Baseline (traditional ML) models
# ======================================================================


class BaselineModels:
    """Traditional ML baselines using TF-IDF features."""

    def __init__(self, max_features: int = 10000):
        """Initialize the baseline models wrapper.

        Args:
            max_features: Maximum number of TF-IDF features.
        """
        self.max_features = max_features
        self.tfidf = TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 2),
            stop_words="english",
            sublinear_tf=True,
        )

    def get_tfidf_features(
        self, texts: List[str], fit: bool = True
    ) -> np.ndarray:
        """Compute TF-IDF feature matrix.

        Args:
            texts: List of text strings.
            fit: If True, fit the vectorizer; otherwise only transform.

        Returns:
            Sparse TF-IDF feature matrix.
        """
        if fit:
            return self.tfidf.fit_transform(texts)
        return self.tfidf.transform(texts)

    def train_svm(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> SVC:
        """Train a Support Vector Machine classifier.

        Args:
            X_train: TF-IDF feature matrix for training data.
            y_train: Training labels.

        Returns:
            Trained SVM model.
        """
        model = SVC(
            kernel="linear",
            C=1.0,
            probability=True,
            class_weight="balanced",
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    def train_logistic_regression(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> LogisticRegression:
        """Train a Logistic Regression classifier.

        Args:
            X_train: TF-IDF feature matrix for training data.
            y_train: Training labels.

        Returns:
            Trained Logistic Regression model.
        """
        model = LogisticRegression(
            max_iter=1000,
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    def train_random_forest(
        self, X_train: np.ndarray, y_train: np.ndarray
    ) -> RandomForestClassifier:
        """Train a Random Forest classifier.

        Args:
            X_train: TF-IDF feature matrix for training data.
            y_train: Training labels.

        Returns:
            Trained Random Forest model.
        """
        model = RandomForestClassifier(
            n_estimators=200,
            max_depth=None,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )
        model.fit(X_train, y_train)
        return model

    def predict(
        self, model, X_test: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Generate predictions and probability estimates.

        Args:
            model: A trained sklearn classifier.
            X_test: TF-IDF feature matrix for test data.

        Returns:
            Tuple of (predicted labels, predicted probabilities).
        """
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)
        return y_pred, y_prob


# ======================================================================
# BERT-based classifier
# ======================================================================


class BERTClassifier(nn.Module):
    """BERT-based binary classifier for fake news detection."""

    def __init__(
        self,
        model_name: str = "bert-base-uncased",
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize the BERT classifier.

        Args:
            model_name: HuggingFace BERT model identifier.
            num_classes: Number of output classes.
            dropout: Dropout probability for the classification head.
        """
        super().__init__()
        self.bert = AutoModel.from_pretrained(model_name)
        hidden_size = self.bert.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through BERT and classification head.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len).
            attention_mask: Attention mask tensor of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits


# ======================================================================
# RoBERTa-based classifier
# ======================================================================


class RoBERTaClassifier(nn.Module):
    """RoBERTa-based binary classifier for fake news detection."""

    def __init__(
        self,
        model_name: str = "roberta-base",
        num_classes: int = 2,
        dropout: float = 0.3,
    ):
        """Initialize the RoBERTa classifier.

        Args:
            model_name: HuggingFace RoBERTa model identifier.
            num_classes: Number of output classes.
            dropout: Dropout probability for the classification head.
        """
        super().__init__()
        self.roberta = AutoModel.from_pretrained(model_name)
        hidden_size = self.roberta.config.hidden_size

        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(hidden_size // 2, num_classes),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass through RoBERTa and classification head.

        Args:
            input_ids: Token IDs tensor of shape (batch, seq_len).
            attention_mask: Attention mask tensor of shape (batch, seq_len).

        Returns:
            Logits tensor of shape (batch, num_classes).
        """
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        # Use the <s> token representation (equivalent to [CLS])
        cls_output = outputs.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_output)
        return logits
