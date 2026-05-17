"""Text preprocessing pipeline for COVID-19 Fake News Detection."""

import re
import string
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from transformers import AutoTokenizer


class TextProcessor:
    """Handles text cleaning, linguistic feature extraction, and BERT tokenization."""

    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 256):
        """Initialize the text processor.

        Args:
            model_name: HuggingFace model name for tokenization.
            max_length: Maximum token sequence length for BERT.
        """
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        # Simple positive/negative word lists for sentiment heuristics
        self._positive_words = {
            "good", "great", "excellent", "positive", "confirmed", "safe",
            "effective", "recovery", "recovered", "cure", "healthy", "hope",
            "success", "successful", "approved", "beneficial", "protect",
            "reliable", "trust", "accurate", "verified", "official",
        }
        self._negative_words = {
            "bad", "terrible", "fake", "hoax", "dangerous", "death", "die",
            "conspiracy", "lie", "false", "scam", "fraud", "panic", "fear",
            "threat", "kill", "toxic", "harmful", "illegal", "worst",
            "misinformation", "disinformation", "propaganda", "unverified",
        }

    # ------------------------------------------------------------------
    # Cleaning
    # ------------------------------------------------------------------

    def clean_text(self, text: str) -> str:
        """Remove URLs, HTML tags, special characters, and normalize whitespace.

        Args:
            text: Raw input text.

        Returns:
            Cleaned text string.
        """
        if not isinstance(text, str):
            return ""

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove HTML tags
        text = re.sub(r"<[^>]+>", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove mentions and hashtags symbols (keep the text)
        text = re.sub(r"[@#]", "", text)

        # Remove special characters but keep basic punctuation
        text = re.sub(r"[^\w\s.,!?;:'\"-]", "", text)

        # Normalize whitespace
        text = re.sub(r"\s+", " ", text).strip()

        return text

    # ------------------------------------------------------------------
    # Linguistic features
    # ------------------------------------------------------------------

    def extract_linguistic_features(self, text: str) -> Dict[str, float]:
        """Extract linguistic features using simple heuristics.

        Features extracted:
            - avg_word_length: average number of characters per word
            - sentence_count: approximate number of sentences
            - exclamation_count: number of exclamation marks
            - question_mark_count: number of question marks
            - uppercase_ratio: ratio of uppercase characters to all alphabetic characters
            - sentiment_polarity: simple polarity score in [-1, 1]

        Args:
            text: Input text (ideally already cleaned).

        Returns:
            Dictionary mapping feature names to their values.
        """
        if not isinstance(text, str) or len(text.strip()) == 0:
            return {
                "avg_word_length": 0.0,
                "sentence_count": 0,
                "exclamation_count": 0,
                "question_mark_count": 0,
                "uppercase_ratio": 0.0,
                "sentiment_polarity": 0.0,
            }

        words = text.split()
        word_count = len(words) if words else 1

        # Average word length
        avg_word_length = (
            sum(len(w.strip(string.punctuation)) for w in words) / word_count
        )

        # Sentence count (split on sentence-ending punctuation)
        sentences = re.split(r"[.!?]+", text)
        sentence_count = max(len([s for s in sentences if s.strip()]), 1)

        # Punctuation counts
        exclamation_count = text.count("!")
        question_mark_count = text.count("?")

        # Uppercase ratio
        alpha_chars = [c for c in text if c.isalpha()]
        uppercase_ratio = (
            sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)
            if alpha_chars
            else 0.0
        )

        # Simple sentiment polarity
        lower_words = {w.lower().strip(string.punctuation) for w in words}
        pos_count = len(lower_words & self._positive_words)
        neg_count = len(lower_words & self._negative_words)
        total_sentiment = pos_count + neg_count
        sentiment_polarity = (
            (pos_count - neg_count) / total_sentiment if total_sentiment > 0 else 0.0
        )

        return {
            "avg_word_length": round(avg_word_length, 4),
            "sentence_count": sentence_count,
            "exclamation_count": exclamation_count,
            "question_mark_count": question_mark_count,
            "uppercase_ratio": round(uppercase_ratio, 4),
            "sentiment_polarity": round(sentiment_polarity, 4),
        }

    # ------------------------------------------------------------------
    # BERT tokenization
    # ------------------------------------------------------------------

    def tokenize_for_bert(
        self, texts: List[str]
    ) -> Dict[str, "torch.Tensor"]:  # noqa: F821
        """Tokenize a list of texts for BERT using the HuggingFace tokenizer.

        Args:
            texts: List of text strings to tokenize.

        Returns:
            Dictionary with 'input_ids', 'attention_mask', and 'token_type_ids' tensors.
        """
        encodings = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return encodings

    # ------------------------------------------------------------------
    # Full pipeline
    # ------------------------------------------------------------------

    def process_dataset(
        self,
        df: pd.DataFrame,
        text_column: str = "text",
    ) -> Tuple[pd.DataFrame, Optional[Dict]]:
        """Run the full preprocessing pipeline: clean, extract features, tokenize.

        Args:
            df: Input DataFrame containing at least the text column.
            text_column: Name of the column holding raw text.

        Returns:
            A tuple of:
                - DataFrame with cleaned text and linguistic feature columns added.
                - Dictionary of BERT tokenization outputs (tensors).
        """
        result_df = df.copy()

        # 1. Clean text
        result_df["cleaned_text"] = result_df[text_column].apply(self.clean_text)

        # 2. Extract linguistic features
        features = result_df["cleaned_text"].apply(self.extract_linguistic_features)
        features_df = pd.DataFrame(features.tolist())
        result_df = pd.concat([result_df, features_df], axis=1)

        # 3. Tokenize for BERT
        cleaned_texts = result_df["cleaned_text"].tolist()
        encodings = self.tokenize_for_bert(cleaned_texts)

        return result_df, encodings
