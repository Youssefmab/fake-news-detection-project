"""
Unit tests for text processing utilities.

Run with:
    pytest tests/test_text_processor.py -v
"""

import re
import sys
from pathlib import Path

import numpy as np
import pytest

# ---------------------------------------------------------------------------
# Ensure project root is on sys.path so we can import helpers.
# The functions under test are defined inline here for portability, but they
# mirror the implementations used in the Streamlit app and the API.
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ═══════════════════════════════════════════════════════════════════════════
# Functions under test (self-contained copies for test isolation)
# ═══════════════════════════════════════════════════════════════════════════
def clean_text(text: str) -> str:
    """Remove URLs, HTML tags, and excessive whitespace."""
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def extract_linguistic_features(text: str) -> dict:
    """Extract simple linguistic features from raw text."""
    words = text.split()
    sentences = re.split(r"[.!?]+", text)
    sentences = [s.strip() for s in sentences if s.strip()]

    exclamation_count = text.count("!")
    question_count = text.count("?")
    uppercase_ratio = sum(1 for c in text if c.isupper()) / max(len(text), 1)
    avg_word_length = float(np.mean([len(w) for w in words])) if words else 0.0

    clickbait_keywords = [
        "breaking", "exposed", "shocking", "secret", "conspiracy",
        "urgent", "banned", "miracle", "cure", "they don't want you to know",
    ]
    clickbait_score = sum(1 for kw in clickbait_keywords if kw in text.lower())

    return {
        "word_count": len(words),
        "sentence_count": len(sentences),
        "avg_word_length": round(avg_word_length, 2),
        "exclamation_marks": exclamation_count,
        "question_marks": question_count,
        "uppercase_ratio": round(uppercase_ratio, 4),
        "clickbait_score": clickbait_score,
    }


def tokenize_for_bert(text: str, max_length: int = 128):
    """
    Simulate BERT tokenization output shapes using a simple whitespace
    tokenizer when transformers is not available, or a real tokenizer when
    it is.
    """
    try:
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        encoding = tokenizer(
            text,
            return_tensors="np",
            truncation=True,
            max_length=max_length,
            padding="max_length",
        )
        return {
            "input_ids": encoding["input_ids"],
            "attention_mask": encoding["attention_mask"],
        }
    except Exception:
        # Fallback: return numpy arrays with the correct shapes
        words = text.split()[:max_length]
        ids = np.array([[hash(w) % 30000 for w in words] + [0] * (max_length - len(words))])
        mask = np.array([[1] * len(words) + [0] * (max_length - len(words))])
        return {"input_ids": ids, "attention_mask": mask}


# ═══════════════════════════════════════════════════════════════════════════
# Tests — clean_text
# ═══════════════════════════════════════════════════════════════════════════
class TestCleanText:
    """Tests for the clean_text function."""

    def test_removes_http_urls(self):
        text = "Check this link http://example.com for more info."
        result = clean_text(text)
        assert "http" not in result
        assert "example.com" not in result

    def test_removes_https_urls(self):
        text = "Visit https://www.example.org/page?q=1 now!"
        result = clean_text(text)
        assert "https" not in result
        assert "example.org" not in result

    def test_removes_www_urls(self):
        text = "Go to www.fake-news-site.com for details."
        result = clean_text(text)
        assert "www." not in result

    def test_removes_html_tags(self):
        text = "<p>This is <b>bold</b> text.</p>"
        result = clean_text(text)
        assert "<" not in result
        assert ">" not in result
        assert "bold" in result

    def test_collapses_whitespace(self):
        text = "Too   many     spaces   here."
        result = clean_text(text)
        assert "  " not in result

    def test_strips_leading_trailing_whitespace(self):
        text = "   hello world   "
        result = clean_text(text)
        assert result == "hello world"

    def test_combined(self):
        text = "  <div>Visit http://x.com  for   info</div>  "
        result = clean_text(text)
        assert result == "Visit for info"

    def test_empty_string(self):
        assert clean_text("") == ""


# ═══════════════════════════════════════════════════════════════════════════
# Tests — extract_linguistic_features
# ═══════════════════════════════════════════════════════════════════════════
class TestExtractLinguisticFeatures:
    """Tests for the extract_linguistic_features function."""

    def test_returns_dict(self):
        result = extract_linguistic_features("Hello world.")
        assert isinstance(result, dict)

    def test_expected_keys(self):
        result = extract_linguistic_features("Sample text.")
        expected_keys = {
            "word_count",
            "sentence_count",
            "avg_word_length",
            "exclamation_marks",
            "question_marks",
            "uppercase_ratio",
            "clickbait_score",
        }
        assert set(result.keys()) == expected_keys

    def test_word_count(self):
        result = extract_linguistic_features("one two three four")
        assert result["word_count"] == 4

    def test_sentence_count(self):
        result = extract_linguistic_features("First sentence. Second one! Third?")
        assert result["sentence_count"] == 3

    def test_exclamation_marks(self):
        result = extract_linguistic_features("Wow! Amazing! Incredible!")
        assert result["exclamation_marks"] == 3

    def test_question_marks(self):
        result = extract_linguistic_features("Really? Are you sure? Why?")
        assert result["question_marks"] == 3

    def test_uppercase_ratio_type(self):
        result = extract_linguistic_features("HELLO world")
        assert isinstance(result["uppercase_ratio"], float)
        assert 0.0 <= result["uppercase_ratio"] <= 1.0

    def test_clickbait_score(self):
        result = extract_linguistic_features(
            "BREAKING: Shocking conspiracy exposed!"
        )
        assert result["clickbait_score"] >= 3  # breaking, shocking, conspiracy, exposed

    def test_avg_word_length_type(self):
        result = extract_linguistic_features("hi there friend")
        assert isinstance(result["avg_word_length"], float)

    def test_empty_string(self):
        result = extract_linguistic_features("")
        assert result["word_count"] == 0
        assert result["avg_word_length"] == 0.0


# ═══════════════════════════════════════════════════════════════════════════
# Tests — tokenize_for_bert
# ═══════════════════════════════════════════════════════════════════════════
class TestTokenizeForBert:
    """Tests for the tokenize_for_bert function."""

    def test_output_keys(self):
        result = tokenize_for_bert("Hello world", max_length=32)
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_output_shapes(self):
        max_len = 64
        result = tokenize_for_bert("A short sentence.", max_length=max_len)
        assert result["input_ids"].shape == (1, max_len)
        assert result["attention_mask"].shape == (1, max_len)

    def test_dtype_is_numeric(self):
        result = tokenize_for_bert("Test input", max_length=16)
        assert np.issubdtype(result["input_ids"].dtype, np.integer) or \
               np.issubdtype(result["input_ids"].dtype, np.floating)

    def test_attention_mask_values(self):
        result = tokenize_for_bert("word", max_length=16)
        mask = result["attention_mask"].flatten()
        # At least one token should be attended to
        assert mask.sum() >= 1
        # All values should be 0 or 1
        assert set(mask.tolist()).issubset({0, 1})
