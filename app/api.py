"""
COVID-19 Fake News Detector — Flask REST API

Endpoints
---------
POST /predict   Classify a text as fake or real.
GET  /health    Health-check / readiness probe.

Fixes vs. original
------------------
- LABEL_MAP corrected: 0=real, 1=fake  (matches run_pipeline.py)
- Model loading uses .pkl (pickle) files, not .joblib
- LinearSVC fallback uses decision_function → sigmoid (no predict_proba)
- DistilBERT loading: tries distilbert_best/ (HF), then distilbert_model.pt
"""

import os
import pickle
import re
import sys
from pathlib import Path

import numpy as np
from flask import Flask, jsonify, request
from flask_cors import CORS

# ─── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models"

# run_pipeline.py: label_map = {'real': 0, 'fake': 1}
LABEL_MAP = {0: "real", 1: "fake"}

# ─── optional imports ─────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DistilBertModel,
        DistilBertTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

# ─── Flask app ────────────────────────────────────────────────────────────────
app = Flask(__name__)
CORS(app)

_model      = None
_tokenizer  = None
_vectorizer = None
_model_type = None   # "lr" | "svm" | "distilbert_hf" | "distilbert_custom"


# ─── custom DistilBERT architecture (mirrors run_pipeline.py) ─────────────────
if TRANSFORMERS_AVAILABLE:
    class _DistilBERTClassifier(nn.Module):
        def __init__(self, model_name="distilbert-base-uncased",
                     num_classes=2, dropout=0.3):
            super().__init__()
            self.bert    = DistilBertModel.from_pretrained(model_name)
            self.dropout = nn.Dropout(dropout)
            self.fc1     = nn.Linear(768, 256)
            self.fc2     = nn.Linear(256, num_classes)
            self.relu    = nn.ReLU()

        def forward(self, input_ids, attention_mask):
            out = self.bert(input_ids=input_ids, attention_mask=attention_mask)
            cls = out.last_hidden_state[:, 0, :]
            x   = self.dropout(cls)
            x   = self.relu(self.fc1(x))
            x   = self.dropout(x)
            return self.fc2(x)


# ─── text utilities ───────────────────────────────────────────────────────────
def clean_text(text: str) -> str:
    text = re.sub(r"http\S+|www\.\S+", "", text)
    text = re.sub(r"<.*?>", "", text)
    return re.sub(r"\s+", " ", text).strip()


def extract_linguistic_features(text: str) -> dict:
    words     = text.split()
    sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    aw_len    = float(np.mean([len(w) for w in words])) if words else 0.0
    clickbait = ["breaking", "exposed", "shocking", "secret", "conspiracy",
                 "urgent", "banned", "miracle", "cure",
                 "they don't want you to know"]
    return {
        "word_count":       len(words),
        "sentence_count":   len(sentences),
        "avg_word_length":  round(aw_len, 2),
        "exclamation_marks": text.count("!"),
        "question_marks":   text.count("?"),
        "uppercase_ratio":  round(sum(c.isupper() for c in text) / max(len(text), 1), 4),
        "clickbait_score":  sum(kw in text.lower() for kw in clickbait),
    }


# ─── model loading ────────────────────────────────────────────────────────────
def _pickle_load(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _first_existing(candidates):
    for name in candidates:
        p = MODEL_DIR / name
        if p.exists():
            return p
    return None


def load_model() -> None:
    global _model, _tokenizer, _vectorizer, _model_type

    # 1. DistilBERT HuggingFace format
    if TRANSFORMERS_AVAILABLE:
        hf_path = MODEL_DIR / "distilbert_best"
        if hf_path.exists():
            try:
                _tokenizer  = AutoTokenizer.from_pretrained(str(hf_path))
                _model      = AutoModelForSequenceClassification.from_pretrained(str(hf_path))
                _model.eval()
                _model_type = "distilbert_hf"
                print("[API] DistilBERT (HF) loaded.")
                return
            except Exception as exc:
                print(f"[API] Could not load distilbert_best: {exc}")

    # 2. DistilBERT custom checkpoint
    if TRANSFORMERS_AVAILABLE:
        ckpt_path = MODEL_DIR / "distilbert_model.pt"
        if ckpt_path.exists():
            try:
                ckpt       = torch.load(str(ckpt_path), map_location="cpu")
                model_name = ckpt.get("model_name", "distilbert-base-uncased")
                _model     = _DistilBERTClassifier(model_name)
                _model.load_state_dict(ckpt["model_state_dict"])
                _model.eval()
                tok_path   = MODEL_DIR / "tokenizer"
                _tokenizer = (
                    DistilBertTokenizer.from_pretrained(str(tok_path))
                    if tok_path.exists()
                    else DistilBertTokenizer.from_pretrained(model_name)
                )
                _model_type = "distilbert_custom"
                print("[API] DistilBERT (custom checkpoint) loaded.")
                return
            except Exception as exc:
                print(f"[API] Could not load distilbert_model.pt: {exc}")

    # 3. Logistic Regression (has predict_proba)
    vec_path = _first_existing(["tfidf_vectorizer.pkl", "tfidf_vectorizer.joblib"])
    lr_path  = _first_existing(["lr_model.pkl", "logistic_regression.pkl"])
    if vec_path and lr_path:
        _vectorizer = _pickle_load(vec_path)
        _model      = _pickle_load(lr_path)
        _model_type = "lr"
        print("[API] Logistic Regression loaded.")
        return

    # 4. SVM / LinearSVC fallback
    svm_path = _first_existing(["svm_model.pkl", "svm_linearsvc.pkl"])
    if vec_path and svm_path:
        _vectorizer = _pickle_load(vec_path)
        _model      = _pickle_load(svm_path)
        _model_type = "svm"
        print("[API] LinearSVC loaded.")
        return

    print("[API] WARNING: No model found. /predict will return errors.")


# ─── prediction ───────────────────────────────────────────────────────────────
def predict(text: str) -> dict:
    if _model is None:
        raise RuntimeError(
            "No model loaded. Train models first (run run_pipeline.py) "
            "or place model files in the 'models/' directory."
        )

    cleaned  = clean_text(text)
    features = extract_linguistic_features(text)

    if _model_type in ("distilbert_hf", "distilbert_custom"):
        inputs = _tokenizer(
            cleaned, return_tensors="pt",
            truncation=True, max_length=128, padding=True,
        )
        with torch.no_grad():
            if _model_type == "distilbert_hf":
                logits = _model(**inputs).logits
            else:
                logits = _model(inputs["input_ids"], inputs["attention_mask"])
        probs      = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
        prediction = int(np.argmax(probs))
        confidence = float(probs[prediction])

    elif _model_type == "lr":
        vec        = _vectorizer.transform([cleaned])
        prediction = int(_model.predict(vec)[0])
        probs      = _model.predict_proba(vec)[0]
        confidence = float(probs.max())

    else:  # svm / LinearSVC
        vec        = _vectorizer.transform([cleaned])
        prediction = int(_model.predict(vec)[0])
        dec        = _model.decision_function(vec)[0]
        p_pos      = float(1.0 / (1.0 + np.exp(-dec)))
        confidence = max(p_pos, 1.0 - p_pos)

    return {
        "prediction": LABEL_MAP.get(prediction, "unknown"),
        "confidence": round(confidence, 4),
        "model":      _model_type,
        "features":   features,
    }


# ─── routes ───────────────────────────────────────────────────────────────────
@app.route("/predict", methods=["POST"])
def predict_route():
    """Classify text.

    Request JSON::

        {"text": "…"}

    Response JSON::

        {"prediction": "fake"|"real", "confidence": 0.95,
         "model": "lr", "features": {…}}
    """
    data = request.get_json(silent=True)
    if not data or "text" not in data:
        return jsonify({"error": "Request must contain JSON with a 'text' field."}), 400
    text = data["text"].strip()
    if not text:
        return jsonify({"error": "'text' must not be empty."}), 400
    try:
        return jsonify(predict(text)), 200
    except RuntimeError as exc:
        return jsonify({"error": str(exc)}), 503
    except Exception as exc:
        return jsonify({"error": f"Prediction failed: {exc}"}), 500


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status":     "ready" if _model is not None else "no_model",
        "model_type": _model_type,
    }), 200


# ─── entry point ──────────────────────────────────────────────────────────────
if __name__ == "__main__":
    load_model()
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
