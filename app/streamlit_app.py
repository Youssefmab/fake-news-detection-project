"""
COVID-19 Fake News Detector — Streamlit Application

Fixes vs. original:
  - LABEL_MAP corrected: 0=REAL, 1=FAKE  (matches run_pipeline.py label_map)
  - Model loading updated to .pkl files (pickle, not joblib)
  - Baseline uses lr_model.pkl (has predict_proba); falls back to LinearSVC
    via decision_function → sigmoid when only svm_model.pkl is available
  - DistilBERT support: tries distilbert_best/ (HF format) then
    distilbert_model.pt (custom checkpoint)
  - Model availability shown in sidebar
  - Richer result card with risk-level messaging
"""

import pickle
import re
import sys
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn

# ─── paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models"

# ─── optional heavy dependencies ──────────────────────────────────────────────
try:
    from transformers import (
        AutoModelForSequenceClassification,
        AutoTokenizer,
        DistilBertModel,
        DistilBertTokenizer,
    )
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False

try:
    import plotly.graph_objects as go
    PLOTLY_AVAILABLE = True
except ImportError:
    PLOTLY_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

try:
    from lime.lime_text import LimeTextExplainer
    LIME_AVAILABLE = True
except ImportError:
    LIME_AVAILABLE = False

# ─── constants ────────────────────────────────────────────────────────────────
# run_pipeline.py trains with label_map = {'real': 0, 'fake': 1}
LABEL_MAP = {0: "REAL", 1: "FAKE"}
COLOR_MAP  = {"FAKE": "#e74c3c", "REAL": "#2ecc71"}

EXAMPLES = {
    "🔴 Fake — 5G conspiracy": (
        "BREAKING: Scientists confirm that 5G towers are spreading COVID-19 "
        "across the world. Exposed documents show a global conspiracy between "
        "telecom companies and governments."
    ),
    "🟢 Real — WHO pandemic": (
        "The World Health Organization declared COVID-19 a pandemic on March "
        "11, 2020. Governments worldwide implemented public health measures "
        "including social distancing and vaccination campaigns."
    ),
    "🔴 Fake — bleach cure": (
        "Exposed! Drinking bleach can cure COVID-19 in 24 hours according to "
        "top secret government research the mainstream media is hiding from you."
    ),
    "🟢 Real — Pfizer efficacy": (
        "Pfizer and BioNTech announced their COVID-19 vaccine candidate was "
        "more than 90% effective in preventing COVID-19 in participants without "
        "evidence of prior SARS-CoV-2 infection in the first interim analysis."
    ),
}

# Pre-loaded batch news list (id, text, expected label)
BATCH_NEWS: list[dict] = [
    {
        "id": 1,
        "category": "Conspiracy",
        "expected": "FAKE",
        "text": (
            "URGENT: Doctors are hiding the truth! Garlic and lemon juice can "
            "completely cure COVID-19 in just 48 hours. The government has BANNED "
            "this information because it threatens Big Pharma profits."
        ),
    },
    {
        "id": 2,
        "category": "Vaccine science",
        "expected": "REAL",
        "text": (
            "Researchers at Oxford University published findings showing that the "
            "AstraZeneca COVID-19 vaccine provides strong protection against severe "
            "disease and hospitalization. The peer-reviewed study in The Lancet "
            "analyzed data from over 17,000 participants."
        ),
    },
    {
        "id": 3,
        "category": "Microchip hoax",
        "expected": "FAKE",
        "text": (
            "BREAKING: Exposed documents reveal that COVID-19 vaccines contain "
            "microchips designed to track your location and monitor your thoughts. "
            "Whistleblower scientists have been silenced by the deep state."
        ),
    },
    {
        "id": 4,
        "category": "CDC guidance",
        "expected": "REAL",
        "text": (
            "The Centers for Disease Control and Prevention recommends that "
            "individuals wear well-fitting masks in indoor public settings in areas "
            "with high COVID-19 transmission. Vaccination remains the most effective "
            "tool to prevent severe illness, hospitalization, and death."
        ),
    },
    {
        "id": 5,
        "category": "Miracle cure",
        "expected": "FAKE",
        "text": (
            "Secret miracle drug hydroxychloroquine CURES coronavirus 100% of the "
            "time but mainstream media is suppressing the shocking truth. Thousands "
            "of doctors have been banned from sharing this information online."
        ),
    },
    {
        "id": 6,
        "category": "Variant report",
        "expected": "REAL",
        "text": (
            "The Omicron variant of SARS-CoV-2 was first reported to the World "
            "Health Organization from South Africa on 24 November 2021. Early "
            "studies suggest it carries a higher number of mutations in the spike "
            "protein compared to previous variants."
        ),
    },
    {
        "id": 7,
        "category": "5G hoax",
        "expected": "FAKE",
        "text": (
            "EXPOSED: 5G radiation is activating the COVID-19 virus inside your "
            "body right now. Countries without 5G infrastructure have zero cases — "
            "the evidence is being suppressed by telecom billionaires. Share this "
            "before it gets deleted!"
        ),
    },
    {
        "id": 8,
        "category": "Lockdown policy",
        "expected": "REAL",
        "text": (
            "Several European countries reimposed lockdown restrictions in late 2021 "
            "amid a surge in COVID-19 cases driven by the Delta variant. Health "
            "officials urged unvaccinated populations to get inoculated as hospitals "
            "in some regions approached capacity."
        ),
    },
    {
        "id": 9,
        "category": "Population control",
        "expected": "FAKE",
        "text": (
            "Bill Gates admitted in a shocking leaked video that COVID vaccines are "
            "designed for population control and will sterilize millions of people "
            "worldwide. The globalist agenda is finally being exposed by brave "
            "patriots standing up to the New World Order."
        ),
    },
    {
        "id": 10,
        "category": "Booster study",
        "expected": "REAL",
        "text": (
            "A study published in the New England Journal of Medicine found that a "
            "booster dose of the BNT162b2 mRNA vaccine substantially restored "
            "protection against COVID-19 that had waned after the primary two-dose "
            "series, reducing symptomatic infection by approximately 95 percent."
        ),
    },
    {
        "id": 11,
        "category": "Ivermectin claim",
        "expected": "FAKE",
        "text": (
            "Ivermectin DESTROYS COVID-19 in 48 hours but the FDA is banning it to "
            "protect vaccine profits. Doctors who prescribe it are being arrested. "
            "This banned miracle cure is available in veterinary stores — stock up "
            "now before they remove it!"
        ),
    },
    {
        "id": 12,
        "category": "Long COVID research",
        "expected": "REAL",
        "text": (
            "A study by the UK Office for National Statistics found that around "
            "1.9 million people in the United Kingdom reported experiencing long "
            "COVID symptoms lasting more than four weeks after their initial "
            "infection. Fatigue and shortness of breath were the most commonly "
            "reported symptoms."
        ),
    },
]

# ─── custom DistilBERT (matches run_pipeline.py architecture) ─────────────────
if TRANSFORMERS_AVAILABLE:
    class _DistilBERTClassifier(nn.Module):
        def __init__(self, model_name: str = "distilbert-base-uncased",
                     num_classes: int = 2, dropout: float = 0.3) -> None:
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
                 "urgent", "banned", "miracle", "cure", "they don't want you to know"]
    return {
        "word_count":      len(words),
        "sentence_count":  len(sentences),
        "avg_word_length": round(aw_len, 2),
        "exclamation_marks": text.count("!"),
        "question_marks":  text.count("?"),
        "uppercase_ratio": round(sum(c.isupper() for c in text) / max(len(text), 1), 4),
        "clickbait_score": sum(kw in text.lower() for kw in clickbait),
    }


# ─── model loading ────────────────────────────────────────────────────────────
def _pickle_load(path: Path):
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _first_existing(candidates: list[str]) -> "Path | None":
    for name in candidates:
        p = MODEL_DIR / name
        if p.exists():
            return p
    return None


@st.cache_resource(show_spinner="Loading baseline model…")
def load_baseline():
    """
    Return (model, vectorizer, kind) where kind is 'lr' or 'svm'.
    Prefers LR (has predict_proba); falls back to LinearSVC.
    """
    vec_path = _first_existing(["tfidf_vectorizer.pkl", "tfidf_vectorizer.joblib"])
    if vec_path is None:
        return None, None, None
    vectorizer = _pickle_load(vec_path)

    # Prefer LR — it has predict_proba
    lr_path = _first_existing(["lr_model.pkl", "logistic_regression.pkl", "lr_model.joblib"])
    if lr_path:
        return _pickle_load(lr_path), vectorizer, "lr"

    # Fall back to SVM (LinearSVC — no predict_proba, use decision_function)
    svm_path = _first_existing(["svm_model.pkl", "svm_linearsvc.pkl", "svm_model.joblib"])
    if svm_path:
        return _pickle_load(svm_path), vectorizer, "svm"

    return None, None, None


@st.cache_resource(show_spinner="Loading DistilBERT…")
def load_distilbert():
    """
    Return (model, tokenizer, fmt) where fmt is 'hf' or 'custom'.
    Tries distilbert_best/ (HuggingFace AutoModel) first,
    then distilbert_model.pt (custom checkpoint from run_pipeline.py).
    """
    if not TRANSFORMERS_AVAILABLE:
        return None, None, None

    # 1 — HuggingFace SavedModel format
    hf_path = MODEL_DIR / "distilbert_best"
    if hf_path.exists():
        try:
            tok   = AutoTokenizer.from_pretrained(str(hf_path))
            model = AutoModelForSequenceClassification.from_pretrained(str(hf_path))
            model.eval()
            return model, tok, "hf"
        except Exception:
            pass

    # 2 — Custom checkpoint
    ckpt_path = MODEL_DIR / "distilbert_model.pt"
    if ckpt_path.exists():
        try:
            ckpt       = torch.load(str(ckpt_path), map_location="cpu")
            model_name = ckpt.get("model_name", "distilbert-base-uncased")
            model      = _DistilBERTClassifier(model_name)
            model.load_state_dict(ckpt["model_state_dict"])
            model.eval()

            tok_path = MODEL_DIR / "tokenizer"
            tok = (
                DistilBertTokenizer.from_pretrained(str(tok_path))
                if tok_path.exists()
                else DistilBertTokenizer.from_pretrained(model_name)
            )
            return model, tok, "custom"
        except Exception:
            pass

    return None, None, None


# ─── prediction ───────────────────────────────────────────────────────────────
def predict_baseline(text: str) -> "tuple[str | None, float | None]":
    model, vectorizer, kind = load_baseline()
    if model is None:
        return None, None
    vec = vectorizer.transform([clean_text(text)])

    if kind == "lr" or hasattr(model, "predict_proba"):
        pred  = int(model.predict(vec)[0])
        probs = model.predict_proba(vec)[0]
        conf  = float(probs.max())
    else:  # LinearSVC
        pred  = int(model.predict(vec)[0])
        dec   = model.decision_function(vec)[0]
        p_pos = float(1.0 / (1.0 + np.exp(-dec)))
        conf  = max(p_pos, 1.0 - p_pos)

    return LABEL_MAP.get(pred, "UNKNOWN"), conf


def predict_distilbert(text: str) -> "tuple[str | None, float | None]":
    model, tok, fmt = load_distilbert()
    if model is None:
        return None, None
    inputs = tok(
        clean_text(text), return_tensors="pt",
        truncation=True, max_length=128, padding=True,
    )
    with torch.no_grad():
        logits = model(**inputs).logits if fmt == "hf" else model(
            inputs["input_ids"], inputs["attention_mask"]
        )
    probs = torch.softmax(logits, dim=-1).squeeze().cpu().numpy()
    pred  = int(np.argmax(probs))
    return LABEL_MAP.get(pred, "UNKNOWN"), float(probs[pred])


# ─── LIME predict_proba wrappers ──────────────────────────────────────────────
def _baseline_proba_fn():
    model, vectorizer, kind = load_baseline()
    if model is None:
        return None
    if kind == "lr" or hasattr(model, "predict_proba"):
        def fn(texts):
            return model.predict_proba(vectorizer.transform(texts))
    else:
        def fn(texts):
            decs  = model.decision_function(vectorizer.transform(texts))
            probs = 1.0 / (1.0 + np.exp(-decs))
            return np.column_stack([1.0 - probs, probs])
    return fn


def _distilbert_proba_fn():
    model, tok, fmt = load_distilbert()
    if model is None:
        return None
    def fn(texts):
        results = []
        for t in texts:
            inp = tok(clean_text(t), return_tensors="pt",
                      truncation=True, max_length=128, padding=True)
            with torch.no_grad():
                lgts = model(**inp).logits if fmt == "hf" else model(
                    inp["input_ids"], inp["attention_mask"]
                )
            results.append(torch.softmax(lgts, dim=-1).squeeze().cpu().numpy())
        return np.array(results)
    return fn


# ─── visualization helpers ────────────────────────────────────────────────────
def render_confidence_bar(label: str, confidence: float) -> None:
    color = COLOR_MAP.get(label, "#95a5a6")
    if PLOTLY_AVAILABLE:
        fig = go.Figure(go.Bar(
            x=[confidence], y=["Confidence"], orientation="h",
            marker_color=color,
            text=f"{confidence:.1%}", textposition="auto",
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 1], tickformat=".0%"),
            yaxis=dict(visible=False),
            height=88, margin=dict(l=0, r=0, t=0, b=0),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        w = int(confidence * 100)
        st.markdown(
            f'<div style="background:{color};width:{w}%;padding:9px;'
            f'color:white;border-radius:6px;text-align:center;font-weight:bold;">'
            f"{confidence:.1%}</div>",
            unsafe_allow_html=True,
        )


def render_lime(text: str, proba_fn) -> None:
    if not LIME_AVAILABLE:
        st.info("Install `lime` to see word-level explanations.")
        return
    if not MATPLOTLIB_AVAILABLE:
        st.info("Install `matplotlib` to render LIME charts.")
        return
    try:
        exp = LimeTextExplainer(class_names=["REAL", "FAKE"]).explain_instance(
            text, proba_fn, num_features=10, num_samples=200,
        )
        fig = exp.as_pyplot_figure()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception as exc:
        st.warning(f"LIME unavailable: {exc}")


def render_feature_table(features: dict) -> None:
    import pandas as pd
    st.table(pd.DataFrame(list(features.items()), columns=["Feature", "Value"]))


# ─── model availability check ─────────────────────────────────────────────────
def _model_status() -> dict:
    d = MODEL_DIR
    vec_ok = any((d / n).exists()
                 for n in ["tfidf_vectorizer.pkl", "tfidf_vectorizer.joblib"])
    base_ok = vec_ok and any(
        (d / n).exists()
        for n in ["lr_model.pkl", "logistic_regression.pkl", "svm_model.pkl"]
    )
    bert_ok = (d / "distilbert_best").exists() or (d / "distilbert_model.pt").exists()
    return {"Baseline (LR / SVM)": base_ok, "DistilBERT": bert_ok}


# ─── batch analysis helper ────────────────────────────────────────────────────
def run_batch(items: list[dict], model_choice: str) -> list[dict]:
    """Run predictions on a list of news dicts and return enriched results."""
    results = []
    for item in items:
        text = item["text"]
        if model_choice == "Baseline (LR / SVM)":
            label, conf = predict_baseline(text)
        elif model_choice == "DistilBERT":
            label, conf = predict_distilbert(text)
        else:
            label, conf = None, None

        expected = item.get("expected", "")
        match    = (label == expected) if label else None
        feats    = extract_linguistic_features(text)

        results.append({
            "#":          item["id"],
            "Category":   item.get("category", ""),
            "Text":       text[:90] + "…" if len(text) > 90 else text,
            "Expected":   expected,
            "Predicted":  label or "—",
            "Confidence": f"{conf:.1%}" if conf is not None else "—",
            "Match":      ("✅" if match else "❌") if match is not None else "—",
            "Clickbait":  feats["clickbait_score"],
            "CAPS ratio": f"{feats['uppercase_ratio']:.2%}",
            "_label":     label,
            "_conf":      conf,
            "_match":     match,
        })
    return results


# ─── main app ─────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="COVID-19 Fake News Detector",
        page_icon="🔍",
        layout="wide",
    )

    # ── sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.title("⚙️ Configuration")

        st.markdown("**Model availability**")
        status = _model_status()
        for name, ok in status.items():
            st.markdown(f"{'✅' if ok else '❌'} {name}")

        if not TRANSFORMERS_AVAILABLE:
            st.warning("`transformers` not installed — DistilBERT disabled.")

        st.divider()

        available = [n for n, ok in status.items()
                     if ok and (n != "DistilBERT" or TRANSFORMERS_AVAILABLE)]
        if not available:
            st.error("No models found.  Run `run_pipeline.py` to train.")
            available = ["(none available)"]

        model_choice = st.selectbox("Select model", available)

        st.divider()
        st.markdown(
            "**About**\n\n"
            "Trained on ~7,000 COVID-19 news samples from Kaggle.  "
            "Classifies text as **REAL** or **FAKE** using classical ML "
            "and/or DistilBERT fine-tuning."
        )

    # ── header ───────────────────────────────────────────────────────────────
    st.title("🔍 COVID-19 Fake News Detector")

    tab_single, tab_batch = st.tabs(["📝 Single Analysis", "📋 Batch News List"])

    # ════════════════════════════════════════════════════════════════════════
    # TAB 1 — Single Analysis
    # ════════════════════════════════════════════════════════════════════════
    with tab_single:
        st.markdown(
            "Paste a news article or social-media post and click **Analyze** "
            "to check whether it is likely **real** or **fake**."
        )

        st.subheader("Quick examples")
        cols = st.columns(len(EXAMPLES))
        selected = ""
        for col, (name, content) in zip(cols, EXAMPLES.items()):
            with col:
                if st.button(name, use_container_width=True, key=f"ex_{name}"):
                    selected = content

        user_text = st.text_area(
            "Enter news text",
            value=selected,
            height=175,
            placeholder="Paste a headline, tweet, or article excerpt here…",
        )

        if st.button("🔍 Analyze", type="primary", key="single_analyze"):
            if not user_text.strip():
                st.error("Please enter some text before clicking Analyze.")
            elif model_choice == "(none available)":
                st.error("No model available. Please run `run_pipeline.py` first.")
            else:
                with st.spinner("Analyzing…"):
                    lime_fn = None
                    if model_choice == "Baseline (LR / SVM)":
                        label, confidence = predict_baseline(user_text)
                        lime_fn = _baseline_proba_fn()
                    else:
                        label, confidence = predict_distilbert(user_text)
                        lime_fn = _distilbert_proba_fn()

                if label is None:
                    st.warning(f"**{model_choice}** could not produce a result.")
                else:
                    color = COLOR_MAP.get(label, "#95a5a6")
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.markdown(
                            f"### Verdict: "
                            f'<span style="color:{color};font-size:1.4em;'
                            f'font-weight:900;">{label}</span>',
                            unsafe_allow_html=True,
                        )
                        st.metric("Confidence", f"{confidence:.1%}")
                        if label == "FAKE":
                            if confidence > 0.85:
                                st.error("⚠️ High misinformation probability")
                            elif confidence > 0.65:
                                st.warning("⚠️ Likely fake — verify before sharing")
                            else:
                                st.info("ℹ️ Possibly fake — low confidence")
                        else:
                            if confidence > 0.85:
                                st.success("✅ Appears credible")
                            else:
                                st.info("ℹ️ Likely real — moderate confidence")
                    with col2:
                        st.markdown("**Confidence score**")
                        render_confidence_bar(label, confidence)

                    st.divider()
                    col3, col4 = st.columns(2)
                    with col3:
                        st.subheader("Linguistic features")
                        render_feature_table(extract_linguistic_features(user_text))
                    with col4:
                        st.subheader("Word-level explanation (LIME)")
                        if lime_fn:
                            render_lime(clean_text(user_text), lime_fn)
                        else:
                            st.info("Explanation not available for this model.")

    # ════════════════════════════════════════════════════════════════════════
    # TAB 2 — Batch News List
    # ════════════════════════════════════════════════════════════════════════
    with tab_batch:
        st.markdown(
            "Select news items from the list below and click **Run Batch Analysis** "
            "to classify all of them at once.  You can also add your own custom news."
        )

        # ── pre-loaded news list ─────────────────────────────────────────
        st.subheader("Pre-loaded news list")

        col_sel, col_desel = st.columns([1, 1])
        with col_sel:
            select_all = st.button("☑ Select all", key="sel_all")
        with col_desel:
            deselect_all = st.button("☐ Deselect all", key="desel_all")

        selected_ids: list[int] = []
        for item in BATCH_NEWS:
            badge  = "🔴 FAKE" if item["expected"] == "FAKE" else "🟢 REAL"
            label  = f"**#{item['id']}** · {badge} · *{item['category']}*"
            default = not deselect_all   # True unless "Deselect all" was just clicked
            checked = st.checkbox(
                label,
                value=default,
                key=f"chk_{item['id']}",
                help=item["text"],
            )
            if checked:
                selected_ids.append(item["id"])
            with st.expander(f"Preview #{item['id']}", expanded=False):
                st.write(item["text"])

        st.divider()

        # ── custom news entry ────────────────────────────────────────────
        st.subheader("Add your own news items")
        st.markdown(
            "Enter one news item per line — each line will be analyzed separately."
        )
        custom_input = st.text_area(
            "Custom news (one per line)",
            height=120,
            placeholder=(
                "Paste your own headlines here, one per line…\n"
                "e.g.: Scientists develop new COVID treatment with 80% efficacy\n"
                "e.g.: SHOCKING: Government hiding alien cure for coronavirus!"
            ),
        )

        # ── run button ───────────────────────────────────────────────────
        n_selected = len(selected_ids)
        custom_lines = [l.strip() for l in custom_input.splitlines() if l.strip()]
        n_custom = len(custom_lines)
        total = n_selected + n_custom

        st.markdown(f"**{n_selected}** pre-loaded + **{n_custom}** custom = **{total}** items selected")

        if st.button("▶ Run Batch Analysis", type="primary", key="batch_run",
                     disabled=(total == 0)):
            if model_choice == "(none available)":
                st.error("No model available. Please run `run_pipeline.py` first.")
            else:
                # Build items list
                items_to_run = [
                    item for item in BATCH_NEWS if item["id"] in selected_ids
                ]
                for i, line in enumerate(custom_lines, start=1):
                    items_to_run.append({
                        "id":       f"C{i}",
                        "category": "Custom",
                        "expected": "?",
                        "text":     line,
                    })

                with st.spinner(f"Analyzing {len(items_to_run)} items…"):
                    results = run_batch(items_to_run, model_choice)

                # ── summary metrics ──────────────────────────────────────
                st.subheader("Results")

                scored = [r for r in results if r["_match"] is not None]
                correct = sum(1 for r in scored if r["_match"])
                n_fake  = sum(1 for r in results if r["Predicted"] == "FAKE")
                n_real  = sum(1 for r in results if r["Predicted"] == "REAL")

                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Total analyzed", len(results))
                m2.metric("Detected FAKE",  n_fake)
                m3.metric("Detected REAL",  n_real)
                if scored:
                    m4.metric("Accuracy vs expected",
                              f"{correct}/{len(scored)} ({correct/len(scored):.0%})")
                else:
                    m4.metric("Accuracy vs expected", "N/A")

                st.divider()

                # ── styled results table ─────────────────────────────────
                import pandas as pd

                display_cols = [
                    "#", "Category", "Text", "Expected",
                    "Predicted", "Confidence", "Match",
                    "Clickbait", "CAPS ratio",
                ]
                df = pd.DataFrame(results)[display_cols]

                def _row_style(row):
                    if row["Predicted"] == "FAKE":
                        bg = "background-color: #3d1a1a; color: #ff8080"
                    elif row["Predicted"] == "REAL":
                        bg = "background-color: #1a3d1a; color: #80ff80"
                    else:
                        bg = ""
                    return [bg] * len(row)

                st.dataframe(
                    df.style.apply(_row_style, axis=1),
                    use_container_width=True,
                    hide_index=True,
                )

                # ── per-item detail expanders ────────────────────────────
                st.subheader("Item details")
                for item, res in zip(items_to_run, results):
                    verdict_icon = "🔴" if res["Predicted"] == "FAKE" else "🟢"
                    match_icon   = res["Match"]
                    with st.expander(
                        f"{verdict_icon} #{res['#']} · {res['Category']} · "
                        f"{res['Predicted']} ({res['Confidence']})  {match_icon}",
                        expanded=False,
                    ):
                        st.markdown(f"**Full text:**\n\n{item['text']}")
                        if res["_conf"] is not None:
                            render_confidence_bar(res["Predicted"], res["_conf"])
                        feats = extract_linguistic_features(item["text"])
                        st.table(
                            pd.DataFrame(
                                list(feats.items()), columns=["Feature", "Value"]
                            )
                        )

                # ── CSV download ─────────────────────────────────────────
                csv = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇ Download results as CSV",
                    data=csv,
                    file_name="batch_results.csv",
                    mime="text/csv",
                )


if __name__ == "__main__":
    main()
