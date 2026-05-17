# Détection de Fake News COVID-19 par Apprentissage Automatique

> **Niveau :** 4CS
> **Encadrant :** Prof. Salem Trabelsi

## Description

Ce projet vise à développer un système de détection automatique de fausses
informations (fake news) liées à la pandémie de COVID-19. En combinant des
techniques classiques de machine learning (SVM, TF-IDF) avec des modèles de
langage pré-entraînés (BERT, RoBERTa), le système est capable de classifier
un texte comme étant **réel** ou **faux** avec un haut niveau de confiance.

Le projet comprend :

- Une analyse exploratoire approfondie du jeu de données.
- Un pipeline de prétraitement et d'extraction de caractéristiques linguistiques.
- L'entraînement et l'évaluation de plusieurs modèles (SVM, BERT, RoBERTa).
- Des visualisations d'explicabilité (LIME, SHAP, attention).
- Une application Streamlit interactive et une API REST Flask.

## Structure du projet

```
fake-news-detection-project/
│
├── README.md                              ← Ce fichier
├── LICENSE                                ← Licence MIT
├── requirements.txt                       ← Dépendances Python
├── download_dataset.py                    ← Script de téléchargement du dataset
├── Presentation_Pipeline_FakeNews_COVID19.pptx  ← Présentation PowerPoint
│
├── data/
│   ├── README.md                          ← Documentation du jeu de données
│   ├── raw/                               ← Données brutes (CSV — CONSTRAINT dataset)
│   └── processed/                         ← Données nettoyées + métadonnées
│
├── notebooks/
│   ├── 00_pipeline_complet.ipynb          ← ★ NOTEBOOK COMPLET (soumission)
│   ├── 01_exploration.ipynb               ← Analyse exploratoire (EDA)
│   ├── 02_preprocessing.ipynb             ← Nettoyage et prétraitement
│   ├── 03_baseline_models.ipynb           ← SVM, LR, RF, XGBoost + TF-IDF
│   ├── 03b_baseline_improvements.ipynb    ← Améliorations baselines
│   ├── 04_advanced_models.ipynb           ← Fine-tuning DistilBERT / BERT
│   ├── 05_evaluation.ipynb                ← Évaluation complète (ROC, MCC, McNemar)
│   ├── 06_explainability.ipynb            ← Explicabilité (LIME, SHAP, Attention)
│   └── 06b_pdp_global_explainability.ipynb ← PDP & explicabilité globale
│
├── src/
│   ├── pipeline.py                        ← Pipeline sklearn end-to-end
│   ├── data/text_processor.py             ← Prétraitement de texte
│   ├── models/text_models.py              ← Architectures de modèles
│   ├── evaluation/metrics.py              ← Métriques d'évaluation
│   └── explainability/interpretability.py ← LIME, SHAP, Attention
│
├── app/
│   ├── streamlit_app.py                   ← Application Streamlit interactive
│   └── api.py                             ← API REST Flask
│
├── models/                                ← Modèles sauvegardés
│   ├── *.pkl                              ← Modèles baseline (SVM, LR, RF, XGBoost)
│   ├── tfidf_vectorizer.pkl               ← Vectoriseur TF-IDF
│   ├── distilbert_best/                   ← DistilBERT fine-tuné (config + tokenizer)
│   └── *.csv                              ← Résultats d'évaluation
│
├── reports/
│   ├── figures/                           ← ~50 figures de résultats
│   └── final_results.csv                  ← Tableau récapitulatif
│
└── tests/
    ├── test_text_processor.py             ← Tests unitaires — prétraitement
    ├── test_pipeline.py                   ← Tests unitaires — pipeline
    └── test_metrics.py                    ← Tests unitaires — évaluation
```

## Installation

### Prérequis

- **Python 3.9+** (3.10 ou 3.11 recommandé)
- **pip** (gestionnaire de paquets Python)
- **Git**
- Un compte [Kaggle](https://www.kaggle.com/) (pour le téléchargement des données)

### Étapes

1. **Cloner le dépôt :**

   ```bash
   git clone <URL_DU_DEPOT>
   cd fake-news-detection-project
   ```

2. **Créer un environnement virtuel :**

   ```bash
   python -m venv venv
   source venv/bin/activate        # Linux / macOS
   venv\Scripts\activate           # Windows
   ```

3. **Installer les dépendances :**

   ```bash
   pip install -r requirements.txt
   ```

## Téléchargement du dataset

### Option 1 — Script automatique (Kaggle CLI)

1. Placez votre fichier `kaggle.json` dans `~/.kaggle/` (Linux/macOS) ou
   `C:\Users\<username>\.kaggle\` (Windows). Ce fichier est téléchargeable
   depuis <https://www.kaggle.com/settings> → *API → Create New Token*.

2. Exécutez :

   ```bash
   python download_dataset.py
   ```

### Option 2 — Téléchargement manuel

1. Rendez-vous sur
   <https://www.kaggle.com/datasets/elroyggj/covid19-fake-news-dataset-nlp>.
2. Cliquez sur **Download**.
3. Décompressez l'archive dans `data/raw/`.

## Exécution des notebooks

Les notebooks sont conçus pour être exécutés **dans l'ordre** :

| #   | Notebook                          | Description                               |
| --- | --------------------------------- | ----------------------------------------- |
| 00  | `00_pipeline_complet.ipynb`       | ★ Pipeline complet — notebook de soumission |
| 01  | `01_exploration.ipynb`            | Analyse exploratoire du dataset (EDA)     |
| 02  | `02_preprocessing.ipynb`          | Nettoyage, features linguistiques, BERT   |
| 03  | `03_baseline_models.ipynb`        | SVM, LR, Random Forest, XGBoost + TF-IDF |
| 03b | `03b_baseline_improvements.ipynb` | Améliorations et grid search              |
| 04  | `04_advanced_models.ipynb`        | Fine-tuning DistilBERT / BERT             |
| 05  | `05_evaluation.ipynb`             | Évaluation complète (ROC, MCC, McNemar)   |
| 06  | `06_explainability.ipynb`         | Explicabilité (LIME, SHAP, Attention)     |
| 06b | `06b_pdp_global_explainability.ipynb` | PDP et explicabilité globale          |

Pour lancer Jupyter :

```bash
jupyter notebook
```

Puis ouvrez chaque notebook depuis l'interface web.

## Lancer l'application Streamlit

```bash
streamlit run app/streamlit_app.py
```

L'application sera accessible à l'adresse <http://localhost:8501>.

## Utiliser l'API REST

### Démarrer le serveur

```bash
python app/api.py
```

Le serveur démarre sur le port **5000** par défaut.

### Endpoints

#### `GET /health`

Vérification de l'état du service.

```bash
curl http://localhost:5000/health
```

Réponse :

```json
{
  "status": "ready",
  "model_type": "roberta"
}
```

#### `POST /predict`

Classification d'un texte.

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Breaking news: 5G causes COVID-19!"}'
```

Réponse :

```json
{
  "prediction": "fake",
  "confidence": 0.9732,
  "model": "roberta",
  "features": {
    "word_count": 7,
    "sentence_count": 1,
    "avg_word_length": 5.14,
    "exclamation_marks": 1,
    "question_marks": 0,
    "uppercase_ratio": 0.1212,
    "clickbait_score": 1
  }
}
```

## Technologies utilisées

| Catégorie       | Outils                                    |
| --------------- | ----------------------------------------- |
| ML classique    | scikit-learn, XGBoost                     |
| Deep Learning   | PyTorch, Hugging Face Transformers        |
| NLP             | NLTK, Tokenizers, Sentence-Transformers   |
| Explicabilité   | LIME, SHAP, Captum                        |
| Visualisation   | Matplotlib, Seaborn, Plotly, WordCloud    |
| Application     | Streamlit, Flask                          |
| Tests           | pytest                                    |

## Licence

Ce projet est distribué sous licence **MIT**. Voir le fichier [LICENSE](LICENSE).
