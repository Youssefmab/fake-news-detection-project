"""Build a single consolidated master notebook from the 6 project notebooks."""
import json

nbs = {}
for nb_path in [
    'notebooks/01_exploration.ipynb',
    'notebooks/02_preprocessing.ipynb',
    'notebooks/03_baseline_models.ipynb',
    'notebooks/04_advanced_models.ipynb',
    'notebooks/05_evaluation.ipynb',
    'notebooks/06_explainability.ipynb',
]:
    nbs[nb_path] = json.load(open(nb_path, encoding='utf-8'))


def md(text):
    return {"cell_type": "markdown", "metadata": {}, "source": [text]}


def all_cells(nb_path, skip_first_md=True):
    cells = nbs[nb_path]['cells']
    if skip_first_md and cells and cells[0]['cell_type'] == 'markdown':
        cells = cells[1:]
    return cells


# ── Cover ────────────────────────────────────────────────────────────────────
cover = """# Pipeline Complet — Détection de Fake News COVID-19
## Thème : Sécurité IA — Pipelines des Modèles ML

---

**Niveau :** 4CS
**Encadrant :** Mme Nadia
**Date :** Mai 2026

---

## Résumé

Ce notebook présente le **pipeline end-to-end** de détection de fausses informations COVID-19.
Il couvre chaque étape, de l'exploration des données brutes jusqu'à l'explicabilité des prédictions.

### Architecture du Pipeline

```
Données brutes CSV
      ↓
1. Analyse Exploratoire (EDA)
      ↓
2. Prétraitement & Feature Engineering
      ↓
3. Modèles Baselines : SVM · LR · RF · XGBoost  (TF-IDF)
      ↓
4. Modèles Avancés  : DistilBERT  (Fine-tuning)
      ↓
5. Évaluation complète : ROC · MCC · McNemar
      ↓
6. Explicabilité : LIME · SHAP · Attention
```

### Meilleurs résultats

| Modèle | Accuracy | F1 | MCC |
|---|---|---|---|
| **SVM + TF-IDF** | **93.9 %** | 0.939 | 0.877 |
| DistilBERT | 92.9 % | 0.928 | 0.857 |
| Logistic Regression | 92.6 % | 0.926 | 0.852 |
| XGBoost | 90.4 % | 0.904 | 0.810 |
| Random Forest | 90.1 % | 0.901 | 0.803 |

---
"""

cells = [md(cover)]

# ── Section 1 — EDA ──────────────────────────────────────────────────────────
cells.append(md("""---
# Section 1 — Analyse Exploratoire des Données (EDA)

**Objectif :** Comprendre la structure, la distribution et les caractéristiques textuelles
du dataset COVID-19 Fake News (~10 700 articles classés **fake** / **real**).

**Ce que nous analysons :**
- Distribution des classes et équilibre du dataset
- Distribution des longueurs de texte par classe
- Fréquence des mots les plus discriminants
- Nuages de mots (WordCloud) par classe
- Bigrammes et trigrammes les plus fréquents
- Tests statistiques (Mann-Whitney U, test t de Welch)

---
"""))
cells += all_cells('notebooks/01_exploration.ipynb')

# ── Section 2 — Preprocessing ────────────────────────────────────────────────
cells.append(md("""---
# Section 2 — Prétraitement des Données & Feature Engineering

**Objectif :** Nettoyer les textes bruts et extraire les représentations numériques.

**Pipeline de nettoyage (classe TextCleaner) :**
1. Suppression des URLs, mentions (@), hashtags (#)
2. Mise en minuscules + suppression de la ponctuation
3. Suppression des stop words (NLTK)
4. Lemmatisation (WordNetLemmatizer)

**Features linguistiques extraites :**
- `word_count`, `char_count`, `avg_word_length`, `unique_word_ratio`
- `exclamation_count`, `question_count`, `uppercase_ratio`
- `hashtag_count`, `mention_count`, `url_count`

**Tokenisation BERT :** distribution de longueur des tokens (max = 128)

---
"""))
cells += all_cells('notebooks/02_preprocessing.ipynb')

# ── Section 3 — Baseline Models ──────────────────────────────────────────────
cells.append(md("""---
# Section 3 — Modèles de Base (Baselines)

**Objectif :** Établir des performances de référence avec des classifieurs classiques.

**Représentation des features :**
- TF-IDF : max 10 000 features, unigrammes + bigrammes, sublinear_tf=True
- Features linguistiques numériques normalisées (StandardScaler)
- Combinées via ColumnTransformer (pipeline sklearn)

**Modèles entraînés :**
| Modèle | Hyperparamètre clé | Sélection |
|---|---|---|
| SVM LinearSVC | C ∈ {0.01, 0.1, 1, 5, 10} | F1 sur validation |
| Régression Logistique | C ∈ {0.01, 0.1, 1, 5, 10} | F1 sur validation |
| Random Forest | 200 arbres, max_depth=20 | — |
| XGBoost | 200 estimateurs, lr=0.1 | — |

---
"""))
cells += all_cells('notebooks/03_baseline_models.ipynb')

# ── Section 4 — Advanced Models ──────────────────────────────────────────────
cells.append(md("""---
# Section 4 — Modèles Avancés (Transformers)

**Objectif :** Fine-tuner des modèles pré-entraînés de type Transformer pour tirer
parti du contexte sémantique profond.

**Modèles fine-tunés :**
- **DistilBERT** (`distilbert-base-uncased`) — 40 % plus rapide que BERT, 97 % des performances
- **BERT** (`bert-base-uncased`) — modèle de référence si GPU disponible
- **RoBERTa** — optionnel (meilleure robustesse)

**Hyperparamètres d'entraînement :**
| Paramètre | Valeur |
|---|---|
| Optimizer | AdamW |
| Learning rate | 2e-5 |
| Batch size | 16 |
| Epochs | 3 |
| Max tokens | 128 |
| Scheduler | Linear warmup (10 %) |

---
"""))
cells += all_cells('notebooks/04_advanced_models.ipynb')

# ── Section 5 — Evaluation ───────────────────────────────────────────────────
cells.append(md("""---
# Section 5 — Évaluation Complète des Modèles

**Objectif :** Comparer rigoureusement tous les modèles sur plusieurs axes.

**Métriques calculées :**
| Métrique | Pourquoi |
|---|---|
| Accuracy | Mesure globale |
| F1-Score macro | Équilibre précision/rappel |
| MCC | Robuste aux classes déséquilibrées |
| AUC-ROC | Capacité discriminante globale |
| Average Precision | Utile pour la classe minoritaire |

**Analyses complémentaires :**
- Matrices de confusion côte à côte
- Analyse des erreurs (faux positifs / faux négatifs)
- Analyse de la confiance des prédictions
- Tests de McNemar (significativité statistique entre paires de modèles)

---
"""))
cells += all_cells('notebooks/05_evaluation.ipynb')

# ── Section 6 — Explainability ───────────────────────────────────────────────
cells.append(md("""---
# Section 6 — Explicabilité des Modèles (XAI)

**Objectif :** Interpréter les décisions du modèle pour garantir transparence et confiance.

**Méthodes implémentées :**

| Méthode | Type | Description |
|---|---|---|
| **Attention** | Locale (token) | Poids d'attention BERT par token |
| **LIME** | Locale | Approximation linéaire locale |
| **SHAP** | Globale + locale | Valeurs de Shapley (théorie des jeux) |

**Analyse des biais :**
- Tests de sensibilité à des termes géographiques (Chine, USA, Europe)
- Termes médicaux vs termes de conspiration
- Comparaison des décisions sur des textes quasi-identiques

**Considérations éthiques :**
- Risque de censure et faux positifs à fort impact social
- Biais linguistiques et culturels du corpus d'entraînement
- Limites de généralisation hors domaine COVID-19

---
"""))
cells += all_cells('notebooks/06_explainability.ipynb')

# ── Conclusion ───────────────────────────────────────────────────────────────
cells.append(md("""---
# Conclusion & Synthèse Finale

## Tableau de Bord des Performances

| Modèle | Accuracy | F1 | MCC | AUC-ROC | Temps train |
|---|---|---|---|---|---|
| **SVM + TF-IDF** | **93.9 %** | **0.939** | **0.877** | — | < 1 min |
| DistilBERT | 92.9 % | 0.928 | 0.857 | 0.976 | ~30 min GPU |
| Logistic Regression | 92.6 % | 0.926 | 0.852 | 0.976 | < 1 min |
| XGBoost | 90.4 % | 0.904 | 0.810 | 0.968 | ~5 min |
| Random Forest | 90.1 % | 0.901 | 0.803 | 0.969 | ~3 min |

## Conclusions Clés

1. **SVM + TF-IDF** : meilleure accuracy (93.9 %) à coût computationnel minimal.
   Démontre l'efficacité des représentations bag-of-words sur textes courts spécialisés.

2. **DistilBERT** : comprend le contexte sémantique profond (attention sur les tokens
   médicaux) mais n'apporte pas de gain significatif sur ce dataset.

3. **Les modèles baselines sont compétitifs** pour des corpus à vocabulaire contrôlé
   comme les fake news COVID (jargon médical + marqueurs stylistiques forts).

4. **LIME & SHAP** confirment que les mots-clés médicaux et les structures
   d'urgence/sensationnalisme sont les features les plus discriminantes.

5. **Biais identifiés** : le modèle est plus sensible aux termes chinois (biais corpus)
   — à corriger avant tout déploiement en production.

## Travaux Futurs

- Entraînement multilingue (arabe, français, anglais)
- Intégration de fact-checking APIs (Snopes, PolitiFact)
- Détection en temps réel via l'API Flask + Streamlit déployée
- Modèles multi-modaux (texte + métadonnées sociales)
- Apprentissage contrastif pour améliorer la robustesse cross-domaine
"""))

# ── Build final notebook ──────────────────────────────────────────────────────
master_nb = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3 (ipykernel)",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.14.3",
        },
    },
    "cells": cells,
}

out_path = 'notebooks/00_pipeline_complet.ipynb'
with open(out_path, 'w', encoding='utf-8') as f:
    json.dump(master_nb, f, ensure_ascii=False, indent=1)

code_cells = sum(1 for c in cells if c['cell_type'] == 'code')
md_cells   = sum(1 for c in cells if c['cell_type'] == 'markdown')
print(f"Created : {out_path}")
print(f"Total   : {len(cells)} cells  ({code_cells} code, {md_cells} markdown)")
