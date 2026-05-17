"""
Full pipeline: Preprocessing + Baseline Models + BERT + Evaluation + Explainability
COVID-19 Fake News Detection Project
"""
import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

import pandas as pd
import numpy as np
import re, os, warnings, pickle, time, json
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (classification_report, accuracy_score, f1_score,
                              precision_score, recall_score, confusion_matrix,
                              roc_auc_score, matthews_corrcoef, roc_curve,
                              precision_recall_curve, average_precision_score)
from sklearn.model_selection import cross_val_score
import xgboost as xgb

os.makedirs('data/processed', exist_ok=True)
os.makedirs('models', exist_ok=True)
os.makedirs('reports/figures', exist_ok=True)

# ================================================================
# PHASE 1 & 2: LOAD + PREPROCESS
# ================================================================
print("=" * 60)
print("PHASE 1-2: CHARGEMENT ET PREPROCESSING")
print("=" * 60)

train = pd.read_csv('data/raw/Constraint_Train.csv')
val = pd.read_csv('data/raw/Constraint_Val.csv')
test = pd.read_csv('data/raw/Constraint_Test.csv')

print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")
print(f"Total: {len(train)+len(val)+len(test)} echantillons")
print(f"Classes: {dict(train['label'].value_counts())}")


def clean_text(text):
    if not isinstance(text, str):
        return ""
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def extract_features(text):
    words = text.split()
    sentences = re.split(r'[.!?]+', text)
    return {
        'word_count': len(words),
        'char_count': len(text),
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,
        'exclamation_count': text.count('!'),
        'question_count': text.count('?'),
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),
        'sentence_count': max(len([s for s in sentences if s.strip()]), 1),
        'unique_word_ratio': len(set(words)) / max(len(words), 1),
    }


# Clean
train['clean_text'] = train['tweet'].apply(clean_text)
val['clean_text'] = val['tweet'].apply(clean_text)

# Encode
label_map = {'real': 0, 'fake': 1}
train['label_enc'] = train['label'].map(label_map)
val['label_enc'] = val['label'].map(label_map)

# Extract features
train_feats = pd.DataFrame(train['clean_text'].apply(extract_features).tolist())
val_feats = pd.DataFrame(val['clean_text'].apply(extract_features).tolist())

# Save
train.to_csv('data/processed/train_processed.csv', index=False)
val.to_csv('data/processed/val_processed.csv', index=False)
print("Preprocessing done. Saved to data/processed/\n")

# ================================================================
# PHASE 3: BASELINE MODELS
# ================================================================
print("=" * 60)
print("PHASE 3: MODELES BASELINE")
print("=" * 60)

X_train_text = train['clean_text'].values
X_val_text = val['clean_text'].values
y_train = train['label_enc'].values
y_val = val['label_enc'].values

# TF-IDF
print("\n[1] TF-IDF Vectorization...")
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, max_df=0.95,
                         sublinear_tf=True)
X_train_tfidf = tfidf.fit_transform(X_train_text)
X_val_tfidf = tfidf.transform(X_val_text)
print(f"    Shape: {X_train_tfidf.shape}")

with open('models/tfidf_vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

results = {}
all_preds = {}
all_probs = {}

# --- SVM ---
print("\n[2] SVM (LinearSVC)...")
svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=10000)
svm.fit(X_train_tfidf, y_train)
y_pred = svm.predict(X_val_tfidf)
all_preds['SVM'] = y_pred
results['SVM'] = {
    'Accuracy': accuracy_score(y_val, y_pred),
    'F1': f1_score(y_val, y_pred, average='weighted'),
    'Precision': precision_score(y_val, y_pred, average='weighted'),
    'Recall': recall_score(y_val, y_pred, average='weighted'),
    'MCC': matthews_corrcoef(y_val, y_pred),
}
print(f"    Acc: {results['SVM']['Accuracy']:.4f}, F1: {results['SVM']['F1']:.4f}")
with open('models/svm_model.pkl', 'wb') as f:
    pickle.dump(svm, f)

# --- Logistic Regression ---
print("\n[3] Logistic Regression...")
lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs')
lr.fit(X_train_tfidf, y_train)
y_pred = lr.predict(X_val_tfidf)
y_prob = lr.predict_proba(X_val_tfidf)[:, 1]
all_preds['LR'] = y_pred
all_probs['LR'] = y_prob
results['Logistic Regression'] = {
    'Accuracy': accuracy_score(y_val, y_pred),
    'F1': f1_score(y_val, y_pred, average='weighted'),
    'Precision': precision_score(y_val, y_pred, average='weighted'),
    'Recall': recall_score(y_val, y_pred, average='weighted'),
    'MCC': matthews_corrcoef(y_val, y_pred),
    'AUC-ROC': roc_auc_score(y_val, y_prob),
}
print(f"    Acc: {results['Logistic Regression']['Accuracy']:.4f}, F1: {results['Logistic Regression']['F1']:.4f}, AUC: {results['Logistic Regression']['AUC-ROC']:.4f}")
with open('models/lr_model.pkl', 'wb') as f:
    pickle.dump(lr, f)

# --- Random Forest ---
print("\n[4] Random Forest...")
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)
rf.fit(X_train_tfidf, y_train)
y_pred = rf.predict(X_val_tfidf)
y_prob = rf.predict_proba(X_val_tfidf)[:, 1]
all_preds['RF'] = y_pred
all_probs['RF'] = y_prob
results['Random Forest'] = {
    'Accuracy': accuracy_score(y_val, y_pred),
    'F1': f1_score(y_val, y_pred, average='weighted'),
    'Precision': precision_score(y_val, y_pred, average='weighted'),
    'Recall': recall_score(y_val, y_pred, average='weighted'),
    'MCC': matthews_corrcoef(y_val, y_pred),
    'AUC-ROC': roc_auc_score(y_val, y_prob),
}
print(f"    Acc: {results['Random Forest']['Accuracy']:.4f}, F1: {results['Random Forest']['F1']:.4f}, AUC: {results['Random Forest']['AUC-ROC']:.4f}")

# --- XGBoost ---
print("\n[5] XGBoost...")
scale = np.sum(y_train == 0) / np.sum(y_train == 1)
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                                scale_pos_weight=scale, eval_metric='logloss',
                                random_state=42)
xgb_model.fit(X_train_tfidf, y_train)
y_pred = xgb_model.predict(X_val_tfidf)
y_prob = xgb_model.predict_proba(X_val_tfidf)[:, 1]
all_preds['XGB'] = y_pred
all_probs['XGB'] = y_prob
results['XGBoost'] = {
    'Accuracy': accuracy_score(y_val, y_pred),
    'F1': f1_score(y_val, y_pred, average='weighted'),
    'Precision': precision_score(y_val, y_pred, average='weighted'),
    'Recall': recall_score(y_val, y_pred, average='weighted'),
    'MCC': matthews_corrcoef(y_val, y_pred),
    'AUC-ROC': roc_auc_score(y_val, y_prob),
}
print(f"    Acc: {results['XGBoost']['Accuracy']:.4f}, F1: {results['XGBoost']['F1']:.4f}, AUC: {results['XGBoost']['AUC-ROC']:.4f}")

# Cross-validation
print("\n[6] 5-Fold Cross-Validation (SVM)...")
cv_scores = cross_val_score(LinearSVC(C=1.0, class_weight='balanced', max_iter=10000),
                             X_train_tfidf, y_train, cv=5, scoring='f1_weighted')
print(f"    CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# Results table
print("\n" + "-" * 60)
print("RESULTATS BASELINES:")
print("-" * 60)
results_df = pd.DataFrame(results).T
print(results_df.round(4).to_string())
results_df.to_csv('reports/baseline_results.csv')

# ================================================================
# PHASE 3 FIGURES
# ================================================================

# Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))
model_names = ['SVM', 'LR', 'RF', 'XGB']
display_names = ['SVM', 'Logistic Reg.', 'Random Forest', 'XGBoost']
for ax, key, dname in zip(axes, model_names, display_names):
    cm = confusion_matrix(y_val, all_preds[key])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])
    acc = accuracy_score(y_val, all_preds[key])
    ax.set_title(f'{dname}\n(Acc: {acc:.3f})', fontsize=11)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
plt.tight_layout()
plt.savefig('reports/figures/06_confusion_matrices_baselines.png', dpi=150, bbox_inches='tight')
plt.close()

# Model comparison
fig, ax = plt.subplots(figsize=(10, 6))
metrics_list = ['Accuracy', 'F1', 'Precision', 'Recall']
x = np.arange(len(metrics_list))
width = 0.18
colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
for i, (model_name, metrics) in enumerate(results.items()):
    values = [metrics[m] for m in metrics_list]
    ax.bar(x + i * width, values, width, label=model_name, color=colors_bar[i])
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels(metrics_list, fontsize=12)
ax.set_ylim(0.7, 1.0)
ax.set_title('Comparaison des Modeles Baseline', fontsize=14)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/07_baseline_comparison.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))
roc_colors = ['#e74c3c', '#2ecc71', '#f39c12']
for (name, prob), color in zip(all_probs.items(), roc_colors):
    display = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'XGB': 'XGBoost'}[name]
    fpr, tpr, _ = roc_curve(y_val, prob)
    auc_val = roc_auc_score(y_val, prob)
    ax.plot(fpr, tpr, label=f'{display} (AUC={auc_val:.3f})', color=color, linewidth=2)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Courbes ROC - Modeles Baseline', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/08_roc_curves_baselines.png', dpi=150, bbox_inches='tight')
plt.close()

# Feature importance (LR)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
feature_names = np.array(tfidf.get_feature_names_out())
coefs = lr.coef_[0]
top_real_idx = np.argsort(coefs)[:20]
top_fake_idx = np.argsort(coefs)[-20:][::-1]
axes[0].barh(range(20), coefs[top_real_idx], color='#2ecc71')
axes[0].set_yticks(range(20))
axes[0].set_yticklabels(feature_names[top_real_idx])
axes[0].set_title('Top 20 Mots -> REAL', fontsize=13)
axes[0].invert_yaxis()
axes[1].barh(range(20), coefs[top_fake_idx], color='#e74c3c')
axes[1].set_yticks(range(20))
axes[1].set_yticklabels(feature_names[top_fake_idx])
axes[1].set_title('Top 20 Mots -> FAKE', fontsize=13)
axes[1].invert_yaxis()
plt.tight_layout()
plt.savefig('reports/figures/09_feature_importance_lr.png', dpi=150, bbox_inches='tight')
plt.close()

print("\n[OK] Baseline figures saved (06-09)")

# ================================================================
# PHASE 4: BERT MODEL
# ================================================================
print("\n" + "=" * 60)
print("PHASE 4: MODELE BERT (Fine-tuning)")
print("=" * 60)

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import (DistilBertTokenizer, DistilBertModel,
                           get_linear_schedule_with_warmup)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Use DistilBERT (faster on CPU, good performance)
MODEL_NAME = 'distilbert-base-uncased'
print(f"Model: {MODEL_NAME}")

print("\n[1] Loading tokenizer...")
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)

MAX_LEN = 128
BATCH_SIZE = 32
EPOCHS = 3
LR = 2e-5


class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            self.texts[idx],
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }


class DistilBERTClassifier(nn.Module):
    def __init__(self, model_name, num_classes=2, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(768, 256)
        self.fc2 = nn.Linear(256, num_classes)
        self.relu = nn.ReLU()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]
        x = self.dropout(cls_output)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        logits = self.fc2(x)
        return logits


print("[2] Creating datasets...")
train_dataset = NewsDataset(train['clean_text'].tolist(), y_train.tolist(), tokenizer, MAX_LEN)
val_dataset = NewsDataset(val['clean_text'].tolist(), y_val.tolist(), tokenizer, MAX_LEN)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print("[3] Loading model...")
model = DistilBERTClassifier(MODEL_NAME).to(device)

# Class weights for imbalanced data
class_counts = np.bincount(y_train)
class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float).to(device)
class_weights = class_weights / class_weights.sum() * 2

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),
                                             num_training_steps=total_steps)

print(f"[4] Training for {EPOCHS} epochs...")
train_losses = []
val_losses = []
val_accs = []
best_f1 = 0
best_model_state = None

for epoch in range(EPOCHS):
    # Train
    model.train()
    epoch_loss = 0
    correct = 0
    total = 0
    start_time = time.time()

    for batch_idx, batch in enumerate(train_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        epoch_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        if (batch_idx + 1) % 50 == 0:
            print(f"    Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    train_acc = correct / total
    avg_train_loss = epoch_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # Validate
    model.eval()
    val_loss = 0
    all_val_preds = []
    all_val_probs_bert = []

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            all_val_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_val_probs_bert.extend(probs[:, 1].cpu().numpy())

    avg_val_loss = val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    val_acc = accuracy_score(y_val, all_val_preds)
    val_f1 = f1_score(y_val, all_val_preds, average='weighted')
    val_accs.append(val_acc)

    elapsed = time.time() - start_time
    print(f"\n  Epoch {epoch+1}/{EPOCHS} ({elapsed:.0f}s):")
    print(f"    Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"    Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")

    if val_f1 > best_f1:
        best_f1 = val_f1
        best_model_state = model.state_dict().copy()
        print(f"    -> New best model (F1: {best_f1:.4f})")

# Restore best model
if best_model_state:
    model.load_state_dict(best_model_state)

# Save model
torch.save({
    'model_state_dict': model.state_dict(),
    'model_name': MODEL_NAME,
    'max_len': MAX_LEN,
    'num_classes': 2,
}, 'models/distilbert_model.pt')
print(f"\n[OK] Model saved to models/distilbert_model.pt")

# Final evaluation
model.eval()
all_val_preds_bert = []
all_val_probs_bert = []
with torch.no_grad():
    for batch in val_loader:
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        logits = model(input_ids, attention_mask)
        probs = torch.softmax(logits, dim=1)
        all_val_preds_bert.extend(logits.argmax(dim=1).cpu().numpy())
        all_val_probs_bert.extend(probs[:, 1].cpu().numpy())

all_val_preds_bert = np.array(all_val_preds_bert)
all_val_probs_bert = np.array(all_val_probs_bert)

bert_results = {
    'Accuracy': accuracy_score(y_val, all_val_preds_bert),
    'F1': f1_score(y_val, all_val_preds_bert, average='weighted'),
    'Precision': precision_score(y_val, all_val_preds_bert, average='weighted'),
    'Recall': recall_score(y_val, all_val_preds_bert, average='weighted'),
    'MCC': matthews_corrcoef(y_val, all_val_preds_bert),
    'AUC-ROC': roc_auc_score(y_val, all_val_probs_bert),
}
results['DistilBERT'] = bert_results
print(f"\nDistilBERT -> Acc: {bert_results['Accuracy']:.4f}, F1: {bert_results['F1']:.4f}, AUC: {bert_results['AUC-ROC']:.4f}")

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(range(1, EPOCHS+1), train_losses, 'b-o', label='Train Loss')
axes[0].plot(range(1, EPOCHS+1), val_losses, 'r-o', label='Val Loss')
axes[0].set_title('Learning Curves - Loss', fontsize=13)
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(range(1, EPOCHS+1), val_accs, 'g-o', label='Val Accuracy')
axes[1].set_title('Validation Accuracy', fontsize=13)
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()
axes[1].grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/10_bert_training_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Figure 10 saved")

# ================================================================
# PHASE 5: COMPREHENSIVE EVALUATION
# ================================================================
print("\n" + "=" * 60)
print("PHASE 5: EVALUATION COMPLETE")
print("=" * 60)

# Final results table
print("\n" + "-" * 60)
print("RESULTATS FINAUX - TOUS LES MODELES:")
print("-" * 60)
final_df = pd.DataFrame(results).T
print(final_df.round(4).to_string())
final_df.to_csv('reports/final_results.csv')

# Classification reports
print("\n--- Classification Report: Best Baseline (LR) ---")
print(classification_report(y_val, all_preds['LR'], target_names=['Real', 'Fake']))

print("\n--- Classification Report: DistilBERT ---")
print(classification_report(y_val, all_val_preds_bert, target_names=['Real', 'Fake']))

# Confusion matrices side by side (best baseline vs BERT)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, preds_arr, title in [
    (axes[0], all_preds['LR'], 'Logistic Regression'),
    (axes[1], all_val_preds_bert, 'DistilBERT')
]:
    cm = confusion_matrix(y_val, preds_arr)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],
                annot_kws={'size': 14})
    acc = accuracy_score(y_val, preds_arr)
    f1_val = f1_score(y_val, preds_arr, average='weighted')
    ax.set_title(f'{title}\nAcc: {acc:.3f}, F1: {f1_val:.3f}', fontsize=13)
    ax.set_xlabel('Predicted', fontsize=11)
    ax.set_ylabel('Actual', fontsize=11)
plt.tight_layout()
plt.savefig('reports/figures/11_confusion_best_vs_bert.png', dpi=150, bbox_inches='tight')
plt.close()

# ROC comparison: all models
fig, ax = plt.subplots(figsize=(9, 7))
all_roc = {
    'Logistic Regression': all_probs['LR'],
    'Random Forest': all_probs['RF'],
    'XGBoost': all_probs['XGB'],
    'DistilBERT': all_val_probs_bert,
}
roc_colors_all = ['#e74c3c', '#2ecc71', '#f39c12', '#3498db']
for (name, prob), color in zip(all_roc.items(), roc_colors_all):
    fpr, tpr, _ = roc_curve(y_val, prob)
    auc_val = roc_auc_score(y_val, prob)
    lw = 3 if name == 'DistilBERT' else 2
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', color=color, linewidth=lw)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Courbes ROC - Tous les Modeles', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/12_roc_all_models.png', dpi=150, bbox_inches='tight')
plt.close()

# Precision-Recall curves
fig, ax = plt.subplots(figsize=(9, 7))
for (name, prob), color in zip(all_roc.items(), roc_colors_all):
    prec_curve, rec_curve, _ = precision_recall_curve(y_val, prob)
    ap = average_precision_score(y_val, prob)
    lw = 3 if name == 'DistilBERT' else 2
    ax.plot(rec_curve, prec_curve, label=f'{name} (AP={ap:.3f})', color=color, linewidth=lw)
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Courbes Precision-Recall', fontsize=14)
ax.legend(fontsize=11)
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/13_precision_recall_curves.png', dpi=150, bbox_inches='tight')
plt.close()

# Final heatmap comparison
fig, ax = plt.subplots(figsize=(10, 5))
metrics_for_heatmap = ['Accuracy', 'F1', 'Precision', 'Recall', 'MCC']
heatmap_data = final_df[metrics_for_heatmap].astype(float)
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,
            linewidths=0.5, vmin=0.7, vmax=1.0)
ax.set_title('Comparaison des Performances - Tous les Modeles', fontsize=14)
plt.tight_layout()
plt.savefig('reports/figures/14_performance_heatmap.png', dpi=150, bbox_inches='tight')
plt.close()

# Error analysis
print("\n--- Analyse d'erreurs (DistilBERT) ---")
errors_mask = all_val_preds_bert != y_val
error_df = val[errors_mask].copy()
error_df['predicted'] = all_val_preds_bert[errors_mask]
error_df['true_label'] = y_val[errors_mask]
print(f"Total errors: {len(error_df)} / {len(y_val)} ({100*len(error_df)/len(y_val):.1f}%)")
print(f"False Positives (real->fake): {((all_val_preds_bert == 1) & (y_val == 0)).sum()}")
print(f"False Negatives (fake->real): {((all_val_preds_bert == 0) & (y_val == 1)).sum()}")

# Confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
correct_probs = all_val_probs_bert[~errors_mask]
wrong_probs = all_val_probs_bert[errors_mask]
axes[0].hist(correct_probs, bins=30, alpha=0.7, color='#2ecc71', label='Correct')
axes[0].hist(wrong_probs, bins=30, alpha=0.7, color='#e74c3c', label='Wrong')
axes[0].set_title('Distribution de Confiance (DistilBERT)', fontsize=13)
axes[0].set_xlabel('Probabilite Fake')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Error by text length
val_copy = val.copy()
val_copy['word_count'] = val_copy['tweet'].astype(str).apply(lambda x: len(x.split()))
val_copy['is_error'] = errors_mask.astype(int)
bins = pd.qcut(val_copy['word_count'], q=5, duplicates='drop')
error_by_length = val_copy.groupby(bins)['is_error'].mean()
error_by_length.plot(kind='bar', ax=axes[1], color='#e74c3c', alpha=0.7)
axes[1].set_title("Taux d'erreur par longueur de texte", fontsize=13)
axes[1].set_xlabel('Groupe de longueur (mots)')
axes[1].set_ylabel("Taux d'erreur")
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig('reports/figures/15_error_analysis.png', dpi=150, bbox_inches='tight')
plt.close()

print("[OK] Figures 11-15 saved")

# ================================================================
# PHASE 6: EXPLAINABILITY (LIME)
# ================================================================
print("\n" + "=" * 60)
print("PHASE 6: EXPLICABILITE")
print("=" * 60)

from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names=['Real', 'Fake'], random_state=42)


def predict_proba_for_lime(texts):
    """Prediction function for LIME using DistilBERT."""
    model.eval()
    all_probs_list = []
    for text in texts:
        encoding = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                             truncation=True, return_tensors='pt')
        with torch.no_grad():
            logits = model(encoding['input_ids'].to(device),
                          encoding['attention_mask'].to(device))
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
        all_probs_list.append(probs)
    return np.array(all_probs_list)


# Explain 2 examples from each class
print("\n[1] LIME Explanations...")
sample_indices = {
    'real': val[val['label'] == 'real'].index[:2].tolist(),
    'fake': val[val['label'] == 'fake'].index[:2].tolist(),
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
for row, label in enumerate(['real', 'fake']):
    for col, idx in enumerate(sample_indices[label]):
        text = val.loc[idx, 'clean_text'] if 'clean_text' in val.columns else clean_text(val.loc[idx, 'tweet'])
        exp = explainer.explain_instance(text, predict_proba_for_lime, num_features=10, num_samples=200)

        features = exp.as_list()
        words = [f[0] for f in features]
        weights = [f[1] for f in features]
        colors_lime = ['#e74c3c' if w > 0 else '#2ecc71' for w in weights]

        axes[row][col].barh(range(len(words)), weights, color=colors_lime)
        axes[row][col].set_yticks(range(len(words)))
        axes[row][col].set_yticklabels(words)
        axes[row][col].invert_yaxis()
        pred_class = 'Fake' if exp.predict_proba[1] > 0.5 else 'Real'
        axes[row][col].set_title(f'True: {label.upper()}, Pred: {pred_class}\n(conf: {max(exp.predict_proba):.2f})',
                                  fontsize=11)
        axes[row][col].axvline(x=0, color='black', linewidth=0.5)
        axes[row][col].grid(axis='x', alpha=0.3)

plt.suptitle('LIME Explanations - DistilBERT', fontsize=14, y=1.01)
plt.tight_layout()
plt.savefig('reports/figures/16_lime_explanations.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Figure 16 saved")

# Attention visualization
print("\n[2] Attention Visualization...")


def get_attention_weights(text):
    encoding = tokenizer(text, max_length=MAX_LEN, padding='max_length',
                         truncation=True, return_tensors='pt')
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])
    # Get actual token length (non-padding)
    real_len = (encoding['attention_mask'][0] == 1).sum().item()
    tokens = tokens[:real_len]

    model.eval()
    with torch.no_grad():
        outputs = model.bert(
            input_ids=encoding['input_ids'].to(device),
            attention_mask=encoding['attention_mask'].to(device),
            output_attentions=True
        )
    # Last layer attention, average over heads, CLS token attention
    attention = outputs.attentions[-1][0].mean(dim=0)  # avg over heads
    cls_attention = attention[0, :real_len].cpu().numpy()
    return tokens, cls_attention


fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for idx, label in enumerate(['real', 'fake']):
    sample_idx = sample_indices[label][0]
    text = val.loc[sample_idx, 'clean_text'] if 'clean_text' in val.columns else clean_text(val.loc[sample_idx, 'tweet'])
    tokens, attn = get_attention_weights(text)
    # Keep only first 40 tokens for readability
    max_tokens = min(40, len(tokens))
    tokens = tokens[:max_tokens]
    attn = attn[:max_tokens]
    attn = attn / attn.max()  # normalize

    color = '#2ecc71' if label == 'real' else '#e74c3c'
    axes[idx].bar(range(len(tokens)), attn, color=color, alpha=0.7)
    axes[idx].set_xticks(range(len(tokens)))
    axes[idx].set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)
    axes[idx].set_title(f'Attention Weights - {label.upper()} News', fontsize=13)
    axes[idx].set_ylabel('Attention')
    axes[idx].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('reports/figures/17_attention_visualization.png', dpi=150, bbox_inches='tight')
plt.close()
print("[OK] Figure 17 saved")

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 60)
print("RESUME FINAL DU PROJET")
print("=" * 60)
print(f"\nDataset: COVID-19 Fake News (10,700 echantillons)")
print(f"Train: 6,420 | Val: 2,140 | Test: 2,140")
print(f"\nMeilleurs resultats sur validation:")
print("-" * 45)
best_model = max(results.items(), key=lambda x: x[1]['F1'])
for name, r in sorted(results.items(), key=lambda x: x[1]['F1'], reverse=True):
    marker = " <-- BEST" if name == best_model[0] else ""
    auc_str = f", AUC: {r.get('AUC-ROC', 0):.4f}" if 'AUC-ROC' in r else ""
    print(f"  {name:25s} Acc: {r['Accuracy']:.4f}, F1: {r['F1']:.4f}{auc_str}{marker}")

print(f"\nFigures sauvees: reports/figures/ ({len(os.listdir('reports/figures/'))} fichiers)")
print(f"Modeles sauves: models/ ({len(os.listdir('models/'))} fichiers)")
print(f"Donnees traitees: data/processed/")
print(f"Resultats CSV: reports/final_results.csv")
print("\n" + "=" * 60)
print("PROJET TERMINE AVEC SUCCES!")
print("=" * 60)
