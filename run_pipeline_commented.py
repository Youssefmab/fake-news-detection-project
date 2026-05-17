"""
Full pipeline: Preprocessing + Baseline Models + BERT + Evaluation + Explainability
COVID-19 Fake News Detection Project
"""  # Module docstring describing the project's full pipeline

import sys, io  # Import system and I/O modules for stdout reconfiguration
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')  # Force UTF-8 encoding on stdout to avoid encoding errors

import pandas as pd  # Import pandas for data manipulation and CSV handling
import numpy as np  # Import numpy for numerical operations
import re, os, warnings, pickle, time, json  # Import utilities: regex, OS, warnings, serialization, timing, JSON
warnings.filterwarnings('ignore')  # Suppress all warning messages for cleaner output

import matplotlib  # Import matplotlib base module
matplotlib.use('Agg')  # Set non-interactive backend (no GUI needed, saves to file only)
import matplotlib.pyplot as plt  # Import pyplot for creating plots and figures
import seaborn as sns  # Import seaborn for enhanced statistical visualizations

from sklearn.feature_extraction.text import TfidfVectorizer  # Import TF-IDF text vectorizer
from sklearn.svm import LinearSVC  # Import linear Support Vector Machine classifier
from sklearn.linear_model import LogisticRegression  # Import Logistic Regression classifier
from sklearn.ensemble import RandomForestClassifier  # Import Random Forest ensemble classifier
from sklearn.metrics import (classification_report, accuracy_score, f1_score,  # Import evaluation metrics (part 1)
                              precision_score, recall_score, confusion_matrix,  # Import evaluation metrics (part 2)
                              roc_auc_score, matthews_corrcoef, roc_curve,  # Import evaluation metrics (part 3)
                              precision_recall_curve, average_precision_score)  # Import evaluation metrics (part 4)
from sklearn.model_selection import cross_val_score  # Import cross-validation scorer
import xgboost as xgb  # Import XGBoost gradient boosting library

os.makedirs('data/processed', exist_ok=True)  # Create directory for processed data (skip if exists)
os.makedirs('models', exist_ok=True)  # Create directory for saved models (skip if exists)
os.makedirs('reports/figures', exist_ok=True)  # Create directory for output figures (skip if exists)

# ================================================================
# PHASE 1 & 2: LOAD + PREPROCESS
# ================================================================
print("=" * 60)  # Print separator line of 60 '=' characters
print("PHASE 1-2: CHARGEMENT ET PREPROCESSING")  # Print phase title (Loading and Preprocessing)
print("=" * 60)  # Print closing separator line

train = pd.read_csv('data/raw/Constraint_Train.csv')  # Load the training dataset from CSV
val = pd.read_csv('data/raw/Constraint_Val.csv')  # Load the validation dataset from CSV
test = pd.read_csv('data/raw/Constraint_Test.csv')  # Load the test dataset from CSV

print(f"Train: {train.shape}, Val: {val.shape}, Test: {test.shape}")  # Print the shape (rows, cols) of each dataset
print(f"Total: {len(train)+len(val)+len(test)} echantillons")  # Print total number of samples across all sets
print(f"Classes: {dict(train['label'].value_counts())}")  # Print class distribution (real vs fake counts)


def clean_text(text):  # Define text cleaning/preprocessing function
    if not isinstance(text, str):  # Check if input is not a string
        return ""  # Return empty string for non-string inputs
    text = re.sub(r'http\S+|www\.\S+', '', text)  # Remove URLs (http:// and www.)
    text = re.sub(r'<[^>]+>', '', text)  # Remove HTML tags
    text = re.sub(r'@\w+', '', text)  # Remove @mentions
    text = re.sub(r'#(\w+)', r'\1', text)  # Remove '#' from hashtags but keep the word
    text = re.sub(r'[^\w\s.,!?;:\'-]', ' ', text)  # Remove special characters except common punctuation
    text = re.sub(r'\s+', ' ', text).strip()  # Collapse multiple spaces into one and strip edges
    return text.lower()  # Convert to lowercase and return


def extract_features(text):  # Define function to extract hand-crafted text features
    words = text.split()  # Split text into a list of words
    sentences = re.split(r'[.!?]+', text)  # Split text into sentences by punctuation
    return {  # Return a dictionary of computed features
        'word_count': len(words),  # Total number of words
        'char_count': len(text),  # Total number of characters
        'avg_word_length': np.mean([len(w) for w in words]) if words else 0,  # Average word length (0 if empty)
        'exclamation_count': text.count('!'),  # Number of exclamation marks
        'question_count': text.count('?'),  # Number of question marks
        'uppercase_ratio': sum(1 for c in text if c.isupper()) / max(len(text), 1),  # Ratio of uppercase characters
        'sentence_count': max(len([s for s in sentences if s.strip()]), 1),  # Number of non-empty sentences (min 1)
        'unique_word_ratio': len(set(words)) / max(len(words), 1),  # Ratio of unique words to total words
    }


# Clean
train['clean_text'] = train['tweet'].apply(clean_text)  # Apply text cleaning to training set tweets
val['clean_text'] = val['tweet'].apply(clean_text)  # Apply text cleaning to validation set tweets

# Encode
label_map = {'real': 0, 'fake': 1}  # Define label-to-integer mapping (real=0, fake=1)
train['label_enc'] = train['label'].map(label_map)  # Encode training labels as integers
val['label_enc'] = val['label'].map(label_map)  # Encode validation labels as integers

# Extract features
train_feats = pd.DataFrame(train['clean_text'].apply(extract_features).tolist())  # Extract features for training set into DataFrame
val_feats = pd.DataFrame(val['clean_text'].apply(extract_features).tolist())  # Extract features for validation set into DataFrame

# Save
train.to_csv('data/processed/train_processed.csv', index=False)  # Save processed training data to CSV
val.to_csv('data/processed/val_processed.csv', index=False)  # Save processed validation data to CSV
print("Preprocessing done. Saved to data/processed/\n")  # Print confirmation message

# ================================================================
# PHASE 3: BASELINE MODELS
# ================================================================
print("=" * 60)  # Print separator line
print("PHASE 3: MODELES BASELINE")  # Print phase title (Baseline Models)
print("=" * 60)  # Print closing separator line

X_train_text = train['clean_text'].values  # Extract cleaned training texts as numpy array
X_val_text = val['clean_text'].values  # Extract cleaned validation texts as numpy array
y_train = train['label_enc'].values  # Extract encoded training labels as numpy array
y_val = val['label_enc'].values  # Extract encoded validation labels as numpy array

# TF-IDF
print("\n[1] TF-IDF Vectorization...")  # Print step indicator for TF-IDF
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2), min_df=2, max_df=0.95,  # Initialize TF-IDF with 15k features, uni+bigrams, frequency thresholds
                         sublinear_tf=True)  # Enable sublinear TF scaling (1 + log(tf))
X_train_tfidf = tfidf.fit_transform(X_train_text)  # Fit TF-IDF on training data and transform it
X_val_tfidf = tfidf.transform(X_val_text)  # Transform validation data using fitted TF-IDF
print(f"    Shape: {X_train_tfidf.shape}")  # Print shape of TF-IDF feature matrix

with open('models/tfidf_vectorizer.pkl', 'wb') as f:  # Open file to save TF-IDF vectorizer
    pickle.dump(tfidf, f)  # Serialize and save the TF-IDF vectorizer

results = {}  # Initialize dictionary to store all model results
all_preds = {}  # Initialize dictionary to store all model predictions
all_probs = {}  # Initialize dictionary to store all model probabilities

# --- SVM ---
print("\n[2] SVM (LinearSVC)...")  # Print step indicator for SVM
svm = LinearSVC(C=1.0, class_weight='balanced', max_iter=10000)  # Initialize SVM with balanced class weights
svm.fit(X_train_tfidf, y_train)  # Train SVM on TF-IDF training data
y_pred = svm.predict(X_val_tfidf)  # Predict labels for validation set
all_preds['SVM'] = y_pred  # Store SVM predictions
results['SVM'] = {  # Compute and store SVM evaluation metrics
    'Accuracy': accuracy_score(y_val, y_pred),  # Calculate accuracy
    'F1': f1_score(y_val, y_pred, average='weighted'),  # Calculate weighted F1 score
    'Precision': precision_score(y_val, y_pred, average='weighted'),  # Calculate weighted precision
    'Recall': recall_score(y_val, y_pred, average='weighted'),  # Calculate weighted recall
    'MCC': matthews_corrcoef(y_val, y_pred),  # Calculate Matthews Correlation Coefficient
}
print(f"    Acc: {results['SVM']['Accuracy']:.4f}, F1: {results['SVM']['F1']:.4f}")  # Print SVM results
with open('models/svm_model.pkl', 'wb') as f:  # Open file to save SVM model
    pickle.dump(svm, f)  # Serialize and save the SVM model

# --- Logistic Regression ---
print("\n[3] Logistic Regression...")  # Print step indicator for Logistic Regression
lr = LogisticRegression(C=1.0, class_weight='balanced', max_iter=1000, solver='lbfgs')  # Initialize LR with balanced weights and LBFGS solver
lr.fit(X_train_tfidf, y_train)  # Train Logistic Regression on TF-IDF training data
y_pred = lr.predict(X_val_tfidf)  # Predict labels for validation set
y_prob = lr.predict_proba(X_val_tfidf)[:, 1]  # Get probability of class 1 (fake) for validation set
all_preds['LR'] = y_pred  # Store LR predictions
all_probs['LR'] = y_prob  # Store LR probabilities
results['Logistic Regression'] = {  # Compute and store LR evaluation metrics
    'Accuracy': accuracy_score(y_val, y_pred),  # Calculate accuracy
    'F1': f1_score(y_val, y_pred, average='weighted'),  # Calculate weighted F1 score
    'Precision': precision_score(y_val, y_pred, average='weighted'),  # Calculate weighted precision
    'Recall': recall_score(y_val, y_pred, average='weighted'),  # Calculate weighted recall
    'MCC': matthews_corrcoef(y_val, y_pred),  # Calculate Matthews Correlation Coefficient
    'AUC-ROC': roc_auc_score(y_val, y_prob),  # Calculate Area Under ROC Curve
}
print(f"    Acc: {results['Logistic Regression']['Accuracy']:.4f}, F1: {results['Logistic Regression']['F1']:.4f}, AUC: {results['Logistic Regression']['AUC-ROC']:.4f}")  # Print LR results
with open('models/lr_model.pkl', 'wb') as f:  # Open file to save LR model
    pickle.dump(lr, f)  # Serialize and save the LR model

# --- Random Forest ---
print("\n[4] Random Forest...")  # Print step indicator for Random Forest
rf = RandomForestClassifier(n_estimators=200, class_weight='balanced', n_jobs=-1, random_state=42)  # Initialize RF with 200 trees, balanced weights, parallel processing
rf.fit(X_train_tfidf, y_train)  # Train Random Forest on TF-IDF training data
y_pred = rf.predict(X_val_tfidf)  # Predict labels for validation set
y_prob = rf.predict_proba(X_val_tfidf)[:, 1]  # Get probability of class 1 (fake) for validation set
all_preds['RF'] = y_pred  # Store RF predictions
all_probs['RF'] = y_prob  # Store RF probabilities
results['Random Forest'] = {  # Compute and store RF evaluation metrics
    'Accuracy': accuracy_score(y_val, y_pred),  # Calculate accuracy
    'F1': f1_score(y_val, y_pred, average='weighted'),  # Calculate weighted F1 score
    'Precision': precision_score(y_val, y_pred, average='weighted'),  # Calculate weighted precision
    'Recall': recall_score(y_val, y_pred, average='weighted'),  # Calculate weighted recall
    'MCC': matthews_corrcoef(y_val, y_pred),  # Calculate Matthews Correlation Coefficient
    'AUC-ROC': roc_auc_score(y_val, y_prob),  # Calculate Area Under ROC Curve
}
print(f"    Acc: {results['Random Forest']['Accuracy']:.4f}, F1: {results['Random Forest']['F1']:.4f}, AUC: {results['Random Forest']['AUC-ROC']:.4f}")  # Print RF results

# --- XGBoost ---
print("\n[5] XGBoost...")  # Print step indicator for XGBoost
scale = np.sum(y_train == 0) / np.sum(y_train == 1)  # Calculate class imbalance ratio for positive weight scaling
xgb_model = xgb.XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,  # Initialize XGBoost with 200 trees, max depth 6, lr 0.1
                                scale_pos_weight=scale, eval_metric='logloss',  # Set class imbalance weight and log loss metric
                                random_state=42)  # Set random seed for reproducibility
xgb_model.fit(X_train_tfidf, y_train)  # Train XGBoost on TF-IDF training data
y_pred = xgb_model.predict(X_val_tfidf)  # Predict labels for validation set
y_prob = xgb_model.predict_proba(X_val_tfidf)[:, 1]  # Get probability of class 1 (fake) for validation set
all_preds['XGB'] = y_pred  # Store XGBoost predictions
all_probs['XGB'] = y_prob  # Store XGBoost probabilities
results['XGBoost'] = {  # Compute and store XGBoost evaluation metrics
    'Accuracy': accuracy_score(y_val, y_pred),  # Calculate accuracy
    'F1': f1_score(y_val, y_pred, average='weighted'),  # Calculate weighted F1 score
    'Precision': precision_score(y_val, y_pred, average='weighted'),  # Calculate weighted precision
    'Recall': recall_score(y_val, y_pred, average='weighted'),  # Calculate weighted recall
    'MCC': matthews_corrcoef(y_val, y_pred),  # Calculate Matthews Correlation Coefficient
    'AUC-ROC': roc_auc_score(y_val, y_prob),  # Calculate Area Under ROC Curve
}
print(f"    Acc: {results['XGBoost']['Accuracy']:.4f}, F1: {results['XGBoost']['F1']:.4f}, AUC: {results['XGBoost']['AUC-ROC']:.4f}")  # Print XGBoost results

# Cross-validation
print("\n[6] 5-Fold Cross-Validation (SVM)...")  # Print step indicator for cross-validation
cv_scores = cross_val_score(LinearSVC(C=1.0, class_weight='balanced', max_iter=10000),  # Run 5-fold CV with a fresh SVM
                             X_train_tfidf, y_train, cv=5, scoring='f1_weighted')  # Use weighted F1 as scoring metric
print(f"    CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")  # Print mean and std of CV F1 scores

# Results table
print("\n" + "-" * 60)  # Print separator line
print("RESULTATS BASELINES:")  # Print section title (Baseline Results)
print("-" * 60)  # Print closing separator line
results_df = pd.DataFrame(results).T  # Create results DataFrame (models as rows, metrics as columns)
print(results_df.round(4).to_string())  # Print results rounded to 4 decimal places
results_df.to_csv('reports/baseline_results.csv')  # Save baseline results to CSV

# ================================================================
# PHASE 3 FIGURES
# ================================================================

# Confusion matrices
fig, axes = plt.subplots(1, 4, figsize=(20, 4.5))  # Create figure with 4 subplots for confusion matrices
model_names = ['SVM', 'LR', 'RF', 'XGB']  # List of model keys for predictions dictionary
display_names = ['SVM', 'Logistic Reg.', 'Random Forest', 'XGBoost']  # List of display names for plot titles
for ax, key, dname in zip(axes, model_names, display_names):  # Loop over each model's subplot
    cm = confusion_matrix(y_val, all_preds[key])  # Compute confusion matrix for this model
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,  # Plot heatmap with integer annotations
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'])  # Set axis labels
    acc = accuracy_score(y_val, all_preds[key])  # Calculate accuracy for title
    ax.set_title(f'{dname}\n(Acc: {acc:.3f})', fontsize=11)  # Set subplot title with model name and accuracy
    ax.set_xlabel('Predicted')  # Set x-axis label
    ax.set_ylabel('Actual')  # Set y-axis label
plt.tight_layout()  # Adjust layout to prevent overlap
plt.savefig('reports/figures/06_confusion_matrices_baselines.png', dpi=150, bbox_inches='tight')  # Save figure to file
plt.close()  # Close figure to free memory

# Model comparison
fig, ax = plt.subplots(figsize=(10, 6))  # Create figure for grouped bar chart
metrics_list = ['Accuracy', 'F1', 'Precision', 'Recall']  # Metrics to compare
x = np.arange(len(metrics_list))  # Create x positions for metric groups
width = 0.18  # Set bar width
colors_bar = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']  # Define colors for each model
for i, (model_name, metrics) in enumerate(results.items()):  # Loop over each model
    values = [metrics[m] for m in metrics_list]  # Get metric values for this model
    ax.bar(x + i * width, values, width, label=model_name, color=colors_bar[i])  # Plot grouped bar for this model
ax.set_xticks(x + width * 1.5)  # Center x-tick positions across grouped bars
ax.set_xticklabels(metrics_list, fontsize=12)  # Set x-tick labels as metric names
ax.set_ylim(0.7, 1.0)  # Set y-axis range to focus on high-performance region
ax.set_title('Comparaison des Modeles Baseline', fontsize=14)  # Set plot title (Baseline Model Comparison)
ax.legend(fontsize=10)  # Add legend for model colors
ax.grid(axis='y', alpha=0.3)  # Add light horizontal grid lines
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/07_baseline_comparison.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure

# ROC Curves
fig, ax = plt.subplots(figsize=(8, 6))  # Create figure for ROC curves
roc_colors = ['#e74c3c', '#2ecc71', '#f39c12']  # Define colors for each model's ROC curve
for (name, prob), color in zip(all_probs.items(), roc_colors):  # Loop over each model with probabilities
    display = {'LR': 'Logistic Regression', 'RF': 'Random Forest', 'XGB': 'XGBoost'}[name]  # Map short name to display name
    fpr, tpr, _ = roc_curve(y_val, prob)  # Compute false positive rate and true positive rate
    auc_val = roc_auc_score(y_val, prob)  # Compute AUC score
    ax.plot(fpr, tpr, label=f'{display} (AUC={auc_val:.3f})', color=color, linewidth=2)  # Plot ROC curve with AUC in label
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Plot diagonal baseline (random classifier)
ax.set_xlabel('False Positive Rate', fontsize=12)  # Set x-axis label
ax.set_ylabel('True Positive Rate', fontsize=12)  # Set y-axis label
ax.set_title('Courbes ROC - Modeles Baseline', fontsize=14)  # Set plot title (ROC Curves - Baseline Models)
ax.legend(fontsize=11)  # Add legend
ax.grid(alpha=0.3)  # Add light grid
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/08_roc_curves_baselines.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure

# Feature importance (LR)
fig, axes = plt.subplots(1, 2, figsize=(14, 6))  # Create figure with 2 subplots for top features
feature_names = np.array(tfidf.get_feature_names_out())  # Get all TF-IDF feature names
coefs = lr.coef_[0]  # Get LR coefficient weights for each feature
top_real_idx = np.argsort(coefs)[:20]  # Get indices of top 20 features indicating REAL (most negative coefficients)
top_fake_idx = np.argsort(coefs)[-20:][::-1]  # Get indices of top 20 features indicating FAKE (most positive coefficients)
axes[0].barh(range(20), coefs[top_real_idx], color='#2ecc71')  # Plot horizontal bars for top REAL features
axes[0].set_yticks(range(20))  # Set y-tick positions
axes[0].set_yticklabels(feature_names[top_real_idx])  # Label bars with feature names
axes[0].set_title('Top 20 Mots -> REAL', fontsize=13)  # Set title (Top 20 Words -> REAL)
axes[0].invert_yaxis()  # Invert y-axis so highest importance is at top
axes[1].barh(range(20), coefs[top_fake_idx], color='#e74c3c')  # Plot horizontal bars for top FAKE features
axes[1].set_yticks(range(20))  # Set y-tick positions
axes[1].set_yticklabels(feature_names[top_fake_idx])  # Label bars with feature names
axes[1].set_title('Top 20 Mots -> FAKE', fontsize=13)  # Set title (Top 20 Words -> FAKE)
axes[1].invert_yaxis()  # Invert y-axis so highest importance is at top
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/09_feature_importance_lr.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure

print("\n[OK] Baseline figures saved (06-09)")  # Print confirmation that baseline figures are saved

# ================================================================
# PHASE 4: BERT MODEL
# ================================================================
print("\n" + "=" * 60)  # Print separator line
print("PHASE 4: MODELE BERT (Fine-tuning)")  # Print phase title (BERT Model Fine-tuning)
print("=" * 60)  # Print closing separator line

import torch  # Import PyTorch core library
import torch.nn as nn  # Import PyTorch neural network module
from torch.utils.data import Dataset, DataLoader  # Import Dataset and DataLoader for batching
from transformers import (DistilBertTokenizer, DistilBertModel,  # Import DistilBERT tokenizer and model from HuggingFace
                           get_linear_schedule_with_warmup)  # Import learning rate scheduler with warmup

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Set device to GPU if available, else CPU
print(f"Device: {device}")  # Print which device is being used

# Use DistilBERT (faster on CPU, good performance)
MODEL_NAME = 'distilbert-base-uncased'  # Define the pretrained model name
print(f"Model: {MODEL_NAME}")  # Print the model name

print("\n[1] Loading tokenizer...")  # Print step indicator for tokenizer loading
tokenizer = DistilBertTokenizer.from_pretrained(MODEL_NAME)  # Load the pretrained DistilBERT tokenizer

MAX_LEN = 128  # Set maximum token sequence length
BATCH_SIZE = 32  # Set batch size for DataLoader
EPOCHS = 3  # Set number of training epochs
LR = 2e-5  # Set learning rate for AdamW optimizer


class NewsDataset(Dataset):  # Define custom PyTorch Dataset class for news data
    def __init__(self, texts, labels, tokenizer, max_len):  # Constructor takes texts, labels, tokenizer, and max length
        self.texts = texts  # Store list of text strings
        self.labels = labels  # Store list of integer labels
        self.tokenizer = tokenizer  # Store tokenizer reference
        self.max_len = max_len  # Store max sequence length

    def __len__(self):  # Return dataset size
        return len(self.texts)  # Number of samples

    def __getitem__(self, idx):  # Get a single sample by index
        encoding = self.tokenizer(  # Tokenize the text at index idx
            self.texts[idx],  # Input text string
            max_length=self.max_len,  # Truncate/pad to max_len
            padding='max_length',  # Pad to max_length
            truncation=True,  # Truncate if longer than max_length
            return_tensors='pt'  # Return PyTorch tensors
        )
        return {  # Return dictionary with input_ids, attention_mask, and label
            'input_ids': encoding['input_ids'].squeeze(),  # Token IDs (remove batch dimension)
            'attention_mask': encoding['attention_mask'].squeeze(),  # Attention mask (remove batch dimension)
            'label': torch.tensor(self.labels[idx], dtype=torch.long)  # Label as long tensor
        }


class DistilBERTClassifier(nn.Module):  # Define custom DistilBERT classifier model
    def __init__(self, model_name, num_classes=2, dropout=0.3):  # Constructor with model name, classes, dropout rate
        super().__init__()  # Initialize parent nn.Module
        self.bert = DistilBertModel.from_pretrained(model_name)  # Load pretrained DistilBERT model
        self.dropout = nn.Dropout(dropout)  # Dropout layer for regularization
        self.fc1 = nn.Linear(768, 256)  # First fully connected layer (768 BERT hidden size -> 256)
        self.fc2 = nn.Linear(256, num_classes)  # Second fully connected layer (256 -> number of classes)
        self.relu = nn.ReLU()  # ReLU activation function

    def forward(self, input_ids, attention_mask):  # Forward pass method
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)  # Run input through DistilBERT
        cls_output = outputs.last_hidden_state[:, 0, :]  # Extract [CLS] token representation (first token)
        x = self.dropout(cls_output)  # Apply dropout to CLS output
        x = self.relu(self.fc1(x))  # Apply first FC layer + ReLU activation
        x = self.dropout(x)  # Apply dropout again
        logits = self.fc2(x)  # Apply second FC layer to get class logits
        return logits  # Return raw logits (unnormalized scores)


print("[2] Creating datasets...")  # Print step indicator for dataset creation
train_dataset = NewsDataset(train['clean_text'].tolist(), y_train.tolist(), tokenizer, MAX_LEN)  # Create training dataset
val_dataset = NewsDataset(val['clean_text'].tolist(), y_val.tolist(), tokenizer, MAX_LEN)  # Create validation dataset

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # Create training DataLoader with shuffling
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)  # Create validation DataLoader without shuffling

print("[3] Loading model...")  # Print step indicator for model loading
model = DistilBERTClassifier(MODEL_NAME).to(device)  # Instantiate classifier and move to device (GPU/CPU)

# Class weights for imbalanced data
class_counts = np.bincount(y_train)  # Count samples per class in training set
class_weights = torch.tensor([1.0 / c for c in class_counts], dtype=torch.float).to(device)  # Compute inverse frequency weights
class_weights = class_weights / class_weights.sum() * 2  # Normalize weights to sum to 2 (one per class)

criterion = nn.CrossEntropyLoss(weight=class_weights)  # Define weighted cross-entropy loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)  # Initialize AdamW optimizer with weight decay
total_steps = len(train_loader) * EPOCHS  # Calculate total training steps (batches * epochs)
scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps),  # Create LR scheduler with 10% warmup
                                             num_training_steps=total_steps)  # Total steps for linear decay

print(f"[4] Training for {EPOCHS} epochs...")  # Print training start indicator
train_losses = []  # List to track training loss per epoch
val_losses = []  # List to track validation loss per epoch
val_accs = []  # List to track validation accuracy per epoch
best_f1 = 0  # Track best validation F1 score
best_model_state = None  # Store state dict of best model

for epoch in range(EPOCHS):  # Loop over each epoch
    # Train
    model.train()  # Set model to training mode (enables dropout, batch norm)
    epoch_loss = 0  # Accumulator for epoch training loss
    correct = 0  # Counter for correct predictions
    total = 0  # Counter for total samples
    start_time = time.time()  # Record epoch start time

    for batch_idx, batch in enumerate(train_loader):  # Loop over each training batch
        input_ids = batch['input_ids'].to(device)  # Move input IDs to device
        attention_mask = batch['attention_mask'].to(device)  # Move attention mask to device
        labels = batch['label'].to(device)  # Move labels to device

        optimizer.zero_grad()  # Reset gradients to zero
        logits = model(input_ids, attention_mask)  # Forward pass to get logits
        loss = criterion(logits, labels)  # Compute loss
        loss.backward()  # Backpropagate gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # Clip gradients to prevent exploding gradients
        optimizer.step()  # Update model parameters
        scheduler.step()  # Update learning rate

        epoch_loss += loss.item()  # Accumulate batch loss
        preds = logits.argmax(dim=1)  # Get predicted class (index of max logit)
        correct += (preds == labels).sum().item()  # Count correct predictions in this batch
        total += labels.size(0)  # Count total samples in this batch

        if (batch_idx + 1) % 50 == 0:  # Print progress every 50 batches
            print(f"    Batch {batch_idx+1}/{len(train_loader)}, Loss: {loss.item():.4f}")  # Print batch number and loss

    train_acc = correct / total  # Compute epoch training accuracy
    avg_train_loss = epoch_loss / len(train_loader)  # Compute average training loss
    train_losses.append(avg_train_loss)  # Append to training loss history

    # Validate
    model.eval()  # Set model to evaluation mode (disables dropout)
    val_loss = 0  # Accumulator for validation loss
    all_val_preds = []  # List to collect all validation predictions
    all_val_probs_bert = []  # List to collect all validation probabilities

    with torch.no_grad():  # Disable gradient computation for validation
        for batch in val_loader:  # Loop over each validation batch
            input_ids = batch['input_ids'].to(device)  # Move input IDs to device
            attention_mask = batch['attention_mask'].to(device)  # Move attention mask to device
            labels = batch['label'].to(device)  # Move labels to device

            logits = model(input_ids, attention_mask)  # Forward pass
            loss = criterion(logits, labels)  # Compute loss
            val_loss += loss.item()  # Accumulate validation loss

            probs = torch.softmax(logits, dim=1)  # Convert logits to probabilities via softmax
            all_val_preds.extend(logits.argmax(dim=1).cpu().numpy())  # Collect predicted classes
            all_val_probs_bert.extend(probs[:, 1].cpu().numpy())  # Collect probability of fake class

    avg_val_loss = val_loss / len(val_loader)  # Compute average validation loss
    val_losses.append(avg_val_loss)  # Append to validation loss history

    val_acc = accuracy_score(y_val, all_val_preds)  # Compute validation accuracy
    val_f1 = f1_score(y_val, all_val_preds, average='weighted')  # Compute validation weighted F1
    val_accs.append(val_acc)  # Append to validation accuracy history

    elapsed = time.time() - start_time  # Compute epoch duration in seconds
    print(f"\n  Epoch {epoch+1}/{EPOCHS} ({elapsed:.0f}s):")  # Print epoch number and duration
    print(f"    Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.4f}")  # Print training metrics
    print(f"    Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.4f}, Val F1: {val_f1:.4f}")  # Print validation metrics

    if val_f1 > best_f1:  # Check if this epoch achieved best F1 so far
        best_f1 = val_f1  # Update best F1
        best_model_state = model.state_dict().copy()  # Save model state dict
        print(f"    -> New best model (F1: {best_f1:.4f})")  # Print notification of new best model

# Restore best model
if best_model_state:  # If we saved a best model state
    model.load_state_dict(best_model_state)  # Restore the best model weights

# Save model
torch.save({  # Save model checkpoint to disk
    'model_state_dict': model.state_dict(),  # Save model weights
    'model_name': MODEL_NAME,  # Save model name
    'max_len': MAX_LEN,  # Save max sequence length
    'num_classes': 2,  # Save number of classes
}, 'models/distilbert_model.pt')  # Path to save the checkpoint
print(f"\n[OK] Model saved to models/distilbert_model.pt")  # Print confirmation

# Final evaluation
model.eval()  # Set model to evaluation mode
all_val_preds_bert = []  # List for final validation predictions
all_val_probs_bert = []  # List for final validation probabilities
with torch.no_grad():  # Disable gradient computation
    for batch in val_loader:  # Loop over validation batches
        input_ids = batch['input_ids'].to(device)  # Move input IDs to device
        attention_mask = batch['attention_mask'].to(device)  # Move attention mask to device
        logits = model(input_ids, attention_mask)  # Forward pass
        probs = torch.softmax(logits, dim=1)  # Compute class probabilities
        all_val_preds_bert.extend(logits.argmax(dim=1).cpu().numpy())  # Collect predictions
        all_val_probs_bert.extend(probs[:, 1].cpu().numpy())  # Collect fake-class probabilities

all_val_preds_bert = np.array(all_val_preds_bert)  # Convert predictions list to numpy array
all_val_probs_bert = np.array(all_val_probs_bert)  # Convert probabilities list to numpy array

bert_results = {  # Compute all evaluation metrics for DistilBERT
    'Accuracy': accuracy_score(y_val, all_val_preds_bert),  # Calculate accuracy
    'F1': f1_score(y_val, all_val_preds_bert, average='weighted'),  # Calculate weighted F1 score
    'Precision': precision_score(y_val, all_val_preds_bert, average='weighted'),  # Calculate weighted precision
    'Recall': recall_score(y_val, all_val_preds_bert, average='weighted'),  # Calculate weighted recall
    'MCC': matthews_corrcoef(y_val, all_val_preds_bert),  # Calculate Matthews Correlation Coefficient
    'AUC-ROC': roc_auc_score(y_val, all_val_probs_bert),  # Calculate AUC-ROC score
}
results['DistilBERT'] = bert_results  # Add DistilBERT results to the global results dictionary
print(f"\nDistilBERT -> Acc: {bert_results['Accuracy']:.4f}, F1: {bert_results['F1']:.4f}, AUC: {bert_results['AUC-ROC']:.4f}")  # Print DistilBERT results

# Training curves
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create figure with 2 subplots for training curves
axes[0].plot(range(1, EPOCHS+1), train_losses, 'b-o', label='Train Loss')  # Plot training loss curve
axes[0].plot(range(1, EPOCHS+1), val_losses, 'r-o', label='Val Loss')  # Plot validation loss curve
axes[0].set_title('Learning Curves - Loss', fontsize=13)  # Set title
axes[0].set_xlabel('Epoch')  # Set x-axis label
axes[0].set_ylabel('Loss')  # Set y-axis label
axes[0].legend()  # Add legend
axes[0].grid(alpha=0.3)  # Add light grid

axes[1].plot(range(1, EPOCHS+1), val_accs, 'g-o', label='Val Accuracy')  # Plot validation accuracy curve
axes[1].set_title('Validation Accuracy', fontsize=13)  # Set title
axes[1].set_xlabel('Epoch')  # Set x-axis label
axes[1].set_ylabel('Accuracy')  # Set y-axis label
axes[1].legend()  # Add legend
axes[1].grid(alpha=0.3)  # Add light grid
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/10_bert_training_curves.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure
print("[OK] Figure 10 saved")  # Print confirmation

# ================================================================
# PHASE 5: COMPREHENSIVE EVALUATION
# ================================================================
print("\n" + "=" * 60)  # Print separator line
print("PHASE 5: EVALUATION COMPLETE")  # Print phase title (Comprehensive Evaluation)
print("=" * 60)  # Print closing separator line

# Final results table
print("\n" + "-" * 60)  # Print separator line
print("RESULTATS FINAUX - TOUS LES MODELES:")  # Print section title (Final Results - All Models)
print("-" * 60)  # Print closing separator line
final_df = pd.DataFrame(results).T  # Create DataFrame with all models' results (transposed)
print(final_df.round(4).to_string())  # Print results rounded to 4 decimals
final_df.to_csv('reports/final_results.csv')  # Save final results to CSV

# Classification reports
print("\n--- Classification Report: Best Baseline (LR) ---")  # Print header for LR classification report
print(classification_report(y_val, all_preds['LR'], target_names=['Real', 'Fake']))  # Print detailed LR classification report

print("\n--- Classification Report: DistilBERT ---")  # Print header for DistilBERT classification report
print(classification_report(y_val, all_val_preds_bert, target_names=['Real', 'Fake']))  # Print detailed DistilBERT classification report

# Confusion matrices side by side (best baseline vs BERT)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create figure with 2 subplots for confusion matrices
for ax, preds_arr, title in [  # Loop over LR and DistilBERT
    (axes[0], all_preds['LR'], 'Logistic Regression'),  # First subplot: Logistic Regression
    (axes[1], all_val_preds_bert, 'DistilBERT')  # Second subplot: DistilBERT
]:
    cm = confusion_matrix(y_val, preds_arr)  # Compute confusion matrix
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,  # Plot confusion matrix heatmap
                xticklabels=['Real', 'Fake'], yticklabels=['Real', 'Fake'],  # Set axis labels
                annot_kws={'size': 14})  # Set annotation font size
    acc = accuracy_score(y_val, preds_arr)  # Compute accuracy
    f1_val = f1_score(y_val, preds_arr, average='weighted')  # Compute weighted F1
    ax.set_title(f'{title}\nAcc: {acc:.3f}, F1: {f1_val:.3f}', fontsize=13)  # Set title with metrics
    ax.set_xlabel('Predicted', fontsize=11)  # Set x-axis label
    ax.set_ylabel('Actual', fontsize=11)  # Set y-axis label
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/11_confusion_best_vs_bert.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure

# ROC comparison: all models
fig, ax = plt.subplots(figsize=(9, 7))  # Create figure for combined ROC curves
all_roc = {  # Dictionary of model names to probability arrays for ROC
    'Logistic Regression': all_probs['LR'],  # LR probabilities
    'Random Forest': all_probs['RF'],  # RF probabilities
    'XGBoost': all_probs['XGB'],  # XGBoost probabilities
    'DistilBERT': all_val_probs_bert,  # DistilBERT probabilities
}
roc_colors_all = ['#e74c3c', '#2ecc71', '#f39c12', '#3498db']  # Colors for each model's curve
for (name, prob), color in zip(all_roc.items(), roc_colors_all):  # Loop over each model
    fpr, tpr, _ = roc_curve(y_val, prob)  # Compute FPR and TPR
    auc_val = roc_auc_score(y_val, prob)  # Compute AUC
    lw = 3 if name == 'DistilBERT' else 2  # Use thicker line for DistilBERT
    ax.plot(fpr, tpr, label=f'{name} (AUC={auc_val:.3f})', color=color, linewidth=lw)  # Plot ROC curve
ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)  # Plot random classifier diagonal
ax.set_xlabel('False Positive Rate', fontsize=12)  # Set x-axis label
ax.set_ylabel('True Positive Rate', fontsize=12)  # Set y-axis label
ax.set_title('Courbes ROC - Tous les Modeles', fontsize=14)  # Set title (ROC Curves - All Models)
ax.legend(fontsize=11)  # Add legend
ax.grid(alpha=0.3)  # Add light grid
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/12_roc_all_models.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure

# Precision-Recall curves
fig, ax = plt.subplots(figsize=(9, 7))  # Create figure for Precision-Recall curves
for (name, prob), color in zip(all_roc.items(), roc_colors_all):  # Loop over each model
    prec_curve, rec_curve, _ = precision_recall_curve(y_val, prob)  # Compute precision and recall at various thresholds
    ap = average_precision_score(y_val, prob)  # Compute Average Precision (AP) score
    lw = 3 if name == 'DistilBERT' else 2  # Use thicker line for DistilBERT
    ax.plot(rec_curve, prec_curve, label=f'{name} (AP={ap:.3f})', color=color, linewidth=lw)  # Plot PR curve
ax.set_xlabel('Recall', fontsize=12)  # Set x-axis label
ax.set_ylabel('Precision', fontsize=12)  # Set y-axis label
ax.set_title('Courbes Precision-Recall', fontsize=14)  # Set title (Precision-Recall Curves)
ax.legend(fontsize=11)  # Add legend
ax.grid(alpha=0.3)  # Add light grid
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/13_precision_recall_curves.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure

# Final heatmap comparison
fig, ax = plt.subplots(figsize=(10, 5))  # Create figure for performance heatmap
metrics_for_heatmap = ['Accuracy', 'F1', 'Precision', 'Recall', 'MCC']  # Metrics to display in heatmap
heatmap_data = final_df[metrics_for_heatmap].astype(float)  # Extract metrics as float DataFrame
sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='YlOrRd', ax=ax,  # Plot heatmap with 3-decimal annotations
            linewidths=0.5, vmin=0.7, vmax=1.0)  # Set cell borders and color scale range
ax.set_title('Comparaison des Performances - Tous les Modeles', fontsize=14)  # Set title (Performance Comparison - All Models)
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/14_performance_heatmap.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure

# Error analysis
print("\n--- Analyse d'erreurs (DistilBERT) ---")  # Print header for error analysis section
errors_mask = all_val_preds_bert != y_val  # Boolean mask: True where DistilBERT predicted incorrectly
error_df = val[errors_mask].copy()  # Create DataFrame of misclassified samples
error_df['predicted'] = all_val_preds_bert[errors_mask]  # Add predicted label column
error_df['true_label'] = y_val[errors_mask]  # Add true label column
print(f"Total errors: {len(error_df)} / {len(y_val)} ({100*len(error_df)/len(y_val):.1f}%)")  # Print total error count and percentage
print(f"False Positives (real->fake): {((all_val_preds_bert == 1) & (y_val == 0)).sum()}")  # Print number of false positives
print(f"False Negatives (fake->real): {((all_val_preds_bert == 0) & (y_val == 1)).sum()}")  # Print number of false negatives

# Confidence distribution
fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Create figure with 2 subplots for error analysis
correct_probs = all_val_probs_bert[~errors_mask]  # Probabilities for correctly classified samples
wrong_probs = all_val_probs_bert[errors_mask]  # Probabilities for incorrectly classified samples
axes[0].hist(correct_probs, bins=30, alpha=0.7, color='#2ecc71', label='Correct')  # Plot histogram of correct prediction confidence
axes[0].hist(wrong_probs, bins=30, alpha=0.7, color='#e74c3c', label='Wrong')  # Plot histogram of wrong prediction confidence
axes[0].set_title('Distribution de Confiance (DistilBERT)', fontsize=13)  # Set title (Confidence Distribution)
axes[0].set_xlabel('Probabilite Fake')  # Set x-axis label (Fake Probability)
axes[0].legend()  # Add legend
axes[0].grid(alpha=0.3)  # Add light grid

# Error by text length
val_copy = val.copy()  # Create copy of validation DataFrame to avoid modifying original
val_copy['word_count'] = val_copy['tweet'].astype(str).apply(lambda x: len(x.split()))  # Compute word count per tweet
val_copy['is_error'] = errors_mask.astype(int)  # Add binary error indicator column
bins = pd.qcut(val_copy['word_count'], q=5, duplicates='drop')  # Divide word counts into 5 quantile bins
error_by_length = val_copy.groupby(bins)['is_error'].mean()  # Compute error rate per length bin
error_by_length.plot(kind='bar', ax=axes[1], color='#e74c3c', alpha=0.7)  # Plot bar chart of error rate by text length
axes[1].set_title("Taux d'erreur par longueur de texte", fontsize=13)  # Set title (Error Rate by Text Length)
axes[1].set_xlabel('Groupe de longueur (mots)')  # Set x-axis label (Length Group in words)
axes[1].set_ylabel("Taux d'erreur")  # Set y-axis label (Error Rate)
axes[1].tick_params(axis='x', rotation=45)  # Rotate x-tick labels 45 degrees
axes[1].grid(axis='y', alpha=0.3)  # Add light horizontal grid
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/15_error_analysis.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure

print("[OK] Figures 11-15 saved")  # Print confirmation that evaluation figures are saved

# ================================================================
# PHASE 6: EXPLAINABILITY (LIME)
# ================================================================
print("\n" + "=" * 60)  # Print separator line
print("PHASE 6: EXPLICABILITE")  # Print phase title (Explainability)
print("=" * 60)  # Print closing separator line

from lime.lime_text import LimeTextExplainer  # Import LIME text explainer for model interpretability

explainer = LimeTextExplainer(class_names=['Real', 'Fake'], random_state=42)  # Initialize LIME explainer with class names


def predict_proba_for_lime(texts):  # Define prediction function compatible with LIME's API
    """Prediction function for LIME using DistilBERT."""  # Docstring explaining function purpose
    model.eval()  # Set model to evaluation mode
    all_probs_list = []  # List to collect probabilities for each text
    for text in texts:  # Loop over each input text
        encoding = tokenizer(text, max_length=MAX_LEN, padding='max_length',  # Tokenize the text
                             truncation=True, return_tensors='pt')  # Return as PyTorch tensor
        with torch.no_grad():  # Disable gradient computation
            logits = model(encoding['input_ids'].to(device),  # Forward pass through model
                          encoding['attention_mask'].to(device))  # Pass attention mask
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]  # Convert logits to probabilities
        all_probs_list.append(probs)  # Append probabilities for this text
    return np.array(all_probs_list)  # Return all probabilities as numpy array


# Explain 2 examples from each class
print("\n[1] LIME Explanations...")  # Print step indicator for LIME explanations
sample_indices = {  # Select sample indices for explanation
    'real': val[val['label'] == 'real'].index[:2].tolist(),  # First 2 real samples
    'fake': val[val['label'] == 'fake'].index[:2].tolist(),  # First 2 fake samples
}

fig, axes = plt.subplots(2, 2, figsize=(16, 12))  # Create 2x2 figure for LIME explanations
for row, label in enumerate(['real', 'fake']):  # Loop over each class (rows)
    for col, idx in enumerate(sample_indices[label]):  # Loop over each sample index (columns)
        text = val.loc[idx, 'clean_text'] if 'clean_text' in val.columns else clean_text(val.loc[idx, 'tweet'])  # Get cleaned text (or clean it on the fly)
        exp = explainer.explain_instance(text, predict_proba_for_lime, num_features=10, num_samples=200)  # Generate LIME explanation with 10 features, 200 perturbation samples

        features = exp.as_list()  # Get list of (feature, weight) tuples
        words = [f[0] for f in features]  # Extract feature words
        weights = [f[1] for f in features]  # Extract feature weights
        colors_lime = ['#e74c3c' if w > 0 else '#2ecc71' for w in weights]  # Red for fake-indicating, green for real-indicating

        axes[row][col].barh(range(len(words)), weights, color=colors_lime)  # Plot horizontal bar chart of feature weights
        axes[row][col].set_yticks(range(len(words)))  # Set y-tick positions
        axes[row][col].set_yticklabels(words)  # Label y-ticks with feature words
        axes[row][col].invert_yaxis()  # Invert y-axis so top features appear at top
        pred_class = 'Fake' if exp.predict_proba[1] > 0.5 else 'Real'  # Determine predicted class label
        axes[row][col].set_title(f'True: {label.upper()}, Pred: {pred_class}\n(conf: {max(exp.predict_proba):.2f})',  # Set title with true/predicted labels and confidence
                                  fontsize=11)  # Set font size
        axes[row][col].axvline(x=0, color='black', linewidth=0.5)  # Add vertical line at x=0 (decision boundary)
        axes[row][col].grid(axis='x', alpha=0.3)  # Add light vertical grid

plt.suptitle('LIME Explanations - DistilBERT', fontsize=14, y=1.01)  # Add overall figure title
plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/16_lime_explanations.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure
print("[OK] Figure 16 saved")  # Print confirmation

# Attention visualization
print("\n[2] Attention Visualization...")  # Print step indicator for attention visualization


def get_attention_weights(text):  # Define function to extract attention weights from DistilBERT
    encoding = tokenizer(text, max_length=MAX_LEN, padding='max_length',  # Tokenize input text
                         truncation=True, return_tensors='pt')  # Return as PyTorch tensor
    tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'][0])  # Convert token IDs back to token strings
    # Get actual token length (non-padding)
    real_len = (encoding['attention_mask'][0] == 1).sum().item()  # Count non-padded tokens
    tokens = tokens[:real_len]  # Keep only non-padded tokens

    model.eval()  # Set model to evaluation mode
    with torch.no_grad():  # Disable gradient computation
        outputs = model.bert(  # Run through DistilBERT backbone (not classifier head)
            input_ids=encoding['input_ids'].to(device),  # Pass input IDs
            attention_mask=encoding['attention_mask'].to(device),  # Pass attention mask
            output_attentions=True  # Request attention weight outputs
        )
    # Last layer attention, average over heads, CLS token attention
    attention = outputs.attentions[-1][0].mean(dim=0)  # Average attention across all heads in last layer
    cls_attention = attention[0, :real_len].cpu().numpy()  # Extract CLS token's attention to all other tokens
    return tokens, cls_attention  # Return token strings and their attention weights


fig, axes = plt.subplots(2, 1, figsize=(14, 8))  # Create figure with 2 vertical subplots
for idx, label in enumerate(['real', 'fake']):  # Loop over real and fake samples
    sample_idx = sample_indices[label][0]  # Get the first sample index for this class
    text = val.loc[sample_idx, 'clean_text'] if 'clean_text' in val.columns else clean_text(val.loc[sample_idx, 'tweet'])  # Get cleaned text
    tokens, attn = get_attention_weights(text)  # Extract tokens and attention weights
    # Keep only first 40 tokens for readability
    max_tokens = min(40, len(tokens))  # Limit to 40 tokens max for display
    tokens = tokens[:max_tokens]  # Truncate token list
    attn = attn[:max_tokens]  # Truncate attention weights
    attn = attn / attn.max()  # Normalize attention weights to [0, 1]

    color = '#2ecc71' if label == 'real' else '#e74c3c'  # Green for real, red for fake
    axes[idx].bar(range(len(tokens)), attn, color=color, alpha=0.7)  # Plot bar chart of attention weights
    axes[idx].set_xticks(range(len(tokens)))  # Set x-tick positions
    axes[idx].set_xticklabels(tokens, rotation=45, ha='right', fontsize=8)  # Label x-ticks with tokens (rotated)
    axes[idx].set_title(f'Attention Weights - {label.upper()} News', fontsize=13)  # Set subplot title
    axes[idx].set_ylabel('Attention')  # Set y-axis label
    axes[idx].grid(axis='y', alpha=0.3)  # Add light horizontal grid

plt.tight_layout()  # Adjust layout
plt.savefig('reports/figures/17_attention_visualization.png', dpi=150, bbox_inches='tight')  # Save figure
plt.close()  # Close figure
print("[OK] Figure 17 saved")  # Print confirmation

# ================================================================
# FINAL SUMMARY
# ================================================================
print("\n" + "=" * 60)  # Print separator line
print("RESUME FINAL DU PROJET")  # Print title (Final Project Summary)
print("=" * 60)  # Print closing separator line
print(f"\nDataset: COVID-19 Fake News (10,700 echantillons)")  # Print dataset description
print(f"Train: 6,420 | Val: 2,140 | Test: 2,140")  # Print dataset split sizes
print(f"\nMeilleurs resultats sur validation:")  # Print header (Best results on validation)
print("-" * 45)  # Print separator line
best_model = max(results.items(), key=lambda x: x[1]['F1'])  # Find the model with the highest F1 score
for name, r in sorted(results.items(), key=lambda x: x[1]['F1'], reverse=True):  # Loop over models sorted by F1 (descending)
    marker = " <-- BEST" if name == best_model[0] else ""  # Add "BEST" marker to top model
    auc_str = f", AUC: {r.get('AUC-ROC', 0):.4f}" if 'AUC-ROC' in r else ""  # Format AUC string if available
    print(f"  {name:25s} Acc: {r['Accuracy']:.4f}, F1: {r['F1']:.4f}{auc_str}{marker}")  # Print model name and metrics

print(f"\nFigures sauvees: reports/figures/ ({len(os.listdir('reports/figures/'))} fichiers)")  # Print number of saved figures
print(f"Modeles sauves: models/ ({len(os.listdir('models/'))} fichiers)")  # Print number of saved models
print(f"Donnees traitees: data/processed/")  # Print processed data directory
print(f"Resultats CSV: reports/final_results.csv")  # Print path to final results CSV
print("\n" + "=" * 60)  # Print separator line
print("PROJET TERMINE AVEC SUCCES!")  # Print success message (Project Completed Successfully!)
print("=" * 60)  # Print closing separator line
