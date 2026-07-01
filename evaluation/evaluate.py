"""
evaluate.py — Offline model comparison script.

Run this to generate the three comparison charts saved in evaluation/results/:
  - model_comparison.png      : Accuracy and AUC bar chart for all three models
  - confusion_matrix.png      : Confusion matrix for the DME (GradientBoosting) model
  - crossval_comparison.png   : 5-fold CV AUC with error bars

This is an academic/research script, not part of the live Streamlit app.
It compares three approaches:
  1. Baseline: always predicts the majority class (trivial, sets the floor)
  2. Logistic Regression: a simple linear model with no CBR memory
  3. DME (GradientBoosting + memory): the actual system, using CBR alongside ML

All three are trained and evaluated on the same MIMIC-IV demo dataset so the
comparison is fair. Class imbalance (87:13) is handled via sample weights
for the GBC model and class_weight='balanced' for Logistic Regression.
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (accuracy_score, roc_auc_score,
                             classification_report, confusion_matrix,
                             ConfusionMatrixDisplay)
from sklearn.utils.class_weight import compute_sample_weight
from modules.outcome_linker import get_completed_traces_with_outcomes
from modules.feature_pipeline import preprocess, get_feature_columns

# Make sure the output folder exists before we try to save anything
os.makedirs('evaluation/results', exist_ok=True)

print("Loading data...")
df         = get_completed_traces_with_outcomes()
processed  = preprocess(df, fit_scaler=True)
feature_cols = get_feature_columns(processed)
# Drop any feature columns that didn't survive preprocessing (edge case)
feature_cols = [c for c in feature_cols if c in processed.columns]

X = processed[feature_cols].fillna(0)
y = processed['outcome_value']

# Stratified split preserves the 87:13 class ratio in both train and test halves
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Sample weights for the GBC — it doesn't accept class_weight directly,
# so we compute per-sample weights that achieve the same effect
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Three models to compare.
# The Dummy baseline is important — it tells us what "always predict majority"
# gets you, which is effectively the floor any real model must beat.
models = {
    'Baseline\n(Always Predict Majority)': DummyClassifier(strategy='most_frequent'),
    'Logistic Regression\n(No Memory)':    LogisticRegression(
                                               max_iter=500, class_weight='balanced'),
    'DME\n(Gradient Boosting + Memory)':   GradientBoostingClassifier(
                                               n_estimators=100, random_state=42,
                                               subsample=0.8, min_samples_leaf=5),
}

results = {}
print("\n=== MODEL EVALUATION RESULTS ===\n")

for name, model in models.items():
    # GBC needs sample weights instead of class_weight, the others don't
    if 'Gradient' in name:
        model.fit(X_train, y_train, sample_weight=sample_weights)
    else:
        model.fit(X_train, y_train)

    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]
    acc   = accuracy_score(y_test, preds)
    auc   = roc_auc_score(y_test, probs)

    # Cross-validation on the full dataset gives a more stable AUC estimate
    # than just the single test split (especially with n=100)
    cv_scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')

    results[name] = {
        'accuracy': acc,
        'auc':      auc,
        'cv_auc':   cv_scores.mean(),
        'cv_std':   cv_scores.std(),
    }

    clean_name = name.replace('\n', ' ')
    print(f"{clean_name}")
    print(f"  Accuracy       : {acc:.3f}")
    print(f"  AUC-ROC        : {auc:.3f}")
    print(f"  CV AUC (5-fold): {cv_scores.mean():.3f} +/- {cv_scores.std():.3f}")
    print(f"  Classification Report:")
    print(classification_report(y_test, preds, zero_division=0))
    print()

# Chart 1: Accuracy and AUC side-by-side bar chart.
# Color-coded: red = baseline, amber = logistic, green = DME
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle('DME Prototype — Model Performance Comparison', fontsize=14, fontweight='bold')

model_names = list(results.keys())
accuracies  = [results[m]['accuracy'] for m in model_names]
aucs        = [results[m]['auc']      for m in model_names]
colors      = ['#d9534f', '#f0ad4e', '#5cb85c']

axes[0].bar(model_names, accuracies, color=colors, width=0.5, edgecolor='white')
axes[0].set_title('Accuracy', fontsize=12)
axes[0].set_ylim(0, 1.1)
axes[0].set_ylabel('Score')
axes[0].axhline(y=0.9, color='gray', linestyle='--', alpha=0.5, label='90% line')
for i, v in enumerate(accuracies):
    axes[0].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

axes[1].bar(model_names, aucs, color=colors, width=0.5, edgecolor='white')
axes[1].set_title('AUC-ROC (Discriminative Power)', fontsize=12)
axes[1].set_ylim(0, 1.1)
axes[1].set_ylabel('Score')
axes[1].axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random baseline')
for i, v in enumerate(aucs):
    axes[1].text(i, v + 0.02, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')

plt.tight_layout()
plt.savefig('evaluation/results/model_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: evaluation/results/model_comparison.png")

# Chart 2: Confusion matrix for the DME model only.
# We show the DME model specifically because that's what the system actually uses
dme_model = list(models.values())[2]
cm   = confusion_matrix(y_test, dme_model.predict(X_test))
fig2, ax2 = plt.subplots(figsize=(6, 5))
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                               display_labels=['Did Not Improve', 'Improved'])
disp.plot(ax=ax2, colorbar=False, cmap='Blues')
ax2.set_title('DME Model — Confusion Matrix', fontsize=12, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation/results/confusion_matrix.png', dpi=150, bbox_inches='tight')
print("Saved: evaluation/results/confusion_matrix.png")

# Chart 3: Cross-validation AUC with error bars.
# Error bars show +/- 1 std across the 5 folds — a wide bar means the model
# is sensitive to which patients end up in each fold (common with n=100)
fig3, ax3 = plt.subplots(figsize=(8, 5))
cv_means = [results[m]['cv_auc'] for m in model_names]
cv_stds  = [results[m]['cv_std'] for m in model_names]
ax3.bar(model_names, cv_means, yerr=cv_stds, color=colors,
        width=0.5, edgecolor='white', capsize=8)
ax3.set_title('5-Fold Cross-Validation AUC-ROC', fontsize=12, fontweight='bold')
ax3.set_ylim(0, 1.1)
ax3.set_ylabel('AUC-ROC Score')
ax3.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
for i, (v, s) in enumerate(zip(cv_means, cv_stds)):
    ax3.text(i, v + s + 0.03, f'{v:.3f}', ha='center', fontsize=10, fontweight='bold')
plt.tight_layout()
plt.savefig('evaluation/results/crossval_comparison.png', dpi=150, bbox_inches='tight')
print("Saved: evaluation/results/crossval_comparison.png")

print("\n=== EVALUATION COMPLETE ===")
print("All charts saved to evaluation/results/")
print("\nSummary Table:")
print(f"{'Model':<35} {'Accuracy':>10} {'AUC-ROC':>10} {'CV AUC':>10}")
print("-" * 70)
for name, r in results.items():
    clean = name.replace('\n', ' ')
    print(f"{clean:<35} {r['accuracy']:>10.3f} {r['auc']:>10.3f} {r['cv_auc']:>10.3f}")
