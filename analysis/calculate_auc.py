from pathlib import Path

import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, auc
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "evaluate_skill_model_vFINAL.pkl"
DATA_PATH = PROJECT_ROOT / "data" / "final_processed_data.csv"
OUTPUT_PATH = Path(__file__).resolve().parent / "roc_curve_plot.png"

# Load the trained model
print("Loading trained model...")
model_data = joblib.load(MODEL_PATH)
model = model_data['model']
features = model_data['features']

# Load data
print("Loading data...")
df = pd.read_csv(DATA_PATH)

X = df[features].fillna(0)
y = df['correct']

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Get predictions and probabilities
print("Calculating AUC...")
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate AUC
auc_score = roc_auc_score(y_test, y_prob)

print("\n" + "="*60)
print("MODEL AUC (Area Under the Curve)")
print("="*60)
print(f"AUC Score: {auc_score:.4f}")
print("="*60)

# Generate ROC curve
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Create ROC curve plot
fig, ax = plt.subplots(figsize=(10, 8))

# Plot ROC curve
ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.4f})')
ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')

# Styling
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
ax.set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
ax.set_title('ROC Curve - Model Performance', fontsize=14, fontweight='bold', pad=20)
ax.legend(loc="lower right", fontsize=11)
ax.grid(alpha=0.3, linestyle='--')
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

plt.tight_layout()

# Save the ROC curve plot
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\nROC curve plot saved as '{OUTPUT_PATH.name}'")
