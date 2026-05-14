
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import warnings
import logging
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

# 1. Environment setup (clean output)
warnings.filterwarnings("ignore")
logging.getLogger('lightgbm').setLevel(logging.ERROR)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "final_processed_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "evaluate_skill_model_vFINAL.pkl"

# 2. Load the data
print(f"Loading: {DATA_PATH.name} ...")
try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"Error: File '{DATA_PATH.name}' not found.")
    exit()

# Feature list (the 14 agreed features used for training)
FEATURES = [
    'decayed_mastery', 'skill_difficulty_avg', 'student_skill_history',
    'complexity_gap', 'learning_momentum', 'consecutive_correct',
    'opportunity_count', 'weighted_consistency', 'total_experience_score',
    'mastery_history_gap', 'weighted_streak', 'user_success_rate',
    'performance_efficiency', 'consistency_success_sync'
]

X = df[FEATURES].fillna(0)
y = df['correct']

# Split the data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# 3. Build the model (Grounded Strategy)
print("Training Grounded LightGBM Model (Final Version)...")
model = LGBMClassifier(
    n_estimators=3000,      
    learning_rate=0.01,      # Low rate builds confidence gradually.
    num_leaves=63,           # Moderate complexity reduces overfitting.
    max_depth=8,             # Depth that captures relationships without excess complexity.
    min_child_samples=40,    # Protects against decisions based on too few samples.
    
    # Class weights keep the model attentive to weak students.
    class_weight={0: 1.8, 1: 1.0}, 

    # Regularization reduces overconfidence.
    reg_alpha=0.5,           
    reg_lambda=0.5,
    
    boost_from_average=True, # Start from the base rate for realistic probabilities.
    feature_fraction=0.8,    
    importance_type='gain', 
    random_state=42,
    verbosity=-1             
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    eval_metric='binary_logloss'
)

# 4. Generate raw probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# 5. Display the full analysis table for threshold selection
print("\n" + "="*115)
print(f"{'THR':<6} | {'Accuracy':<8} | {'P(0)':<6} | {'R(0)':<6} | {'P(1)':<6} | {'R(1)':<6} | {'Insight / Recommendation'}")
print("-" * 115)

# Sweep from 0.1 to 0.95 to inspect model behavior
thresholds = np.arange(0.1, 1.0, 0.05)

for thr in thresholds:
    y_pred = (y_prob >= thr).astype(int)
    acc = accuracy_score(y_test, y_pred)
    
    # Compute precision and recall for each class (0 and 1)
    precision, recall, _, _ = precision_recall_fscore_support(
        y_test, y_pred, labels=[0, 1], zero_division=0
    )
    
    # Simple recommendation for each threshold
    if recall[0] > 0.80 and recall[1] > 0.65:
        status = "Best balance for the project"
    elif recall[0] > 0.90:
        status = "Strict: high security (detects most weak students)"
    elif recall[1] > 0.85:
        status = "Generous: easier to pass"
    else:
        status = "Standard calibration"

    print(f"{thr:<6.2f} | {acc:<8.4f} | {precision[0]:<6.2f} | {recall[0]:<6.2f} | {precision[1]:<6.2f} | {recall[1]:<6.2f} | {status}")

print("-" * 115)

# 6. Manual threshold selection
while True:
    try:
        user_input = input("\nBased on the metrics above, enter your chosen threshold (e.g., 0.60): ")
        chosen_thr = float(user_input)
        if 0 < chosen_thr < 1:
            break
        else:
            print("Value must be between 0.0 and 1.0.")
    except ValueError:
        print("Invalid format. Please enter a decimal number.")

# 7. Save the final model package
output_file = MODEL_PATH
joblib.dump({
    'model': model,
    'features': FEATURES,
    'threshold': chosen_thr,
    'metadata': {
        'version': '4.8 (Final Balanced)',
        'calibration': 'Grounded Anti-Optimism',
        'selected_threshold': chosen_thr
    }
}, output_file)

print(f"\nSUCCESS! Your final model is ready.")
print(f"File saved as: {output_file.name}")
print(f"Embedded threshold: {chosen_thr}")