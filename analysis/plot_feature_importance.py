from pathlib import Path

import joblib
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import numpy as np

# Use non-interactive backend
matplotlib.use('Agg')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "evaluate_skill_model_vFINAL.pkl"
OUTPUT_PATH = Path(__file__).resolve().parent / "feature_importance_plot.png"

# Load the trained model
print("Loading trained model...")
model_data = joblib.load(MODEL_PATH)
model = model_data['model']

# Feature names
FEATURES = [
    'decayed_mastery', 'skill_difficulty_avg', 'student_skill_history',
    'complexity_gap', 'learning_momentum', 'consecutive_correct',
    'opportunity_count', 'weighted_consistency', 'total_experience_score',
    'mastery_history_gap', 'weighted_streak', 'user_success_rate',
    'performance_efficiency', 'consistency_success_sync'
]

# Get feature importance
feature_importance = model.feature_importances_

# Convert to percentages
importance_percentage = (feature_importance / feature_importance.sum()) * 100

# Create a DataFrame for better visualization
importance_df = pd.DataFrame({
    'Feature': FEATURES,
    'Importance (%)': importance_percentage
}).sort_values('Importance (%)', ascending=True)

print("\n" + "="*60)
print("Feature Importance (Percentage)")
print("="*60)
for idx, row in importance_df.sort_values('Importance (%)', ascending=False).iterrows():
    print(f"{row['Feature']:<35} : {row['Importance (%)']:>6.2f}%")
print("="*60)

# Create visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Create horizontal bar plot
colors = plt.cm.viridis(np.linspace(0, 1, len(importance_df)))
bars = ax.barh(importance_df['Feature'], importance_df['Importance (%)'], color=colors, edgecolor='black', linewidth=1.2)

# Add percentage labels on the bars
for i, (bar, value) in enumerate(zip(bars, importance_df['Importance (%)'])):
    ax.text(value + 0.2, bar.get_y() + bar.get_height()/2, 
            f'{value:.2f}%', va='center', fontsize=10, fontweight='bold')

# Styling
ax.set_xlabel('Importance (%)', fontsize=12, fontweight='bold')
ax.set_ylabel('Features', fontsize=12, fontweight='bold')
ax.set_title('Feature Importance in LightGBM Model (by Percentage)', fontsize=14, fontweight='bold', pad=20)
ax.set_xlim(0, importance_percentage.max() + 5)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add a subtle background
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

plt.tight_layout()

# Save the plot
plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches='tight')
print(f"\nPlot saved as '{OUTPUT_PATH.name}'")
