from pathlib import Path
import sys

import joblib
import pandas as pd
import numpy as np

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# 1. Load the model and its saved metadata
MODEL_NAME = Path(__file__).resolve().parent.parent / 'models' / 'evaluate_skill_model_vFINAL.pkl'

try:
    # Load the full saved payload
    payload = joblib.load(MODEL_NAME)
    
    model = payload['model']
    features_ordered = payload['features']
    best_threshold = payload['threshold']
    
    print(f"Model loaded successfully: {MODEL_NAME.name}")
    print(f"Saved threshold: {best_threshold}")
    print("-" * 60)
except Exception as e:
    print(f"Error loading model: {e}")
    # Default values keep the script running if loading fails
    best_threshold = 0.70
    features_ordered = []

def predict_student_status(student_data_dict, custom_threshold=None):
    current_threshold = custom_threshold if custom_threshold is not None else best_threshold
    
    # Convert the input to a DataFrame and keep the feature order stable
    input_df = pd.DataFrame([student_data_dict])
    
    # Ensure all required columns exist and fill missing values with zero
    input_df = input_df.reindex(columns=features_ordered, fill_value=0)
    
    # Compute the probability
    probability = model.predict_proba(input_df)[0, 1]
    
    # Confidence bands
    if probability < current_threshold:
        status = "Weak Skill 🔴"
    elif probability < (current_threshold + (1 - current_threshold) / 2):
        status = "Developing 🟡"
    else:
        status = "Strong Skill 🟢"
        
    return {"prob": round(probability * 100, 2), "status": status}

# ==========================================
# 2. Test scenarios
# ==========================================
# These are realistic test cases aligned with the current feature set

precise_tests = {
    "SUPER_EXPERT (شاطر جداً)": {
        'decayed_mastery': 0.98, 'skill_difficulty_avg': 0.85, 
        'student_skill_history': 0.95,
        # 'user_success_rate': 0.98, 
        'weighted_streak': 20.0, 'consecutive_correct': 15,
        'opportunity_count': 100, 'learning_momentum': 0.05, 
        'weighted_consistency': 0.99, 'total_experience_score': 5.0, 
        'complexity_gap': 0.10, 'mastery_history_gap': 0.01,
        'performance_efficiency': 0.98, 'consistency_success_sync': 0.98
    },
    "TYPICAL_AVERAGE (متوسط منطقي)": {
        # التعديل: رفع الثبات ونسبة النجاح لضمان تخطي حاجز الـ 60%
        'decayed_mastery': 0.72,          # إتقان تراكمي جيد جداً
        'skill_difficulty_avg': 0.50, 
        'student_skill_history': 0.68, 
        # 'user_success_rate': 0.75,        # نجاح في 3/4 الأسئلة تقريباً
        'weighted_streak': 7.0,           # استمرارية واضحة
        'consecutive_correct': 7,
        'opportunity_count': 50,          # خبرة متوسطة كافية
        'learning_momentum': 0.15, 
        'weighted_consistency': 0.82,     # الثبات العالي هو ما يرفعه للمنطقة الخضراء
        'total_experience_score': 3.5, 
        'complexity_gap': 0.20,           # مستواه أعلى من متوسط صعوبة المادة
        'mastery_history_gap': 0.04,
        'performance_efficiency': 0.78,
        'consistency_success_sync': 0.75
    },
    "TOTAL_FAILURE (فاشل جداً)": {
        'decayed_mastery': 0.05, 'skill_difficulty_avg': 0.30, 
        'student_skill_history': 0.05, # 'user_success_rate': 0.12, 
        'weighted_streak': 0.0, 'consecutive_correct': 0,
        'opportunity_count': 15, 'learning_momentum': -0.85, 
        'weighted_consistency': 0.05, 'total_experience_score': 0.4, 
        'complexity_gap': -0.45, 'mastery_history_gap': -0.50,
        'performance_efficiency': 0.10, 'consistency_success_sync': 0.05
    }
}


evolution_snapshots = {
    "Early_Beginner (بداية متعثرة)": {
        'decayed_mastery': 0.18, 'skill_difficulty_avg': 0.40, 
        'student_skill_history': 0.12, 'opportunity_count': 5, 
        'weighted_streak': 1.0, # 'user_success_rate': 0.22,
        'weighted_consistency': 0.10, 'consecutive_correct': 1,
        'performance_efficiency': 0.15, 'consistency_success_sync': 0.10,
        'total_experience_score': 0.2, 'learning_momentum': 0.05, 
        'complexity_gap': -0.25, 'mastery_history_gap': 0.10
    },
    
    "Intermediate_Struggler (مرحلة الصعود)": {
        # التعديل: رفع "الزخم" و"الثبات" ليكون في الستينيات
        'decayed_mastery': 0.60, 
        'skill_difficulty_avg': 0.50, 
        'student_skill_history': 0.55, 
        'opportunity_count': 35,
        'weighted_streak': 5.0, 
        # 'user_success_rate': 0.68, 
        'weighted_consistency': 0.75,     # أهم عامل للنجاح
        'consecutive_correct': 5,
        'performance_efficiency': 0.70, 
        'consistency_success_sync': 0.68,
        'total_experience_score': 2.6, 
        'learning_momentum': 0.65,        # زخم انفجاري يدل على تحسن حقيقي
        'complexity_gap': 0.15, 
        'mastery_history_gap': 0.15
    },

   "Transformed_Expert_20_Opps": {
        "decayed_mastery": 0.94,
        "skill_difficulty_avg": 0.75,
        "student_skill_history": 0.88,
        "opportunity_count": 20, 
        "consecutive_correct": 12, 
        "total_experience_score": 3.0445, 
        "weighted_streak": 16.24, 
        "weighted_consistency": 0.925,
        "performance_efficiency": 1.10,
        "consistency_success_sync": 0.3525, 
        "learning_momentum": 0.06,
        "complexity_gap": 0.13,
        "mastery_history_gap": 0.06,
        # "user_success_rate": 0.92
    }
}

# --- تشغيل الاختبارات ---
print("\n" + " TARGET SCENARIOS ".center(60, "="))
for name, data in precise_tests.items():
    res = predict_student_status(data)
    print(f"{name:<35} | Prob: {res['prob']:>6}% | Status: {res['status']}")

print("\n" + " EVOLUTION JOURNEY ".center(60, "="))
for name, data in evolution_snapshots.items():
    res = predict_student_status(data)
    print(f"{name:<35} | Prob: {res['prob']:>6}% | Status: {res['status']}")