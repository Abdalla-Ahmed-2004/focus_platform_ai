


from pathlib import Path

import pandas as pd
import numpy as np
import os

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT_FILE = DATA_DIR / "2015_100_skill_builders_main_problems.csv"
DEFAULT_OUTPUT_FILE = DATA_DIR / "final_processed_data.csv"

def load_assistments_file(file_path):
    file_path = str(file_path)
    for encoding in ['utf-8', 'ISO-8859-1', 'cp1252']:
        try:
            if file_path.endswith('.csv'):
                return pd.read_csv(file_path, encoding=encoding, low_memory=False)
            else:
                return pd.read_excel(file_path)
        except Exception:
            continue
    raise ValueError(f"Unable to read file: {file_path}")

def preprocess_skill_centric(file_path, output_path=None):
    print("Starting Final Calibration Preprocessing (v3.3)...")
    
    if output_path is None:
        output_path = DEFAULT_OUTPUT_FILE

    df = load_assistments_file(file_path)
    
    # 1. Clean and prepare columns
    rename_map = {'sequence_id': 'skill_id', 'log_id': 'order_id'}
    df = df.rename(columns=rename_map)
    
    df['skill_id'] = df['skill_id'].astype(str).str.split(r'[,;_-]')
    df = df.explode('skill_id')
    df['skill_id'] = pd.to_numeric(df['skill_id'], errors='coerce')
    df = df.dropna(subset=['skill_id', 'user_id', 'correct'])
    
    if 'order_id' not in df.columns:
        df['order_id'] = df.index

    df['correct'] = df['correct'].astype(int)
    df = df.sort_values(by=['user_id', 'order_id']).reset_index(drop=True)

    # 2. Compute features with probability boosting
    grouped_us = df.groupby(['user_id', 'skill_id'])

    # --- Core features ---
    df['opportunity_count'] = grouped_us.cumcount()

    def get_streak(series):
        streak, current = [], 0
        for val in series.values:
            streak.append(current)
            current = (current + 1) if val == 1 else 0
        return pd.Series(streak, index=series.index)
    df['consecutive_correct'] = grouped_us['correct'].transform(get_streak)

    # Increase alpha so the model rewards strong performance faster
    def get_decayed(series, alpha=0.55): 
        mastery = []
        current_m = 0.5 
        for val in series.values:
            mastery.append(current_m)
            current_m = current_m + alpha * (val - current_m)
        return pd.Series(mastery, index=series.index)
    df['decayed_mastery'] = grouped_us['correct'].transform(get_decayed)

    # Skill difficulty
    skill_stats = df.groupby('skill_id')['correct'].mean().reset_index()
    skill_stats.columns = ['skill_id', 'success_rate']
    skill_stats['skill_difficulty_avg'] = (1 - skill_stats['success_rate']).round(3)
    df = df.merge(skill_stats[['skill_id', 'skill_difficulty_avg']], on='skill_id', how='left')
    
    df['student_skill_history'] = grouped_us['correct'].transform(
        lambda x: x.shift(1).expanding(min_periods=1).mean()
    ).fillna(1 - df['skill_difficulty_avg'])

    # --- Feature improvements to increase probabilities ---
    
    # Performance efficiency: raise the ceiling to distinguish experts clearly
    df['performance_efficiency'] = (df['student_skill_history'] / (df['skill_difficulty_avg'] + 0.05)).clip(0, 15)
    
    # Combine mastery and success into a strong stability signal
    df['consistency_success_sync'] = (df['decayed_mastery'] * (1 - df['skill_difficulty_avg']) * 1.5).clip(0, 1)

    # Increase streak impact
    raw_streak = df['consecutive_correct'] * np.power(2.0, df['skill_difficulty_avg'])
    df['weighted_streak'] = raw_streak.clip(upper=15.0)

    # Momentum and gap features
    df['complexity_gap'] = (df['student_skill_history'] - df['skill_difficulty_avg']).clip(-1, 1)
    df['learning_momentum'] = (df['decayed_mastery'] - df['student_skill_history']).clip(-1, 1)
    
    # Use a softer log scale to reward consistency
    df['weighted_consistency'] = (np.log2(df['consecutive_correct'] + 1) * (1 - df['skill_difficulty_avg']))
    
    df['total_experience_score'] = np.log1p(df['opportunity_count'])
    df['mastery_history_gap'] = (df['decayed_mastery'] - df['student_skill_history'])

    df['user_success_rate'] = df.groupby('user_id')['correct'].transform(
        lambda x: x.shift(1).expanding().mean()
    ).fillna(0.5)

    # --- Final cleanup ---
    df = df.fillna(0.0)
    
    final_cols = [
        'order_id', 'user_id', 'skill_id', 'correct',
        'decayed_mastery', 'skill_difficulty_avg', 'student_skill_history', 
        'complexity_gap', 'learning_momentum', 'consecutive_correct', 
        'opportunity_count', 'weighted_consistency', 'total_experience_score',
        'mastery_history_gap', 'weighted_streak', 'user_success_rate',
        'performance_efficiency', 'consistency_success_sync'
    ]
    
    df = df[final_cols].round(4)
    df.to_csv(output_path, index=False)
    
    print(f"Preprocessing v3.3 complete! Probabilities should now be more generous.")
    return df

if __name__ == "__main__":
    if os.path.exists(DEFAULT_INPUT_FILE):
        preprocess_skill_centric(DEFAULT_INPUT_FILE)