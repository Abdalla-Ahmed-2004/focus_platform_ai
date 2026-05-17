from pathlib import Path
import pandas as pd
import numpy as np
import os
import joblib

# 1. Set project paths automatically (so the code runs on any machine without path edits)
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_INPUT_FILE = DATA_DIR / "2015_100_skill_builders_main_problems.csv"

# Output files that this script will create and save as fast NumPy arrays
X_OUTPUT_PATH = DATA_DIR / "lstm_X.npy"
Y_OUTPUT_PATH = DATA_DIR / "lstm_y.npy"
METADATA_PATH = DATA_DIR / "lstm_metadata.pkl"

def load_assistments_file(file_path):
    """Robust reader for CSV files trying multiple encodings to avoid read errors."""
    file_path = str(file_path)
    for encoding in ['utf-8', 'ISO-8859-1', 'cp1252']:
        try:
            return pd.read_csv(file_path, encoding=encoding, low_memory=False)
        except Exception:
            continue
    raise ValueError(f"❌ Could not read file, ensure it exists at: {file_path}")

def preprocess_for_lstm(file_path, max_seq_len=50):
    print(f"🚀 Starting preprocessing for LSTM... (fixed sequence length: {max_seq_len})")
    
    # Step A: read the original data file
    df = load_assistments_file(file_path)
    
    # Step B: clean and normalize column names to match code expectations
    df = df.rename(columns={'sequence_id': 'skill_id', 'log_id': 'order_id'})
    df['skill_id'] = df['skill_id'].astype(str)
    df['user_id'] = df['user_id'].astype(str)
    
    # Convert the answer column to strict 1 (correct) / 0 (incorrect)
    df['correct'] = (df['correct'] >= 1.0).astype(int)
    
    # Step C: compute skill difficulty statistically from data (1 - overall correct rate)
    print("⏳ Calculating skill difficulty statistics...")
    global_difficulty = df.groupby('skill_id')['correct'].mean().to_dict()
    for k in global_difficulty:
        global_difficulty[k] = round(1.0 - global_difficulty[k], 4)
        
    # Inject difficulty column into the data; default to 0.5 for skills with no data
    df['skill_difficulty_avg'] = df['skill_id'].map(global_difficulty).fillna(0.5)
    
    # Step D: strict chronological ordering (group by student then by answer order)
    df = df.sort_values(by=['user_id', 'order_id']).reset_index(drop=True)
    
    X_sequences = []
    y_sequences = []
    
    # Split data and aggregate per student
    grouped = df.groupby('user_id')
    
    print("⏳ Slicing student arrays and building sequential time-series...")
    for user_id, group in grouped:
        corrects = group['correct'].values
        difficulties = group['skill_difficulty_avg'].values
        
        # LSTM learns from sequences; skip students with fewer than 2 steps
        if len(corrects) < 2:
            continue
            
        # combine two features per step [current_response, current_question_difficulty]
        features = np.column_stack((corrects, difficulties))
        
        # Deep Knowledge Tracing feature engineering:
        # Inputs X: from first question up to the penultimate
        # Targets y: student's actual answers from second question to the end (predict next-step)
        X_student = features[:-1]
        y_student = corrects[1:]
        
        # split very long sequences into fixed chunks of 50 questions to avoid LSTM memory issues
        for i in range(0, len(X_student), max_seq_len):
            X_chunk = X_student[i:i+max_seq_len]
            y_chunk = y_student[i:i+max_seq_len]
            
            # Apply padding (-1.0) if the student solved fewer than max_seq_len questions
            if len(X_chunk) < max_seq_len:
                pad_width = max_seq_len - len(X_chunk)
                # padding with -1.0 allows the LSTM Masking layer to ignore these slots
                X_chunk = np.pad(X_chunk, ((0, pad_width), (0, 0)), mode='constant', constant_values=-1.0)
                y_chunk = np.pad(y_chunk, (0, pad_width), mode='constant', constant_values=-1.0)
                
            X_sequences.append(X_chunk)
            y_sequences.append(y_chunk)

    # Convert lists to final NumPy arrays ready for deep learning
    X_final = np.array(X_sequences, dtype=np.float32)
    y_final = np.array(y_sequences, dtype=np.float32)
    
    # Step E: save final files to disk
    os.makedirs(os.path.dirname(X_OUTPUT_PATH), exist_ok=True)
    np.save(X_OUTPUT_PATH, X_final)
    np.save(Y_OUTPUT_PATH, y_final)
    
    # Save metadata (used by the API to look up difficulties later)
    metadata = {
        'max_seq_len': max_seq_len,
        'global_difficulty': global_difficulty,
        'features_count': X_final.shape[2]  # will be 2 [response, difficulty]
    }
    joblib.dump(metadata, METADATA_PATH)
    print("\n" + "="*60)
    print("✅ Preprocessing for LSTM completed successfully!")
    print(f"📦 Saved input array (X) -> {X_OUTPUT_PATH} | shape: {X_final.shape}")
    print(f"📦 Saved target array (y) -> {Y_OUTPUT_PATH} | shape: {y_final.shape}")
    print(f"⚙️ Saved metadata and difficulties -> {METADATA_PATH}")
    print("="*60)

if __name__ == "__main__":
    preprocess_for_lstm(DEFAULT_INPUT_FILE, max_seq_len=50)