from pathlib import Path
import numpy as np
import tensorflow as tf
import joblib

# 1. Setup automatic absolute paths matching the project structure
PROJECT_ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(".")
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODEL_PATH = MODELS_DIR / "lstm_knowledge_tracing_model.h5"
METADATA_PATH = DATA_DIR / "lstm_metadata.pkl"

def run_test_prediction():
    print("⏳ Loading trained LSTM model and metadata...")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model file not found at: {MODEL_PATH}\nPlease run train_lstm.py first.")
        
    # Load the trained network
    model = tf.keras.models.load_model(str(MODEL_PATH))
    
    # Load metadata sequence configurations
    max_seq_len = 50
    if METADATA_PATH.exists():
        metadata = joblib.load(str(METADATA_PATH))
        max_seq_len = metadata.get('max_seq_len', 50)
        
    print("✅ Model and metadata loaded successfully!\n")

    # 2. Simulate the exact JSON Payload coming from Laravel after a Quiz
    sample_payload = [
        {
            "skill_id": "201",
            "skill_name": "Kinematic Matrices",
            "skill_difficulty_avg": 0.45,
            "student_history": [1, 0, 1, 1, 1, 1, 1]  # High performance profile
        },
        {
            "skill_id": "202",
            "skill_name": "Advanced Probability",
            "skill_difficulty_avg": 0.75,
            "student_history": [0, 1, 0, 0, 0]        # Struggling profile
        },
        {
            "skill_id": "203",
            "skill_name": "Logarithmic Functions",
            "skill_difficulty_avg": 0.60,
            "student_history": [0, 0, 0, 1, 1, 1, 1]  # High Learning Curve / Recency Momentum
        }
    ]

    print("🚀 Simulating LSTM Predictions for Laravel Payload...")
    print("=" * 75)

    for item in sample_payload:
        skill_id = item["skill_id"]
        skill_name = item["skill_name"]
        difficulty = float(item["skill_difficulty_avg"])
        student_history = item["student_history"]
        
        # Calculate static metrics 
        total_attempts = len(student_history)
        total_correct = sum(student_history)
        
        if total_attempts == 0:
            # Default cold-start formula if history is empty
            mastery_score = round((1.0 - difficulty) * 100, 2)
        else:
            # Build the structural [correct, difficulty] timestep features
            features = [[float(ans), difficulty] for ans in student_history]
            features = np.array(features, dtype=np.float32)
            actual_len = len(features)
            
            # Apply Padding or Truncation to fit the fixed 50 timesteps required by LSTM layers
            if actual_len < max_seq_len:
                pad_width = max_seq_len - actual_len
                padded_features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant', constant_values=-1.0)
            else:
                padded_features = features[-max_seq_len:]
                actual_len = max_seq_len
                
            # Reshape tensor to match Keras input dimensions: [Batch_Size=1, Timesteps=50, Feature_Count=2]
            input_tensor = np.expand_dims(padded_features, axis=0)
            
            # Execute Feed-forward prediction
            predictions = model.predict(input_tensor, verbose=0)
            
            # Extract probability from the last active timestep cell before padding
            last_step_index = actual_len - 1
            mastery_prob = predictions[0, last_step_index, 0]
            mastery_score = round(float(mastery_prob) * 100, 2)
            
        # Classify mastery threshold states
        if mastery_score < 50.0:
            status = "Red (Requires Support)"
        elif mastery_score < 75.0:
            status = "Developing (On Track)"
        else:
            status = "Green (Mastered)"
            
        # Display formatted output
        print(f"📘 Skill: {skill_name} (ID: {skill_id})")
        print(f"   📊 Attempts Count : {total_attempts}")
        print(f"   ✅ Correct Answers : {total_correct}")
        print(f"   📈 LSTM Mastery Score: {mastery_score}%")
        print(f"   🏷️ Student Status   : {status}")
        print("-" * 75)

if __name__ == "__main__":
    run_test_prediction()