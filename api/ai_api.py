



# from pathlib import Path
# import sys
# import os
# import joblib
# import numpy as np
# import tensorflow as tf
# from flask import Flask, jsonify, request

# app = Flask(__name__)

# # Set project root paths
# PROJECT_ROOT = Path(__file__).resolve().parent.parent
# if str(PROJECT_ROOT) not in sys.path: 
#     sys.path.insert(0, str(PROJECT_ROOT))

# MODEL_DIR = PROJECT_ROOT / "models"
# MODEL_PATH = MODEL_DIR / "lstm_knowledge_tracing_model.h5"
# METADATA_PATH = PROJECT_ROOT / "data" / "lstm_metadata.pkl"

# # Load model and metadata on server start
# lstm_model = None
# max_seq_len = 50

# try:
#     if MODEL_PATH.exists():
#         lstm_model = tf.keras.models.load_model(str(MODEL_PATH))
#         print(" [SUCCESS] LSTM Knowledge Tracing Model Loaded Successfully!")
#     else:
#         print(" [WARNING] Model file not found. Please train the LSTM first.")
# except Exception as e:
#     print(f" [ERROR] Failed to load the LSTM model: {str(e)}")


# def calculate_lstm_mastery(student_history, skill_difficulty):
#     """
#     Calculate the mastery percentage using the LSTM, apply proper padding,
#     and normalize the score so that 90% from the model equals 100% for the student.
#     """
#     if not student_history:
#         return round((1.0 - skill_difficulty) * 100, 2)
        
#     # build features [correct, difficulty] for each timestep
#     features = [[float(ans), float(skill_difficulty)] for ans in student_history]
#     features = np.array(features, dtype=np.float32)
#     actual_len = len(features)
    
#     # Apply padding or truncating to fix sequence length at max_seq_len timesteps
#     if actual_len < max_seq_len:
#         pad_width = max_seq_len - actual_len
#         padded_features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant', constant_values=-1.0)
#     else:
#         padded_features = features[-max_seq_len:]
#         actual_len = max_seq_len
        
#     # Reshape to LSTM input -> [Batch=1, Timesteps=max_seq_len, Features=2]
#     input_tensor = np.expand_dims(padded_features, axis=0)
    
#     # Predict through the deep network
#     predictions = lstm_model.predict(input_tensor, verbose=0)
    
#     # Extract probability from the last actual timestep answered by the student
#     last_step_index = actual_len - 1
#     mastery_prob = predictions[0, last_step_index, 0]
    
#     # 1. Convert raw model probability to percentage (0 - 100)
#     raw_score = float(mastery_prob) * 100
    
#     # 2. Set the custom saturation threshold (90% model score = 100% user score)
#     MAX_MODEL_SCORE = 80.0
    
#     # 3. Apply Min-Max / Linear Scaling Normalization
#     normalized_score = (raw_score / MAX_MODEL_SCORE) * 100
    
#     # 4. Safeguard to ensure the score never exceeds 100.0%
#     final_score = min(normalized_score, 100.0)
    
#     return round(raw_score, 2)


# def _status(prob_percentage):
#     """Determine mastery status based on the normalized percentage"""
#     if prob_percentage < 50.0: 
#         return "Red (weak skill)  "
#     if prob_percentage < 75.0: 
#         return "Developing (On Track)"
#     return "Green (Mastered)"


# def _normalize_skill_batch(payload):
#     """Normalizes incoming payloads from Laravel to a unified format"""
#     if isinstance(payload, list):
#         if payload and isinstance(payload[0], dict) and "student_history" in payload[0]:
#             return payload

#         grouped = {}
#         for item in payload:
#             if not isinstance(item, dict):
#                 continue

#             skill_id = str(item.get("skill_id", ""))
#             if not skill_id:
#                 continue

#             if skill_id not in grouped:
#                 grouped[skill_id] = {
#                     "skill_id": skill_id,
#                     "skill_name": item.get("skill_name", "Unknown Skill"),
#                     "skill_difficulty_avg": float(item.get("skill_difficulty_avg", 0.50)),
#                     "student_history": [],
#                 }

#             grouped[skill_id]["student_history"].append(int(item.get("is_correct", item.get("correctness", 0))))

#         return list(grouped.values()) if grouped else payload

#     if not isinstance(payload, dict):
#         return [payload]

#     if isinstance(payload.get("student_history"), list):
#         return [payload]

#     source_items = None
#     for key in ("items", "history_records", "predictions", "skills", "tests"):
#         if isinstance(payload.get(key), list):
#             source_items = payload[key]
#             break

#     if source_items is None:
#         return [payload]

#     skills_difficulty = payload.get("skills_difficulty") or payload.get("skills_meta") or {}
#     normalized_difficulty = {}
#     for skill_id, difficulty in skills_difficulty.items():
#         if isinstance(difficulty, dict):
#             difficulty = difficulty.get("skill_difficulty_avg", difficulty.get("difficulty", 0.50))
#         normalized_difficulty[str(skill_id)] = float(difficulty)

#     grouped = {}
#     for item in source_items:
#         if not isinstance(item, dict):
#             continue

#         skill_id = str(item.get("skill_id", ""))
#         if not skill_id:
#             continue

#         if skill_id not in grouped:
#             grouped[skill_id] = {
#                 "skill_id": skill_id,
#                 "skill_name": item.get("skill_name", "Unknown Skill"),
#                 "skill_difficulty_avg": normalized_difficulty.get(skill_id, float(item.get("skill_difficulty_avg", 0.50))),
#                 "student_history": [],
#             }

#         grouped[skill_id]["student_history"].append(int(item.get("is_correct", item.get("correctness", 0))))

#     return list(grouped.values()) if grouped else [payload]


# @app.route("/predict", methods=["POST"])
# def predict():
#     """
#     Main endpoint to receive the batched skill payload from Laravel after a quiz.
#     """
#     if lstm_model is None:
#         return jsonify({"error": "LSTM model is not trained or initialized."}), 500

#     payload = request.get_json(silent=True)
#     if not payload: 
#         return jsonify({"error": "Invalid JSON Payload"}), 400

#     skills_input = _normalize_skill_batch(payload)
    
#     results = []
    
#     for item in skills_input:
#         skill_id = str(item.get("skill_id", ""))
#         skill_name = item.get("skill_name", "Unknown Skill")
        
#         # receive the overall answer history for the skill [0, 1, 1, 0, ...]
#         student_history = item.get("student_history", [])
        
#         # receive the skill difficulty sent from Laravel (default to 0.5 if missing)
#         skill_difficulty = float(item.get("skill_difficulty_avg", 0.50))
        
#         try:
#             # 1. compute the required statistics automatically from history
#             total_attempts = len(student_history)
#             total_correct = int(sum(student_history)) # sum of 1s gives number correct
            
#             # 2. compute normalized mastery percentage via the LSTM
#             mastery_score = calculate_lstm_mastery(student_history, skill_difficulty)
#             status_str = _status(mastery_score)
            
#             # 3. build the skill report with the exact required fields
#             results.append({
#                 "skill_id": skill_id,
#                 "skill_name": skill_name,
#                 "total_attempts": total_attempts,   # number of questions for the skill
#                 "total_correct": total_correct,     # number of correct answers
#                 "mastery_score": mastery_score,     # mastery probability percentage (Normalized)
#                 "status": status_str
#             })
            
#         except Exception as e:
#             results.append({
#                 "skill_id": skill_id,
#                 "skill_name": skill_name,
#                 "error": f"Failed to predict for this skill: {str(e)}"
#             })

#     # return the aggregated result back to Laravel
#     return jsonify(results if isinstance(payload, list) else results[0]), 200


# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=5000, debug=True)



from pathlib import Path
import sys
import os
import joblib
import numpy as np
import tensorflow as tf
from flask import Flask, jsonify, request

app = Flask(__name__)

# Set project root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path: 
    sys.path.insert(0, str(PROJECT_ROOT))

MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODEL_DIR / "lstm_knowledge_tracing_model.h5"

# Load model on server start
lstm_model = None
max_seq_len = 50

try:
    if MODEL_PATH.exists():
        lstm_model = tf.keras.models.load_model(str(MODEL_PATH))
        print(" [SUCCESS] LSTM Model Loaded!")
except Exception as e:
    print(f" [ERROR] Failed to load model: {str(e)}")


# def calculate_lstm_mastery(student_history, skill_difficulty):
#     """
#     Final Balanced Mastery Calculation:
#     - Smart Conditional Weighting for improvement.
#     - Dynamic Weighting based on volume.
#     - Mastery Scaling for high achievers.
#     """
#     if not student_history:
#         return round((1.0 - skill_difficulty) * 100, 2)
        
#     features = [[float(ans), float(skill_difficulty)] for ans in student_history]
#     features = np.array(features, dtype=np.float32)
#     actual_len = len(features)
    
#     if actual_len < max_seq_len:
#         pad_width = max_seq_len - actual_len
#         padded_features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant', constant_values=-1.0)
#     else:
#         padded_features = features[-max_seq_len:]
#         actual_len = max_seq_len
        
#     input_tensor = np.expand_dims(padded_features, axis=0)
#     predictions = lstm_model.predict(input_tensor, verbose=0)
    
#     # 1. Model Prediction
#     mastery_prob = float(predictions[0, actual_len - 1, 0])
    
#     # 2. Actual Performance
#     total_attempts = len(student_history)
#     actual_success_rate = sum(student_history) / total_attempts
    
#     # 3. Dynamic Smart Weighting
#     if mastery_prob > actual_success_rate:
#         final_score = mastery_prob # Reward Improvement
#     else:
#         # Dynamic balancing based on volume
#         if total_attempts >= 50:
#             final_score = (mastery_prob * 0.30) + (actual_success_rate * 0.70)
#         elif total_attempts >= 20:
#             final_score = (mastery_prob * 0.50) + (actual_success_rate * 0.50)
#         elif actual_success_rate >= 0.75 and skill_difficulty >= 0.70:
#             final_score = (mastery_prob * 0.30) + (actual_success_rate * 0.70)
#         else:
#             final_score = (mastery_prob * 0.80) + (actual_success_rate * 0.20)
    
#     # 4. Final Scaling for high achievers
#     final_percentage = min(final_score * 100 * 1.15, 100.0)
    
#     return round(final_percentage, 2)

def calculate_lstm_mastery(student_history, skill_difficulty):
    if not student_history:
        return round((1.0 - skill_difficulty) * 100, 2)
    
        # If the student answered questions but got 0 correct, force their score to 0
    if sum(student_history) == 0:
        return 0.0
    features = [[float(ans), float(skill_difficulty)] for ans in student_history]
    features = np.array(features, dtype=np.float32)
    actual_len = len(features)
    
    if actual_len < max_seq_len:
        pad_width = max_seq_len - actual_len
        padded_features = np.pad(features, ((0, pad_width), (0, 0)), mode='constant', constant_values=-1.0)
    else:
        padded_features = features[-max_seq_len:]
        actual_len = max_seq_len
        
    input_tensor = np.expand_dims(padded_features, axis=0)
    predictions = lstm_model.predict(input_tensor, verbose=0)
    
    # 1. Model Prediction
    mastery_prob = float(predictions[0, actual_len - 1, 0])
    
    # 2. Actual Performance
    total_attempts = len(student_history)
    actual_success_rate = sum(student_history) / total_attempts
    
    # 3. Dynamic Smart Weighting (التعديل الذي طلبته)
    if mastery_prob > actual_success_rate:
        final_score = mastery_prob 
    else:
        if total_attempts >= 50:
            final_score = (mastery_prob * 0.30) + (actual_success_rate * 0.70)
        elif total_attempts >= 20:
            final_score = (mastery_prob * 0.50) + (actual_success_rate * 0.50)
        elif actual_success_rate >= 0.75 and skill_difficulty >= 0.70:
            final_score = (mastery_prob * 0.30) + (actual_success_rate * 0.70)
        else:
            final_score = (mastery_prob * 0.80) + (actual_success_rate * 0.20)
    
    # 4. Elastic Scaling (حل مشكلة الثبات عند 100%)
    # إذا كانت النتيجة فوق 80%، نعطيها دفعة للوصول لـ 100%
    # إذا كانت أقل من 80%، نتركها تعكس مستواه الحقيقي بدقة
    if final_score > 0.80:
        final_percentage = min(final_score * 100 + (final_score * 100 - 80) * 0.75, 100.0)
    else:
        final_percentage = final_score * 100
    
    return round(final_percentage, 2)
def _status(prob_percentage):
    if prob_percentage < 50.0: return "Red (weak skill)"
    if prob_percentage < 85.0: return "Developing (On Track)"
    return "Green (Mastered)"


def _normalize_skill_batch(payload):
    # (تم الإبقاء على دالة التوزيع كما هي لضمان توافق Laravel)
    if isinstance(payload, list):
        if payload and isinstance(payload[0], dict) and "student_history" in payload[0]: return payload
        grouped = {}
        for item in payload:
            if not isinstance(item, dict): continue
            skill_id = str(item.get("skill_id", ""))
            if not skill_id: continue
            if skill_id not in grouped:
                grouped[skill_id] = {"skill_id": skill_id, "skill_name": item.get("skill_name", "Unknown"), "skill_difficulty_avg": float(item.get("skill_difficulty_avg", 0.50)), "student_history": []}
            grouped[skill_id]["student_history"].append(int(item.get("is_correct", item.get("correctness", 0))))
        return list(grouped.values()) if grouped else payload
    
    if isinstance(payload, dict) and "student_history" in payload: return [payload]
    
    source_items = None
    for key in ("items", "history_records", "predictions", "skills", "tests"):
        if isinstance(payload.get(key), list): source_items = payload[key]; break
    if source_items is None: return [payload]
    
    grouped = {}
    for item in source_items:
        skill_id = str(item.get("skill_id", ""))
        if not skill_id: continue
        if skill_id not in grouped:
            grouped[skill_id] = {"skill_id": skill_id, "skill_name": item.get("skill_name", "Unknown"), "skill_difficulty_avg": float(item.get("skill_difficulty_avg", 0.50)), "student_history": []}
        grouped[skill_id]["student_history"].append(int(item.get("is_correct", item.get("correctness", 0))))
    return list(grouped.values())


@app.route("/predict", methods=["POST"])
def predict():
    if lstm_model is None: return jsonify({"error": "Model not loaded"}), 500
    payload = request.get_json(silent=True)
    if not payload: return jsonify({"error": "Invalid JSON"}), 400

    skills_input = _normalize_skill_batch(payload)
    results = []
    
    for item in skills_input:
        student_history = item.get("student_history", [])
        skill_difficulty = float(item.get("skill_difficulty_avg", 0.50))
        
        mastery_score = calculate_lstm_mastery(student_history, skill_difficulty)
        
        results.append({
            "skill_id": item.get("skill_id"),
            "skill_name": item.get("skill_name"),
            "total_attempts": len(student_history),
            "total_correct": int(sum(student_history)),
            "mastery_score": mastery_score,
            "status": _status(mastery_score)
        })

    return jsonify(results if isinstance(payload, list) else results[0]), 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)