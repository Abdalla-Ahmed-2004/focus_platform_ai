from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request


app = Flask(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_DIR = PROJECT_ROOT / "models"
MODEL_PATHS = [
    MODEL_DIR / "evaluate_skill_model_vFINAL.pkl",
  
]

# Final feature list used by train.py and embedded in the saved model bundle.
DEFAULT_FEATURES = [
    "decayed_mastery",
    "skill_difficulty_avg",
    "student_skill_history",
    "user_success_rate",
    "weighted_streak",
    "consecutive_correct",
    "opportunity_count",
    "learning_momentum",
    "weighted_consistency",
    "total_experience_score",
    "complexity_gap",
    "mastery_history_gap",
    "performance_efficiency",
    "consistency_success_sync",
]


def _to_float(payload, key):
    value = payload.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        raise ValueError(f"Invalid numeric value for '{key}'")


def _build_feature_row(payload):
    features = model_features or DEFAULT_FEATURES
    row = {}

    for feature in features:
        value = payload.get(feature, 0)
        row[feature] = _to_float({feature: value}, feature)

    return row


def _extract_metadata(payload):
    """Extract non-feature fields from payload (user_id, skill_id, skill_name)."""
    metadata = {}
    if "user_id" in payload:
        metadata["user_id"] = payload["user_id"]
    if "skill_id" in payload:
        metadata["skill_id"] = payload["skill_id"]
    if "skill_name" in payload:
        metadata["skill_name"] = payload["skill_name"]
    return metadata


def _normalize_payloads(payload):
    if isinstance(payload, dict):
        for key in ("items", "predictions", "skills"):
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]

    if isinstance(payload, list):
        return payload

    raise ValueError("Request body must be a JSON object or array.")


def _status(probability, threshold):
    if probability < threshold:
        return "Weak Skill"
    if probability < (threshold + (1 - threshold) / 2):
        return "Developing"
    return "Strong Skill"


model = None
best_threshold = 0.70
model_features = DEFAULT_FEATURES
model_error = None

load_errors = []
for model_path in MODEL_PATHS:
    if not model_path.exists():
        load_errors.append(f"{model_path.name} not found")
        continue

    try:
        model_data = joblib.load(model_path)
        model = model_data["model"]
        best_threshold = float(model_data.get("threshold", model_data.get("best_threshold", 0.70)))
        model_features = model_data.get("features") or DEFAULT_FEATURES
        model_error = None
        break
    except Exception as exc:
        load_errors.append(f"{model_path.name}: {exc}")

if model is None:
    model_error = "; ".join(load_errors) if load_errors else "Unknown model loading error"


@app.get("/health")
def health():
    if model is None:
        return jsonify({"status": "error", "message": f"Model load failed: {model_error}"}), 500

    return jsonify(
        {
            "status": "ok",
            "threshold": best_threshold,
            "train_features": DEFAULT_FEATURES,
            "model_features": model_features,
        }
    )


@app.post("/predict")
def predict():
    if model is None:
        return jsonify({"error": f"Model load failed: {model_error}"}), 500

    payload = request.get_json(silent=True)

    try:
        payloads = _normalize_payloads(payload)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400

    if not payloads:
        return jsonify({"error": "Request body contains no prediction items."}), 400

    results = []

    try:
        feature_rows = []
        metadatas = []
        for index, p in enumerate(payloads):
            if not isinstance(p, dict):
                raise ValueError(f"Prediction item at index {index} must be an object.")

            feature_row = _build_feature_row(p)
            metadata = _extract_metadata(p)
            feature_rows.append(feature_row)
            metadatas.append(metadata)

        input_df = pd.DataFrame(feature_rows, columns=model_features)
        probabilities = model.predict_proba(input_df)[:, 1]

        for i, prob in enumerate(probabilities):
            result = {
                "prob": round(float(prob) * 100, 2),
                "status": _status(float(prob), best_threshold),
            }
            result.update(metadatas[i])
            results.append(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 400

    return jsonify({
        "count": len(results),
        "data": results,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)