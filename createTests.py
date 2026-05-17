import numpy as np
import json

def get_reasonable_score(features):
    """
    A heuristics engine that simulates the real model weights to produce a realistic percentage.
    Weights depend on features: Mastery, History, and Difficulty.
    """
    # Convert values to percentage (0-100)
    # 1. Effect of mastery and student history (positive)
    mastery_weight = features['decayed_mastery'] * 40 
    history_weight = features['student_skill_history'] * 20
    
    # 2. Effect of difficulty (negative)
    difficulty_penalty = features['skill_difficulty_avg'] * 15
    
    # 3. Effect of continuity and momentum (positive)
    momentum_bonus = (features['learning_momentum'] + 1) * 10 # map from (-1,1) to (0,20)
    streak_bonus = min(features['weighted_streak'] * 2, 15) # capped at 15%
    
    # compute the final score
    raw_score = 30 + mastery_weight + history_weight - difficulty_penalty + momentum_bonus + streak_bonus
    
    # ensure the percentage stays between 5% and 98% for realism (no 100% certainty in learning)
    final_percentage = np.clip(raw_score, 5.0, 98.0)
    return round(float(final_percentage), 2)

def generate_final_test():
    print("="*60)
    print("📊 AI DATA GENERATOR & PERCENTAGE PREDICTOR")
    print("="*60)

    try:
        # Terminal inputs
        is_correct = int(input("Current Answer (1/0): "))
        skill_diff = float(input("Skill Difficulty (0.0-1.0): "))
        prev_streak = int(input("Previous Streak: "))
        prev_opps = int(input("Previous Practice Count: "))
        skill_hist = float(input("Skill History Rate (0.0-1.0): "))
        prev_mastery = float(input("Previous Mastery (0.0-1.0): "))
        total_success = float(input("Overall Success Rate (0.0-1.0): "))

        # compute derived features (Logic v3.3)
        alpha = 0.55
        new_mastery = prev_mastery + alpha * (is_correct - prev_mastery)
        
        # derived features
        perf_eff = np.clip(skill_hist / (skill_diff + 0.05), 0, 15)
        consist_sync = np.clip(new_mastery * (1 - skill_diff) * 1.5, 0, 1)
        w_streak = np.clip(prev_streak * np.power(2.0, skill_diff), 0, 15.0)
        comp_gap = np.clip(skill_hist - skill_diff, -1, 1)
        learn_mom = np.clip(new_mastery - skill_hist, -1, 1)
        w_consist = np.log2(prev_streak + 1) * (1 - skill_diff)
        exp_score = np.log1p(prev_opps)
        mast_hist_gap = new_mastery - skill_hist

        # build the JSON snapshot
        snapshot = {
            "decayed_mastery": round(new_mastery, 4),
            "skill_difficulty_avg": round(skill_diff, 4),
            "student_skill_history": round(skill_hist, 4),
            "complexity_gap": round(comp_gap, 4),
            "learning_momentum": round(learn_mom, 4),
            "consecutive_correct": prev_streak,
            "opportunity_count": prev_opps,
            "weighted_consistency": round(w_consist, 4),
            "total_experience_score": round(exp_score, 4),
            "mastery_history_gap": round(mast_hist_gap, 4),
            "weighted_streak": round(w_streak, 4),
            "user_success_rate": round(total_success, 4),
            "performance_efficiency": round(perf_eff, 4),
            "consistency_success_sync": round(consist_sync, 4)
        }

        # get the heuristic percentage
        predicted_percentage = get_reasonable_score(snapshot)

        # final output
        final_json = {
            "status": "success",
            "data_payload": snapshot,
            "prediction": {
                "success_probability": f"{predicted_percentage}%",
                "recommendation": "Pass to Laravel API"
            }
        }
        return final_json
        # print("\n" + "✨" * 20)
        # print(json.dumps(final_json, indent=4))
        # print("✨" * 20)

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print(generate_final_test())