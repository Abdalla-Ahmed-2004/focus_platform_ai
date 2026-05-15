from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from urllib import error, request

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from preprocessing.skill_snapshot_preprocessor import MODEL_FEATURES, build_ai_prediction_payload


DEFAULT_FIXTURE = ROOT / "data" / "student_answers_test.json"
DEFAULT_API_URL = os.environ.get("AI_PREDICTOR_URL", "http://127.0.0.1:5000/predict")


def load_fixture(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def validate_snapshots(snapshots: list[dict], fixture: dict) -> None:
    expected_skill_ids = [item["skill_id"] for item in fixture["items"]]
    unique_expected_skills = []
    for skill_id in expected_skill_ids:
        if skill_id not in unique_expected_skills:
            unique_expected_skills.append(skill_id)

    assert len(snapshots) == len(unique_expected_skills), (
        f"Expected {len(unique_expected_skills)} snapshots, got {len(snapshots)}"
    )

    snapshot_by_skill = {snapshot["skill_id"]: snapshot for snapshot in snapshots}
    for skill_id in unique_expected_skills:
        snapshot = snapshot_by_skill[skill_id]

        for feature_name in MODEL_FEATURES:
            assert feature_name in snapshot, f"Missing feature '{feature_name}' for skill {skill_id}"

        expected_difficulty = round(float(fixture["skills_difficulty"][str(skill_id)]), 4)
        assert snapshot["skill_difficulty_avg"] == expected_difficulty, (
            f"Difficulty mismatch for skill {skill_id}: expected {expected_difficulty}, got {snapshot['skill_difficulty_avg']}"
        )


def post_to_flask(api_url: str, payload: dict) -> dict:
    body = json.dumps(payload).encode("utf-8")
    req = request.Request(api_url, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with request.urlopen(req, timeout=20) as response:
        response_text = response.read().decode("utf-8")
        return json.loads(response_text)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the student answer to skill snapshot flow.")
    parser.add_argument("--fixture", type=Path, default=DEFAULT_FIXTURE, help="Path to the sample student answer JSON file.")
    parser.add_argument("--post", action="store_true", help="Post the fixture to the Flask /predict endpoint after preprocessing.")
    parser.add_argument("--api-url", default=DEFAULT_API_URL, help="Flask prediction URL.")
    args = parser.parse_args()

    fixture = load_fixture(args.fixture)
    snapshots = build_ai_prediction_payload(fixture)

    print("Generated skill snapshots:")
    print(json.dumps(snapshots, indent=2, ensure_ascii=False))

    validate_snapshots(snapshots, fixture)
    print(f"Validated {len(snapshots)} skill snapshots against the model feature list.")

    if args.post:
        api_response = post_to_flask(args.api_url, fixture)
        print("Flask /predict response:")
        print(json.dumps(api_response, indent=2, ensure_ascii=False))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
