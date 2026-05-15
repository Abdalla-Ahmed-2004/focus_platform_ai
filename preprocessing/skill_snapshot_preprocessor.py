from __future__ import annotations

from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import numpy as np
import pandas as pd

MODEL_FEATURES = [
    'decayed_mastery',
    'skill_difficulty_avg',
    'student_skill_history',
    'complexity_gap',
    'learning_momentum',
    'consecutive_correct',
    'opportunity_count',
    'weighted_consistency',
    'total_experience_score',
    'mastery_history_gap',
    'weighted_streak',
    'performance_efficiency',
    'consistency_success_sync',
]


def _coerce_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or value == '':
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _coerce_int(value: Any, default: int = 0) -> int:
    try:
        if value is None or value == '':
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _normalize_correct_value(row: Mapping[str, Any]) -> int:
    for key in ('is_correct', 'correct', 'correctness'):
        if key in row:
            return 1 if _coerce_int(row.get(key), 0) else 0
    return 0


def _normalize_payload(payload: Any) -> Tuple[List[Dict[str, Any]], Optional[Dict[Any, Any]]]:
    skills_difficulty = None

    if isinstance(payload, pd.DataFrame):
        return payload.to_dict('records'), skills_difficulty

    if isinstance(payload, list):
        return payload, skills_difficulty

    if isinstance(payload, dict):
        maybe_difficulty = payload.get('skills_difficulty')
        if isinstance(maybe_difficulty, dict):
            skills_difficulty = maybe_difficulty

        for key in ('items', 'answers', 'predictions', 'skills', 'tests'):
            value = payload.get(key)
            if isinstance(value, list):
                return value, skills_difficulty

        return [payload], skills_difficulty

    raise ValueError('Payload must be a list, DataFrame, or JSON object.')


def _order_key(row: Mapping[str, Any]) -> Tuple[float, str]:
    order_id = row.get('order_id')
    try:
        order_value = float(order_id)
    except (TypeError, ValueError):
        order_value = float('inf')

    return order_value, str(row.get('skill_id', ''))


def _group_rows_by_skill(records: Iterable[Mapping[str, Any]]) -> List[Tuple[Any, List[Dict[str, Any]]]]:
    grouped: Dict[Any, List[Dict[str, Any]]] = {}
    ordered_skill_ids: List[Any] = []

    for row in sorted(records, key=_order_key):
        skill_id = row.get('skill_id')
        if skill_id is None:
            continue

        if skill_id not in grouped:
            grouped[skill_id] = []
            ordered_skill_ids.append(skill_id)

        grouped[skill_id].append(dict(row))

    return [(skill_id, grouped[skill_id]) for skill_id in ordered_skill_ids]


def _resolve_skill_difficulty(skill_id: Any, rows: List[Dict[str, Any]], skills_difficulty: Optional[Dict[Any, Any]]) -> float:
    if isinstance(skills_difficulty, dict):
        candidate_keys = [skill_id, str(skill_id), _coerce_int(skill_id, skill_id)]
        for candidate_key in candidate_keys:
            if candidate_key in skills_difficulty:
                return _coerce_float(skills_difficulty[candidate_key], 0.0)

    for row in rows:
        if row.get('skill_difficulty_avg') is not None:
            return _coerce_float(row.get('skill_difficulty_avg'), 0.0)

    return 0.0


def _build_skill_snapshot(skill_rows: List[Dict[str, Any]], skill_difficulty_avg: float, include_history: bool = False) -> Dict[str, Any]:
    ordered_rows = sorted(skill_rows, key=_order_key)
    if not ordered_rows:
        return {}

    alpha = 0.55
    current_mastery = 0.5
    streak = 0
    previous_corrects: List[int] = []
    history: List[Dict[str, Any]] = []
    last_snapshot: Dict[str, Any] = {}
    first_row = ordered_rows[0]

    for position, row in enumerate(ordered_rows):
        current_correct = _normalize_correct_value(row)
        student_skill_history = float(np.mean(previous_corrects)) if previous_corrects else (1 - skill_difficulty_avg)

        last_snapshot = {
            'order_id': row.get('order_id'),
            'skill_id': row.get('skill_id'),
            'skill_name': row.get('skill_name') or first_row.get('skill_name'),
            'user_id': row.get('user_id'),
            'decayed_mastery': round(float(current_mastery), 4),
            'skill_difficulty_avg': round(float(skill_difficulty_avg), 4),
            'student_skill_history': round(float(student_skill_history), 4),
            'complexity_gap': round(float(np.clip(student_skill_history - skill_difficulty_avg, -1, 1)), 4),
            'learning_momentum': round(float(np.clip(current_mastery - student_skill_history, -1, 1)), 4),
            'consecutive_correct': int(streak),
            'opportunity_count': int(position),
            'weighted_consistency': round(float(np.log2(streak + 1) * (1 - skill_difficulty_avg)), 4),
            'total_experience_score': round(float(np.log1p(position)), 4),
            'mastery_history_gap': round(float(current_mastery - student_skill_history), 4),
            'weighted_streak': round(float(min(streak * np.power(2.0, skill_difficulty_avg), 15.0)), 4),
            'performance_efficiency': round(float(np.clip(student_skill_history / (skill_difficulty_avg + 0.05), 0, 15)), 4),
            'consistency_success_sync': round(float(np.clip(current_mastery * (1 - skill_difficulty_avg) * 1.5, 0, 1)), 4),
        }

        if include_history:
            history.append({**last_snapshot, 'is_correct': current_correct})

        current_mastery = current_mastery + alpha * (current_correct - current_mastery)
        previous_corrects.append(current_correct)
        streak = streak + 1 if current_correct == 1 else 0

    last_snapshot['answer_count'] = len(ordered_rows)
    if include_history:
        last_snapshot['history'] = history

    return last_snapshot


def build_skill_snapshots(payload: Any, skills_difficulty: Optional[Dict[Any, Any]] = None, include_history: bool = False) -> List[Dict[str, Any]]:
    records, extracted_skill_difficulty = _normalize_payload(payload)
    if skills_difficulty is None:
        skills_difficulty = extracted_skill_difficulty

    snapshots: List[Dict[str, Any]] = []
    for skill_id, rows in _group_rows_by_skill(records):
        difficulty = _resolve_skill_difficulty(skill_id, rows, skills_difficulty)
        snapshot = _build_skill_snapshot(rows, difficulty, include_history=include_history)
        if snapshot:
            snapshot['skill_id'] = skill_id
            snapshot['skill_difficulty_avg'] = round(float(difficulty), 4)
            snapshots.append(snapshot)

    return snapshots


def build_ai_prediction_payload(payload: Any, skills_difficulty: Optional[Dict[Any, Any]] = None, include_history: bool = False) -> List[Dict[str, Any]]:
    return build_skill_snapshots(payload=payload, skills_difficulty=skills_difficulty, include_history=include_history)
