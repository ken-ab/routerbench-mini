from __future__ import annotations

import json
import re
from typing import Any

from .normalization import canonical_answer, extract_choice, extract_number, normalize_tool_call
from .providers import ModelResponse
from .tasks import TaskExample


def is_correct(task: TaskExample, response: ModelResponse | str) -> bool:
    text = response.answer if isinstance(response, ModelResponse) else response
    expected = canonical_answer(task.task_type, task.answer)

    if task.task_type == "math":
        predicted = extract_number(text)
        return predicted is not None and predicted == expected

    if task.is_multiple_choice:
        predicted = extract_choice(text)
        return predicted is not None and predicted == expected

    if task.task_type == "vqa":
        return _open_answers_equal(text, task.answer)

    if task.task_type == "tool":
        predicted = normalize_tool_call(text)
        return _tool_calls_equal(predicted, expected)

    return str(text).strip().lower() == str(expected).strip().lower()


def _open_answers_equal(predicted: str, expected: Any) -> bool:
    predicted_number = extract_number(predicted)
    expected_number = extract_number(str(expected))
    if predicted_number is not None and expected_number is not None:
        left = float(predicted_number)
        right = float(expected_number)
        tolerance = max(1e-6, abs(right) * 0.05)
        return abs(left - right) <= tolerance
    return _normalize_text(predicted) == _normalize_text(str(expected))


def _normalize_text(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^\w\s.-]", "", value)
    return re.sub(r"\s+", " ", value)


def _tool_calls_equal(predicted: dict[str, Any] | None, expected: dict[str, Any] | None) -> bool:
    if predicted is None or expected is None:
        return False
    if predicted.get("name") != expected.get("name"):
        return False
    predicted_args = predicted.get("arguments") or {}
    expected_args = expected.get("arguments") or {}
    for key, value in expected_args.items():
        if key not in predicted_args:
            return False
        if _canonical_value(predicted_args[key]) != _canonical_value(value):
            return False
    return True


def _canonical_value(value: Any) -> str:
    if isinstance(value, str):
        return value.strip().lower()
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).lower()
