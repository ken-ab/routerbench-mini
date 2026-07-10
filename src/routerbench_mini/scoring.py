from __future__ import annotations

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

    if task.task_type == "vqa":
        predicted = extract_choice(text)
        return predicted is not None and predicted == expected

    if task.task_type == "tool":
        predicted = normalize_tool_call(text)
        return _tool_calls_equal(predicted, expected)

    return str(text).strip().lower() == str(expected).strip().lower()


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
        if str(predicted_args[key]).strip().lower() != str(value).strip().lower():
            return False
    return True

