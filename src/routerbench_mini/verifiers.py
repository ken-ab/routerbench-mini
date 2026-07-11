from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .normalization import extract_choice, extract_number, normalize_tool_call
from .providers import ModelResponse
from .tasks import TaskExample


@dataclass(frozen=True)
class VerificationResult:
    valid_format: bool
    should_escalate: bool
    reason: str
    features: dict[str, Any]


def verify_response(task: TaskExample, response: ModelResponse, confidence_threshold: float = 0.55) -> VerificationResult:
    features: dict[str, Any] = {
        "task_type": task.task_type,
        "has_image": task.requires_vision,
        "question_length": len(task.question.split()),
        "confidence": response.confidence,
        "model_role": response.role,
        "self_check_pass": bool(response.metadata.get("self_check_pass", True)),
    }
    valid_format = _format_is_valid(task, response, features)
    reasons: list[str] = []

    if not valid_format:
        reasons.append("invalid_answer_format")
    if response.confidence < confidence_threshold:
        reasons.append("low_confidence")
    if not features["self_check_pass"]:
        reasons.append("self_check_failed")

    should_escalate = bool(reasons)
    return VerificationResult(
        valid_format=valid_format,
        should_escalate=should_escalate,
        reason=";".join(reasons) if reasons else "accepted",
        features=features,
    )


def _format_is_valid(task: TaskExample, response: ModelResponse, features: dict[str, Any]) -> bool:
    if task.task_type == "math":
        predicted = extract_number(response.answer)
        features["answer_format_valid"] = predicted is not None
        return predicted is not None

    if task.is_multiple_choice:
        predicted = extract_choice(response.answer)
        valid_choices = {chr(65 + idx) for idx in range(len(task.choices))}
        features["answer_format_valid"] = predicted in valid_choices
        return features["answer_format_valid"]

    if task.task_type == "vqa":
        features["answer_format_valid"] = bool(response.answer.strip())
        return features["answer_format_valid"]

    if task.task_type == "tool":
        parsed = normalize_tool_call(response.answer)
        available_tools = {tool.get("name") for tool in task.tools}
        required_by_tool = {
            tool.get("name"): set(
                tool.get("required")
                or (tool.get("parameters") or {}).get("required")
                or []
            )
            for tool in task.tools
        }
        features["json_valid"] = parsed is not None
        if parsed is None:
            features["answer_format_valid"] = False
            features["missing_required_args"] = None
            return False

        tool_name = parsed["name"]
        args = parsed.get("arguments") or {}
        missing = sorted(required_by_tool.get(tool_name, set()) - set(args))
        features["tool_name_valid"] = tool_name in available_tools
        features["missing_required_args"] = missing
        features["answer_format_valid"] = tool_name in available_tools and not missing
        return features["answer_format_valid"]

    features["answer_format_valid"] = bool(response.answer.strip())
    return features["answer_format_valid"]
