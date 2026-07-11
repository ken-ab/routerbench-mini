from __future__ import annotations

import re
from typing import Any

from .tasks import TaskExample


CHART_TERMS = {
    "axis",
    "bar chart",
    "chart",
    "diagram",
    "figure",
    "graph",
    "legend",
    "plot",
    "table",
    "trend",
}
OCR_TERMS = {
    "author",
    "book",
    "brand",
    "label",
    "logo",
    "printed",
    "read",
    "sign",
    "text",
    "title",
    "written",
}
LOGIC_TERMS = {
    "directly above",
    "directly below",
    "leftmost",
    "logical",
    "order",
    "rightmost",
    "sequence",
}
MATH_TERMS = {
    "average",
    "calculate",
    "difference",
    "divided",
    "each",
    "minus",
    "percent",
    "plus",
    "product",
    "sum",
    "times",
    "total",
}


def observable_task_features(task: TaskExample) -> dict[str, float]:
    """Extract inference-time features without dataset identity or labels."""

    question = task.question.lower()
    numeric_mentions = len(re.findall(r"-?\d+(?:\.\d+)?", question))
    required_args = sum(len(_required_arguments(tool)) for tool in task.tools)
    schema_depth = max((_schema_depth(_tool_schema(tool)) for tool in task.tools), default=0)
    return {
        "has_image": float(task.requires_vision),
        "is_math": float(
            _contains_any(question, MATH_TERMS)
            or bool(re.search(r"\d\s*[-+*/=]\s*\d", question))
            or numeric_mentions >= 2
        ),
        "is_tool": float(bool(task.tools)),
        "is_multiple_choice": float(task.is_multiple_choice),
        "question_words": float(len(task.question.split())),
        "numeric_mentions": float(numeric_mentions),
        "choice_count": float(len(task.choices)),
        "tool_count": float(len(task.tools)),
        "required_arg_count": float(required_args),
        "schema_depth": float(schema_depth),
        "chart_cue": float(_contains_any(question, CHART_TERMS)),
        "ocr_cue": float(_contains_any(question, OCR_TERMS)),
        "logic_cue": float(_contains_any(question, LOGIC_TERMS)),
    }


def task_risk_score(task: TaskExample) -> float:
    """Compute a transparent difficulty prior from observable request features."""

    features = observable_task_features(task)
    score = 0.0

    if features["is_math"]:
        score += 3.0
    if features["logic_cue"]:
        score += 2.0
    if features["question_words"] >= 50:
        score += 1.0

    if features["has_image"]:
        score += 2.0 * max(features["chart_cue"], features["ocr_cue"])
        if features["numeric_mentions"] >= 3:
            score += 1.0
        if features["choice_count"] >= 5:
            score += 1.0

    if features["is_tool"]:
        if features["tool_count"] >= 3:
            score += 2.0
        elif features["tool_count"] == 2:
            score += 1.0
        if features["required_arg_count"] >= 4:
            score += 1.0
        if features["schema_depth"] >= 4:
            score += 1.0

    return score


def task_feature_vector(task: TaskExample) -> list[float]:
    features = observable_task_features(task)
    return [features[name] for name in sorted(features)]


def task_feature_names() -> list[str]:
    return sorted(observable_task_features(_empty_task()))


def _contains_any(text: str, terms: set[str]) -> bool:
    return any(term in text for term in terms)


def _tool_schema(tool: dict[str, Any]) -> dict[str, Any]:
    parameters = tool.get("parameters")
    return parameters if isinstance(parameters, dict) else tool


def _required_arguments(tool: dict[str, Any]) -> list[str]:
    schema = _tool_schema(tool)
    required = schema.get("required") or tool.get("required") or []
    return [str(value) for value in required]


def _schema_depth(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + max((_schema_depth(item) for item in value.values()), default=0)
    if isinstance(value, list):
        return 1 + max((_schema_depth(item) for item in value), default=0)
    return 0


def _empty_task() -> TaskExample:
    return TaskExample(id="feature-schema", dataset="", task_type="text", question="", answer="")
