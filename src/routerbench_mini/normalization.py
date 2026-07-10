from __future__ import annotations

import json
import re
from typing import Any


NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
CHOICE_RE = re.compile(r"\b([A-D])\b", re.IGNORECASE)


def extract_number(text: str) -> str | None:
    matches = NUMBER_RE.findall(text.replace(",", ""))
    if not matches:
        return None
    value = matches[-1]
    if value.endswith(".0"):
        value = value[:-2]
    return value


def extract_choice(text: str) -> str | None:
    match = CHOICE_RE.search(text.strip())
    if not match:
        return None
    return match.group(1).upper()


def parse_json_object(text: str) -> dict[str, Any] | None:
    text = text.strip()
    candidates = [text]
    if "```" in text:
        fenced = re.findall(r"```(?:json)?\s*(.*?)```", text, flags=re.DOTALL | re.IGNORECASE)
        candidates = fenced + candidates
    brace_start = text.find("{")
    brace_end = text.rfind("}")
    if brace_start >= 0 and brace_end > brace_start:
        candidates.insert(0, text[brace_start : brace_end + 1])

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def normalize_tool_call(value: Any) -> dict[str, Any] | None:
    if isinstance(value, dict):
        parsed = value
    elif isinstance(value, str):
        parsed = parse_json_object(value)
        if parsed is None:
            return None
    else:
        return None

    if "function" in parsed and isinstance(parsed["function"], dict):
        parsed = parsed["function"]

    name = parsed.get("name") or parsed.get("function_name")
    arguments = parsed.get("arguments") or parsed.get("args") or {}
    if not isinstance(name, str) or not isinstance(arguments, dict):
        return None
    return {"name": name, "arguments": arguments}


def canonical_answer(task_type: str, answer: Any) -> Any:
    if task_type == "math":
        if isinstance(answer, (int, float)):
            return str(int(answer)) if float(answer).is_integer() else str(answer)
        return extract_number(str(answer)) or str(answer).strip()
    if task_type == "vqa":
        return extract_choice(str(answer)) or str(answer).strip().upper()
    if task_type == "tool":
        return normalize_tool_call(answer)
    return str(answer).strip().lower()

