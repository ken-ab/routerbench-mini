from __future__ import annotations

import base64
import hashlib
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from .normalization import canonical_answer
from .tasks import TaskExample


@dataclass
class ModelResponse:
    role: str
    model: str
    answer: str
    raw_text: str
    confidence: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class Provider(Protocol):
    role: str
    model: str

    def generate(self, task: TaskExample) -> ModelResponse:
        ...


def _stable_score(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockProvider:
    """Deterministic provider used to test the benchmark without API keys."""

    ACCURACY_BY_ROLE_AND_TASK = {
        "cheap_text": {"math": 0.50, "vqa": 0.05, "tool": 0.45},
        "strong_text": {"math": 0.86, "vqa": 0.20, "tool": 0.76},
        "cheap_vlm": {"math": 0.40, "vqa": 0.62, "tool": 0.25},
        "strong_vlm": {"math": 0.72, "vqa": 0.84, "tool": 0.55},
    }

    LATENCY_BY_ROLE = {
        "cheap_text": 350.0,
        "strong_text": 1200.0,
        "cheap_vlm": 700.0,
        "strong_vlm": 1800.0,
    }

    def __init__(self, role: str, model: str) -> None:
        self.role = role
        self.model = model

    def generate(self, task: TaskExample) -> ModelResponse:
        score = _stable_score(f"{self.role}:{task.id}")
        target_accuracy = self.ACCURACY_BY_ROLE_AND_TASK.get(self.role, {}).get(task.task_type, 0.40)
        # Text-only mock models should be heavily penalized on image-grounded tasks.
        if task.requires_vision and self.role.endswith("text"):
            target_accuracy = min(target_accuracy, 0.12)

        correct = score < target_accuracy
        answer = self._correct_answer(task) if correct else self._wrong_answer(task)
        confidence_base = 0.76 if correct else 0.34
        confidence = max(0.05, min(0.98, confidence_base + (_stable_score(f"conf:{self.role}:{task.id}") - 0.5) * 0.22))
        latency_ms = self.LATENCY_BY_ROLE.get(self.role, 900.0)

        return ModelResponse(
            role=self.role,
            model=self.model,
            answer=answer,
            raw_text=answer,
            confidence=round(confidence, 3),
            latency_ms=latency_ms,
            metadata={"mock_correct": correct},
        )

    def _correct_answer(self, task: TaskExample) -> str:
        answer = canonical_answer(task.task_type, task.answer)
        if task.task_type == "tool":
            return _json_dumps(answer)
        if task.task_type == "math":
            return f"The answer is {answer}."
        return str(answer)

    def _wrong_answer(self, task: TaskExample) -> str:
        answer = canonical_answer(task.task_type, task.answer)
        if task.task_type == "math":
            try:
                return f"The answer is {int(float(answer)) + 1}."
            except (TypeError, ValueError):
                return "The answer is 0."
        if task.task_type == "vqa":
            choices = ["A", "B", "C", "D"]
            if answer in choices:
                return choices[(choices.index(answer) + 1) % len(choices)]
            return "A"
        if task.task_type == "tool":
            tools = task.tools or [{"name": "unknown_tool"}]
            return _json_dumps({"name": tools[0]["name"], "arguments": {}})
        return "I am not sure."


class OpenAICompatibleProvider:
    """Minimal OpenAI-compatible chat completions client.

    The provider is intentionally small: it lets the repo run against DashScope,
    OpenRouter, vLLM, or any endpoint exposing /chat/completions.
    """

    def __init__(self, role: str, model: str, api_key: str, base_url: str, timeout_s: int = 60) -> None:
        self.role = role
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s

    def generate(self, task: TaskExample) -> ModelResponse:
        start = time.perf_counter()
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": self._content_for_task(task)}],
            "temperature": 0,
        }
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=self.timeout_s,
        )
        response.raise_for_status()
        data = response.json()
        text = data["choices"][0]["message"]["content"]
        confidence = _extract_confidence(text)
        return ModelResponse(
            role=self.role,
            model=self.model,
            answer=text,
            raw_text=text,
            confidence=confidence,
            latency_ms=(time.perf_counter() - start) * 1000,
            metadata={"usage": data.get("usage", {})},
        )

    def _content_for_task(self, task: TaskExample) -> Any:
        prompt = build_prompt(task)
        if not task.image_path:
            return prompt

        image_path = Path(task.image_path)
        if not image_path.exists():
            return prompt
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]


def build_prompt(task: TaskExample) -> str:
    lines = [
        "Answer the task concisely.",
        "Return only the final answer unless a tool call JSON is required.",
        f"Task type: {task.task_type}",
        f"Question: {task.question}",
    ]
    if task.choices:
        choice_lines = [f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(task.choices)]
        lines.append("Choices:\n" + "\n".join(choice_lines))
        lines.append("Return one letter: A, B, C, or D.")
    if task.tools:
        lines.append("Available tools:")
        for tool in task.tools:
            lines.append(_json_dumps(tool))
        lines.append('Return JSON: {"name": "...", "arguments": {...}}')
    lines.append("Also include confidence as 'confidence: <0-1>' on a final line if possible.")
    return "\n".join(lines)


def provider_from_config(role: str, config: dict[str, Any]) -> Provider:
    provider_type = config.get("type", "mock")
    model = str(config.get("model", role))
    if provider_type == "mock":
        return MockProvider(role=role, model=model)
    if provider_type == "openai_compatible":
        api_key = os.environ.get(str(config.get("api_key_env", "OPENAI_API_KEY")))
        base_url = os.environ.get(str(config.get("base_url_env", "OPENAI_BASE_URL")))
        if not api_key or not base_url:
            raise RuntimeError(
                f"Provider {role} needs env vars {config.get('api_key_env')} and {config.get('base_url_env')}."
            )
        return OpenAICompatibleProvider(role=role, model=model, api_key=api_key, base_url=base_url)
    raise ValueError(f"Unknown provider type for {role}: {provider_type}")


def _extract_confidence(text: str) -> float:
    lowered = text.lower()
    marker = "confidence:"
    if marker not in lowered:
        return 0.5
    try:
        raw = lowered.rsplit(marker, 1)[1].strip().split()[0]
        return max(0.0, min(1.0, float(raw)))
    except (IndexError, ValueError):
        return 0.5


def _json_dumps(value: Any) -> str:
    import json

    return json.dumps(value, ensure_ascii=False, sort_keys=True)

