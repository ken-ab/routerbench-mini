from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Protocol

import requests

from .normalization import canonical_answer, normalize_tool_call, parse_json_object
from .tasks import TaskExample


PROMPT_VERSION = "unified-multimodal-v2-review-correct"


@dataclass
class ModelResponse:
    role: str
    model: str
    answer: str
    raw_text: str
    confidence: float
    latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ModelResponse":
        return cls(**value)


class Provider(Protocol):
    role: str
    model: str

    def generate(self, task: TaskExample) -> ModelResponse:
        ...

    def review_and_correct(self, task: TaskExample, candidate: ModelResponse) -> ModelResponse:
        ...


def _stable_score(key: str) -> float:
    digest = hashlib.sha256(key.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF


class MockProvider:
    """Deterministic unified-multimodal provider for tests and smoke runs."""

    ACCURACY_BY_ROLE_AND_TASK = {
        "cheap": {"math": 0.55, "mcq": 0.66, "vqa": 0.58, "tool": 0.52},
        "strong": {"math": 0.88, "mcq": 0.88, "vqa": 0.84, "tool": 0.82},
    }
    LATENCY_BY_ROLE = {"cheap": 500.0, "strong": 1500.0}

    def __init__(self, role: str, model: str) -> None:
        self.role = role
        self.model = model

    def generate(self, task: TaskExample) -> ModelResponse:
        score = _stable_score(f"{self.role}:{task.id}")
        target_accuracy = self.ACCURACY_BY_ROLE_AND_TASK.get(self.role, {}).get(task.task_type, 0.50)
        correct = score < target_accuracy
        answer = self._correct_answer(task) if correct else self._wrong_answer(task)
        confidence_base = 0.78 if correct else 0.36
        confidence = max(
            0.05,
            min(0.98, confidence_base + (_stable_score(f"conf:{self.role}:{task.id}") - 0.5) * 0.22),
        )
        return ModelResponse(
            role=self.role,
            model=self.model,
            answer=answer,
            raw_text=answer,
            confidence=round(confidence, 3),
            latency_ms=self.LATENCY_BY_ROLE.get(self.role, 900.0),
            metadata={"mock_correct": correct, "self_check_pass": correct},
        )

    def review_and_correct(self, task: TaskExample, candidate: ModelResponse) -> ModelResponse:
        from .scoring import is_correct

        if is_correct(task, candidate):
            return ModelResponse(
                role=self.role,
                model=self.model,
                answer=candidate.answer,
                raw_text=candidate.raw_text,
                confidence=max(candidate.confidence, 0.9),
                latency_ms=self.LATENCY_BY_ROLE.get(self.role, 900.0),
                metadata={
                    "mock_correct": True,
                    "self_check_pass": True,
                    "inference_mode": "review_and_correct",
                    "review_action": "keep",
                    "review_changed": False,
                },
            )
        corrected = self.generate(task)
        corrected.metadata = {
            **corrected.metadata,
            "inference_mode": "review_and_correct",
            "review_action": "correct",
            "review_changed": corrected.answer != candidate.answer,
        }
        return corrected

    def _correct_answer(self, task: TaskExample) -> str:
        answer = canonical_answer(task.task_type, task.answer)
        if task.task_type == "tool":
            return _json_dumps(answer)
        return str(answer)

    def _wrong_answer(self, task: TaskExample) -> str:
        answer = canonical_answer(task.task_type, task.answer)
        if task.task_type == "math":
            try:
                return str(int(float(answer)) + 1)
            except (TypeError, ValueError):
                return "0"
        if task.is_multiple_choice:
            choices = [chr(65 + idx) for idx in range(len(task.choices))]
            if answer in choices:
                return choices[(choices.index(answer) + 1) % len(choices)]
            return "A"
        if task.task_type == "tool":
            tools = task.tools or [{"name": "unknown_tool"}]
            return _json_dumps({"name": tools[0]["name"], "arguments": {}})
        return "unknown"


class OpenAICompatibleProvider:
    """OpenAI-compatible client with fixed inference settings and disk caching."""

    def __init__(
        self,
        role: str,
        model: str,
        api_key: str,
        base_url: str,
        *,
        timeout_s: int = 120,
        temperature: float = 0.2,
        top_p: float | None = None,
        system_prompt: str | None = None,
        max_tokens: int = 256,
        enable_thinking: bool = False,
        cache_dir: str = ".cache/routerbench",
        input_price_per_million: float = 0.0,
        output_price_per_million: float = 0.0,
        currency: str = "CNY",
        retries: int = 4,
    ) -> None:
        self.role = role
        self.model = model
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout_s = timeout_s
        self.temperature = temperature
        self.top_p = top_p
        self.system_prompt = system_prompt
        self.max_tokens = max_tokens
        self.enable_thinking = enable_thinking
        self.cache_dir = Path(cache_dir)
        self.input_price_per_million = input_price_per_million
        self.output_price_per_million = output_price_per_million
        self.currency = currency
        self.retries = retries

    def generate(self, task: TaskExample) -> ModelResponse:
        return self._generate(task, mode="solve", candidate=None)

    def review_and_correct(self, task: TaskExample, candidate: ModelResponse) -> ModelResponse:
        return self._generate(task, mode="review_and_correct", candidate=candidate)

    def _generate(
        self,
        task: TaskExample,
        *,
        mode: str,
        candidate: ModelResponse | None,
    ) -> ModelResponse:
        cache_path = self._cache_path(task, mode=mode, candidate=candidate)
        if cache_path.exists():
            cached = ModelResponse.from_dict(json.loads(cache_path.read_text(encoding="utf-8")))
            if task.tools and "tool_trace" not in cached.metadata:
                parsed_tool = normalize_tool_call(cached.answer)
                cached.metadata["tool_trace"] = [parsed_tool] if parsed_tool is not None else []
            return cached

        prompt = build_prompt(task) if candidate is None else build_review_prompt(task, candidate)
        messages: list[dict[str, Any]] = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.append({"role": "user", "content": self._content_for_task(task, prompt)})
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
        }
        if self.top_p is not None:
            payload["top_p"] = self.top_p
        if task.tools:
            payload["tools"] = [self._openai_tool(tool) for tool in task.tools]
            payload["tool_choice"] = "auto"
        else:
            payload["response_format"] = {"type": "json_object"}

        start = time.perf_counter()
        data, request_attempts = self._request(payload)
        latency_ms = (time.perf_counter() - start) * 1000
        message = data["choices"][0]["message"]
        answer, raw_text, confidence, self_check_pass = self._parse_message(task, message)
        usage = data.get("usage", {})
        cost = self._usage_cost(usage)
        review_changed = candidate is not None and not _answers_equivalent(task, answer, candidate.answer)
        parsed_tool = normalize_tool_call(answer) if task.tools else None
        result = ModelResponse(
            role=self.role,
            model=self.model,
            answer=answer,
            raw_text=raw_text,
            confidence=confidence,
            latency_ms=latency_ms,
            metadata={
                "usage": usage,
                "cost": cost,
                "currency": self.currency,
                "self_check_pass": self_check_pass,
                "prompt_version": PROMPT_VERSION,
                "system_prompt": self.system_prompt,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "max_tokens": self.max_tokens,
                "inference_mode": mode,
                "review_action": "correct" if review_changed else "keep" if candidate is not None else "solve",
                "review_changed": review_changed,
                "endpoint": f"{self.base_url}/chat/completions",
                "api_response_id": data.get("id"),
                "api_created": data.get("created"),
                "api_model": data.get("model"),
                "system_fingerprint": data.get("system_fingerprint"),
                "request_attempts": request_attempts,
                "tool_trace": [parsed_tool] if parsed_tool is not None else [],
            },
        )
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(asdict(result), ensure_ascii=False), encoding="utf-8")
        return result

    def _request(self, payload: dict[str, Any]) -> tuple[dict[str, Any], int]:
        error: Exception | None = None
        for attempt in range(self.retries):
            try:
                response = requests.post(
                    f"{self.base_url}/chat/completions",
                    headers={"Authorization": f"Bearer {self.api_key}", "Content-Type": "application/json"},
                    json=payload,
                    timeout=self.timeout_s,
                )
                if response.status_code == 429 or response.status_code >= 500:
                    raise requests.HTTPError(f"retryable status {response.status_code}", response=response)
                response.raise_for_status()
                return response.json(), attempt + 1
            except (requests.RequestException, ValueError) as exc:
                error = exc
                if attempt + 1 < self.retries:
                    time.sleep(2**attempt)
        assert error is not None
        raise error

    def _parse_message(self, task: TaskExample, message: dict[str, Any]) -> tuple[str, str, float, bool]:
        tool_calls = message.get("tool_calls") or []
        if tool_calls:
            function = tool_calls[0].get("function") or {}
            arguments = function.get("arguments") or "{}"
            try:
                arguments = json.loads(arguments) if isinstance(arguments, str) else arguments
            except json.JSONDecodeError:
                arguments = {}
            call = {"name": function.get("name"), "arguments": arguments}
            valid = normalize_tool_call(call) is not None
            text = _json_dumps(call)
            return text, text, 0.75 if valid else 0.25, valid

        raw_text = str(message.get("content") or "").strip()
        parsed = parse_json_object(raw_text)
        if parsed is not None and "answer" in parsed:
            value = parsed["answer"]
            answer = _json_dumps(value) if isinstance(value, (dict, list)) else str(value)
            confidence = _coerce_confidence(parsed.get("confidence"), default=0.5)
            self_check = str(parsed.get("self_check", "pass")).strip().lower()
            return answer, raw_text, confidence, self_check not in {"fail", "failed", "false", "reject"}
        return raw_text, raw_text, _extract_confidence(raw_text), bool(raw_text)

    def _content_for_task(self, task: TaskExample, prompt: str) -> Any:
        if not task.image_path:
            return prompt
        image_path = Path(task.image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image for {task.id} does not exist: {image_path}")
        mime = "image/png" if image_path.suffix.lower() == ".png" else "image/jpeg"
        encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
        return [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{encoded}"}},
        ]

    def _openai_tool(self, tool: dict[str, Any]) -> dict[str, Any]:
        parameters = tool.get("parameters")
        if not isinstance(parameters, dict):
            parameters = {
                "type": "object",
                "properties": tool.get("properties") or {},
                "required": tool.get("required") or [],
            }
        if parameters.get("type") == "dict":
            parameters = {**parameters, "type": "object"}
        return {
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": parameters,
            },
        }

    def _usage_cost(self, usage: dict[str, Any]) -> float:
        input_tokens = float(usage.get("prompt_tokens", 0))
        output_tokens = float(usage.get("completion_tokens", 0))
        return (
            input_tokens * self.input_price_per_million
            + output_tokens * self.output_price_per_million
        ) / 1_000_000

    def _cache_path(
        self,
        task: TaskExample,
        *,
        mode: str,
        candidate: ModelResponse | None,
    ) -> Path:
        payload = {
            "prompt_version": PROMPT_VERSION,
            "mode": mode,
            "model": self.model,
            "task": task.to_dict(),
            "candidate_answer": candidate.answer if candidate is not None else None,
            "temperature": self.temperature,
            "top_p": self.top_p,
            "system_prompt": self.system_prompt,
            "max_tokens": self.max_tokens,
            "enable_thinking": self.enable_thinking,
        }
        digest = hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()
        return self.cache_dir / self.model / f"{digest}.json"


def build_prompt(task: TaskExample) -> str:
    if task.tools:
        return (
            "Select and call exactly one provided function that satisfies the user request. "
            "Do not answer the request directly."
            f"\nUser request: {task.question}"
        )

    lines = [
        "Solve and self-check internally. Do not output reasoning or explanation.",
        f"Task type: {task.task_type}",
        f"Question: {task.question}",
    ]
    if task.choices:
        lines.append("Choices:")
        lines.extend(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(task.choices))
        answer_instruction = "Set answer to one choice letter only."
    elif task.task_type == "math":
        answer_instruction = "Set answer to the final number only."
    else:
        answer_instruction = "Set answer to a short exact answer only."
    lines.extend(
        [
            answer_instruction,
            "Return one JSON object only, with no markdown:",
            '{"answer":"...","confidence":0.0,"self_check":"pass or fail"}',
            "Confidence must be between 0 and 1 and should reflect uncertainty after self-checking.",
        ]
    )
    return "\n".join(lines)


def build_review_prompt(task: TaskExample, candidate: ModelResponse) -> str:
    if task.tools:
        return (
            "Act as the final reviewer for a candidate function call. Independently verify the candidate "
            "against the user request and every provided function schema. If it is fully correct, preserve "
            "the same function and arguments. If it is wrong, call the correct function with corrected "
            "arguments. Call exactly one provided function and do not answer in plain text."
            f"\nUser request: {task.question}"
            f"\nCandidate function call: {candidate.answer}"
        )

    lines = [
        "Act as the final reviewer for a candidate answer.",
        "Independently solve and verify the task before judging the candidate.",
        "If the candidate is correct, preserve it exactly. If it is wrong, return the corrected answer.",
        "Do not output reasoning or explanation.",
        f"Task type: {task.task_type}",
        f"Question: {task.question}",
    ]
    if task.choices:
        lines.append("Choices:")
        lines.extend(f"{chr(65 + idx)}. {choice}" for idx, choice in enumerate(task.choices))
        answer_instruction = "Set answer to one choice letter only."
    elif task.task_type == "math":
        answer_instruction = "Set answer to the final number only."
    else:
        answer_instruction = "Set answer to a short exact answer only."
    lines.extend(
        [
            f"Candidate answer: {candidate.answer}",
            answer_instruction,
            "Return one JSON object only, with no markdown:",
            '{"answer":"...","confidence":0.0,"self_check":"pass or fail","review_action":"keep or correct"}',
        ]
    )
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
        return OpenAICompatibleProvider(
            role=role,
            model=model,
            api_key=api_key,
            base_url=base_url,
            timeout_s=int(config.get("timeout_s", 120)),
            temperature=float(config.get("temperature", 0.2)),
            top_p=float(config["top_p"]) if config.get("top_p") is not None else None,
            system_prompt=str(config["system_prompt"]) if config.get("system_prompt") else None,
            max_tokens=int(config.get("max_tokens", 256)),
            enable_thinking=bool(config.get("enable_thinking", False)),
            cache_dir=str(config.get("cache_dir", ".cache/routerbench")),
            input_price_per_million=float(config.get("input_price_per_million", 0.0)),
            output_price_per_million=float(config.get("output_price_per_million", 0.0)),
            currency=str(config.get("currency", "CNY")),
            retries=int(config.get("retries", 4)),
        )
    raise ValueError(f"Unknown provider type for {role}: {provider_type}")


def _coerce_confidence(value: Any, default: float) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return default


def _extract_confidence(text: str) -> float:
    lowered = text.lower()
    marker = "confidence:"
    if marker not in lowered:
        return 0.5
    try:
        raw = lowered.rsplit(marker, 1)[1].strip().split()[0]
        return _coerce_confidence(raw, 0.5)
    except IndexError:
        return 0.5


def _answers_equivalent(task: TaskExample, left: str, right: str) -> bool:
    left_value = canonical_answer(task.task_type, left)
    right_value = canonical_answer(task.task_type, right)
    if isinstance(left_value, dict) or isinstance(right_value, dict):
        return _json_dumps(left_value) == _json_dumps(right_value)
    return str(left_value).strip().lower() == str(right_value).strip().lower()


def _json_dumps(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)
