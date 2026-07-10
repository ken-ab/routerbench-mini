from __future__ import annotations

from routerbench_mini.providers import ModelResponse
from routerbench_mini.tasks import TaskExample
from routerbench_mini.verifiers import verify_response


def _response(answer: str, confidence: float = 0.9, role: str = "cheap_text") -> ModelResponse:
    return ModelResponse(
        role=role,
        model=f"{role}-unit",
        answer=answer,
        raw_text=answer,
        confidence=confidence,
        latency_ms=1.0,
    )


def test_verifier_escalates_invalid_math_format() -> None:
    task = TaskExample(id="math-1", dataset="unit", task_type="math", question="2 + 2?", answer="4")

    result = verify_response(task, _response("I am not sure.", confidence=0.9))

    assert not result.valid_format
    assert result.should_escalate
    assert "invalid_answer_format" in result.reason


def test_verifier_escalates_low_confidence_even_with_valid_format() -> None:
    task = TaskExample(id="math-2", dataset="unit", task_type="math", question="2 + 2?", answer="4")

    result = verify_response(task, _response("The answer is 4.", confidence=0.2))

    assert result.valid_format
    assert result.should_escalate
    assert "low_confidence" in result.reason


def test_verifier_flags_missing_required_tool_arguments() -> None:
    task = TaskExample(
        id="tool-1",
        dataset="unit",
        task_type="tool",
        question="Get Paris weather.",
        answer={"name": "get_weather", "arguments": {"city": "Paris"}},
        tools=[{"name": "get_weather", "required": ["city"], "properties": {"city": "string"}}],
    )

    result = verify_response(task, _response('{"name": "get_weather", "arguments": {}}'))

    assert not result.valid_format
    assert result.should_escalate
    assert result.features["missing_required_args"] == ["city"]


def test_verifier_escalates_text_model_on_vision_task() -> None:
    task = TaskExample(
        id="vqa-1",
        dataset="unit",
        task_type="vqa",
        question="Which option is correct?",
        answer="A",
        choices=["Alpha", "Beta", "Gamma", "Delta"],
        metadata={"has_image": True},
    )

    result = verify_response(task, _response("A", confidence=0.95, role="cheap_text"))

    assert result.valid_format
    assert result.should_escalate
    assert "vision_task_answered_by_text_model" in result.reason
