from __future__ import annotations

from routerbench_mini.providers import (
    ModelResponse,
    OpenAICompatibleProvider,
    build_review_prompt,
)
from routerbench_mini.tasks import TaskExample


def _candidate(answer: str) -> ModelResponse:
    return ModelResponse(
        role="cheap",
        model="cheap-test",
        answer=answer,
        raw_text=answer,
        confidence=0.7,
        latency_ms=1.0,
    )


def test_review_prompt_contains_candidate_and_preservation_instruction() -> None:
    task = TaskExample(
        id="mcq-1",
        dataset="unit",
        task_type="mcq",
        question="Which value is even?",
        answer=1,
        choices=["3", "4", "5", "7"],
    )

    prompt = build_review_prompt(task, _candidate("B"))

    assert "Candidate answer: B" in prompt
    assert "preserve it exactly" in prompt
    assert "If it is wrong" in prompt


def test_review_cache_key_depends_on_mode_and_candidate() -> None:
    task = TaskExample(id="math-1", dataset="unit", task_type="math", question="2 + 2?", answer="4")
    provider = OpenAICompatibleProvider(
        role="strong",
        model="strong-test",
        api_key="not-used",
        base_url="https://example.invalid/v1",
    )

    solve_path = provider._cache_path(task, mode="solve", candidate=None)
    review_four = provider._cache_path(task, mode="review_and_correct", candidate=_candidate("4"))
    review_five = provider._cache_path(task, mode="review_and_correct", candidate=_candidate("5"))

    assert len({solve_path, review_four, review_five}) == 3
